"""Project-aligned inference runner for Email Triage OpenEnv."""

from __future__ import annotations

import json
import os
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple
import requests
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.1-8B-Instruct"

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")

ENV_BASE_URL = (os.getenv("ENV_BASE_URL") or "http://localhost:7860").rstrip("/")

# Define all tasks that must be evaluated
ALL_TASKS = ["easy_triage", "medium_triage", "hard_triage"]
BENCHMARK = os.getenv("BENCHMARK") or "email-triage"

MAX_STEPS = int(os.getenv("MAX_STEPS", "5"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "350"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.70"))

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert B2B SaaS support triage agent.
    Return exactly one JSON object with keys:
    priority, category, routing_target, sentiment, requires_followup,
    summary, suggested_response_tone, tags.

    Valid values:
    priority: urgent|high|normal|low
    category: billing|technical_support|feature_request|bug_report|account_management|sales_inquiry|general_inquiry|spam
    routing_target: tier1_support|tier2_support|billing_team|sales_team|engineering|account_management|spam_filter|escalation
    sentiment: positive|neutral|negative|very_negative
    suggested_response_tone: formal|friendly|empathetic|concise
    tags: list of 0-5 short lower-case strings

    Rules:
    - summary should be concise and useful for a support rep.
    - avoid markdown, backticks, and extra text outside JSON.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    """Emit [START] line exactly as required."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Emit [STEP] line exactly as required."""
    done_val = "true" if done else "false"
    error_val = error if error else "null"
    # Ensure reward is never exactly 0 or 1, clamp to [0.001, 0.999]
    reward_clamped = max(0.001, min(0.999, reward))
    print(
        f"[STEP] step={step} action={action} reward={reward_clamped:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit [END] line exactly as required - NO extra fields."""
    rewards_str = ",".join(f"{max(0.001, min(0.999, r)):.2f}" for r in rewards)
    # IMPORTANT: Do NOT include task= or any other field here
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def _safe_json_dumps(data: Dict[str, Any]) -> str:
    return json.dumps(data, separators=(",", ":"), ensure_ascii=True)


def _clamp_reward(reward: float) -> float:
    """Ensure reward is never exactly 0 or 1."""
    return max(0.001, min(0.999, reward))


def _extract_observation_fields(observation: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "email_id",
        "subject",
        "body",
        "sender_email",
        "sender_name",
        "account_tier",
        "prior_tickets",
        "thread_length",
        "has_attachment",
        "feedback",
    ]
    return {key: observation.get(key) for key in keys}


def build_user_prompt(step: int, observation: Dict[str, Any], history: List[str]) -> str:
    obs_block = json.dumps(_extract_observation_fields(observation), indent=2, ensure_ascii=True)
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Current observation:
        {obs_block}

        Recent history:
        {history_block}

        Produce the best triage action as valid JSON only.
        """
    ).strip()


def _default_action() -> Dict[str, Any]:
    return {
        "priority": "normal",
        "category": "general_inquiry",
        "routing_target": "tier1_support",
        "sentiment": "neutral",
        "requires_followup": False,
        "summary": "General customer inquiry requiring standard support review.",
        "suggested_response_tone": "friendly",
        "tags": ["general_inquiry"],
    }


def _normalize_action(candidate: Dict[str, Any]) -> Dict[str, Any]:
    action = _default_action()

    if isinstance(candidate.get("priority"), str):
        action["priority"] = candidate["priority"].strip().lower()
    if isinstance(candidate.get("category"), str):
        action["category"] = candidate["category"].strip().lower()
    if isinstance(candidate.get("routing_target"), str):
        action["routing_target"] = candidate["routing_target"].strip().lower()
    if isinstance(candidate.get("sentiment"), str):
        action["sentiment"] = candidate["sentiment"].strip().lower()

    action["requires_followup"] = bool(candidate.get("requires_followup", action["requires_followup"]))

    summary = str(candidate.get("summary", action["summary"]))
    action["summary"] = summary.strip()[:300]

    if isinstance(candidate.get("suggested_response_tone"), str):
        action["suggested_response_tone"] = candidate["suggested_response_tone"].strip().lower()

    tags = candidate.get("tags", action["tags"])
    if isinstance(tags, list):
        normalized_tags: List[str] = []
        for tag in tags:
            clean = str(tag).strip().lower()
            if clean and clean not in normalized_tags:
                normalized_tags.append(clean)
            if len(normalized_tags) == 5:
                break
        action["tags"] = normalized_tags or action["tags"]

    return action


def get_model_action(client: OpenAI, step: int, observation: Dict[str, Any], history: List[str]) -> Tuple[Dict[str, Any], Optional[str]]:
    prompt = build_user_prompt(step, observation, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
            response_format={"type": "json_object"},
        )
        text = (completion.choices[0].message.content or "").strip()
        parsed = json.loads(text) if text else {}
        return _normalize_action(parsed), None
    except Exception as exc:
        return _default_action(), str(exc)


def reset_env(task_name: str) -> Tuple[str, Dict[str, Any]]:
    payload = {"task_name": task_name}
    response = requests.post(f"{ENV_BASE_URL}/reset", json=payload, timeout=30)
    response.raise_for_status()
    body = response.json()
    return body["session_id"], body["observation"]


def step_env(session_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
    payload = {"session_id": session_id, "action": action}
    response = requests.post(f"{ENV_BASE_URL}/step", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def close_session(session_id: Optional[str]) -> None:
    if not session_id:
        return
    try:
        requests.delete(f"{ENV_BASE_URL}/session/{session_id}", timeout=15)
    except Exception:
        pass


def wait_for_env(url: str, timeout: int = 60) -> bool:
    """Wait for the environment to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{url}/health", timeout=5)
            if resp.status_code == 200:
                return True
        except:
            pass
        time.sleep(2)
    return False


def run_single_task(client: OpenAI, task_name: str) -> Tuple[bool, int, float, List[float]]:
    """Run inference for a single task and return results."""
    session_id: Optional[str] = None
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        session_id, observation = reset_env(task_name)

        for step in range(1, MAX_STEPS + 1):
            if observation.get("done", False):
                break

            action, model_error = get_model_action(client, step, observation, history)
            step_result = step_env(session_id, action)
            observation = step_result.get("observation", {})

            reward = float(step_result.get("reward", 0.0) or 0.0)
            reward = _clamp_reward(reward)  # Ensure never 0 or 1
            done = bool(step_result.get("done", False))

            rewards.append(reward)
            steps_taken = step

            action_str = _safe_json_dumps(action)
            log_step(step=step, action=action_str, reward=reward, done=done, error=model_error)

            history.append(
                f"step={step} reward={reward:.2f} done={str(done).lower()} feedback={observation.get('feedback', '')}"
            )

            if done:
                break

        # Calculate score as average reward across steps
        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = _clamp_reward(score)  # Ensure in [0.001, 0.999]
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        if steps_taken == 0:
            log_step(
                step=1,
                action="null",
                reward=0.001,
                done=True,
                error=str(exc),
            )
            steps_taken = 1
            rewards.append(0.001)
            score = 0.001
            success = False
    finally:
        close_session(session_id)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return success, steps_taken, score, rewards


def main() -> None:
    """Run inference for ALL tasks defined in openenv.yaml."""
    # Wait for environment to be ready
    if not wait_for_env(ENV_BASE_URL):
        print(f"[ERROR] Environment not reachable at {ENV_BASE_URL}", flush=True)
        exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    all_results = {}
    overall_success = True
    
    print(f"\n{'='*60}", flush=True)
    print(f"Running inference for {len(ALL_TASKS)} tasks: {ALL_TASKS}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    for task_name in ALL_TASKS:
        print(f"\n--- Running task: {task_name} ---\n", flush=True)
        success, steps, score, rewards = run_single_task(client, task_name)
        all_results[task_name] = {
            "success": success,
            "steps": steps,
            "score": score,
            "rewards": rewards,
        }
        if not success:
            overall_success = False
    
    # Exit with non-zero code if any task failed
    exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()