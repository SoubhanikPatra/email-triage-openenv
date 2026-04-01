#!/usr/bin/env python3
"""
Email Triage Environment — Baseline Inference Script

Runs an LLM agent against all three tasks (easy, medium, hard) using
the OpenAI-compatible client. Outputs the mandatory stdout format.

Environment variables
---------------------
API_BASE_URL   LLM API base URL  (default: https://router.huggingface.co/v1)
MODEL_NAME     Model identifier   (default: meta-llama/Llama-3.1-8B-Instruct)
HF_TOKEN       API key
ENV_BASE_URL   Email triage server URL (default: http://localhost:7860)

Stdout format (mandatory)
-------------------------
[START] task=<task_name> env=email-triage model=<model>
[STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or "hf_placeholder"
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:7860").rstrip("/")

TASKS = ["easy_triage", "medium_triage", "hard_triage"]
MAX_STEPS_PER_TASK = 10          # generous upper bound; each task has 5 emails
TEMPERATURE = 0.0
MAX_TOKENS = 512

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ---------------------------------------------------------------------------
# Stdout helpers (mandatory format)
# ---------------------------------------------------------------------------

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env=email-triage model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    err_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={err_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------

def env_reset(task_name: str) -> Dict[str, Any]:
    r = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_name": task_name},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def env_step(session_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"session_id": session_id, "action": action},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def env_close(session_id: str) -> None:
    try:
        requests.delete(f"{ENV_BASE_URL}/session/{session_id}", timeout=10)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert customer support triage agent for a B2B SaaS company.

Your job is to analyze incoming support emails and output a structured triage decision.

## Valid field values

priority: urgent | high | normal | low
category: billing | technical_support | feature_request | bug_report | account_management | sales_inquiry | general_inquiry | spam
routing_target: tier1_support | tier2_support | billing_team | sales_team | engineering | account_management | spam_filter | escalation
sentiment: positive | neutral | negative | very_negative
suggested_response_tone: formal | friendly | empathetic | concise

## Routing guide
- billing issues → billing_team
- simple how-to questions → tier1_support
- technical bugs / API issues → tier2_support
- sales / pricing / upgrade inquiries → sales_team
- angry enterprise customer at churn-risk or legal data request → escalation
- unsubscribe / account deletion → account_management
- spam → spam_filter

## Output format
Respond with ONLY a valid JSON object, no markdown, no explanation:
{
  "priority": "...",
  "category": "...",
  "routing_target": "...",
  "sentiment": "...",
  "requires_followup": true|false,
  "summary": "One or two sentence summary for the support rep.",
  "suggested_response_tone": "...",
  "tags": ["tag1", "tag2"]
}
"""


def build_user_prompt(obs: Dict[str, Any]) -> str:
    email_obs = obs.get("observation", obs)
    lines = [
        f"From: {email_obs.get('sender_name', '')} <{email_obs.get('sender_email', '')}>",
        f"Subject: {email_obs.get('subject', '')}",
        f"Account tier: {email_obs.get('account_tier', 'unknown')}",
        f"Prior open tickets: {email_obs.get('prior_tickets', 0)}",
        f"Thread depth: {email_obs.get('thread_length', 0)} prior messages",
        f"Has attachment: {email_obs.get('has_attachment', False)}",
        "",
        "--- Email body ---",
        email_obs.get("body", ""),
    ]
    if email_obs.get("feedback"):
        lines += ["", f"[Previous feedback: {email_obs['feedback']}]"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call with retry
# ---------------------------------------------------------------------------

DEFAULT_ACTION = {
    "priority": "normal",
    "category": "general_inquiry",
    "routing_target": "tier1_support",
    "sentiment": "neutral",
    "requires_followup": False,
    "summary": "Unable to parse email.",
    "suggested_response_tone": "friendly",
    "tags": [],
}


def call_llm(user_prompt: str) -> Dict[str, Any]:
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            text = completion.choices[0].message.content or ""
            # Strip markdown fences if present
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text.strip())
        except json.JSONDecodeError:
            if attempt == 2:
                return DEFAULT_ACTION
            time.sleep(1)
        except Exception as exc:
            print(f"[DEBUG] LLM error (attempt {attempt+1}): {exc}", file=sys.stderr)
            if attempt == 2:
                return DEFAULT_ACTION
            time.sleep(2)
    return DEFAULT_ACTION


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

def run_task(task_name: str) -> None:
    log_start(task=task_name, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    session_id: Optional[str] = None
    error_msg: Optional[str] = None

    try:
        reset_data = env_reset(task_name)
        session_id = reset_data["session_id"]
        obs = reset_data

        for step in range(1, MAX_STEPS_PER_TASK + 1):
            email_obs = obs.get("observation", obs)
            if email_obs.get("done", False):
                break

            user_prompt = build_user_prompt(obs)
            action = call_llm(user_prompt)

            step_data = env_step(session_id, action)
            reward = float(step_data.get("reward", 0.0))
            done = bool(step_data.get("done", False))
            error_msg = None  # no hard errors in this env

            rewards.append(reward)
            steps_taken = step

            action_str = json.dumps(action, separators=(",", ":"))
            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            obs = step_data
            if done:
                success = reward > 0.0 or (len(rewards) > 0 and sum(rewards) / len(rewards) >= 0.5)
                break

        # Episode-level success: average reward ≥ 0.5
        if rewards:
            success = (sum(rewards) / len(rewards)) >= 0.5

    except Exception as exc:
        error_msg = str(exc)
        print(f"[DEBUG] Episode error: {exc}", file=sys.stderr)
        success = False

    finally:
        if session_id:
            env_close(session_id)
        log_end(success=success, steps=steps_taken, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Quick health check
    try:
        r = requests.get(f"{ENV_BASE_URL}/health", timeout=10)
        r.raise_for_status()
    except Exception as exc:
        print(f"[ERROR] Cannot reach environment at {ENV_BASE_URL}: {exc}", file=sys.stderr)
        sys.exit(1)

    for task in TASKS:
        run_task(task)
        time.sleep(1)   # polite pause between tasks


if __name__ == "__main__":
    main()