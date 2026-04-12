"""
Email Triage Environment — FastAPI Server
Exposes the standard OpenEnv HTTP endpoints:
  POST /reset
  POST /step
  GET  /state
  GET  /health
  GET  /tasks    (bonus: list available tasks)
"""

from __future__ import annotations
import uuid
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from email_triage_env.models import TriageAction, EmailObservation, TriageState
from server.environment import EmailTriageEnvironment
from email_triage_env.email_data import TASK_EMAIL_MAP
from server.graders import GRADERS

app = FastAPI(
    title="Email Triage Environment",
    description=(
        "An OpenEnv-compatible RL environment for training agents to "
        "triage B2B SaaS support emails."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Session store — one env instance per session_id
# ---------------------------------------------------------------------------

_sessions: Dict[str, EmailTriageEnvironment] = {}


def _get_or_create(session_id: Optional[str]) -> tuple[str, EmailTriageEnvironment]:
    if session_id and session_id in _sessions:
        return session_id, _sessions[session_id]
    sid = str(uuid.uuid4())
    env = EmailTriageEnvironment()
    _sessions[sid] = env
    return sid, env


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_name: str = "easy_triage"
    seed: Optional[int] = None
    session_id: Optional[str] = None


class StepRequest(BaseModel):
    action: TriageAction
    session_id: Optional[str] = None


class ResetResponse(BaseModel):
    session_id: str
    observation: EmailObservation


class StepResponse(BaseModel):
    session_id: str
    observation: EmailObservation
    reward: float
    done: bool


class StateResponse(BaseModel):
    session_id: str
    state: TriageState

class GraderRequest(BaseModel):
    task_id: str
    state: Dict[str, Any]
    reward: float


class GraderResponse(BaseModel):
    task_id: str
    score: float
# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy", "environment": "email-triage"}

@app.get("/")
def root():
    return {"status": "ok", "message": "Email Triage Environment running"}

@app.get("/tasks")
def list_tasks():
    descriptions = {
        "easy_triage": "5 clear-signal emails. Unambiguous priority, category, and routing.",
        "medium_triage": "5 emails with mixed signals, thread context, and multi-part issues.",
        "hard_triage": "5 ambiguous emails with churn risk, legal implications, and competing signals.",
    }
    difficulties = {
        "easy_triage": "easy",
        "medium_triage": "medium",
        "hard_triage": "hard",
    }
    grader_map = {
        "easy_triage": "server.graders:grade_easy_triage",
        "medium_triage": "server.graders:grade_medium_triage",
        "hard_triage": "server.graders:grade_hard_triage",
    }

    return {
        "tasks": [
            {
                "id": task_name,
                "task_id": task_name,
                "name": task_name,
                "description": descriptions.get(task_name, ""),
                "difficulty": difficulties.get(task_name, "unknown"),
                "email_count": len(emails),
                "grader": grader_map[task_name],
                "graders": [grader_map[task_name]],
                "max_reward": 1.0,
                "reset_params": {"task_name": task_name},
                "max_steps": 5,
            }
            for task_name, emails in TASK_EMAIL_MAP.items()
        ]
    }

@app.post("/reset", response_model=ResetResponse)
def reset(req: Optional[ResetRequest] = None) -> ResetResponse:
    task_name = req.task_name if req else "easy_triage"
    seed = req.seed if req else None
    session_id = req.session_id if req else None

    sid, env = _get_or_create(session_id)

    obs = env.reset(
        task_name=task_name,
        seed=seed,
        episode_id=sid,
    )

    return ResetResponse(session_id=sid, observation=obs)


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    sid = req.session_id
    if not sid or sid not in _sessions:
        raise HTTPException(
            status_code=400,
            detail="No active session. Call /reset first.",
        )
    env = _sessions[sid]
    obs = env.step(req.action)
    return StepResponse(
        session_id=sid,
        observation=obs,
        reward=obs.reward if obs.reward is not None else 0.0001,
        done=obs.done,
    )


@app.get("/state", response_model=StateResponse)
def state(session_id: str) -> StateResponse:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    env = _sessions[session_id]
    return StateResponse(session_id=session_id, state=env.state)


@app.delete("/session/{session_id}")
def close_session(session_id: str) -> Dict[str, str]:
    existed = session_id in _sessions
    _sessions.pop(session_id, None)
    return {
        "status": "closed" if existed else "not_found",
        "session_id": session_id,
    }

@app.post("/grader", response_model=GraderResponse)
def grader_endpoint(req: GraderRequest) -> GraderResponse:
    grader_fn = GRADERS.get(req.task_id)
    if grader_fn is None:
        raise HTTPException(
            status_code=404,
            detail=f"No grader found for task_id={req.task_id}",
        )

    score = grader_fn(req.state, req.reward)
    score = max(0.001, min(0.999, round(float(score), 3)))

    return GraderResponse(task_id=req.task_id, score=score)

def main() -> None:
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()