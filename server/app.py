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
    sid = session_id or str(uuid.uuid4())
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy", "environment": "email-triage"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "name": "easy_triage",
                "description": "5 clear-signal emails. Unambiguous priority, category, and routing.",
                "difficulty": "easy",
                "email_count": 5,
            },
            {
                "name": "medium_triage",
                "description": "5 emails with mixed signals, thread context, and multi-part issues.",
                "difficulty": "medium",
                "email_count": 5,
            },
            {
                "name": "hard_triage",
                "description": "5 ambiguous emails with churn risk, legal implications, and competing signals.",
                "difficulty": "hard",
                "email_count": 5,
            },
        ]
    }


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest) -> ResetResponse:
    sid, env = _get_or_create(req.session_id)
    obs = env.reset(
        task_name=req.task_name,
        seed=req.seed,
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
        reward=obs.reward if obs.reward is not None else 0.0,
        done=obs.done,
    )


@app.get("/state", response_model=StateResponse)
def state(session_id: str) -> StateResponse:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    env = _sessions[session_id]
    return StateResponse(session_id=session_id, state=env.state)


@app.delete("/session/{session_id}")
def close_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"status": "closed", "session_id": session_id}