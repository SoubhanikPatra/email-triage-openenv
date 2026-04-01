"""
Email Triage Environment — Server-Side Logic

Implements reset(), step(), and state property.
SUPPORTS_CONCURRENT_SESSIONS = True so multiple clients can run simultaneously.
"""

from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List, Optional

from email_triage_env.email_data import TASK_EMAIL_MAP, EMAILS
from email_triage_env.grader import grade
from email_triage_env.models import (
    EmailObservation,
    TriageAction,
    TriageState,
)

VALID_TASKS = list(TASK_EMAIL_MAP.keys())


class EmailTriageEnvironment:
    """
    Episode structure
    -----------------
    Each episode is one 'inbox' — a sequence of emails drawn from the
    task's pool.  The agent must triage each email in turn.

    reset(task_name=...) selects the task and loads the inbox.
    step(action) scores the action against the current email and
        advances to the next.
    state returns episode-level metadata.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._state = TriageState()
        self._emails: List[Dict[str, Any]] = []
        self._cursor: int = 0
        self._done: bool = False

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        task_name: str = "easy_triage",
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> EmailObservation:
        if task_name not in VALID_TASKS:
            task_name = "easy_triage"

        rng = random.Random(seed)
        pool = list(TASK_EMAIL_MAP[task_name])
        rng.shuffle(pool)

        self._emails = pool
        self._cursor = 0
        self._done = False
        self._state = TriageState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_name=task_name,
            cumulative_reward=0.0,
            emails_correct=0,
            emails_total=len(pool),
        )

        return self._make_observation(reward=None, done=False, feedback="", score_breakdown={})

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: TriageAction, **kwargs: Any) -> EmailObservation:
        if self._done or self._cursor >= len(self._emails):
            return self._make_observation(
                reward=0.0, done=True, feedback="Episode already finished.", score_breakdown={}
            )

        email = self._emails[self._cursor]
        action_dict = action.model_dump()

        total_score, breakdown, feedback = grade(action_dict, email)

        self._state.step_count += 1
        self._state.cumulative_reward += total_score
        if total_score >= 0.75:
            self._state.emails_correct += 1

        self._cursor += 1
        done = self._cursor >= len(self._emails)
        self._done = done

        return self._make_observation(
            reward=total_score,
            done=done,
            feedback=feedback,
            score_breakdown=breakdown,
        )

    # ------------------------------------------------------------------
    # state property
    # ------------------------------------------------------------------

    @property
    def state(self) -> TriageState:
        return self._state

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _current_email(self) -> Optional[Dict[str, Any]]:
        if self._cursor < len(self._emails):
            return self._emails[self._cursor]
        return None

    def _make_observation(
        self,
        reward: Optional[float],
        done: bool,
        feedback: str,
        score_breakdown: Dict[str, float],
    ) -> EmailObservation:
        email = self._current_email()
        if email is None:
            return EmailObservation(
                done=True,
                reward=reward,
                step_number=self._state.step_count,
                total_emails=self._state.emails_total,
                feedback=feedback,
                score_breakdown=score_breakdown,
            )
        return EmailObservation(
            done=done,
            reward=reward,
            email_id=email["email_id"],
            subject=email["subject"],
            body=email["body"],
            sender_email=email["sender_email"],
            sender_name=email["sender_name"],
            thread_length=email["thread_length"],
            has_attachment=email["has_attachment"],
            account_tier=email["account_tier"],
            prior_tickets=email["prior_tickets"],
            step_number=self._state.step_count,
            total_emails=self._state.emails_total,
            feedback=feedback,
            score_breakdown=score_breakdown,
        )