"""
Email Triage Environment — Typed Models

All data contracts between client and server live here.
Action, Observation, State are Pydantic BaseModel subclasses.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared enums (plain strings — no stdlib enum needed)
# ---------------------------------------------------------------------------

PRIORITY_LEVELS = ["urgent", "high", "normal", "low"]
CATEGORIES = [
    "billing",
    "technical_support",
    "feature_request",
    "bug_report",
    "account_management",
    "sales_inquiry",
    "general_inquiry",
    "spam",
]
ROUTING_TARGETS = [
    "tier1_support",
    "tier2_support",
    "billing_team",
    "sales_team",
    "engineering",
    "account_management",
    "spam_filter",
    "escalation",
]
SENTIMENT_LABELS = ["positive", "neutral", "negative", "very_negative"]


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class TriageAction(BaseModel):
    """
    The agent's triage decision for the current email.

    Fields
    ------
    priority : str
        One of: urgent | high | normal | low
    category : str
        One of the CATEGORIES list.
    routing_target : str
        One of the ROUTING_TARGETS list.
    sentiment : str
        One of the SENTIMENT_LABELS list.
    requires_followup : bool
        Whether a follow-up action is needed within 24 h.
    summary : str
        A ≤ 2-sentence summary the agent writes for the support rep.
    suggested_response_tone : str
        One of: formal | friendly | empathetic | concise
    tags : List[str]
        Free-form tags (0–5). Used to score keyword extraction.
    """

    priority: str = Field(..., description="urgent | high | normal | low")
    category: str = Field(..., description="Email category")
    routing_target: str = Field(..., description="Team / queue to route to")
    sentiment: str = Field(default="neutral", description="Customer sentiment")
    requires_followup: bool = Field(default=False)
    summary: str = Field(default="", description="≤ 2-sentence summary")
    suggested_response_tone: str = Field(default="friendly")
    tags: List[str] = Field(default_factory=list, description="0–5 keyword tags")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class EmailObservation(BaseModel):
    """
    What the agent sees each step.

    Fields
    ------
    done : bool
    reward : float | None
    email_id : str
    subject : str
    body : str
    sender_email : str
    sender_name : str
    thread_length : int        Number of prior emails in this thread.
    has_attachment : bool
    account_tier : str         free | starter | pro | enterprise
    prior_tickets : int        How many open tickets this sender has.
    step_number : int
    total_emails : int         How many emails remain in this episode.
    feedback : str             Human-readable grader feedback on last action.
    score_breakdown : Dict     Per-dimension scores from last step.
    """

    done: bool = False
    reward: Optional[float] = None

    email_id: str = ""
    subject: str = ""
    body: str = ""
    sender_email: str = ""
    sender_name: str = ""
    thread_length: int = 0
    has_attachment: bool = False
    account_tier: str = "free"
    prior_tickets: int = 0

    step_number: int = 0
    total_emails: int = 0

    feedback: str = ""
    score_breakdown: Dict[str, float] = Field(default_factory=dict)

    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class TriageState(BaseModel):
    """
    Episode-level metadata (not the per-step observation).

    Fields
    ------
    episode_id : str
    step_count : int
    task_name : str            easy_triage | medium_triage | hard_triage
    cumulative_reward : float
    emails_correct : int
    emails_total : int
    """

    episode_id: Optional[str] = None
    step_count: int = 0

    task_name: str = "easy_triage"
    cumulative_reward: float = 0.0
    emails_correct: int = 0
    emails_total: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)