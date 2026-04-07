"""
Email Triage Environment — Typed Models
All data contracts between client and server live here.
Action, Observation, State are Pydantic BaseModel subclasses.
"""

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Shared enums (plain strings — no stdlib enum needed)
# ---------------------------------------------------------------------------

PRIORITY_LITERAL = Literal["urgent", "high", "normal", "low"]
CATEGORY_LITERAL = Literal[
    "billing",
    "technical_support",
    "feature_request",
    "bug_report",
    "account_management",
    "sales_inquiry",
    "general_inquiry",
    "spam",
]
ROUTING_LITERAL = Literal[
    "tier1_support",
    "tier2_support",
    "billing_team",
    "sales_team",
    "engineering",
    "account_management",
    "spam_filter",
    "escalation",
]
SENTIMENT_LITERAL = Literal["positive", "neutral", "negative", "very_negative"]
TONE_LITERAL = Literal["formal", "friendly", "empathetic", "concise"]

# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class TriageAction(BaseModel):
    
    """
    The agent's triage decision for the current email.
    Fields
    ------
    priority : Literal["urgent", "high", "normal", "low"]
        Urgency level of the email.
    category : Literal[...]
        One of the predefined email categories (billing, technical_support, etc.).
    routing_target : Literal[...]
        Target team/queue to route the email to.
    sentiment : Literal["positive", "neutral", "negative", "very_negative"]
        Detected customer sentiment (default: "neutral").
    requires_followup : bool
        Whether a follow-up action is required within 24 hours.
    summary : str
        A concise (≤ 2-sentence, ≤ 300 chars) summary for the support agent.
    suggested_response_tone : Literal["formal", "friendly", "empathetic", "concise"]
        Tone the agent recommends for replying.
    tags : List[str]
        Up to 5 normalized keyword tags extracted from the email.
    metadata : Dict[str, Any]
        Optional extra information (not used in grading).
    """
    
    priority: PRIORITY_LITERAL
    category: CATEGORY_LITERAL
    routing_target: ROUTING_LITERAL
    sentiment: SENTIMENT_LITERAL = "neutral"
    requires_followup: bool = False
    summary: str = Field(default="", description="≤ 2-sentence summary")
    suggested_response_tone: TONE_LITERAL = "friendly"
    tags: List[str] = Field(default_factory=list, description="0–5 keyword tags")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        value = value.strip()
        if len(value) > 300:
            raise ValueError("summary must be <= 300 characters")
        return value

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: List[str]) -> List[str]:
        cleaned = []
        seen = set()
        for tag in value[:5]:
            tag = str(tag).strip().lower()
            if not tag:
                continue
            if len(tag) > 40:
                tag = tag[:40]
            if tag not in seen:
                seen.add(tag)
                cleaned.append(tag)
        return cleaned
    
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
    remaining_emails: int = 0
    task_name: str = "easy_triage"
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