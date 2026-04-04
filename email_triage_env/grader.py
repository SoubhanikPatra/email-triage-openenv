"""
Email Triage Grader:
Scores an agent's TriageAction against gold-standard labels.
Returns a float in [0.0, 1.0] plus a breakdown dict and feedback string.

Dimensions and weights
----------------------
priority      0.25   exact match on urgent/high/normal/low
category      0.25   exact match across 8 categories
routing       0.20   exact match across routing targets
sentiment     0.10   exact or adjacent match
followup      0.10   binary exact match
tag_overlap   0.10   Jaccard similarity on tag sets

Partial credit
--------------
- Priority: adjacent tiers (e.g. high vs urgent) score 0.5
- Sentiment: adjacent labels score 0.5
- Tag overlap: Jaccard > 0 gives proportional credit
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
# ── weight table ────────────────────────────────────────────────────────────
WEIGHTS = {
    "priority": 0.21,
    "category": 0.21,
    "routing": 0.18,
    "sentiment": 0.08,
    "followup": 0.08,
    "tag_overlap": 0.12,
    "summary": 0.07,
    "tone": 0.05,
}
# ordered priority scale — used for adjacency partial credit
_PRIORITY_ORDER = ["low", "normal", "high", "urgent"]
_SENTIMENT_ORDER = ["positive", "neutral", "negative", "very_negative"]

def _adjacent_score(value: str, gold: str, scale: List[str]) -> float:
    """1.0 if exact, 0.5 if one step away, 0.0 otherwise."""
    try:
        vi, gi = scale.index(value), scale.index(gold)
    except ValueError:
        return 0.0
    diff = abs(vi - gi)
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.5
    return 0.0

def _jaccard(predicted: List[str], gold: List[str]) -> float:
    p = {t.lower().strip() for t in predicted}
    g = {t.lower().strip() for t in gold}
    if not g:
        return 1.0  # no tags required → full marks
    union = p | g
    if not union:
        return 0.0
    return len(p & g) / len(union)

def grade(action_dict: Dict[str, Any], gold: Dict[str, Any]) -> Tuple[float, Dict[str, float], str]:
    """
    Parameters
    ----------
    action_dict : dict  — the agent's TriageAction as a dict
    gold        : dict  — one email entry from email_data.EMAILS

    Returns
    -------
    (total_score, breakdown, feedback_message)
    """
    scores: Dict[str, float] = {}
    feedback_parts: List[str] = []
    # 1. Priority
    pred_priority = str(action_dict.get("priority", "")).lower().strip()
    gold_priority = gold["gold_priority"]
    s_priority = _adjacent_score(pred_priority, gold_priority, _PRIORITY_ORDER)
    scores["priority"] = s_priority
    if s_priority < 1.0:
        feedback_parts.append(
            f"Priority: predicted '{pred_priority}' but gold is '{gold_priority}' "
            f"(score {s_priority:.2f})"
        )
    # 2. Category
    pred_cat = str(action_dict.get("category", "")).lower().strip()
    gold_cat = gold["gold_category"]
    s_cat = 1.0 if pred_cat == gold_cat else 0.0
    scores["category"] = s_cat
    if s_cat < 1.0:
        feedback_parts.append(
            f"Category: predicted '{pred_cat}' but gold is '{gold_cat}'"
        )
    # 3. Routing
    pred_route = str(action_dict.get("routing_target", "")).lower().strip()
    gold_route = gold["gold_routing"]
    s_route = 1.0 if pred_route == gold_route else 0.0
    scores["routing"] = s_route
    if s_route < 1.0:
        feedback_parts.append(
            f"Routing: predicted '{pred_route}' but gold is '{gold_route}'"
        )
    # 4. Sentiment
    pred_sent = str(action_dict.get("sentiment", "neutral")).lower().strip()
    gold_sent = gold["gold_sentiment"]
    s_sent = _adjacent_score(pred_sent, gold_sent, _SENTIMENT_ORDER)
    scores["sentiment"] = s_sent
    if s_sent < 1.0:
        feedback_parts.append(
            f"Sentiment: predicted '{pred_sent}' but gold is '{gold_sent}' "
            f"(score {s_sent:.2f})"
        )
    # 5. Follow-up
    pred_fu = bool(action_dict.get("requires_followup", False))
    gold_fu = bool(gold["gold_requires_followup"])
    s_fu = 1.0 if pred_fu == gold_fu else 0.0
    scores["followup"] = s_fu
    if s_fu < 1.0:
        feedback_parts.append(
            f"Follow-up: predicted {pred_fu} but gold is {gold_fu}"
        )
    # 6. Tag overlap (Jaccard)
    pred_tags = action_dict.get("tags", [])
    gold_tags = gold["gold_tags"]
    s_tags = _jaccard(pred_tags, gold_tags)
    scores["tag_overlap"] = s_tags
    if s_tags < 0.5:
        feedback_parts.append(
            f"Tags: low overlap (Jaccard {s_tags:.2f}). "
            f"Gold tags: {gold_tags}"
        )
    # 7. Summary quality
    pred_summary = str(action_dict.get("summary", ""))
    gold_tags = gold["gold_tags"]
    s_summary = _summary_score(pred_summary, gold_tags)
    scores["summary"] = s_summary
    if s_summary < 1.0:
        feedback_parts.append(
            f"Summary: limited coverage of key issues/tags (score {s_summary:.2f})"
        )
    # 8. Suggested response tone
    pred_tone = str(action_dict.get("suggested_response_tone", "friendly")).lower().strip()
    expected_tone = _expected_tone(gold)
    s_tone = _tone_score(pred_tone, expected_tone)
    scores["tone"] = s_tone
    if s_tone < 1.0:
        feedback_parts.append(
            f"Tone: predicted '{pred_tone}' but expected '{expected_tone}' "
            f"(score {s_tone:.2f})"
        )
    # Weighted total
    total = sum(WEIGHTS[dim] * scores[dim] for dim in WEIGHTS)

    feedback = (
        "Perfect triage!" if not feedback_parts
        else "Issues: " + "; ".join(feedback_parts)
    )
    return round(total, 4), scores, feedback

def _summary_score(summary: str, gold_tags: List[str]) -> float:
    summary = (summary or "").lower().strip()
    if not summary:
        return 0.0

    words = summary.split()
    if len(words) < 4:
        return 0.2
    if len(words) > 50:
        return 0.4

    matches = 0
    for tag in gold_tags:
        normalized = tag.lower().strip()
        if normalized and normalized in summary:
            matches += 1

    if matches >= 2:
        return 1.0
    if matches == 1:
        return 0.7
    return 0.4

def _expected_tone(gold: Dict[str, Any]) -> str:
    if gold["gold_routing"] == "escalation":
        return "empathetic"
    if gold["gold_sentiment"] == "very_negative":
        return "empathetic"
    if "legal" in [t.lower() for t in gold.get("gold_tags", [])]:
        return "formal"
    if gold["gold_category"] in {"sales_inquiry", "feature_request"}:
        return "friendly"
    return "friendly"

def _tone_score(pred_tone: str, expected_tone: str) -> float:
    pred_tone = (pred_tone or "").strip().lower()
    expected_tone = expected_tone.strip().lower()

    if pred_tone == expected_tone:
        return 1.0

    acceptable_pairs = {
        ("formal", "empathetic"),
        ("empathetic", "formal"),
        ("friendly", "concise"),
        ("concise", "friendly"),
    }

    if (pred_tone, expected_tone) in acceptable_pairs:
        return 0.5

    return 0.0