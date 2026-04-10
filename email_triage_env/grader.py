from __future__ import annotations
from typing import Any, Dict, List, Tuple

WEIGHTS = {
    "priority": 0.25,
    "category": 0.25,
    "routing": 0.20,
    "sentiment": 0.10,
    "followup": 0.10,
    "tag_overlap": 0.10,
}

_PRIORITY_ORDER = ["low", "normal", "high", "urgent"]
_SENTIMENT_ORDER = ["positive", "neutral", "negative", "very_negative"]


def _bound_score(value: float) -> float:
    return max(0.001, min(0.999, round(float(value), 3)))


def _adjacent_score(value: str, gold: str, scale: List[str]) -> float:
    try:
        vi, gi = scale.index(value), scale.index(gold)
    except ValueError:
        return 0.001
    diff = abs(vi - gi)
    if diff == 0:
        return 0.999
    if diff == 1:
        return 0.5
    return 0.001


def _jaccard(predicted: List[str], gold: List[str]) -> float:
    p = {str(t).lower().strip() for t in predicted}
    g = {str(t).lower().strip() for t in gold}
    if not g:
        return 0.999
    union = p | g
    if not union:
        return 0.001
    return _bound_score(len(p & g) / len(union))


def grade_detailed(action_dict: Dict[str, Any], gold: Dict[str, Any]) -> Tuple[float, Dict[str, float], str]:
    scores: Dict[str, float] = {}
    feedback_parts: List[str] = []

    pred_priority = str(action_dict.get("priority", "")).lower().strip()
    gold_priority = gold["gold_priority"]
    scores["priority"] = _bound_score(_adjacent_score(pred_priority, gold_priority, _PRIORITY_ORDER))

    pred_cat = str(action_dict.get("category", "")).lower().strip()
    gold_cat = gold["gold_category"]
    scores["category"] = 0.999 if pred_cat == gold_cat else 0.001

    pred_route = str(action_dict.get("routing_target", "")).lower().strip()
    gold_route = gold["gold_routing"]
    scores["routing"] = 0.999 if pred_route == gold_route else 0.001

    pred_sent = str(action_dict.get("sentiment", "neutral")).lower().strip()
    gold_sent = gold["gold_sentiment"]
    scores["sentiment"] = _bound_score(_adjacent_score(pred_sent, gold_sent, _SENTIMENT_ORDER))

    pred_fu = bool(action_dict.get("requires_followup", False))
    gold_fu = bool(gold["gold_requires_followup"])
    scores["followup"] = 0.999 if pred_fu == gold_fu else 0.001

    pred_tags = action_dict.get("tags", [])
    gold_tags = gold["gold_tags"]
    scores["tag_overlap"] = _jaccard(pred_tags, gold_tags)

    total = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)
    total = _bound_score(total)

    if scores["priority"] < 0.999:
        feedback_parts.append(f"Priority mismatch: predicted '{pred_priority}', gold '{gold_priority}'")
    if scores["category"] < 0.999:
        feedback_parts.append(f"Category mismatch: predicted '{pred_cat}', gold '{gold_cat}'")
    if scores["routing"] < 0.999:
        feedback_parts.append(f"Routing mismatch: predicted '{pred_route}', gold '{gold_route}'")
    if scores["sentiment"] < 0.999:
        feedback_parts.append(f"Sentiment mismatch: predicted '{pred_sent}', gold '{gold_sent}'")
    if scores["followup"] < 0.999:
        feedback_parts.append(f"Follow-up mismatch: predicted {pred_fu}, gold {gold_fu}")
    if scores["tag_overlap"] < 0.5:
        feedback_parts.append(f"Low tag overlap with gold tags: {gold_tags}")

    feedback = "Perfect triage!" if not feedback_parts else "Issues: " + "; ".join(feedback_parts)
    return total, scores, feedback


def grade(action_dict: Dict[str, Any], gold: Dict[str, Any]) -> float:
    total, _, _ = grade_detailed(action_dict, gold)
    return total