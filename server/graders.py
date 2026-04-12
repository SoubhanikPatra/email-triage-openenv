"""
Email Triage Environment — Graders for OpenEnv validator
Each grader must have signature (state, reward) -> float
"""

from typing import Dict, Any


def _normalize_reward(reward: float) -> float:
    """Clamp reward to valid OpenEnv range [0.001, 0.999]"""
    return max(0.001, min(0.999, round(float(reward), 3)))


def grade_easy_triage(state: Dict[str, Any], reward: float) -> float:
    """
    Grader for easy_triage task.
    
    Args:
        state: Episode state dict (contains task_name, cumulative_reward, etc.)
        reward: The reward returned from the last step
    
    Returns:
        Normalized score between 0.001 and 0.999
    """
    # Verify we're grading the correct task
    task_name = state.get("task_name") if isinstance(state, dict) else getattr(state, "task_name", None)
    if task_name != "easy_triage":
        return 0.001
    return _normalize_reward(reward)


def grade_medium_triage(state: Dict[str, Any], reward: float) -> float:
    """
    Grader for medium_triage task.
    
    Args:
        state: Episode state dict (contains task_name, cumulative_reward, etc.)
        reward: The reward returned from the last step
    
    Returns:
        Normalized score between 0.001 and 0.999
    """
    task_name = state.get("task_name") if isinstance(state, dict) else getattr(state, "task_name", None)
    if task_name != "medium_triage":
        return 0.001
    return _normalize_reward(reward)


def grade_hard_triage(state: Dict[str, Any], reward: float) -> float:
    """
    Grader for hard_triage task.
    
    Args:
        state: Episode state dict (contains task_name, cumulative_reward, etc.)
        reward: The reward returned from the last step
    
    Returns:
        Normalized score between 0.001 and 0.999
    """
    task_name = state.get("task_name") if isinstance(state, dict) else getattr(state, "task_name", None)
    if task_name != "hard_triage":
        return 0.001
    return _normalize_reward(reward)


# Required for OpenEnv discovery
GRADERS = {
    "easy_triage": grade_easy_triage,
    "medium_triage": grade_medium_triage,
    "hard_triage": grade_hard_triage,
}

# Required for tasks.py compatibility (if validator looks for this)
TASK_GRADER_PAIRS = [
    ("easy_triage", grade_easy_triage),
    ("medium_triage", grade_medium_triage),
    ("hard_triage", grade_hard_triage),
]

__all__ = [
    "grade_easy_triage",
    "grade_medium_triage", 
    "grade_hard_triage",
    "GRADERS",
    "TASK_GRADER_PAIRS",
]