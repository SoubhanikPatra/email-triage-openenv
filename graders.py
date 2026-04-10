from email_triage_env.grader import grade as base_grade
from email_triage_env.email_data import TASK_EMAIL_MAP

def _normalize_reward(reward: float) -> float:
    return min(max(float(reward), 0.0), 1.0)

def _grade_task(task_name: str, state: dict, reward: float) -> float:
    current_task = state.get("task_name") or state.get("task_id")
    if current_task != task_name:
        return 0.0
    return _normalize_reward(reward)

def grade_easy_triage(state: dict, reward: float) -> float:
    return _grade_task("easy_triage", state, reward)

def grade_medium_triage(state: dict, reward: float) -> float:
    return _grade_task("medium_triage", state, reward)

def grade_hard_triage(state: dict, reward: float) -> float:
    return _grade_task("hard_triage", state, reward)

GRADERS = {
    "easy_triage": grade_easy_triage,
    "medium_triage": grade_medium_triage,
    "hard_triage": grade_hard_triage,
}

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