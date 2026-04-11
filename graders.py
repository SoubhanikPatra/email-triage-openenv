from server.graders import (
    EasyGrader,
    MediumGrader,
    HardGrader,
    GRADERS,
)


def grade_easy_triage(state: dict, reward: float) -> float:
    return EasyGrader().grade(state, reward)


def grade_medium_triage(state: dict, reward: float) -> float:
    return MediumGrader().grade(state, reward)


def grade_hard_triage(state: dict, reward: float) -> float:
    return HardGrader().grade(state, reward)

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