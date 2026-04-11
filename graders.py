from server.graders import (
    grade_easy_triage,
    grade_medium_triage,
    grade_hard_triage,
    GRADERS,
)

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