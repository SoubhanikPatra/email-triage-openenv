TASKS = [
    {
        "id": "easy_triage",
        "task_id": "easy_triage",
        "name": "easy_triage",
        "grader": "server.graders:grade_easy_triage",
        "graders": ["server.graders:grade_easy_triage"],
    },
    {
        "id": "medium_triage",
        "task_id": "medium_triage",
        "name": "medium_triage",
        "grader": "server.graders:grade_medium_triage",
        "graders": ["server.graders:grade_medium_triage"],
    },
    {
        "id": "hard_triage",
        "task_id": "hard_triage",
        "name": "hard_triage",
        "grader": "server.graders:grade_hard_triage",
        "graders": ["server.graders:grade_hard_triage"],
    },
]

TASK_GRADER_PAIRS = [
    ("easy_triage", "server.graders:grade_easy_triage"),
    ("medium_triage", "server.graders:grade_medium_triage"),
    ("hard_triage", "server.graders:grade_hard_triage"),
]

__all__ = ["TASKS", "TASK_GRADER_PAIRS"]