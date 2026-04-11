TASKS = [
    {
        "id": "easy_triage",
        "task_id": "easy_triage",
        "name": "easy_triage",
        "grader": "server.graders:EasyGrader",
        "graders": ["server.graders:EasyGrader"],
    },
    {
        "id": "medium_triage",
        "task_id": "medium_triage",
        "name": "medium_triage",
        "grader": "server.graders:MediumGrader",
        "graders": ["server.graders:MediumGrader"],
    },
    {
        "id": "hard_triage",
        "task_id": "hard_triage",
        "name": "hard_triage",
        "grader": "server.graders:HardGrader",
        "graders": ["server.graders:HardGrader"],
    },
]

TASK_GRADER_PAIRS = [
    ("easy_triage", "server.graders:EasyGrader"),
    ("medium_triage", "server.graders:MediumGrader"),
    ("hard_triage", "server.graders:HardGrader"),
]

__all__ = ["TASKS", "TASK_GRADER_PAIRS"]