def _normalize_reward(reward: float) -> float:
    return max(0.001, min(0.999, round(float(reward), 3)))


def _grade_task(task_name: str, state: dict, reward: float) -> float:
    current_task = state.get("task_name") or state.get("task_id")
    if current_task != task_name:
        return 0.001
    return _normalize_reward(reward)


def _extract_state_reward(*args, **kwargs):
    state = kwargs.get("state")
    reward = kwargs.get("reward", 0.0)
    
    if len(args) == 2:
        state, reward = args[0], args[1]
    elif len(args) == 1:
        if isinstance(args[0], dict):
            state = args[0]
        elif hasattr(args[0], "state"):
            state = args[0].state
            
    if state is None:
        state = {}
        
    return state, float(reward)


class EasyGrader:
    def grade(self, *args, **kwargs) -> float:
        state, reward = _extract_state_reward(*args, **kwargs)
        return _grade_task("easy_triage", state, reward)

class MediumGrader:
    def grade(self, *args, **kwargs) -> float:
        state, reward = _extract_state_reward(*args, **kwargs)
        return _grade_task("medium_triage", state, reward)

class HardGrader:
    def grade(self, *args, **kwargs) -> float:
        state, reward = _extract_state_reward(*args, **kwargs)
        return _grade_task("hard_triage", state, reward)


GRADERS = {
    "easy_triage": EasyGrader().grade,
    "medium_triage": MediumGrader().grade,
    "hard_triage": HardGrader().grade,
}

TASK_GRADER_PAIRS = [
    ("easy_triage", EasyGrader().grade),
    ("medium_triage", MediumGrader().grade),
    ("hard_triage", HardGrader().grade),
]