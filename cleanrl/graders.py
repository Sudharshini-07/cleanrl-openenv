"""
Programmatic graders for each task.
Each grader returns a float between 0.0 and 1.0.
"""
from typing import Set


def grade(errors_fixed: Set[str], required_fixes: Set[str]) -> float:
    """
    Universal grader: proportion of required fixes that were applied.
    Score = |correct_fixes| / |required_fixes|
    """
    if not required_fixes:
        return 0.0
    correct = len(errors_fixed & required_fixes)
    score   = correct / len(required_fixes)
    return round(score, 4)


def partial_reward(operation_key: str,
                   errors_fixed: Set[str],
                   required_fixes: Set[str]) -> float:
    """
    Returns incremental reward for a single correct fix.
    Ensures reward is shaped throughout the trajectory, not only at done.
    """
    if operation_key in required_fixes and operation_key not in errors_fixed:
        return round(1.0 / len(required_fixes), 4)
    return 0.0
