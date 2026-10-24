"""ClearPick: score every grasp, take the safest, or push crowded items apart first."""
from brain.strategy import ClearPick

_policy = ClearPick()


def decide(state):
    return _policy.decide(state)
