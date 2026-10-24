"""Grab-only baseline: always suction the highest object at its centre."""
from brain.strategy import GrabOnly

_policy = GrabOnly()


def decide(state):
    return _policy.decide(state)
