"""Reference policy used by the runner's tests and as a stand-in until P3's policies land.

Policy interface (P3 implements the real ones in policies/<name>.py):
    decide(state) -> step dict, or None to stop
    step dict: {"action": "grasp" | "push", "item": id,
                "grasp": one of sim.GRASP_TYPES (grasp only),
                "direction": [dx, dy] (push only),
                "risk": float 0-1 or None, "reason": one short line,
                "stall": True (optional: the robot gave up waiting for a safe grasp; nothing is executed)}
`state` is exactly what Sim.state() returns (contract 3).
"""


def grab_only(state):
    """Always suction the highest object at its centre."""
    if not state["objects"]:
        return None
    item = max(state["objects"], key=lambda o: o["bbox"]["max"][2])
    return {"action": "grasp", "item": item["id"], "grasp": "suction_center", "risk": None,
            "reason": "highest item, suction at centre"}
