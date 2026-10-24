"""Reference policy used by the runner's tests and as a stand-in until P3's policies land.

Policy interface (P3 implements the real ones in policies/<name>.py):
    decide(state) -> step dict, or None to stop
    step dict: {"action": "grasp" | "push", "item": id,
                "grasp": one of sim.GRASP_TYPES (grasp only),
                "direction": [dx, dy] (push only),
                "risk": float 0-1 or None, "reason": one short line}
`state` is exactly what Sim.state() returns.
"""


def grab_only(state):
    """Always suction the highest object at its centre."""
    objs = state["objects"]
    if not objs:
        return None
    item = max(objs, key=lambda o: objs[o]["bbox_max"][2])
    return {"action": "grasp", "item": item, "grasp": "suction_center", "risk": None,
            "reason": "highest item, suction at centre"}
