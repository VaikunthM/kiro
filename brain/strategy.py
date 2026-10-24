"""The two robots. decide(state) -> step dict or None, as sim/reference_policies.py describes."""
import math

from .geometry import normalize
from .score import GRASPS, features, score_grasp

CUTOFF = 0.35
MAX_PUSHES_PER_ITEM = 2


def push_direction(state, item_id):
    """Unit [dx, dy] that moves the item away from what is crowding it."""
    s = normalize(state)
    objs = {o["id"]: o for o in s["objects"]}
    me = objs[item_id]
    f = features(s, item_id)
    tx, ty = s["tote"]["size"][0] / 2, s["tote"]["size"][1] / 2
    if f["below"] > 0:  # slide it off whatever is under it, along its short side
        under = next((o for o in objs.values() if o["id"] != item_id
                      and abs(me["lo"][2] - o["hi"][2]) < 0.01
                      and min(me["hi"][0], o["hi"][0]) > max(me["lo"][0], o["lo"][0])
                      and min(me["hi"][1], o["hi"][1]) > max(me["lo"][1], o["lo"][1])), None)
        axis = 0 if me["size"][0] < me["size"][1] else 1
        lim = (tx, ty)[axis]
        off = me["pos"][axis] - (under["pos"][axis] if under else 0.0)
        sign = 1 if off > 0 else -1 if off < 0 else (-1 if me["pos"][axis] > 0 else 1)
        room = lim - abs(me["pos"][axis]) - me["size"][axis] / 2
        if room < 0.03 and me["pos"][axis] * sign > 0:
            sign = -sign
        d = [0.0, 0.0]
        d[axis] = float(sign)
        return d
    vx = vy = 0.0
    for o in objs.values():
        if o["id"] == item_id:
            continue
        dx, dy = me["pos"][0] - o["pos"][0], me["pos"][1] - o["pos"][1]
        gap = max(-_o(me, o, 0), -_o(me, o, 1))
        if gap < 0.05:
            dist = math.hypot(dx, dy) or 1e-6
            w = 1 / (max(gap, 0.0) + 0.01)
            vx += dx / dist * w
            vy += dy / dist * w
    if f["wall"] > 0:  # toward the tote centre, weighted by how close the wall is
        dist = math.hypot(me["pos"][0], me["pos"][1]) or 1e-6
        vx += -me["pos"][0] / dist * 60 * f["wall"]
        vy += -me["pos"][1] / dist * 60 * f["wall"]
    n = math.hypot(vx, vy)
    return [1.0, 0.0] if n < 1e-9 else [round(vx / n, 3), round(vy / n, 3)]


def _o(a, b, axis):
    return min(a["hi"][axis], b["hi"][axis]) - max(a["lo"][axis], b["lo"][axis])


def _crowd_reason(f):
    if f["below"] > 0.2:
        return "resting on another item: sliding it off"
    if f["wall"] > 0.5 and f["crowd"] > 0.3:
        return "pinned between the wall and a neighbour: pulling it into the open"
    if f["wall"] > 0.5:
        return "against the wall: pulling it into the open"
    if f["small_nb"] > 0.1 or f["big_nb"] > 0.1:
        return "tight against a different-sized neighbour: separating them"
    return "crowded by a neighbour: separating them"


class GrabOnly:
    """Always suction the highest object at its centre."""

    def decide(self, state):
        s = normalize(state)
        if not s["objects"]:
            return None
        item = max(s["objects"], key=lambda o: o["hi"][2])
        return {"action": "grasp", "item": item["id"], "grasp": "suction_center",
                "risk": score_grasp(s, item["id"], "suction_center"),
                "reason": "highest item, suction at centre"}


class ClearPick:
    """Score every grasp for every item; take the safest if under the cutoff, else clear space first."""

    def __init__(self, cutoff=CUTOFF):
        self.cutoff = cutoff
        self._scene, self._n, self._pushes = None, 0, {}

    def _reset_if_new_run(self, s):
        n = len(s["objects"])
        if s["scene_id"] != self._scene or n > self._n:
            self._pushes = {}
        self._scene, self._n = s["scene_id"], n

    def decide(self, state):
        s = normalize(state)
        if not s["objects"]:
            return None
        self._reset_if_new_run(s)
        options = sorted((score_grasp(s, o["id"], g), o["id"], g) for o in s["objects"] for g in GRASPS)
        risk, item, grasp = options[0]
        if risk <= self.cutoff:
            reason = "clear on all sides" if risk < 0.15 else f"lowest-risk option ({grasp.replace('_', ' ')})"
            return {"action": "grasp", "item": item, "grasp": grasp, "risk": risk, "reason": reason}
        cands = []
        for o in s["objects"]:
            f = features(s, o["id"])
            if f["above"] > 0.2 or self._pushes.get(o["id"], 0) >= MAX_PUSHES_PER_ITEM:
                continue
            crowd = f["below"] * 1.5 + f["crowd"] + 0.5 * f["wall"] + f["small_nb"] + f["big_nb"]
            if crowd > 0.05:
                cands.append((crowd, o["id"], f))
        if not cands:
            return {"action": "grasp", "item": item, "grasp": grasp, "risk": risk,
                    "reason": "no way to make more space: taking the safest option"}
        _, pid, f = max(cands, key=lambda c: c[0])
        self._pushes[pid] = self._pushes.get(pid, 0) + 1
        return {"action": "push", "item": pid, "grasp": None, "direction": push_direction(s, pid),
                "risk": risk, "reason": _crowd_reason(f)}
