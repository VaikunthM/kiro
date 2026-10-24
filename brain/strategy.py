"""The two robots. decide(state) -> step dict or None, as sim/reference_policies.py describes."""
import math

from .geometry import normalize
from .score import GRASPS, features, score_grasp

CUTOFF = 0.35
MAX_PUSHES_PER_ITEM = 2
PUSH_PREDICT = 0.06   # assumed push distance (m) when predicting overlap; check with brain.diagnose


def push_direction(state, item_id):
    """Unit [dx, dy] that moves the item away from what is crowding it."""
    s = normalize(state)
    objs = {o["id"]: o for o in s["objects"]}
    me = objs[item_id]
    f = features(s, item_id)
    tx, ty = s["tote"]["size"][0] / 2, s["tote"]["size"][1] / 2
    if f["below"] > 0:  # slide it off whatever is under it: pick the direction that leaves the least overlap
        under = max((o for o in objs.values() if o["id"] != item_id and abs(me["lo"][2] - o["hi"][2]) < 0.01),
                    key=lambda o: _o(me, o, 0) * _o(me, o, 1) if _o(me, o, 0) > 0 and _o(me, o, 1) > 0 else -1,
                    default=None)
        if under is not None:
            lims = (tx, ty)
            best = None
            for axis in (0, 1):
                for sign in (1, -1):
                    room = lims[axis] - me["hi"][axis] if sign > 0 else lims[axis] + me["lo"][axis]
                    move = max(0.0, min(PUSH_PREDICT, room))
                    lo_a, hi_a = me["lo"][axis] + sign * move, me["hi"][axis] + sign * move
                    ov_a = max(0.0, min(hi_a, under["hi"][axis]) - max(lo_a, under["lo"][axis]))
                    ov_b = max(0.0, _o(me, under, 1 - axis))
                    key = (round(ov_a * ov_b, 6), me["size"][axis])
                    if best is None or key < best[0]:
                        best = (key, axis, sign)
            d = [0.0, 0.0]
            d[best[1]] = float(best[2])
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
    # an item flush against a wall cannot move along that axis: toward the wall is blocked, and away from it
    # needs room for the pusher behind the item
    for axis, lim in ((0, tx), (1, ty)):
        if min(lim - me["hi"][axis], lim + me["lo"][axis]) < 0.012:
            if axis == 0:
                vx = 0.0
            else:
                vy = 0.0
    n = math.hypot(vx, vy)
    return None if n < 1e-9 else [round(vx / n, 3), round(vy / n, 3)]


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
        self._last, self._failed = None, {}

    def _reset_if_new_run(self, s):
        n = len(s["objects"])
        if s["scene_id"] != self._scene or n > self._n:
            self._pushes, self._failed, self._last = {}, {}, None
        self._scene, self._n = s["scene_id"], n
        # the last grasp left its item in the tote: it missed, so do not repeat that grasp on that item
        if self._last and any(o["id"] == self._last[0] for o in s["objects"]):
            self._failed.setdefault(self._last[0], set()).add(self._last[1])
        self._last = None

    def decide(self, state):
        s = normalize(state)
        if not s["objects"]:
            return None
        self._reset_if_new_run(s)
        options = sorted((score_grasp(s, o["id"], g), o["id"], g) for o in s["objects"] for g in GRASPS
                         if g not in self._failed.get(o["id"], ()))
        if not options:  # every grasp has failed on every item: start over rather than freeze
            self._failed = {}
            options = sorted((score_grasp(s, o["id"], g), o["id"], g) for o in s["objects"] for g in GRASPS)
        risk, item, grasp = options[0]
        if risk <= self.cutoff:
            reason = "clear on all sides" if risk < 0.15 else f"lowest-risk option ({grasp.replace('_', ' ')})"
            self._last = (item, grasp)
            return {"action": "grasp", "item": item, "grasp": grasp, "risk": risk, "reason": reason}
        cands = []
        for o in s["objects"]:
            f = features(s, o["id"])
            if f["above"] > 0.2 or self._pushes.get(o["id"], 0) >= MAX_PUSHES_PER_ITEM:
                continue
            crowd = f["below"] * 1.5 + f["crowd"] + f["small_nb"] + f["big_nb"]
            if crowd > 0.05 and push_direction(s, o["id"]) is not None:
                cands.append((crowd, o["id"], f))
        if not cands:
            fi = features(s, item)
            why = ("too close to the wall for a suction cup to seal: using the safest grasp"
                   if max(fi["seal_center"], fi["seal_edge"]) > 0.4 and "suction" not in grasp
                   else "no way to make more space: taking the safest option")
            self._last = (item, grasp)
            return {"action": "grasp", "item": item, "grasp": grasp, "risk": risk, "reason": why}
        _, pid, f = max(cands, key=lambda c: c[0])
        self._pushes[pid] = self._pushes.get(pid, 0) + 1
        return {"action": "push", "item": pid, "grasp": None, "direction": push_direction(s, pid),
                "risk": risk, "reason": _crowd_reason(f)}
