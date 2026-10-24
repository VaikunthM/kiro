"""score_grasp(state, item_id, grasp_type) -> risk in [0, 1], from geometry only.

Features (each 0..1): below, above, flat_below, crowd, wall, big_nb, small_nb, flat.
Risk = sigmoid(bias + sum(weight * feature)); weights are hand-set per grasp kind, scaled per failure mode.
Grasp kinds are matched by name so "suction_edge" and "suction_offcenter" both work.
"""
import math

from .geometry import normalize

try:  # use the simulator's own list of grasp types when it can be imported
    from sim.sim import GRASP_TYPES as _GT
    GRASPS = tuple(_GT)
except Exception:  # pragma: no cover
    GRASPS = ("suction_center", "suction_edge", "side_pinch")

EPS = 0.004

BASE = {
    "center":    {"bias": -3.2, "below": 3.0, "above": 4.0, "flat_below": 1.5, "crowd": 1.8,
                  "wall": 0.8, "big_nb": 1.2, "small_nb": 0.8, "flat": 0.0},
    "offcenter": {"bias": -2.8, "below": 2.2, "above": 4.0, "flat_below": 0.8, "crowd": 1.2,
                  "wall": 1.8, "big_nb": 1.0, "small_nb": 0.6, "flat": 0.2},
    "pinch":     {"bias": -2.6, "below": 1.0, "above": 3.0, "flat_below": 0.0, "crowd": 3.2,
                  "wall": 2.4, "big_nb": 1.4, "small_nb": 2.0, "flat": 1.8},
}
MODE_SCALE = {
    "flat_stacked":     {"below": 1.5, "flat_below": 1.5},
    "small_beside_big": {"crowd": 1.4, "big_nb": 1.6, "small_nb": 1.6},
    "against_wall":     {"wall": 1.6, "crowd": 1.2},
}


def _kind(grasp):
    if "pinch" in grasp:
        return "pinch"
    if "off" in grasp or "edge" in grasp:
        return "offcenter"
    if "center" in grasp or "suction" in grasp:
        return "center"
    raise ValueError(f"unknown grasp type {grasp}")


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def _ov(a, b, axis):
    return min(a["hi"][axis], b["hi"][axis]) - max(a["lo"][axis], b["lo"][axis])


def features(state, item_id):
    s = normalize(state)
    objs = {o["id"]: o for o in s["objects"]}
    me = objs[item_id]
    area = me["size"][0] * me["size"][1]
    below = above = big = small = 0.0
    gap_min = math.inf
    for oid, o in objs.items():
        if oid == item_id:
            continue
        ox, oy = _ov(me, o, 0), _ov(me, o, 1)
        if ox > 0 and oy > 0:
            frac = min(ox * oy / area, 1.0)
            if abs(me["lo"][2] - o["hi"][2]) <= EPS * 2.5:
                below = max(below, frac)
                continue
            if abs(o["lo"][2] - me["hi"][2]) <= EPS * 2.5:
                above = max(above, frac)
                continue
        if _ov(me, o, 2) > EPS:  # same height band: a side neighbour
            gap = max(-ox, -oy)
            gap_min = min(gap_min, gap)
            if gap < 0.05:
                ratio = (o["size"][0] * o["size"][1]) / area
                prox = _clamp(1 - gap / 0.05)
                if ratio > 1:
                    big = max(big, _clamp(math.log2(ratio) / 3) * prox)
                elif ratio < 1:
                    small = max(small, _clamp(math.log2(1 / ratio) / 3) * prox)
    tx, ty = s["tote"]["size"][0] / 2, s["tote"]["size"][1] / 2
    wall_dist = min(tx - me["hi"][0], tx + me["lo"][0], ty - me["hi"][1], ty + me["lo"][1])
    flat = _clamp((0.3 - me["size"][2] / max(me["size"][0], me["size"][1])) / 0.3)
    return {"below": below, "above": above, "flat_below": flat * below,
            "crowd": 0.0 if gap_min == math.inf else _clamp(1 - gap_min / 0.03),
            "wall": _clamp(1 - wall_dist / 0.03), "big_nb": big, "small_nb": small, "flat": flat}


def score_grasp(state, item_id, grasp_type):
    s = normalize(state)
    f = features(s, item_id)
    w = BASE[_kind(grasp_type)]
    scale = MODE_SCALE.get(s["mode"], {})
    z = w["bias"] + sum(w[k] * scale.get(k, 1.0) * v for k, v in f.items())
    return round(1 / (1 + math.exp(-z)), 4)
