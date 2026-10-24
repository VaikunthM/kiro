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

CUP_R = 0.02     # suction cup radius (m), from P2: rim within 1.5 cm of a wall = miss, so cup centre within 3.5 cm
SEAL_D = 0.035

BASE = {
    # suction_center: cup at the item centre; a side neighbour within ~1 cm gets lifted too (rule a); wall seal (rule b)
    "center":    {"bias": -3.2, "below": 3.0, "above": 4.0, "flat_below": 1.5, "crowd": 0.3, "crowd_near": 6.0,
                  "seal": 8.0, "wall": 0.2, "big_nb": 0.6, "small_nb": 0.4, "flat": 0.0},
    # suction_edge: cup slides to the side away from walls and neighbours, so it dodges both
    "offcenter": {"bias": -2.8, "below": 2.2, "above": 4.0, "flat_below": 0.8, "crowd": 0.3, "crowd_near": 0.3,
                  "seal": 8.0, "wall": 0.2, "big_nb": 0.3, "small_nb": 0.2, "flat": 0.2},
    "pinch":     {"bias": -2.6, "below": 1.0, "above": 3.0, "flat_below": 0.0, "crowd": 3.2, "crowd_near": 0.0,
                  "seal": 0.0, "wall": 2.4, "big_nb": 1.4, "small_nb": 2.0, "flat": 1.8},
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
    cx, cy = (me["lo"][0] + me["hi"][0]) / 2, (me["lo"][1] + me["hi"][1]) / 2
    d_center = min(tx - abs(cx), ty - abs(cy))
    ex = cx - math.copysign(min(max(0.0, me["size"][0] / 2 - CUP_R), abs(cx)), cx)   # best cup spot, away from walls
    ey = cy - math.copysign(min(max(0.0, me["size"][1] / 2 - CUP_R), abs(cy)), cy)
    d_edge = min(tx - abs(ex), ty - abs(ey))
    above = _clamp(above * 3)  # anything resting on the item rides along when it is lifted
    return {"below": below, "above": above, "flat_below": flat * below,
            "crowd": 0.0 if gap_min == math.inf else _clamp(1 - gap_min / 0.03),
            "wall": _clamp(1 - wall_dist / 0.03), "big_nb": big, "small_nb": small, "flat": flat,
            "crowd_near": 0.0 if gap_min == math.inf else _clamp((0.02 - gap_min) / 0.02),
            "seal_center": _clamp((SEAL_D + 0.005 - d_center) / 0.01),
            "seal_edge": _clamp((SEAL_D + 0.005 - d_edge) / 0.01)}


def score_grasp(state, item_id, grasp_type):
    s = normalize(state)
    f = dict(features(s, item_id))
    kind = _kind(grasp_type)
    f["seal"] = f["seal_center"] if kind == "center" else f["seal_edge"] if kind == "offcenter" else 0.0
    w = BASE[kind]
    scale = MODE_SCALE.get(s["mode"], {})
    z = w["bias"] + sum(v * scale.get(k, 1.0) * f.get(k, 0.0) for k, v in w.items() if k != "bias")
    return round(1 / (1 + math.exp(-z)), 4)
