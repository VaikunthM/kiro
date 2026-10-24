"""Tests for the robot brain. Plain functions: runs with pytest or `python tests/test_brain.py` (no dependencies).

FakeSim is a stand-in for sim.sim.Sim that only exercises the wiring (state shape, step dicts, push/grasp loop).
Its outcomes are sampled from score_grasp, so it proves nothing about real physics.
"""
import copy
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.geometry import normalize
from brain.scene_gen import MODES, TOTE, make_scene
from brain.score import GRASPS, score_grasp
from brain.strategy import ClearPick, GrabOnly


def state_from_scene(scene, as_dict=False, bbox_style="nested"):
    objs = []
    for o in scene["objects"]:
        lo = [o["pos"][i] - o["size"][i] / 2 for i in range(3)]
        hi = [o["pos"][i] + o["size"][i] / 2 for i in range(3)]
        d = {"id": o["id"], "pos": list(o["pos"]), "size": list(o["size"])}
        if bbox_style == "nested":
            d["bbox"] = {"min": lo, "max": hi}
        else:
            d["bbox_min"], d["bbox_max"] = lo, hi
        objs.append(d)
    if as_dict:
        objs = {o.pop("id"): o for o in objs}
    return {"scene_id": scene["scene_id"], "tote": scene["tote"], "objects": objs}


class FakeSim:
    def __init__(self, scene):
        self.scene, self.objs = scene, copy.deepcopy(scene["objects"])
        self.rng = random.Random(scene["seed"])

    def state(self):
        return state_from_scene({**self.scene, "objects": self.objs})

    def _settle(self):
        placed = []
        for o in sorted(self.objs, key=lambda o: o["pos"][2] - o["size"][2] / 2):
            floor = 0.0
            for p in placed:
                if min(o["pos"][0] + o["size"][0] / 2, p["pos"][0] + p["size"][0] / 2) > max(o["pos"][0] - o["size"][0] / 2, p["pos"][0] - p["size"][0] / 2) and \
                   min(o["pos"][1] + o["size"][1] / 2, p["pos"][1] + p["size"][1] / 2) > max(o["pos"][1] - o["size"][1] / 2, p["pos"][1] - p["size"][1] / 2):
                    floor = max(floor, p["pos"][2] + p["size"][2] / 2)
            o["pos"][2] = floor + o["size"][2] / 2
            placed.append(o)

    def grasp(self, item_id, grasp):
        item = next(o for o in self.objs if o["id"] == item_id)
        risk = score_grasp(self.state(), item_id, grasp)
        self.objs.remove(item)
        if self.objs and self.rng.random() < min(1.0, max(0.0, risk + self.rng.uniform(-0.08, 0.08))):
            buddy = min(self.objs, key=lambda o: (o["pos"][0] - item["pos"][0]) ** 2 + (o["pos"][1] - item["pos"][1]) ** 2)
            self.objs.remove(buddy)
            self._settle()
            return "double"
        self._settle()
        return "clean"

    def push(self, item_id, direction):
        item = next(o for o in self.objs if o["id"] == item_id)
        hx, hy = TOTE[0] / 2 - item["size"][0] / 2, TOTE[1] / 2 - item["size"][1] / 2
        item["pos"][0] = max(-hx, min(hx, item["pos"][0] + 0.09 * direction[0]))
        item["pos"][1] = max(-hy, min(hy, item["pos"][1] + 0.09 * direction[1]))
        self._settle()
        return "ok"


def run(policy, scene):
    sim, doubles, pushes = FakeSim(scene), 0, 0
    for _ in range(3 * len(scene["objects"]) + 6):
        step = policy.decide(sim.state())
        if step is None:
            break
        assert step["action"] in ("grasp", "push") and step["reason"] and step["item"]
        if step["action"] == "push":
            assert abs(math.hypot(*step["direction"]) - 1.0) < 1e-2
            sim.push(step["item"], step["direction"])
            pushes += 1
        else:
            assert step["grasp"] in GRASPS
            doubles += sim.grasp(step["item"], step["grasp"]) == "double"
    return doubles, pushes


def test_scenes_valid_and_reproducible():
    for mode in MODES:
        for n in range(1, 21):
            sc = make_scene(mode, n)
            assert sc == make_scene(mode, n)
            assert 4 <= len(sc["objects"]) <= 10, (mode, n, len(sc["objects"]))
            objs = sc["objects"]
            assert len({o["id"] for o in objs}) == len(objs)
            for o in objs:
                assert abs(o["pos"][0]) + o["size"][0] / 2 <= TOTE[0] / 2 + 1e-6, (mode, n, o["id"])
                assert abs(o["pos"][1]) + o["size"][1] / 2 <= TOTE[1] / 2 + 1e-6, (mode, n, o["id"])
                assert o["pos"][2] - o["size"][2] / 2 >= -1e-6
            for i, a in enumerate(objs):  # two objects at the same height must not overlap
                for b in objs[i + 1:]:
                    z_overlap = min(a["pos"][2] + a["size"][2] / 2, b["pos"][2] + b["size"][2] / 2) - \
                        max(a["pos"][2] - a["size"][2] / 2, b["pos"][2] - b["size"][2] / 2)
                    xy_overlap = (abs(a["pos"][0] - b["pos"][0]) < (a["size"][0] + b["size"][0]) / 2 and
                                  abs(a["pos"][1] - b["pos"][1]) < (a["size"][1] + b["size"][1]) / 2)
                    assert not (z_overlap > 1e-6 and xy_overlap), (mode, n, a["id"], b["id"])


def test_flat_stacked_scenes_contain_stacks():
    stacked = sum(any(o["pos"][2] - o["size"][2] / 2 > 0.001 for o in make_scene("flat_stacked", n)["objects"])
                  for n in range(1, 21))
    assert stacked == 20


def test_state_formats_normalize_identically():
    sc = make_scene("flat_stacked", 3)
    a = normalize(state_from_scene(sc))
    b = normalize(state_from_scene(sc, as_dict=True))
    c = normalize(state_from_scene(sc, bbox_style="flat"))
    assert a["objects"] == b["objects"] == c["objects"]
    assert a["mode"] == "flat_stacked"


def test_risk_between_0_and_1():
    for mode in MODES:
        for n in range(1, 101):
            st = state_from_scene(make_scene(mode, n))
            for o in st["objects"]:
                for g in GRASPS:
                    assert 0.0 <= score_grasp(st, o["id"], g) <= 1.0


def test_isolated_scores_lower_than_stacked():
    checked = 0
    for n in range(1, 101):
        sc = make_scene("flat_stacked", n)
        for top in sc["objects"]:
            if top["pos"][2] - top["size"][2] / 2 < 0.001:
                continue
            alone = copy.deepcopy(sc)
            alone["objects"] = [dict(top, pos=[top["pos"][0], top["pos"][1], top["size"][2] / 2])]
            for g in GRASPS:
                assert score_grasp(state_from_scene(alone), top["id"], g) < score_grasp(state_from_scene(sc), top["id"], g)
            checked += 1
    assert checked > 50


def test_clearpick_pushes_a_stack_first():
    mock = {"scene_id": "flat_stacked_000", "mode": "flat_stacked", "seed": 0, "tote": {"size": TOTE}, "objects": [
        {"id": "box_1", "size": [0.14, 0.10, 0.015], "pos": [0.05, 0.0, 0.0075]},
        {"id": "box_2", "size": [0.11, 0.08, 0.015], "pos": [0.055, 0.005, 0.0230]},
        {"id": "gelatin_box_1", "size": [0.085, 0.073, 0.028], "pos": [-0.05, 0.09, 0.014]}]}
    step = ClearPick().decide(state_from_scene(mock))
    assert step["action"] == "grasp" and step["item"] == "gelatin_box_1"  # the free item is safe, so take it first
    mock["objects"] = mock["objects"][:2]
    step = ClearPick().decide(state_from_scene(mock))
    assert step["action"] in ("push", "grasp") and step["item"] in ("box_1", "box_2")
    assert GrabOnly().decide(state_from_scene(mock))["item"] == "box_2"


def test_clearpick_beats_grab_only_in_wiring_check():
    g = c = pushes = 0
    for n in range(1, 21):
        sc = make_scene("flat_stacked", n)
        g += run(GrabOnly(), sc)[0]
        d, p = run(ClearPick(), sc)
        c += d
        pushes += p
    assert c < g and pushes > 0, (c, g, pushes)


def _one(size, pos, mode="against_wall"):
    sc = {"scene_id": f"{mode}_900", "mode": mode, "seed": 0, "tote": {"size": TOTE},
          "objects": [{"id": "a", "size": size, "pos": pos}]}
    return state_from_scene(sc)


def test_wall_seal_rule_hurts_centre_suction_but_not_edge_suction():
    # foam brick flush to the +x wall, its wide side (7.5 cm) facing the wall: centre cup is 3.75 cm out, edge cup can slide away
    st = _one([0.075, 0.05, 0.05], [0.1625, 0.0, 0.025])
    assert score_grasp(st, "a", "suction_center") > score_grasp(st, "a", "suction_edge") + 0.1
    # narrow side (5 cm) facing the wall: neither cup can get 3.5 cm away, both fail
    narrow = _one([0.075, 0.05, 0.05], [0.0, 0.125, 0.025])
    assert score_grasp(narrow, "a", "suction_center") > 0.9 and score_grasp(narrow, "a", "suction_edge") > 0.9
    mid = _one([0.075, 0.05, 0.05], [0.0, 0.0, 0.025])
    assert score_grasp(mid, "a", "suction_center") < 0.1


def test_close_side_neighbour_hurts_centre_suction_only():
    sc = {"scene_id": "small_beside_big_900", "mode": "small_beside_big", "seed": 0, "tote": {"size": TOTE},
          "objects": [{"id": "a", "size": [0.075, 0.05, 0.05], "pos": [-0.05, 0.0, 0.025]},
                      {"id": "b", "size": [0.097, 0.082, 0.05], "pos": [0.041, 0.0, 0.025]}]}
    st = state_from_scene(sc)
    assert score_grasp(st, "a", "suction_center") > 0.5
    assert score_grasp(st, "a", "suction_edge") < score_grasp(st, "a", "suction_center") - 0.2


def test_no_push_away_from_a_wall_the_item_is_flush_against():
    from brain.strategy import push_direction
    sc = {"scene_id": "against_wall_901", "mode": "against_wall", "seed": 0, "tote": {"size": TOTE},
          "objects": [{"id": "a", "size": [0.085, 0.073, 0.028], "pos": [0.0, 0.1135, 0.014]},   # flush to +y wall
                      {"id": "b", "size": [0.085, 0.073, 0.028], "pos": [0.0, 0.03, 0.014]}]}     # neighbour toward the centre
    assert push_direction(state_from_scene(sc), "a") is None   # only way out is away from the wall


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
