"""Simulator behaviour tests, written against the team contract (.kiro/steering shared contracts, section 3).
Scenes are built inline so each test shows exactly what it sets up. Default tote: 0.40 x 0.30 x 0.15."""
import json

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from sim.sim import Sim


def box(oid, size, pos, **kw):
    return {"id": oid, "type": kw.pop("type", "flat_box"), "size": list(size), "pos": list(pos), **kw}


def scene(*objects, sid="test"):
    return {"scene_id": sid, "mode": "test", "objects": list(objects)}


def objs(state):
    return {o["id"]: o for o in state["objects"]}


STACK = (box("bottom", [0.11, 0.09, 0.035], [0, 0, 0.0175]),
         box("top", [0.085, 0.073, 0.028], [0.005, 0, 0.0495]))


@pytest.fixture
def sim():
    return Sim(width=160, height=120)


# ---------- state() shape ----------

def test_state_lists_objects_with_contract_fields(sim):
    st_ = sim.load(scene(box("a", [0.1, 0.08, 0.04], [0, 0, 0.02])))
    assert {"scene_id", "mode", "tote", "objects"} <= st_.keys()
    assert st_["tote"]["size"] == [0.40, 0.30, 0.15]
    (o,) = st_["objects"]
    assert {"id", "type", "size", "pos", "bbox"} <= o.keys()
    assert set(o["bbox"]) == {"min", "max"}


def test_yaw_in_scene_is_the_loaded_orientation(sim):
    st_ = sim.load(scene(box("a", [0.1, 0.05, 0.04], [0, 0, 0.02], yaw=0.5)))
    assert objs(st_)["a"]["yaw"] == pytest.approx(0.5, abs=0.02)


def test_can_types_load_as_cylinders(sim):
    st_ = sim.load(scene(box("c", [0.066, 0.066, 0.101], [0, 0, 0.0505], type="tomato_soup_can")))
    assert objs(st_)["c"]["shape"] == "cylinder"


# ---------- suction ----------

def test_suction_on_isolated_box_is_clean(sim):
    sim.load(scene(box("a", [0.10, 0.08, 0.04], [0, 0, 0.02])))
    assert sim.grasp("a", "suction_center") == "clean"


def test_suction_on_top_of_two_flat_stacked_boxes_is_double(sim):
    sim.load(scene(*STACK))
    assert sim.grasp("top", "suction_center") == "double"


def test_suction_cup_straddling_thin_item_and_same_height_neighbour_is_double(sim):
    sim.load(scene(
        box("thin", [0.03, 0.12, 0.05], [0, 0, 0.025]),
        box("big", [0.12, 0.12, 0.05], [0.0775, 0, 0.025]),  # 2.5 mm gap
    ))
    assert sim.grasp("thin", "suction_center") == "double"


def test_item_resting_on_the_lifted_item_rides_along(sim):
    sim.load(scene(
        box("base", [0.15, 0.12, 0.08], [0, 0, 0.04]),           # thick: no cling
        box("rider", [0.05, 0.05, 0.03], [0.04, 0, 0.096]),      # sits on base, away from the cup
    ))
    assert sim.grasp("base", "suction_edge") == "double"


# ---------- side pinch ----------

def test_pinch_on_isolated_box_is_clean(sim):
    sim.load(scene(box("a", [0.10, 0.06, 0.05], [0, 0, 0.025])))
    assert sim.grasp("a", "side_pinch") == "clean"


def test_pinch_on_box_flush_against_wall_is_miss(sim):
    # short axis is x and the box touches the +x wall (inner wall at x = 0.20)
    sim.load(scene(box("a", [0.06, 0.12, 0.05], [0.17, 0, 0.025])))
    assert sim.grasp("a", "side_pinch") == "miss"


def test_pinch_on_top_of_stack_does_not_take_the_box_below(sim):
    sim.load(scene(*STACK))
    assert sim.grasp("top", "side_pinch") == "clean"


# ---------- picked items leave the tote ----------

def test_clean_pick_removes_item_from_state(sim):
    sim.load(scene(
        box("a", [0.10, 0.08, 0.04], [-0.1, 0, 0.02]),
        box("b", [0.10, 0.08, 0.04], [0.1, 0, 0.02]),
    ))
    sim.grasp("a", "suction_center")
    assert [o["id"] for o in sim.state()["objects"]] == ["b"]


def test_double_removes_both_items(sim):
    sim.load(scene(*STACK))
    sim.grasp("top", "suction_center")
    assert sim.state()["objects"] == []


def test_unknown_grasp_type_raises(sim):
    sim.load(scene(box("a", [0.1, 0.08, 0.04], [0, 0, 0.02])))
    with pytest.raises(ValueError):
        sim.grasp("a", "teleport")


# ---------- push ----------

def test_push_moves_item_in_direction(sim):
    sim.load(scene(box("a", [0.08, 0.08, 0.04], [-0.08, 0, 0.02])))
    x0 = objs(sim.state())["a"]["pos"][0]
    assert sim.push("a", [1, 0]) == "ok"
    assert objs(sim.state())["a"]["pos"][0] - x0 > 0.03


def test_push_separates_a_flat_stack(sim):
    sim.load(scene(*STACK))
    sim.push("top", [1, 0])
    assert sim.grasp("top", "suction_center") == "clean"


def test_push_with_no_room_behind_item_leaves_it_in_place(sim):
    # item flush against the -x wall; pushing toward +x needs the pusher between item and wall
    sim.load(scene(box("a", [0.08, 0.08, 0.04], [-0.16, 0, 0.02])))
    x0 = objs(sim.state())["a"]["pos"][0]
    assert sim.push("a", [1, 0]) == "ok"            # contract: push always returns "ok"
    assert abs(objs(sim.state())["a"]["pos"][0] - x0) < 0.01
    assert sim.last_push_blocked


# ---------- recording and clock ----------

def test_record_path_starts_and_record_none_saves_mp4(sim, tmp_path):
    sim.load(scene(box("a", [0.1, 0.08, 0.04], [0, 0, 0.02])))
    out = tmp_path / "run.mp4"
    sim.record(out)
    sim.grasp("a", "suction_center")
    n = sim.record(None)
    assert out.exists() and out.stat().st_size > 1000
    assert n > 10


def test_record_none_without_start_does_nothing(sim):
    sim.load(scene(box("a", [0.1, 0.08, 0.04], [0, 0, 0.02])))
    assert sim.record(None) == 0


def test_clock_advances_with_moves(sim):
    sim.load(scene(box("a", [0.1, 0.08, 0.04], [0, 0, 0.02])))
    t0 = sim.clock()
    sim.grasp("a", "suction_center")
    assert sim.clock() > t0


# ---------- properties ----------

flat_box = st.tuples(st.floats(0.04, 0.12), st.floats(0.04, 0.10), st.floats(0.02, 0.06))


@pytest.mark.slow
@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(size=flat_box, x=st.floats(-0.09, 0.09), y=st.floats(-0.06, 0.06), yaw=st.floats(0, np.pi))
def test_property_isolated_box_is_always_a_clean_suction_pick(size, x, y, yaw):
    sim = Sim(width=64, height=48)
    sim.load(scene(box("a", size, [x, y, size[2] / 2], yaw=yaw)))
    assert sim.grasp("a", "suction_center") == "clean"


@pytest.mark.slow
@settings(max_examples=12, deadline=None)
@given(size=flat_box, x=st.floats(-0.09, 0.09), y=st.floats(-0.06, 0.06), yaw=st.floats(0, np.pi))
def test_property_state_bbox_contains_centre_and_is_json_serialisable(size, x, y, yaw):
    sim = Sim(width=64, height=48)
    o = sim.load(scene(box("a", size, [x, y, size[2] / 2], yaw=yaw)))["objects"][0]
    assert all(lo <= p <= hi for lo, p, hi in zip(o["bbox"]["min"], o["pos"], o["bbox"]["max"]))
    json.dumps(sim.state())


@pytest.mark.slow
@settings(max_examples=8, deadline=None)
@given(size=flat_box, top_frac=st.floats(0.6, 0.95))
def test_property_two_flat_boxes_stacked_always_double_with_centre_suction(size, top_frac):
    sim = Sim(width=64, height=48)
    sx, sy, sz = size
    h = min(sz, 0.04)
    top = [sx * top_frac, sy * top_frac, h]
    sim.load(scene(
        box("bottom", [sx, sy, h], [0, 0, h / 2]),
        box("top", top, [0, 0, h + top[2] / 2 + 0.001]),
    ))
    assert sim.grasp("top", "suction_center") == "double"
