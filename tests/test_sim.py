"""Simulator behaviour tests. Scenes are built inline so each test shows exactly what it sets up."""
import json

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from sim.sim import Sim


def box(oid, size, pos, **kw):
    return {"id": oid, "shape": "box", "size": list(size), "pos": list(pos), **kw}


def scene(*objects, sid="test"):
    return {"scene_id": sid, "mode": "test", "objects": list(objects)}


@pytest.fixture
def sim():
    return Sim(width=160, height=120, video=False)


# ---------- suction ----------

def test_suction_on_isolated_box_is_clean(sim):
    sim.load(scene(box("a", [0.10, 0.08, 0.04], [0, 0, 0.02])))
    assert sim.grasp("a", "suction_center") == "clean"


def test_suction_on_top_of_two_flat_stacked_boxes_is_double_grab(sim):
    sim.load(scene(
        box("bottom", [0.11, 0.09, 0.035], [0, 0, 0.0175]),
        box("top", [0.085, 0.073, 0.028], [0.005, 0, 0.0495]),
    ))
    assert sim.grasp("top", "suction_center") == "double_grab"


def test_suction_cup_straddling_thin_item_and_same_height_neighbour_is_double_grab(sim):
    sim.load(scene(
        box("thin", [0.03, 0.12, 0.05], [0, 0, 0.025]),
        box("big", [0.12, 0.12, 0.05], [0.0775, 0, 0.025]),  # 2.5 mm gap
    ))
    assert sim.grasp("thin", "suction_center") == "double_grab"


def test_item_resting_on_the_lifted_item_rides_along(sim):
    sim.load(scene(
        box("base", [0.15, 0.12, 0.08], [0, 0, 0.04]),           # thick: no cling
        box("rider", [0.05, 0.05, 0.03], [0.04, 0, 0.096]),      # sits on base, off the cup
    ))
    assert sim.grasp("base", "suction_offcenter") == "double_grab"


# ---------- side pinch ----------

def test_pinch_on_isolated_box_is_clean(sim):
    sim.load(scene(box("a", [0.10, 0.06, 0.05], [0, 0, 0.025])))
    assert sim.grasp("a", "side_pinch") == "clean"


def test_pinch_on_box_flush_against_wall_is_miss(sim):
    # short axis is x and the box touches the +x wall (inner wall at x = 0.30)
    sim.load(scene(box("a", [0.06, 0.12, 0.05], [0.27, 0, 0.025])))
    assert sim.grasp("a", "side_pinch") == "miss"


def test_pinch_on_top_of_stack_does_not_take_the_box_below(sim):
    sim.load(scene(
        box("bottom", [0.11, 0.09, 0.035], [0, 0, 0.0175]),
        box("top", [0.085, 0.073, 0.028], [0.005, 0, 0.0495]),
    ))
    assert sim.grasp("top", "side_pinch") == "clean"


# ---------- picked items leave the tote ----------

def test_clean_pick_removes_item_from_state(sim):
    sim.load(scene(
        box("a", [0.10, 0.08, 0.04], [-0.15, 0, 0.02]),
        box("b", [0.10, 0.08, 0.04], [0.15, 0, 0.02]),
    ))
    sim.grasp("a", "suction_center")
    assert set(sim.state()["objects"]) == {"b"}


def test_double_grab_removes_both_items(sim):
    sim.load(scene(
        box("bottom", [0.11, 0.09, 0.035], [0, 0, 0.0175]),
        box("top", [0.085, 0.073, 0.028], [0.005, 0, 0.0495]),
    ))
    sim.grasp("top", "suction_center")
    assert sim.state()["objects"] == {}


def test_unknown_grasp_type_raises(sim):
    sim.load(scene(box("a", [0.1, 0.08, 0.04], [0, 0, 0.02])))
    with pytest.raises(ValueError):
        sim.grasp("a", "teleport")


# ---------- push ----------

def test_push_moves_item_in_direction(sim):
    sim.load(scene(box("a", [0.08, 0.08, 0.04], [-0.1, 0, 0.02])))
    x0 = sim.state()["objects"]["a"]["pos"][0]
    assert sim.push("a", [1, 0]) == "ok"
    assert sim.state()["objects"]["a"]["pos"][0] - x0 > 0.03


def test_push_separates_a_flat_stack(sim):
    sim.load(scene(
        box("bottom", [0.11, 0.09, 0.035], [0, 0, 0.0175]),
        box("top", [0.085, 0.073, 0.028], [0.005, 0, 0.0495]),
    ))
    sim.push("top", [1, 0])
    assert sim.grasp("top", "suction_center") == "clean"


def test_push_is_blocked_when_pusher_cannot_get_behind_item(sim):
    # item flush against the -x wall; pushing toward +x needs the pusher between item and wall
    sim.load(scene(box("a", [0.08, 0.08, 0.04], [-0.26, 0, 0.02])))
    assert sim.push("a", [1, 0]) == "blocked"


# ---------- recording ----------

def test_record_writes_mp4_of_the_moves(tmp_path):
    sim = Sim(width=160, height=120, video=True)
    sim.load(scene(box("a", [0.1, 0.08, 0.04], [0, 0, 0.02])))
    sim.grasp("a", "suction_center")
    out = tmp_path / "run.mp4"
    n = sim.record(out)
    assert out.exists() and out.stat().st_size > 1000
    assert n > 10


# ---------- properties ----------

flat_box = st.tuples(
    st.floats(0.05, 0.16), st.floats(0.05, 0.12), st.floats(0.02, 0.06),
)


@pytest.mark.slow
@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(size=flat_box, x=st.floats(-0.15, 0.15), y=st.floats(-0.08, 0.08), yaw=st.floats(0, np.pi))
def test_property_isolated_box_is_always_a_clean_suction_pick(size, x, y, yaw):
    sim = Sim(width=64, height=48, video=False)
    q = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    sim.load(scene(box("a", size, [x, y, size[2] / 2], quat=q)))
    assert sim.grasp("a", "suction_center") == "clean"


@pytest.mark.slow
@settings(max_examples=12, deadline=None)
@given(size=flat_box, x=st.floats(-0.15, 0.15), y=st.floats(-0.08, 0.08), yaw=st.floats(0, np.pi))
def test_property_state_bbox_contains_centre_and_is_json_serialisable(size, x, y, yaw):
    sim = Sim(width=64, height=48, video=False)
    q = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    st_ = sim.load(scene(box("a", size, [x, y, size[2] / 2], quat=q)))
    o = st_["objects"]["a"]
    assert all(lo <= p <= hi for lo, p, hi in zip(o["bbox_min"], o["pos"], o["bbox_max"]))
    json.dumps(st_)


@pytest.mark.slow
@settings(max_examples=8, deadline=None)
@given(size=flat_box, top_frac=st.floats(0.6, 0.95))
def test_property_two_flat_boxes_stacked_always_double_grab_with_centre_suction(size, top_frac):
    sim = Sim(width=64, height=48, video=False)
    sx, sy, sz = size
    top = [sx * top_frac, sy * top_frac, min(sz, 0.04)]
    sim.load(scene(
        box("bottom", [sx, sy, min(sz, 0.04)], [0, 0, min(sz, 0.04) / 2]),
        box("top", top, [0, 0, min(sz, 0.04) + top[2] / 2 + 0.001]),
    ))
    assert sim.grasp("top", "suction_center") == "double_grab"
