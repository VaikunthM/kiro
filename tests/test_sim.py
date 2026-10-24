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


# ---------- picked items stay out of frame for both cameras ----------

def _geom_in_frame(sim, oid, camera):
    """Project object `oid`'s current body centre into `camera` and return whether it lands inside the
    rendered image. Picked items are parked far outside the tote; this checks they don't intrude into a
    rendered frame. Uses MuJoCo's resolved camera pose/fovy so it matches what render() actually draws."""
    import mujoco  # noqa: F401  (ensures the model is realised; Sim already imports it)
    cam = sim.model.camera(camera)
    cid = cam.id
    campos = sim.data.cam_xpos[cid].copy()
    camR = sim.data.cam_xmat[cid].reshape(3, 3).copy()
    pt = sim.data.body(oid).xpos.copy()
    local = camR.T @ (pt - campos)          # into the camera frame; it looks down its own -z
    if local[2] >= 0:                       # at or behind the image plane -> not visible
        return False
    w, h = sim.width, sim.height
    f = (h / 2) / np.tan(np.radians(cam.fovy[0] / 2))
    u = f * local[0] / (-local[2]) + w / 2
    v = -f * local[1] / (-local[2]) + h / 2
    return 0 <= u < w and 0 <= v < h


def test_picked_item_is_out_of_frame_for_the_angled_camera(sim):
    # 5.3 says removed items leave the overhead view; the angled demo camera must also not show them.
    sim.load(scene(
        box("a", [0.10, 0.08, 0.04], [-0.1, 0, 0.02]),
        box("b", [0.10, 0.08, 0.04], [0.1, 0, 0.02]),
    ))
    assert _geom_in_frame(sim, "a", "angled")        # sanity: it IS in frame before the pick
    sim.grasp("a", "suction_center")
    assert "a" in sim.removed
    assert not _geom_in_frame(sim, "a", "angled")    # parked out of the angled frame (8.4)
    assert not _geom_in_frame(sim, "a", "overhead")  # overhead view unchanged: still excludes it (5.3)


def test_overhead_pick_removal_still_excludes_the_item(sim):
    # 5.3 regression guard: the overhead exclusion that already held must keep holding.
    sim.load(scene(box("a", [0.10, 0.08, 0.04], [0, 0, 0.02])))
    assert _geom_in_frame(sim, "a", "overhead")
    sim.grasp("a", "suction_center")
    assert not _geom_in_frame(sim, "a", "overhead")


def test_many_parked_items_all_stay_out_of_both_frames(sim):
    # _remove lays picked items on a 6-wide grid; confirm no slot drifts into either camera's frame.
    # 3 cm apart and away from the walls, so rule (a) (neighbour within 1 cm) and rule (b) (wall seal) don't apply
    grid = [(x, y) for y in (-0.06, 0.06) for x in (-0.12, -0.04, 0.04, 0.12)][:7]
    items = [box(f"o{i}", [0.05, 0.05, 0.04], [x, y, 0.02]) for i, (x, y) in enumerate(grid)]
    sim.load(scene(*items))
    for it in items:
        sim.grasp(it["id"], "suction_center")
    assert sorted(sim.removed) == sorted(o["id"] for o in items)
    for it in items:
        assert not _geom_in_frame(sim, it["id"], "angled")
        assert not _geom_in_frame(sim, it["id"], "overhead")


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


# ---------- highlight scaffolding (render-only) ----------

def _highlight_during_grasp(sim, item_id, grasp_type):
    """Run a grasp and capture sim.highlight as seen mid-move (during _lift)."""
    seen = {}
    real_lift = sim._lift

    def spy(tool, offsets):
        seen["mid"] = set(sim.highlight)
        return real_lift(tool, offsets)

    sim._lift = spy
    outcome = sim.grasp(item_id, grasp_type)
    sim._lift = real_lift
    return outcome, seen.get("mid", set())


def test_highlight_is_empty_before_any_grasp(sim):
    sim.load(scene(box("a", [0.10, 0.08, 0.04], [0, 0, 0.02])))
    assert set(sim.highlight) == set()


def test_highlight_records_target_during_grasp_and_clears_after(sim):
    sim.load(scene(box("a", [0.10, 0.08, 0.04], [0, 0, 0.02])))
    outcome, mid = _highlight_during_grasp(sim, "a", "suction_center")
    assert outcome == "clean"
    assert "a" in mid                 # target is marked while the move is in progress
    assert set(sim.highlight) == set()  # cleared once the move ends


def test_highlight_records_clung_ids_during_flat_stack_grasp(sim):
    sim.load(scene(*STACK))
    outcome, mid = _highlight_during_grasp(sim, "top", "suction_center")
    assert outcome == "double"
    assert {"top", "bottom"} <= mid   # target plus the clung item are both marked
    assert set(sim.highlight) == set()


def test_highlight_is_cleared_after_a_miss(sim):
    # short axis is x and the box touches the +x wall -> pinch is a miss
    sim.load(scene(box("a", [0.06, 0.12, 0.05], [0.17, 0, 0.025])))
    outcome, _ = _highlight_during_grasp(sim, "a", "side_pinch")
    assert outcome == "miss"
    assert set(sim.highlight) == set()


def test_highlight_tracking_does_not_change_outcomes(sim):
    # highlight is render-only: outcomes must match the un-spied grasp
    sim.load(scene(*STACK))
    assert sim.grasp("top", "suction_center") == "double"


def _scene_geom_rgba(sim, oid, camera=None):
    """RGBA the renderer draws for object `oid`'s geom after a render (includes any highlight tint)."""
    import mujoco
    sim.render(camera=camera)
    gid = sim.model.geom(oid).id
    s = sim._renderer.scene
    for g in s.geoms[: s.ngeom]:
        if g.objtype == int(mujoco.mjtObj.mjOBJ_GEOM) and g.objid == gid:
            return np.array(g.rgba, dtype=float)
    raise AssertionError(f"no scene geom for {oid}")


def _scene_rgba_from_current_scene(sim, oid):
    """Read object `oid`'s geom RGBA out of whatever the renderer last drew (no new render)."""
    import mujoco
    gid = sim.model.geom(oid).id
    s = sim._renderer.scene
    for g in s.geoms[: s.ngeom]:
        if g.objtype == int(mujoco.mjtObj.mjOBJ_GEOM) and g.objid == gid:
            return np.array(g.rgba, dtype=float)
    raise AssertionError(f"no scene geom for {oid}")


def _grasp_capturing_mid_frame(sim, item_id, grasp_type, read_id, suppress_highlight=False):
    """Run a real grasp and, mid-move (inside _lift), render one frame and read `read_id`'s scene RGBA.
    With suppress_highlight=True the highlight set is emptied for the duration so the render is un-tinted,
    letting a caller confirm the grasp outcome is identical whether highlighting is on or off."""
    captured = {}
    real_lift = sim._lift

    def spy(tool, offsets):
        if suppress_highlight:
            sim.highlight = set()
        sim.render()  # a frame drawn DURING the move
        captured["rgba"] = _scene_rgba_from_current_scene(sim, read_id)
        return real_lift(tool, offsets)

    sim._lift = spy
    outcome = sim.grasp(item_id, grasp_type)
    sim._lift = real_lift
    return outcome, captured.get("rgba")


def test_render_mid_grasp_tints_the_target_geom_vs_baseline(sim):
    # end-to-end: trigger a real grasp, render DURING the move, and compare to the pre-grasp baseline
    sim.load(scene(box("a", [0.10, 0.08, 0.04], [0, 0, 0.02])))
    baseline = _scene_geom_rgba(sim, "a")                 # no grasp in progress -> no tint
    model_rgba = sim.model.geom("a").rgba.copy()

    outcome, mid_rgba = _grasp_capturing_mid_frame(sim, "a", "suction_center", read_id="a")

    assert outcome == "clean"
    assert mid_rgba is not None
    assert not np.allclose(mid_rgba, baseline)            # target is tinted while the move runs
    assert np.allclose(sim.model.geom("a").rgba, model_rgba)  # model geom untouched (render-only)
    assert set(sim.highlight) == set()                    # highlight cleared once the move ended
    assert np.allclose(_scene_geom_rgba(sim, "a"), baseline)  # post-grasp render is back to baseline


def test_grasp_outcome_is_unchanged_with_highlighting_on(sim):
    # outcomes must match whether a mid-grasp frame is tinted (on) or not (off); render-only invariant
    sim.load(scene(*STACK))
    with_highlight, _ = _grasp_capturing_mid_frame(sim, "top", "suction_center", read_id="top")

    sim.load(scene(*STACK))
    without_highlight, off_rgba = _grasp_capturing_mid_frame(
        sim, "top", "suction_center", read_id="top", suppress_highlight=True)

    sim.load(scene(*STACK))
    off_baseline = _scene_geom_rgba(sim, "top")

    assert with_highlight == without_highlight == "double"  # outcome identical with highlighting on/off
    assert np.allclose(off_rgba, off_baseline)              # with highlight suppressed the frame is un-tinted


def test_render_tints_highlighted_geom_only_in_the_scene(sim):
    sim.load(scene(box("a", [0.10, 0.08, 0.04], [0, 0, 0.02])))
    baseline = _scene_geom_rgba(sim, "a")
    model_rgba = sim.model.geom("a").rgba.copy()

    sim.highlight = {"a"}
    tinted = _scene_geom_rgba(sim, "a")

    assert not np.allclose(tinted, baseline)              # render colour changed while highlighted
    assert np.allclose(sim.model.geom("a").rgba, model_rgba)  # model geom untouched (render-only)

    sim.highlight = set()
    assert np.allclose(_scene_geom_rgba(sim, "a"), baseline)  # clears back to baseline


def test_render_leaves_unhighlighted_geoms_untouched(sim):
    sim.load(scene(
        box("a", [0.10, 0.08, 0.04], [-0.1, 0, 0.02]),
        box("b", [0.10, 0.08, 0.04], [0.1, 0, 0.02]),
    ))
    b_baseline = _scene_geom_rgba(sim, "b")
    sim.highlight = {"a"}
    assert not np.allclose(_scene_geom_rgba(sim, "a"), _scene_geom_rgba(sim, "a") * 0)  # sanity: "a" has colour
    assert np.allclose(_scene_geom_rgba(sim, "b"), b_baseline)  # "b" is not highlighted -> unchanged


def test_highlighting_does_not_change_state_or_model(sim):
    sim.load(scene(box("a", [0.10, 0.08, 0.04], [0, 0, 0.02])))
    before = json.dumps(sim.state(), sort_keys=True)
    model_rgba = sim.model.geom("a").rgba.copy()
    sim.highlight = {"a"}
    sim.render()  # render-only tint; must not touch state(), model, or physics
    assert json.dumps(sim.state(), sort_keys=True) == before
    assert np.allclose(sim.model.geom("a").rgba, model_rgba)


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


@pytest.mark.slow
@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(size=flat_box, x=st.floats(-0.09, 0.09), y=st.floats(-0.06, 0.06), yaw=st.floats(0, np.pi))
def test_property_grasp_marks_target_mid_move_and_clears_after(size, x, y, yaw):
    """For any isolated in-bounds box, a grasp marks the target while the move runs and clears it after.
    **Validates: Requirements 8.4**"""
    sim = Sim(width=64, height=48)
    sim.load(scene(box("a", size, [x, y, size[2] / 2], yaw=yaw)))
    outcome, mid = _highlight_during_grasp(sim, "a", "suction_center")
    assert outcome == "clean"
    assert "a" in mid
    assert set(sim.highlight) == set()


@pytest.mark.slow
@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(size=flat_box, x=st.floats(-0.09, 0.09), y=st.floats(-0.06, 0.06), yaw=st.floats(0, np.pi))
def test_property_picked_item_is_out_of_both_camera_frames(size, x, y, yaw):
    """For any isolated in-bounds box, once it is picked the parked body projects outside BOTH the
    overhead and the angled camera frames, so it appears in neither render.
    **Validates: Requirements 5.3, 8.4**"""
    sim = Sim(width=64, height=48)
    sim.load(scene(box("a", size, [x, y, size[2] / 2], yaw=yaw)))
    assert sim.grasp("a", "suction_center") == "clean"
    assert "a" in sim.removed
    assert not _geom_in_frame(sim, "a", "overhead")
    assert not _geom_in_frame(sim, "a", "angled")
