"""Simplified failure rules agreed with P3 so small_beside_big and against_wall produce failures.
(a) suction_center also lifts a side neighbour whose edge is within NEIGHBOUR_DRAG_GAP and which shares
    at least half the shorter item's height band -> "double". suction_edge and side_pinch are unaffected.
(b) any suction whose cup centre is within CUP_RADIUS + WALL_SEAL_MARGIN (3.5 cm) of a tote wall can't
    seal -> "miss". suction_edge picks the side of the item with most clearance from walls and neighbours.
Default tote 0.40 x 0.30 x 0.15: inner walls at x = +/-0.20, y = +/-0.15."""
import pytest

from sim.sim import CUP_RADIUS, NEIGHBOUR_DRAG_GAP, WALL_SEAL_MARGIN, Sim


def box(oid, size, pos, **kw):
    return {"id": oid, "type": kw.pop("type", "flat_box"), "size": list(size), "pos": list(pos), **kw}


def scene(*objects):
    return {"scene_id": "t", "mode": "test", "objects": list(objects)}


def ids(sim):
    return [o["id"] for o in sim.state()["objects"]]


@pytest.fixture
def sim():
    return Sim(width=160, height=120)


def test_rule_constants_match_what_p3_scores():
    assert NEIGHBOUR_DRAG_GAP == pytest.approx(0.01)
    assert CUP_RADIUS + WALL_SEAL_MARGIN == pytest.approx(0.035)


# ---------- (a) side neighbour within ~1 cm is dragged up by centre suction ----------

def big_and_small(gap):
    return scene(box("big", [0.16, 0.12, 0.06], [-0.04, 0, 0.03]),
                 box("small", [0.05, 0.04, 0.03], [0.04 + 0.025 + gap, 0, 0.015]))


def test_centre_suction_with_neighbour_inside_drag_gap_is_double(sim):
    sim.load(big_and_small(gap=NEIGHBOUR_DRAG_GAP * 0.5))
    assert sim.grasp("big", "suction_center") == "double"
    assert ids(sim) == []


def test_centre_suction_with_neighbour_outside_drag_gap_is_clean(sim):
    sim.load(big_and_small(gap=NEIGHBOUR_DRAG_GAP * 2))
    assert sim.grasp("big", "suction_center") == "clean"
    assert ids(sim) == ["small"]


def test_drag_rule_works_from_the_small_item_too(sim):
    sim.load(big_and_small(gap=NEIGHBOUR_DRAG_GAP * 0.5))
    assert sim.grasp("small", "suction_center") == "double"


def test_edge_suction_is_not_affected_by_the_drag_rule(sim):
    sim.load(big_and_small(gap=NEIGHBOUR_DRAG_GAP * 0.5))
    assert sim.grasp("big", "suction_edge") == "clean"


def test_side_pinch_is_not_affected_by_the_drag_rule(sim):
    # small item with its neighbour along its long side; fingers close across x, away from the neighbour
    sim.load(scene(box("a", [0.05, 0.10, 0.04], [0.0, 0.0, 0.02]),
                   box("b", [0.05, 0.10, 0.04], [0.0, 0.105, 0.02])))
    assert sim.grasp("a", "side_pinch") == "clean"


def test_neighbour_in_a_different_height_band_is_not_dragged(sim):
    # item sitting on a thick base, with a short neighbour on the floor right next to the base
    sim.load(scene(box("base", [0.16, 0.12, 0.08], [-0.03, 0, 0.04]),
                   box("top", [0.06, 0.06, 0.03], [0.02, 0, 0.096]),
                   box("floor", [0.05, 0.05, 0.03], [0.05 + 0.025 + 0.005, 0, 0.015])))
    assert sim.grasp("top", "suction_center") == "clean"


# ---------- (b) suction too close to a wall can't seal ----------

def test_centre_suction_near_wall_is_a_miss_and_leaves_the_item(sim):
    # 6 cm wide box flush with the +x wall: cup centre 3 cm from the wall
    sim.load(scene(box("a", [0.06, 0.10, 0.04], [0.17, 0, 0.02])))
    assert sim.grasp("a", "suction_center") == "miss"
    assert ids(sim) == ["a"]


def test_suction_just_outside_the_wall_margin_is_clean(sim):
    d = CUP_RADIUS + WALL_SEAL_MARGIN + 0.01            # cup centre this far from the wall
    sim.load(scene(box("a", [0.06, 0.10, 0.04], [0.20 - d, 0, 0.02])))
    assert sim.grasp("a", "suction_center") == "clean"


def test_edge_suction_slides_away_from_the_wall_and_seals(sim):
    # 6 cm deep box flush with the +y wall: centre is 3 cm from it
    sim.load(scene(box("a", [0.12, 0.06, 0.04], [0.0, 0.12, 0.02])))
    assert sim.grasp("a", "suction_center") == "miss"
    assert sim.grasp("a", "suction_edge") == "clean"


def test_wall_rule_does_not_affect_side_pinch_along_the_wall(sim):
    sim.load(scene(box("a", [0.05, 0.10, 0.04], [0.0, 0.10, 0.02])))
    assert sim.grasp("a", "side_pinch") == "clean"


# ---------- tall items ----------

def test_item_taller_than_the_tote_wall_can_be_picked(sim):
    sim.load(scene(box("tall", [0.09, 0.06, 0.20], [0.0, 0.0, 0.10])))
    assert sim.grasp("tall", "suction_center") == "clean"


def test_tall_item_can_be_pinched(sim):
    sim.load(scene(box("tall", [0.09, 0.06, 0.20], [0.0, 0.0, 0.10])))
    assert sim.grasp("tall", "side_pinch") == "clean"
