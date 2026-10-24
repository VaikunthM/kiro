"""Spec task 3: physics-based suction behind a flag. Acceptance tests (written before the implementation).

Contract: `Sim(suction="physics")` finds what the cup touches from MuJoCo's contact list (the cup geom collides
while descending) and holds it with a weld equality constraint during the lift, instead of pinning positions.
`Sim()` keeps `suction="attach"` as the default. Outcomes still come from counting what rose more than
LIFT_CHECK, and the agreed rules stay: cling (flat stack), (a) neighbour drag, (b) wall seal. Prototype notes:
.kiro/specs/simulator/notes-physics-suction.md.
"""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sim.sim import Sim


def box(oid, size, pos, **kw):
    return {"id": oid, "type": kw.pop("type", "flat_box"), "size": list(size), "pos": list(pos), **kw}


def scene(*objects):
    return {"scene_id": "phys_t", "mode": "test", "objects": list(objects)}


STACK = (box("bottom", [0.11, 0.09, 0.035], [0, 0, 0.0175]),
         box("top", [0.085, 0.073, 0.028], [0.005, 0, 0.0495]))


@pytest.fixture
def phys():
    return Sim(width=160, height=120, suction="physics")


def test_attach_is_the_default_suction_model():
    assert Sim().suction == "attach"


def test_unknown_suction_model_is_rejected():
    with pytest.raises(ValueError):
        Sim(suction="vacuum_magic")


def test_physics_suction_on_isolated_box_is_clean(phys):
    phys.load(scene(box("a", [0.10, 0.08, 0.04], [0, 0, 0.02])))
    assert phys.grasp("a", "suction_center") == "clean"
    assert phys.state()["objects"] == []


def test_physics_suction_keeps_the_flat_stack_double(phys):
    phys.load(scene(*STACK))
    assert phys.grasp("top", "suction_center") == "double"


def test_physics_suction_carries_a_rider(phys):
    phys.load(scene(box("base", [0.15, 0.12, 0.08], [0, 0, 0.04]),
                    box("rider", [0.05, 0.05, 0.03], [0.04, 0, 0.096])))
    assert phys.grasp("base", "suction_edge") == "double"


def test_physics_suction_keeps_the_wall_seal_rule(phys):
    phys.load(scene(box("a", [0.06, 0.10, 0.04], [0.17, 0, 0.02])))
    assert phys.grasp("a", "suction_center") == "miss"


def test_physics_suction_keeps_the_neighbour_drag_rule(phys):
    phys.load(scene(box("big", [0.16, 0.12, 0.06], [-0.04, 0, 0.03]),
                    box("small", [0.05, 0.04, 0.03], [0.07, 0, 0.015])))
    assert phys.grasp("big", "suction_center") == "double"


def test_physics_suction_does_not_disturb_neighbours_while_descending(phys):
    phys.load(scene(box("a", [0.10, 0.08, 0.04], [-0.08, 0, 0.02]),
                    box("b", [0.10, 0.08, 0.04], [0.08, 0, 0.02])))
    b0 = next(o for o in phys.state()["objects"] if o["id"] == "b")["pos"]
    phys.grasp("a", "suction_center")
    b1 = next(o for o in phys.state()["objects"] if o["id"] == "b")["pos"]
    assert np.linalg.norm(np.array(b1) - np.array(b0)) < 0.005


def test_physics_and_attach_agree_on_the_failure_rule_scenes():
    scenes = [scene(*STACK),
              scene(box("a", [0.06, 0.10, 0.04], [0.17, 0, 0.02])),
              scene(box("big", [0.16, 0.12, 0.06], [-0.04, 0, 0.03]),
                    box("small", [0.05, 0.04, 0.03], [0.07, 0, 0.015]))]
    picks = [("top", "suction_center"), ("a", "suction_center"), ("big", "suction_center")]
    for sc, (item, grasp) in zip(scenes, picks):
        a, p = Sim(width=64, height=48), Sim(width=64, height=48, suction="physics")
        a.load(sc)
        p.load(sc)
        assert a.grasp(item, grasp) == p.grasp(item, grasp)


@pytest.mark.slow
@settings(max_examples=10, deadline=None)
@given(sx=st.floats(0.05, 0.12), sy=st.floats(0.05, 0.10), sz=st.floats(0.02, 0.06),
       x=st.floats(-0.08, 0.08), y=st.floats(-0.05, 0.05))
def test_property_isolated_box_is_a_clean_physics_suction_pick(sx, sy, sz, x, y):
    sim = Sim(width=64, height=48, suction="physics")
    sim.load(scene(box("a", [sx, sy, sz], [x, y, sz / 2])))
    assert sim.grasp("a", "suction_center") == "clean"
