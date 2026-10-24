"""Spec task 3, mechanism checks. test_physics_suction.py pins the *outcomes* (which physics mode must share
with attach mode); these pin *how* physics mode gets them, per tasks 3.2-3.3: the cup finds the item through
MuJoCo contacts, and a weld equality (not position pinning) carries it during the lift."""
from sim.sim import Sim


def box(oid, size, pos):
    return {"id": oid, "type": "flat_box", "size": list(size), "pos": list(pos)}


SCENE = {"scene_id": "mech_t", "mode": "test", "objects": [box("a", [0.10, 0.08, 0.04], [0, 0, 0.02])]}


def watch(sim):
    """Record, for every physics step of the next move, how many welds are on and whether the cup has a contact."""
    log = []
    original = sim._physics_step

    def spy(*args, **kwargs):
        out = original(*args, **kwargs)
        cup = None
        try:
            cup = sim.model.geom("cup").id
        except KeyError:
            pass
        touching = cup is not None and any(cup in (c.geom1, c.geom2) for c in sim.data.contact[: sim.data.ncon])
        log.append((int(sim.data.eq_active.sum()) if sim.model.neq else 0, touching))
        return out

    sim._physics_step = spy
    return log


def test_physics_suction_lifts_with_a_weld():
    sim = Sim(width=64, height=48, suction="physics")
    sim.load(SCENE)
    log = watch(sim)
    assert sim.grasp("a", "suction_center") == "clean"
    assert max(w for w, _ in log) >= 1            # a weld was switched on during the move
    assert int(sim.data.eq_active.sum()) == 0      # and released afterwards


def test_physics_suction_finds_the_item_by_contact():
    sim = Sim(width=64, height=48, suction="physics")
    sim.load(SCENE)
    log = watch(sim)
    sim.grasp("a", "suction_center")
    assert any(t for _, t in log)                  # the cup really touched it in MuJoCo


def test_attach_mode_uses_neither_welds_nor_cup_contacts():
    sim = Sim(width=64, height=48)
    sim.load(SCENE)
    log = watch(sim)
    assert sim.grasp("a", "suction_center") == "clean"
    assert max(w for w, _ in log) == 0
    assert not any(t for _, t in log)
