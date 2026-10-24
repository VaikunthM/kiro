"""Spec task 4: optional YCB mesh visuals. Acceptance tests (written before the implementation).

Contract: `Sim(meshes=True)` swaps in textured YCB meshes for the object *types* listed in
assets/ycb/manifest.json, as visual-only geoms named "<id>_visual" (contype = conaffinity = 0), and hides the
collision primitive. Collision, physics, outcomes and state() are identical to meshes=False. Types without a
mesh (e.g. flat_box) and a missing asset folder fall back to the plain primitive without errors.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from sim.sim import Sim

MANIFEST = Path(__file__).resolve().parents[1] / "assets" / "ycb" / "manifest.json"


def ob(i, t, size, pos):
    return {"id": i, "type": t, "size": list(size), "pos": list(pos), "yaw": 0.0}


SCENE = {"scene_id": "mesh_t", "mode": "test", "objects": [
    ob("sugar_1", "sugar_box", [0.175, 0.089, 0.038], [-0.07, -0.08, 0.019]),
    ob("can_1", "tomato_soup_can", [0.066, 0.066, 0.101], [0.12, 0.07, 0.0505]),
    ob("flat_1", "flat_box", [0.12, 0.09, 0.015], [-0.07, 0.07, 0.0075]),
    ob("flat_2", "flat_box", [0.10, 0.08, 0.015], [-0.065, 0.072, 0.0231]),   # stacked on flat_1
    ob("meat_1", "potted_meat_can", [0.097, 0.082, 0.05], [0.12, -0.08, 0.025]),
]}


def geom(sim, name):
    return sim.model.geom(name)


def test_manifest_lists_the_scene_generator_types_with_existing_files():
    m = json.loads(MANIFEST.read_text())
    for t in ("sugar_box", "tomato_soup_can", "gelatin_box", "potted_meat_can", "foam_brick"):
        e = m["types"][t]
        assert (MANIFEST.parent / e["mesh"]).exists() and (MANIFEST.parent / e["texture"]).exists()


def test_meshes_are_off_by_default():
    sim = Sim(width=64, height=48)
    sim.load(SCENE)
    assert sim.model.nmesh == 0


def test_meshes_add_visual_only_geoms_for_known_types():
    sim = Sim(width=64, height=48, meshes=True)
    sim.load(SCENE)
    assert sim.model.nmesh == 3                                   # sugar_box, tomato_soup_can, potted_meat_can
    for oid in ("sugar_1", "can_1", "meat_1"):
        v = geom(sim, f"{oid}_visual")
        assert v.contype[0] == 0 and v.conaffinity[0] == 0        # visual only
        assert v.bodyid[0] == sim.model.body(oid).id


def test_types_without_a_mesh_keep_the_primitive_and_no_visual_geom():
    sim = Sim(width=64, height=48, meshes=True)
    sim.load(SCENE)
    with pytest.raises(KeyError):
        geom(sim, "flat_1_visual")


def test_collision_geoms_are_unchanged_by_meshes():
    plain, meshed = Sim(width=64, height=48), Sim(width=64, height=48, meshes=True)
    plain.load(SCENE)
    meshed.load(SCENE)
    for o in SCENE["objects"]:
        a, b = geom(plain, o["id"]), geom(meshed, o["id"])
        assert a.type[0] == b.type[0]
        assert np.allclose(a.size, b.size)
        assert b.contype[0] == a.contype[0] and b.conaffinity[0] == a.conaffinity[0]


def test_state_and_outcomes_are_identical_with_meshes():
    plain, meshed = Sim(width=64, height=48), Sim(width=64, height=48, meshes=True)
    assert plain.load(SCENE) == meshed.load(SCENE)
    for item, grasp in (("can_1", "suction_center"), ("flat_2", "suction_center"), ("meat_1", "side_pinch")):
        assert plain.grasp(item, grasp) == meshed.grasp(item, grasp)
    assert plain.state() == meshed.state()


def test_render_shows_the_meshes():
    plain, meshed = Sim(width=160, height=120), Sim(width=160, height=120, meshes=True)
    plain.load(SCENE)
    meshed.load(SCENE)
    diff = np.abs(plain.render().astype(int) - meshed.render().astype(int)).mean()
    assert diff > 1.0


def test_missing_asset_folder_falls_back_to_primitives(tmp_path):
    sim = Sim(width=64, height=48, meshes=True, asset_dir=tmp_path / "nowhere")
    sim.load(SCENE)
    assert sim.model.nmesh == 0


def test_highlight_still_marks_the_picked_item_with_meshes():
    sim = Sim(width=160, height=120, meshes=True)
    sim.load(SCENE)
    before = sim.render().astype(int)
    sim.highlight = {"sugar_1"}
    after = sim.render().astype(int)
    assert np.abs(after - before).sum() > 0
