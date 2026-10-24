"""Batch runner tests: one run = one scene x one policy -> runs/<scene_id>_<policy>.json (+ .mp4).
Run JSON follows the team contract, section 4."""
import json

from sim.reference_policies import grab_only
from sim.run_batch import run_batch, run_scene

STACKED = {
    "scene_id": "flat_stacked_t01", "mode": "flat_stacked",
    "objects": [
        {"id": "bottom", "type": "flat_box", "size": [0.11, 0.09, 0.035], "pos": [0, 0, 0.0175]},
        {"id": "top", "type": "flat_box", "size": [0.085, 0.073, 0.028], "pos": [0.005, 0, 0.0495]},
        {"id": "lone", "type": "flat_box", "size": [0.08, 0.06, 0.05], "pos": [-0.13, 0.09, 0.025]},
    ],
}


def test_run_json_follows_the_contract(tmp_path):
    run = run_scene(STACKED, "grab_only", grab_only, tmp_path, video=False)
    saved = json.loads((tmp_path / "flat_stacked_t01_grab_only.json").read_text())
    assert saved == run
    assert {"scene_id", "mode", "policy", "steps", "summary", "video"} <= run.keys()
    step = run["steps"][0]
    assert {"t", "action", "item", "grasp", "direction", "risk", "reason", "outcome"} <= step.keys()
    assert set(run["summary"]) == {"clean", "double_grabs", "stalls", "moves", "misses", "items"}


def test_grab_only_double_grabs_the_flat_stack_and_empties_the_tote(tmp_path):
    s = run_scene(STACKED, "grab_only", grab_only, tmp_path, video=False)["summary"]
    assert (s["double_grabs"], s["clean"], s["moves"], s["items"]) == (1, 1, 2, 3)


def test_run_stops_when_policy_returns_none(tmp_path):
    run = run_scene(STACKED, "idle", lambda state: None, tmp_path, video=False)
    assert run["steps"] == [] and run["summary"]["moves"] == 0


def test_repeated_misses_end_the_run_and_are_counted(tmp_path):
    wall = {"scene_id": "wall_t01", "mode": "against_wall", "objects": [
        {"id": "a", "type": "flat_box", "size": [0.06, 0.12, 0.05], "pos": [0.17, 0, 0.025]}]}

    def stubborn(state):
        return {"action": "grasp", "item": "a", "grasp": "side_pinch", "risk": 0.9, "reason": "test"}

    s = run_scene(wall, "stubborn", stubborn, tmp_path, video=False)["summary"]
    assert s["misses"] == 3 and s["clean"] == 0


def test_policy_stall_steps_are_counted_and_do_not_move_the_robot(tmp_path):
    calls = []

    def waits_then_stops(state):
        calls.append(1)
        if len(calls) > 2:
            return None
        return {"action": "grasp", "item": "top", "grasp": None, "risk": 0.9,
                "reason": "no safe grasp", "stall": True}

    run = run_scene(STACKED, "waiter", waits_then_stops, tmp_path, video=False)
    assert run["summary"]["stalls"] == 2 and run["summary"]["moves"] == 0
    assert all(s["stall"] and s["outcome"] is None for s in run["steps"])


def test_run_batch_runs_every_scene_with_every_policy_and_writes_mp4s(tmp_path):
    scenes = tmp_path / "scenes" / "flat_stacked"
    scenes.mkdir(parents=True)
    (scenes / "000.json").write_text(json.dumps(STACKED))
    out = tmp_path / "runs"
    index = run_batch(tmp_path / "scenes", {"grab_only": grab_only, "copy": grab_only}, out,
                      video=True, width=160, height=120)
    assert len(index) == 2
    for policy in ("grab_only", "copy"):
        run = json.loads((out / f"flat_stacked_t01_{policy}.json").read_text())
        assert run["video"] == f"runs/flat_stacked_t01_{policy}.mp4"
        assert (out / f"flat_stacked_t01_{policy}.mp4").stat().st_size > 1000
    assert json.loads((out / "index.json").read_text()) == index
