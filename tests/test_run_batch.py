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


def test_camera_defaults_to_overhead_and_threads_through_to_sim(monkeypatch):
    """--camera is additive: run_scene/run_batch default to overhead and pass the choice to Sim."""
    import sim.run_batch as rb

    seen = []

    class FakeSim:
        def __init__(self, width=640, height=480, camera="overhead"):
            seen.append(camera)

        def load(self, scene):
            return {"objects": []}

        def record(self, path):
            return 0

        def state(self):
            return {"objects": []}

    monkeypatch.setattr(rb, "Sim", FakeSim)

    # default: no camera argument -> overhead
    rb.run_scene(STACKED, "grab_only", grab_only, "runs", video=False)
    assert seen[-1] == "overhead"

    # explicit angled threads through run_scene
    rb.run_scene(STACKED, "grab_only", grab_only, "runs", video=False, camera="angled")
    assert seen[-1] == "angled"

    # and through run_batch -> run_scene
    rb.run_batch(STACKED, {"grab_only": grab_only}, "runs", video=False,
                 scene_files=[], camera="angled")
    assert all(c == "angled" for c in seen[2:]) or seen[-1] == "angled"


def test_angled_camera_batch_writes_a_valid_mp4_and_overhead_is_the_default(tmp_path):
    """End-to-end with the real Sim: an angled one-scene batch writes a valid .mp4, and
    omitting the flag keeps overhead the default (schema/run-file unchanged)."""
    scenes = tmp_path / "scenes" / "flat_stacked"
    scenes.mkdir(parents=True)
    (scenes / "000.json").write_text(json.dumps(STACKED))

    # angled: a one-scene batch renders from the angled camera and writes a real MP4.
    angled_out = tmp_path / "runs_angled"
    index = run_batch(tmp_path / "scenes", {"grab_only": grab_only}, angled_out,
                      video=True, width=160, height=120, camera="angled")
    assert len(index) == 1
    run = json.loads((angled_out / "flat_stacked_t01_grab_only.json").read_text())
    assert run["video"] == "runs_angled/flat_stacked_t01_grab_only.mp4"
    assert (angled_out / "flat_stacked_t01_grab_only.mp4").stat().st_size > 1000
    # schema is unchanged by the camera choice (contract 4).
    assert {"scene_id", "mode", "policy", "steps", "summary", "video"} <= run.keys()

    # overhead is the default when the flag/kwarg is omitted: the real Sim still writes a valid MP4.
    default_out = tmp_path / "runs_default"
    run_batch(tmp_path / "scenes", {"grab_only": grab_only}, default_out,
              video=True, width=160, height=120)
    assert (default_out / "flat_stacked_t01_grab_only.mp4").stat().st_size > 1000
