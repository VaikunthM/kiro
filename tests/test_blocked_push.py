"""A blocked push is not a dead end: the runner keeps going so the policy can grasp next.
Only misses (and policy stalls) end a run early. Regression for flat_stacked_001, where three
blocked pushes in a row ended ClearPick's run with five items still in the tote."""
import sim.run_batch as rb

SCENE = {"scene_id": "blocked_t", "mode": "test",
         "objects": [{"id": f"box_{i}", "type": "flat_box", "size": [0.1, 0.08, 0.02], "pos": [0, 0, 0.01]}
                     for i in range(3)]}


class FakeSim:
    """Every push is blocked; grasps return whatever the test sets and remove the item when clean."""
    grasp_outcome = "clean"

    def __init__(self, **kwargs):
        self.items, self.t, self.last_push_blocked = [], 0.0, False

    def load(self, scene):
        self.items = [o["id"] for o in scene["objects"]]
        return self.state()

    def state(self):
        return {"scene_id": "blocked_t", "objects": [{"id": i} for i in self.items]}

    def clock(self):
        return self.t

    def record(self, path):
        return 0

    def push(self, item, direction):
        self.t += 1
        self.last_push_blocked = True
        return "ok"

    def grasp(self, item, grasp_type):
        self.t += 1
        if self.grasp_outcome == "clean":
            self.items.remove(item)
        return self.grasp_outcome


def pushes_three_times_then_grabs():
    count = {"n": 0}

    def decide(state):
        count["n"] += 1
        item = state["objects"][0]["id"]
        if count["n"] <= 3:
            return {"action": "push", "item": item, "direction": [0, 1]}
        return {"action": "grasp", "item": item, "grasp": "suction_center"}
    return decide


def test_three_blocked_pushes_do_not_end_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(rb, "Sim", FakeSim)
    FakeSim.grasp_outcome = "clean"
    run = rb.run_scene(SCENE, "clearpick", pushes_three_times_then_grabs(), tmp_path, video=False)
    assert run["summary"]["clean"] == 3                       # the tote was emptied
    assert [s["action"] for s in run["steps"]][:4] == ["push"] * 3 + ["grasp"]


def test_three_misses_in_a_row_still_end_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(rb, "Sim", FakeSim)
    FakeSim.grasp_outcome = "miss"
    always_grab = lambda s: {"action": "grasp", "item": s["objects"][0]["id"], "grasp": "suction_center"}
    run = rb.run_scene(SCENE, "grab_only", always_grab, tmp_path, video=False)
    assert run["summary"]["misses"] == rb.MAX_STALLS_IN_A_ROW
    assert len(run["steps"]) == rb.MAX_STALLS_IN_A_ROW
