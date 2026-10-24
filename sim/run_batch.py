"""Run every scene with every policy; write runs/<scene_id>_<policy>.json and .mp4 plus runs/index.json.

Usage (from the repo root):
    python -m sim.run_batch                                   # all scenes, grab_only + clearpick
    python -m sim.run_batch --mode flat_stacked --limit 5     # a quick subset
    python -m sim.run_batch --scene scenes/mock/flat_stacked_000.json
    python -m sim.run_batch --no-video                        # fast, JSON only

Policies are loaded from policies/<name>.py (P3), each exposing decide(state).
grab_only falls back to sim.reference_policies.grab_only until P3's version exists.
"""
import argparse
import importlib
import json
from pathlib import Path

from sim import reference_policies
from sim.sim import Sim

MAX_STALLS_IN_A_ROW = 3


def run_scene(scene, policy_name, decide, out_dir, video=True, width=640, height=480):
    """Run one policy on one scene until the tote is empty, the policy stops, or it stalls out."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sim = Sim(width=width, height=height)
    state = sim.load(scene)
    stem = f"{scene['scene_id']}_{policy_name}"
    if video:
        sim.record(out_dir / f"{stem}.mp4")
    max_steps = 3 * len(scene["objects"])
    steps, no_progress = [], 0
    summary = {"clean": 0, "double_grabs": 0, "stalls": 0, "moves": 0, "misses": 0,
               "items": len(scene["objects"])}

    while state["objects"] and len(steps) < max_steps and no_progress < MAX_STALLS_IN_A_ROW:
        step = decide(state)
        if step is None:
            break
        record = {"t": sim.clock(), "action": step.get("action"), "item": step.get("item"),
                  "grasp": step.get("grasp"), "direction": step.get("direction"),
                  "risk": step.get("risk"), "reason": step.get("reason", ""), "outcome": None}
        if step.get("stall"):  # the policy gave up waiting for a safe grasp: nothing is executed
            record["stall"] = True
            summary["stalls"] += 1
            no_progress += 1
            sim.step(0.5)
        else:
            if step["action"] == "grasp":
                record["outcome"] = sim.grasp(step["item"], step["grasp"])
            elif step["action"] == "push":
                record["outcome"] = sim.push(step["item"], step["direction"])
            else:
                raise ValueError(f"unknown action {step['action']!r} from policy {policy_name}")
            summary["moves"] += 1
            outcome = record["outcome"]
            summary["clean"] += outcome == "clean"
            summary["double_grabs"] += outcome == "double"
            summary["misses"] += outcome == "miss"
            stuck = outcome == "miss" or (step["action"] == "push" and sim.last_push_blocked)
            no_progress = no_progress + 1 if stuck else 0
        steps.append(record)
        state = sim.state()

    video_path = None
    if video:
        sim.record(None)
        video_path = f"{out_dir.name}/{stem}.mp4"
    run = {"scene_id": scene["scene_id"], "mode": scene.get("mode"), "policy": policy_name,
           "steps": steps, "summary": summary, "video": video_path}
    (out_dir / f"{stem}.json").write_text(json.dumps(run, indent=1))
    return run


def run_batch(scenes_root, policies, out_dir, video=True, width=640, height=480, mode=None, limit=None,
              scene_files=None):
    """policies: {name: decide_fn}. Returns and writes the index of all runs."""
    out_dir = Path(out_dir)
    if scene_files is None:
        pattern = f"{mode}/*.json" if mode else "*/*.json"
        scene_files = sorted(Path(scenes_root).glob(pattern))
        if limit:
            by_mode = {}
            for f in scene_files:
                by_mode.setdefault(f.parent.name, []).append(f)
            scene_files = [f for fs in by_mode.values() for f in fs[:limit]]
    index = []
    for f in scene_files:
        scene = json.loads(Path(f).read_text())
        for name, decide in policies.items():
            run = run_scene(scene, name, decide, out_dir, video=video, width=width, height=height)
            index.append({"scene_id": run["scene_id"], "mode": run["mode"], "policy": name,
                          "json": f"{out_dir.name}/{run['scene_id']}_{name}.json",
                          "video": run["video"], "summary": run["summary"]})
            s = run["summary"]
            print(f"{run['scene_id']:28s} {name:10s} clean={s['clean']} double={s['double_grabs']} "
                  f"stalls={s['stalls']} moves={s['moves']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    return index


def load_policy(name, module_root="policies"):
    try:
        return importlib.import_module(f"{module_root}.{name}").decide
    except ModuleNotFoundError as e:
        if e.name in (module_root, f"{module_root}.{name}") and hasattr(reference_policies, name):
            print(f"[run_batch] {module_root}/{name}.py not found, using sim.reference_policies.{name}")
            return getattr(reference_policies, name)
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", default="scenes")
    ap.add_argument("--scene", help="run a single scene JSON")
    ap.add_argument("--mode", help="only scenes/<mode>/")
    ap.add_argument("--limit", type=int, help="max scenes per mode")
    ap.add_argument("--policies", nargs="+", default=["grab_only", "clearpick"])
    ap.add_argument("--out", default="runs")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--size", nargs=2, type=int, default=[640, 480], metavar=("W", "H"))
    a = ap.parse_args()
    pols = {}
    for p in a.policies:
        try:
            pols[p] = load_policy(p)
        except ModuleNotFoundError:
            print(f"[run_batch] skipping {p}: policies/{p}.py not found yet")
    run_batch(a.scenes, pols, a.out, video=not a.no_video, width=a.size[0], height=a.size[1],
              mode=a.mode, limit=a.limit, scene_files=[a.scene] if a.scene else None)
