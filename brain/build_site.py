"""Bundle run results into the Robot page's data file.

    python -m sim.run_batch ...            # produces runs/*.json (and .mp4 for scenes you record)
    python -m brain.build_site             # writes site/data.js and copies videos to site/videos/
    open site/robot.html                   # works straight from disk, no server needed

site/robot.html is hand-written and committed; this script only refreshes site/data.js and site/videos/.
"""
import json
import re
import shutil
import sys
from pathlib import Path

from .evaluate import build as build_scoreboard

MODE_ORDER = ["flat_stacked", "small_beside_big", "against_wall"]


def _scene_files(scenes_dir):
    out = {}
    for p in Path(scenes_dir).rglob("*.json"):
        try:
            sc = json.loads(p.read_text())
        except ValueError:
            continue
        if isinstance(sc, dict) and "scene_id" in sc and "objects" in sc:
            out[sc["scene_id"]] = sc
    return out


def build(runs_dir="runs", scenes_dir="scenes", site_dir="site"):
    site = Path(site_dir)
    (site / "videos").mkdir(parents=True, exist_ok=True)
    scene_files = _scene_files(scenes_dir)
    scenes, videos = {}, 0
    for p in sorted(Path(runs_dir).glob("*.json")):
        try:
            r = json.loads(p.read_text())
        except ValueError:
            continue
        if not isinstance(r, dict) or "steps" not in r or "policy" not in r or "scene_id" not in r:
            continue  # e.g. runs/index.json
        sid, pol = r["scene_id"], r["policy"]
        if not re.fullmatch(r"[a-z_]+_\d{3}", sid) or sid.endswith("_000"):
            continue  # skip mock and test scenes such as flat_stacked_000 and flat_stacked_t01
        base = scene_files.get(sid, {})
        entry = scenes.setdefault(sid, {"mode": r.get("mode") or base.get("mode", ""),
                                        "tote": base.get("tote", {"size": [0.4, 0.3, 0.15]}),
                                        "objects": [{k: o[k] for k in ("id", "type", "size", "pos") if k in o}
                                                    for o in base.get("objects", [])],
                                        "runs": {}})
        mp4 = Path(runs_dir) / f"{sid}_{pol}.mp4"
        video = None
        if mp4.exists():
            shutil.copy2(mp4, site / "videos" / mp4.name)
            video = f"videos/{mp4.name}"
            videos += 1
        entry["runs"][pol] = {"steps": r["steps"], "summary": r.get("summary", {}), "video": video}
    modes = {}
    for sid, e in scenes.items():
        modes.setdefault(e["mode"], []).append(sid)
    ordered = {m: sorted(modes[m]) for m in sorted(modes, key=lambda m: (MODE_ORDER.index(m) if m in MODE_ORDER else 99, m))}
    data = {"scoreboard": build_scoreboard(runs_dir), "modes": ordered, "scenes": scenes}
    (site / "data.js").write_text("window.CLEARPICK = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"site/data.js: {len(scenes)} scenes in {len(ordered)} modes, {videos} videos copied")
    return data


if __name__ == "__main__":
    build(*sys.argv[1:4])
