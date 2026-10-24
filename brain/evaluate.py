"""Per-mode scoreboard from runs/*.json -> scoreboard.json.   python -m brain.evaluate [runs_dir] [out]"""
import json
import sys
from pathlib import Path


def build(runs_dir="runs"):
    board = {}
    for p in sorted(Path(runs_dir).glob("*.json")):
        r = json.loads(p.read_text())
        if not isinstance(r, dict) or "summary" not in r or "policy" not in r:
            continue  # e.g. runs/index.json
        s = r["summary"]
        c = board.setdefault(r["mode"], {}).setdefault(
            r["policy"], {"scenes": 0, "clean": 0, "double_grabs": 0, "misses": 0, "stalls": 0, "pushes": 0, "items": 0})
        c["scenes"] += 1
        c["clean"] += s.get("clean", 0)
        c["double_grabs"] += s.get("double_grabs", 0)
        c["misses"] += s.get("misses", 0)
        c["stalls"] += s.get("stalls", 0)
        c["items"] += s.get("items", 0)
        c["pushes"] += sum(1 for st in r.get("steps", []) if st.get("action") == "push")
    for mode in board.values():
        for c in mode.values():
            picks = c["clean"] + c["double_grabs"] + c["misses"]
            c["double_grab_rate"] = round(c["double_grabs"] / picks, 3) if picks else 0.0
            c["clean_rate"] = round(c["clean"] / picks, 3) if picks else 0.0
    return board


def main(runs_dir="runs", out="scoreboard.json"):
    board = build(runs_dir)
    Path(out).write_text(json.dumps(board, indent=1))
    print(f"{'mode':18}{'policy':11}{'scenes':>7}{'clean':>7}{'double':>8}{'miss':>6}{'push':>6}{'dbl rate':>10}")
    for mode, pols in board.items():
        for pol, c in pols.items():
            print(f"{mode:18}{pol:11}{c['scenes']:>7}{c['clean']:>7}{c['double_grabs']:>8}{c['misses']:>6}"
                  f"{c['pushes']:>6}{c['double_grab_rate']:>10.1%}")
    return board


if __name__ == "__main__":
    main(*sys.argv[1:3])
