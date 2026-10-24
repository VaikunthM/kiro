"""Check p1/site/atlas.json against the team contract. Exit code 1 if anything is wrong.

Run from the repo root:  python3 p1/validate_atlas.py
Contract (from the steering doc): clusters[] has id, name, defect, count, examples;
points[] has image, x, y, cluster.
"""
import json
import os
import sys

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site", "atlas.json")
CLUSTER_KEYS = {"id", "name", "defect", "count", "examples"}
POINT_KEYS = {"image", "x", "y", "cluster"}


def problems(a):
    out = []
    if not isinstance(a.get("clusters"), list) or not isinstance(a.get("points"), list):
        return ["atlas.json needs 'clusters' and 'points' lists"]
    ids = [c.get("id") for c in a["clusters"]]
    if len(ids) != len(set(ids)):
        out.append("duplicate cluster ids")
    for c in a["clusters"]:
        miss = CLUSTER_KEYS - set(c)
        if miss:
            out.append(f"cluster {c.get('id')}: missing {sorted(miss)}")
    for i, p in enumerate(a["points"]):
        miss = POINT_KEYS - set(p)
        if miss:
            out.append(f"point {i}: missing {sorted(miss)}")
            continue
        if p["cluster"] not in ids:
            out.append(f"point {i}: unknown cluster {p['cluster']}")
        if not (0 <= p["x"] <= 1 and 0 <= p["y"] <= 1):
            out.append(f"point {i}: x/y outside [0, 1]")
    if sum(c.get("count", 0) for c in a["clusters"]) != len(a["points"]):
        out.append("cluster counts do not add up to the number of points")
    return out[:20]


def main():
    try:
        a = json.load(open(PATH))
    except Exception as e:
        print(f"atlas.json is not valid JSON: {e}")
        sys.exit(1)
    bad = problems(a)
    if bad:
        print("atlas.json breaks the contract:")
        for b in bad:
            print("  -", b)
        sys.exit(1)
    print(f"atlas.json OK: {len(a['clusters'])} clusters, {len(a['points'])} points")


if __name__ == "__main__":
    main()
