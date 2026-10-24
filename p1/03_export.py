"""Train the failure classifier, export atlas.json and thumbnails into site/.

Usage:  python 03_export.py
Needs:  cache/ from steps 01-02.
Writes: site/atlas.json, site/img/*.jpg, results.md (numbers for the write-up)

atlas.json follows the team contract (clusters: id, name, defect, count, examples;
points: image, x, y, cluster) and adds a few extra keys the Atlas page uses:
share, simulated, detection_rate, type_accuracy, n_test, label (per point), classifier.
"""
import json
import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from modes import NAMES, SIMULATED

THUMB = 200


def thumb(src, dst):
    if os.path.exists(dst):
        return
    try:
        im = Image.open(src).convert("RGB")
        im.thumbnail((THUMB, THUMB))
        im.save(dst, quality=80)
    except Exception:
        Image.new("RGB", (THUMB, THUMB), (200, 200, 200)).save(dst)


REQUIRED_SIMULATED = 1
REQUIRED_SCENE_MODES = 3


def check_simulated_count(clusters):
    """Exactly one cluster (multi-pick) is simulated, and it carries exactly three scene modes."""
    sim = [c for c in clusters if c.get("simulated")]
    if len(sim) != REQUIRED_SIMULATED:
        raise SystemExit(
            f"{len(sim)} cluster(s) marked simulated, but exactly {REQUIRED_SIMULATED} "
            "is required (multi_pick). Fix SIMULATED in modes.py."
        )
    n = len(sim[0].get("scene_modes", []))
    if n != REQUIRED_SCENE_MODES:
        raise SystemExit(
            f"The simulated cluster has {n} scene mode(s), but exactly {REQUIRED_SCENE_MODES} "
            "are required. Fix SIMULATED in modes.py."
        )
    return len(sim)


def main():
    df = pd.read_csv("cache/meta.csv")
    E = np.load("cache/embeddings.npy")
    cl = pd.read_csv("cache/clusters.csv")
    y = df.label.values

    # ---- classifier on frozen embeddings (real-data results) ----
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=0)
    clf = LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced").fit(E[tr], y[tr])
    pred = clf.predict(E[te])
    classes = sorted(set(y))
    acc3 = float((pred == y[te]).mean())
    acc_bin = float(((y[te] != "clean") == (pred != "clean")).mean())
    recall = {c: float((pred[y[te] == c] == c).mean()) for c in classes}
    cm = confusion_matrix(y[te], pred, labels=classes).tolist()

    # per cluster, on held-out failures only
    row2cl = dict(zip(cl.row, cl.cluster))
    te_fail = [(i, p) for i, p in zip(te, pred) if i in row2cl]
    per = {}
    for c in sorted(cl.cluster.unique()):
        items = [(i, p) for i, p in te_fail if row2cl[i] == c]
        if items:
            det = np.mean([p != "clean" for _, p in items])
            typ = np.mean([p == y[i] for i, p in items])
        else:
            det = typ = None
        per[c] = {"n_test": len(items),
                  "detection_rate": None if det is None else round(float(det), 3),
                  "type_accuracy": None if typ is None else round(float(typ), 3)}

    # ---- thumbnails + atlas.json ----
    os.makedirs("site/img", exist_ok=True)
    points, clusters = [], []
    for _, r in cl.iterrows():
        rel = f"img/{int(r.row)}.jpg"
        thumb(r.image, os.path.join("site", rel))
        cid = r.cluster
        points.append({"image": rel, "x": float(r.x), "y": float(r.y),
                       "cluster": cid, "label": r.label})

    total = len(cl)
    for c in sorted(cl.cluster.unique()):
        sub = cl[cl.cluster == c]
        ex = sub.sample(min(10, len(sub)), random_state=0)
        for _, r in ex.iterrows():
            thumb(r.image, os.path.join("site", f"img/{int(r.row)}.jpg"))
        entry = {
            "id": c, "name": NAMES[c],
            "defect": sub.label.value_counts().idxmax(),
            "count": int(len(sub)), "share": round(len(sub) / total, 4),
            "simulated": c in SIMULATED,
            "examples": [f"img/{int(r)}.jpg" for r in ex.row],
            **per[c],
        }
        if c in SIMULATED:
            entry["scene_modes"] = SIMULATED[c]
        clusters.append(entry)
    clusters.sort(key=lambda c: -c["count"])
    check_simulated_count(clusters)

    classifier = {"model": "CLIP ViT-B/32 embeddings + logistic regression",
                  "train": int(len(tr)), "test": int(len(te)),
                  "accuracy_3class": round(acc3, 3), "accuracy_failure_vs_clean": round(acc_bin, 3),
                  "recall": {k: round(v, 3) for k, v in recall.items()},
                  "classes": classes, "confusion": cm}
    # Write to a temp file, re-read it to prove it is valid JSON, then swap it in.
    tmp = "site/atlas.json.tmp"
    with open(tmp, "w") as fh:
        json.dump({"clusters": clusters, "points": points, "classifier": classifier}, fh, allow_nan=False)
    json.load(open(tmp))
    os.replace(tmp, "site/atlas.json")

    # ---- results.md for the write-up ----
    lines = ["# ClearPick real-data results (ARMBench Image defect)", "",
             f"- Train {len(tr)}, held-out test {len(te)}",
             f"- 3-class accuracy: {acc3:.1%}",
             f"- Failure vs clean accuracy: {acc_bin:.1%}"]
    lines += [f"- Recall {k}: {v:.1%}" for k, v in recall.items()]
    lines += ["", "| Cluster | Count | Share | Simulated | Detected (held-out) |", "|---|---|---|---|---|"]
    for c in clusters:
        d = "n/a" if c["detection_rate"] is None else f"{c['detection_rate']:.0%} of {c['n_test']}"
        lines.append(f"| {c['name']} | {c['count']} | {c['share']:.0%} | {'yes' if c['simulated'] else 'no'} | {d} |")
    open("results.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nWrote site/atlas.json, site/img/, results.md")


if __name__ == "__main__":
    main()
