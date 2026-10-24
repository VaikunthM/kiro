"""Train the failure classifier, export atlas.json and thumbnails into site/.

Usage:  python 03_export.py
Needs:  cache/ from steps 01-02, and names.json edited by you.
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


REQUIRED_SIMULATED = 3


def check_simulated_count(names, required=REQUIRED_SIMULATED):
    """Ensure exactly `required` clusters are marked simulate=true in names.json."""
    n = sum(1 for v in names.values() if v.get("simulate", False))
    if n != required:
        raise SystemExit(
            f"names.json marks {n} cluster(s) with simulate=true, "
            f"but exactly {required} are required. Fix names.json."
        )
    return n


def main():
    df = pd.read_csv("cache/meta.csv")
    E = np.load("cache/embeddings.npy")
    cl = pd.read_csv("cache/clusters.csv")
    names = json.load(open("names.json"))
    check_simulated_count(names)
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
        cid = names[str(int(r.cluster))]["id"]
        points.append({"image": rel, "x": float(r.x), "y": float(r.y),
                       "cluster": cid, "label": r.label})

    total = len(cl)
    for c in sorted(cl.cluster.unique()):
        n = names[str(int(c))]
        sub = cl[cl.cluster == c]
        clusters.append({
            "id": n["id"], "name": n["name"],
            "defect": sub.label.value_counts().idxmax(),
            "count": int(len(sub)), "share": round(len(sub) / total, 4),
            "simulated": bool(n.get("simulate", False)),
            "examples": [f"img/{int(r)}.jpg" for r in sub.nsmallest(10, "dist").row],
            **per[c],
        })
    clusters.sort(key=lambda c: -c["count"])

    classifier = {"model": "CLIP ViT-B/32 embeddings + logistic regression",
                  "train": int(len(tr)), "test": int(len(te)),
                  "accuracy_3class": round(acc3, 3), "accuracy_failure_vs_clean": round(acc_bin, 3),
                  "recall": {k: round(v, 3) for k, v in recall.items()},
                  "classes": classes, "confusion": cm}
    json.dump({"clusters": clusters, "points": points, "classifier": classifier},
              open("site/atlas.json", "w"))

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
