"""Cluster the failures, make contact sheets to look at, suggest names.

Usage:  python 02_cluster.py --k 8
Writes: cache/clusters.csv, review/cluster_<n>.jpg, names.json (template, only if missing)

Then: open every review/cluster_<n>.jpg, and edit names.json:
  id        short snake_case id (becomes the mode id for P3, e.g. flat_stacked)
  name      readable name ("Flat items stacked")
  simulate  true for the 3 physical modes you hand to P3, false for everything else
Re-run with a different --k if clusters look mixed (try 6 or 10).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans

TILE = 220


def sheet(paths, out, cols=5):
    rows = (len(paths) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * TILE, rows * TILE), (255, 255, 255))
    for i, p in enumerate(paths):
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((TILE - 6, TILE - 6))
            canvas.paste(im, ((i % cols) * TILE + 3, (i // cols) * TILE + 3))
        except Exception:
            pass
    canvas.save(out, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    df = pd.read_csv("cache/meta.csv")
    E = np.load("cache/embeddings.npy")
    fail = (df.label != "clean").values
    Ef = E[fail]
    dff = df[fail].copy()
    dff["row"] = np.where(fail)[0]
    dff = dff.reset_index(drop=True)
    print(f"Clustering {len(dff)} failures into {args.k} clusters")

    km = KMeans(n_clusters=args.k, n_init=10, random_state=0).fit(Ef)
    try:
        import umap

        xy = umap.UMAP(random_state=0, n_neighbors=30, min_dist=0.1).fit_transform(Ef)
    except ImportError:
        from sklearn.decomposition import PCA

        print("umap-learn not installed, using PCA for the 2D map")
        xy = PCA(n_components=2, random_state=0).fit_transform(Ef)
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0) + 1e-9)

    dff["cluster"] = km.labels_
    dff["x"], dff["y"] = xy[:, 0].round(4), xy[:, 1].round(4)
    dist = ((Ef - km.cluster_centers_[km.labels_]) ** 2).sum(1)
    dff["dist"] = dist
    dff.to_csv("cache/clusters.csv", index=False)

    T = np.load("cache/text_embeddings.npy")
    phrases = json.load(open("cache/phrases.json"))
    os.makedirs("review", exist_ok=True)
    template = {}
    rng = np.random.default_rng(0)
    for c in range(args.k):
        sub = dff[dff.cluster == c]
        cen = km.cluster_centers_[c] / np.linalg.norm(km.cluster_centers_[c])
        top = np.argsort(-(T @ cen))[:3]
        closest = sub.nsmallest(10, "dist").image.tolist()
        rest = sub.drop(sub.nsmallest(10, "dist").index)
        rand = rest.image.iloc[rng.permutation(len(rest))[:5]].tolist() if len(rest) else []
        sheet(closest + rand, f"review/cluster_{c}.jpg")
        mix = sub.label.value_counts().to_dict()
        print(f"\nCluster {c}: {len(sub)} images  ({len(sub) / len(dff):.0%})  labels={mix}")
        for j in top:
            print(f"   suggests: {phrases[j]}")
        template[str(c)] = {"id": f"cluster_{c}", "name": phrases[top[0]], "simulate": False}

    if not os.path.exists("names.json"):
        json.dump(template, open("names.json", "w"), indent=2)
        print("\nWrote names.json template. Edit it after looking at review/*.jpg")
    else:
        print("\nnames.json already exists; not overwritten (delete it if you changed --k)")


if __name__ == "__main__":
    main()
