"""Group the failures by ARMBench's real failure labels and build the 2D map.

Usage:  python3 02_cluster.py
Needs:  cache/embeddings.npy and cache/meta.csv from 01_embed.py
Writes: cache/clusters.csv, review/<mode>.jpg contact sheets

(Earlier we tried k-means on CLIP embeddings, but those clusters grouped by camera
station, not by failure type. Amazon's own sublabels are the real failure modes.)
"""
import os

import numpy as np
import pandas as pd
from PIL import Image

from modes import NAMES, mode_for_image

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
    df = pd.read_csv("cache/meta.csv")
    E = np.load("cache/embeddings.npy")
    fail = (df.label != "clean").values
    Ef = E[fail]
    dff = df[fail].copy()
    dff["row"] = np.where(fail)[0]
    dff = dff.reset_index(drop=True)
    dff["cluster"] = [mode_for_image(p, l) for p, l in zip(dff.image, dff.label)]
    print(f"{len(dff)} failures")

    try:
        import umap

        xy = umap.UMAP(random_state=0, n_neighbors=30, min_dist=0.1).fit_transform(Ef)
    except ImportError:
        from sklearn.decomposition import PCA

        print("umap-learn not installed, using PCA for the 2D map")
        xy = PCA(n_components=2, random_state=0).fit_transform(Ef)
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0) + 1e-9)
    dff["x"], dff["y"] = xy[:, 0].round(4), xy[:, 1].round(4)
    dff.to_csv("cache/clusters.csv", index=False)

    os.makedirs("review", exist_ok=True)
    print("\nFailure modes in the sample:")
    for mode, sub in dff.groupby("cluster"):
        print(f"  {mode:18s} {len(sub):5d}  ({len(sub) / len(dff):.0%})  {NAMES[mode]}")
        sheet(sub.sample(min(15, len(sub)), random_state=0).image.tolist(), f"review/{mode}.jpg")
    print("\nWrote cache/clusters.csv and review/<mode>.jpg")


if __name__ == "__main__":
    main()
