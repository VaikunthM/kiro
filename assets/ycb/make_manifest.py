"""Compute how to fit each YCB visual mesh into the box the simulator collides with.

    python assets/ycb/make_manifest.py        (from the repo root; rewrites assets/ycb/manifest.json)

For every type, the mesh's axes are matched to the scene size's axes by extent (largest to largest), then the
mesh is scaled per axis so its bounding box equals the primitive's full extents and recentred on the object's
origin. Collision stays the primitive box/cylinder: the mesh is visual-only (contype = conaffinity = 0).
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIZES = {  # full extents (x, y, z) used by the scene generator (brain/scene_gen.py SIZES)
    "sugar_box": (0.175, 0.089, 0.038),
    "tomato_soup_can": (0.066, 0.066, 0.101),
    "gelatin_box": (0.085, 0.073, 0.028),
    "potted_meat_can": (0.097, 0.082, 0.050),
    "foam_brick": (0.075, 0.050, 0.050),
}


def obj_vertices(path):
    vs = [list(map(float, line.split()[1:4])) for line in open(path) if line.startswith("v ")]
    return np.array(vs)


def mat_to_quat(R):
    """Rotation matrix -> unit quaternion (w, x, y, z)."""
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def fit(mesh_path, size):
    v = obj_vertices(mesh_path)
    lo, hi = v.min(0), v.max(0)
    ext, centre = hi - lo, (hi + lo) / 2
    size = np.array(size, float)
    mesh_order, scene_order = np.argsort(-ext, kind="stable"), np.argsort(-size, kind="stable")
    R = np.zeros((3, 3))                      # object = R @ mesh
    for m_ax, s_ax in zip(mesh_order, scene_order):
        R[s_ax, m_ax] = 1.0
    if np.linalg.det(R) < 0:                  # keep it a proper rotation (no mirrored labels)
        R[:, mesh_order[-1]] *= -1
    scale = np.array([size[np.argmax(np.abs(R[:, i]))] / ext[i] for i in range(3)])
    pos = -R @ (scale * centre)
    return {"scale": scale.round(5).tolist(), "quat": mat_to_quat(R).round(6).tolist(),
            "pos": pos.round(5).tolist(), "mesh_extent": ext.round(5).tolist()}


if __name__ == "__main__":
    manifest = {"source": "YCB Object and Model Set, Google 16k textured models (ycbbenchmarks.com), "
                          "CC BY 4.0; cleaned copies from github.com/elpis-lab/ycb_dataset (MIT)",
                "visual_only": True, "types": {}}
    for t, size in SIZES.items():
        manifest["types"][t] = {"mesh": f"{t}/textured.obj", "texture": f"{t}/texture_map.png",
                                "size": list(size), **fit(HERE / t / "textured.obj", size)}
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest["types"], indent=1))
