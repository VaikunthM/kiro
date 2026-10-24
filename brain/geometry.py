"""Turn whatever Sim.state() returns into one simple shape the scorer and robots use.

Accepts objects as a list or a dict keyed by id, and bboxes as {"bbox": {"min","max"}} or bbox_min/bbox_max.
Output: {"_norm": True, "scene_id", "mode", "tote": {"size": [x, y, z]},
         "objects": [{"id", "pos", "size", "lo", "hi"}]}   lo/hi = axis-aligned bbox corners.
"""


def normalize(state):
    if state.get("_norm"):
        return state
    raw = state["objects"]
    items = [dict(v, id=k) for k, v in raw.items()] if isinstance(raw, dict) else [dict(o) for o in raw]
    out = []
    for o in items:
        if "bbox" in o:
            lo, hi = o["bbox"]["min"], o["bbox"]["max"]
        elif "bbox_min" in o:
            lo, hi = o["bbox_min"], o["bbox_max"]
        else:
            lo = [o["pos"][i] - o["size"][i] / 2 for i in range(3)]
            hi = [o["pos"][i] + o["size"][i] / 2 for i in range(3)]
        lo, hi = [float(x) for x in lo], [float(x) for x in hi]
        pos = [float(x) for x in o["pos"]] if "pos" in o else [(lo[i] + hi[i]) / 2 for i in range(3)]
        out.append({"id": o["id"], "pos": pos, "size": [hi[i] - lo[i] for i in range(3)], "lo": lo, "hi": hi})
    scene_id = state.get("scene_id", "")
    mode = state.get("mode") or (scene_id.rsplit("_", 1)[0] if "_" in scene_id else "")
    return {"_norm": True, "scene_id": scene_id, "mode": mode,
            "tote": {"size": list(state.get("tote", {}).get("size", [0.40, 0.30, 0.15]))}, "objects": out}
