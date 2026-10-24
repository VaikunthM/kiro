# ClearPick shared contracts (steering doc)

Everyone builds against these. Change them only by telling the whole team.

Units: metres, seconds. Tote is centred at the origin, floor at z = 0.
Default tote: `size = [0.40, 0.30, 0.15]` (x, y, wall height). Object `pos` is the object's CENTRE.
Boxes are treated as axis-aligned for scoring (yaw is stored for rendering only).

## 1. atlas.json (P1 writes)
```json
{"clusters": [{"id": "flat_stacked", "name": "Flat items stacked", "defect": "multi_pick", "count": 0, "examples": ["img/a.jpg"]}],
 "points":   [{"image": "img/a.jpg", "x": 0.1, "y": 0.4, "cluster": "flat_stacked"}]}
```

## 2. scenes/<mode>/<n>.json (P3 writes, P2 loads)
```json
{"scene_id": "flat_stacked_007", "mode": "flat_stacked", "seed": 7,
 "tote": {"size": [0.40, 0.30, 0.15]},
 "objects": [{"id": "box_1", "type": "flat_box", "size": [0.14, 0.10, 0.015],
              "pos": [0.02, -0.03, 0.0075], "yaw": 0.2}]}
```
Modes (default, until P1 changes them): `flat_stacked`, `small_beside_big`, `against_wall`.
`type` is a YCB-style name P2 maps to a mesh or box. Sizes are full extents (x, y, z).

## 3. Simulator API (P2 writes, P3 calls)
```
sim.load(scene)                 # scene dict from contract 2
sim.state() -> {"scene_id", "mode", "tote": {"size": [...]},
                "objects": [{"id", "type", "size": [x,y,z], "pos": [x,y,z],
                             "bbox": {"min": [x,y,z], "max": [x,y,z]}}]}   # only objects still in the tote
sim.grasp(item_id, grasp_type) -> "clean" | "double" | "miss"
        grasp_type in {"suction_center", "suction_edge", "side_pinch"}
sim.push(item_id, direction)    -> "ok"   # direction = [dx, dy] unit vector, world frame
sim.record(path)                # start recording a run to path (.mp4); sim.record(None) stops and saves. No-op in the mock
sim.clock() -> float            # optional: simulated seconds so far
```
"clean" removes exactly one object from the tote. "double" removes two or more
(the item plus whatever came with it). "miss" removes none.

## 4. runs/<scene_id>_<policy>.json (P3 writes, P2 adds the .mp4)
```json
{"scene_id": "flat_stacked_007", "mode": "flat_stacked", "policy": "clearpick",
 "steps": [{"t": 1.2, "action": "push", "item": "box_2", "grasp": null, "direction": [1, 0],
            "risk": 0.72, "reason": "two flat boxes stacked", "outcome": "ok"},
           {"t": 3.8, "action": "grasp", "item": "box_2", "grasp": "suction_center",
            "risk": 0.18, "reason": "clear on all sides", "outcome": "clean"}],
 "summary": {"clean": 0, "double_grabs": 0, "stalls": 0, "moves": 0, "misses": 0, "items": 0},
 "video": "runs/flat_stacked_007_clearpick.mp4"}
```
Additive changes to the original contract (nothing renamed): push steps carry `direction`;
steps may carry `stall: true` (robot gave up waiting for a safe grasp); summary adds `misses` and `items`.
Policies: `grab_only`, `clearpick`.

## 5. score_grasp(state, item_id, grasp_type) -> risk in [0, 1]
Pure function of `state()` geometry. Higher means more likely to double-grab or fail.
The ARMBench image model never sees simulator renders; the weights are hand-set from P1's failure clusters.
