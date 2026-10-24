# Design — ClearPick Simulator (`sim/`)

## Overview

The simulator is a thin, deterministic layer over MuJoCo. A scene JSON is compiled to MJCF
(`build_mjcf`), loaded into a `Sim` instance, and driven through a small verb set (`grasp`, `push`) that
moves mocap "tools" (a suction cup, two pinch fingers, a pusher) and reads back which objects rose.
Outcomes are decided purely by measured vertical displacement, which keeps the model simple and
reproducible. This document describes the existing architecture, defines the correctness properties that
property-based tests (Hypothesis) should enforce, and sketches how the four remaining features fit in
without breaking the team contract.

The public surface is fixed by contract 3 in the shared steering doc and summarised in
`.kiro/steering/simulator.md`: `Sim.load`, `Sim.state`, `Sim.grasp`, `Sim.push`, `Sim.record`,
`Sim.clock`, `Sim.last_push_blocked`, the three grasp types, the outcome strings
(`clean`/`double`/`miss`, and push always `ok`), and the run-file schema (contract 4). The design below
adds behaviour behind that surface or through additive, optional parameters only.

## Architecture

```
scene JSON ──► build_mjcf() ──► MJCF XML ──► MjModel / MjData
                                                  │
                     ┌────────────────────────────┴───────────────────────────┐
                     │                         Sim                              │
                     │  load / step / _physics_step / state / clock             │
                     │  grasp ─► _plan_suction | _plan_pinch ─► _lift ─► _remove │
                     │  push  ─► pusher mocap motion (sets last_push_blocked)    │
                     │  record(path) start / record(None) stop+save             │
                     │  render (overhead | angled)                              │
                     └──────────────────────────────┬───────────────────────────┘
                                                    │ state() dict
          validate_scenes.check_scene ◄─────────────┤
          run_batch.run_scene / run_batch ◄──────────┘ (calls policy.decide(state))
```

Key components:
- **Scene-field helpers** — `obj_shape`, `obj_quat`, `obj_mass`, `camera_poses`. These turn the
  contract-2 scene fields (`type`, `yaw`, `mass`, `shape`, `quat`) into MJCF inputs. `CYLINDER_TYPES`
  and the `_can` suffix (minus `potted_meat_can`) decide cylinder vs box; `DENSITY` fills in mass when
  absent; `camera_poses` scales the `overhead` and `angled` cameras to frame the tote.
- **`build_mjcf(scene)`** — pure function from scene dict to MJCF text. Tote walls, ground, two cameras
  (`overhead`, `angled`), and the three mocap tools are fixed; objects become free-joint bodies. `_v`
  snaps denormal floats so MuJoCo's XML parser accepts them.
- **2D geometry helpers** — `_hull`, `_rect`, `_polys_overlap` (separating-axis test), `_point_in_poly`,
  `_dist_point_poly`, `_overlap_fraction`, `_area`. All grasp/push planning reduces to these footprint
  operations in the x-y plane plus a z-range check.
- **`Sim`** — owns model/data, mocap id lookups, per-object qpos/qvel addresses, the renderer, the
  recorded frame buffer, the recording path/flag, `last_push_blocked`, and the clock origin `_t0`.
  `_physics_step` optionally pins attached objects to a tool (`follow`) so a held group moves rigidly.
- **Outcome rule** — `grasp` snapshots each in-tote object's z, performs the move, then counts objects
  that rose more than `LIFT_CHECK`. 0 → `miss`, 1 → `clean`, 2+ → `double`. `push` always returns `ok`
  and records whether the item actually moved via `last_push_blocked`.
- **Simplified suction rules (agreed with P3, mirrored in `brain.score.score_grasp`)** — two snag/seal
  rules layered on the attach-on-touch model: (a) `suction_center` also lifts any side neighbour whose
  nearest edge is within `NEIGHBOUR_DRAG_GAP` (1 cm) of the target and that shares at least half the
  shorter item's height band, giving a `double`; (b) a suction cup whose centre is within
  `CUP_RADIUS + WALL_SEAL_MARGIN` (3.5 cm) of a tote wall cannot seal and the grasp is a `miss`, while
  `suction_edge` picks the side with the most clearance from both walls and neighbours.
- **Recording** — `record(path)` seeds the frame buffer and sets the recording flag; frames are captured
  every `_frame_every` steps; `record(None)` appends a short tail, writes the `.mp4`, and returns the
  frame count.
- **`validate_scenes`** and **`run_batch`** are consumers of the `Sim` API and the run-file contract.
  `run_scene` drives recording through `record(path)` / `record(None)` and reads `last_push_blocked` to
  decide no-progress.

## Correctness properties (for property-based testing)

These are the invariants a Hypothesis suite should check across randomised scenes. Each is phrased so a
generator can produce inputs and an assertion can check the output. Properties marked *(existing)* are
already covered in `tests/test_sim.py`; the rest are gaps worth filling as behaviour changes. Per the
steering rule, any requirement phrased "for any scene/object" should be a Hypothesis property, not a
single example. Build `Sim` with small frames (e.g. `64x48`) and do not start recording in property
tests, matching the existing `@pytest.mark.slow` tests.

**Grasp outcome properties**
- **P1 — Isolated box is always a clean suction pick.** For any single flat box at any in-bounds position
  and yaw, `grasp(id, "suction_center")` returns `clean`. *(existing)*
- **P2 — Two flat boxes stacked are always a double under centre suction.** For any flat base and a
  smaller flat box centred on top, centre suction on the top returns `double`. *(existing)*
- **P3 — Isolated box is always a clean pinch.** For any single box whose shorter horizontal side fits
  the jaw and is clear of walls, `side_pinch` returns `clean`.
- **P4 — Outcome is one of the three strings.** For any scene and any valid grasp type, the return value
  is in `{"clean", "double", "miss"}` and never raises except for the documented `ValueError`s.
- **P5 — Lift count agrees with removals.** The number of objects removed from `state()` by a grasp
  equals the integer implied by its outcome (clean→1, double→2+, miss→0).

- **P20 — Near side neighbour makes a double (rule a).** For a target under `suction_center` with a
  side neighbour whose nearest edge is within `NEIGHBOUR_DRAG_GAP` and that shares at least half the
  shorter item's height band, the outcome is `double`; moving the neighbour just beyond the gap makes
  it `clean`.
- **P21 — Cup too close to a wall misses (rule b).** For any suction grasp whose cup centre is within
  `CUP_RADIUS + WALL_SEAL_MARGIN` of a tote wall, the outcome is `miss`; `suction_edge` chooses the
  side with the most clearance from walls and neighbours.

**State / geometry properties**
- **P6 — Bbox contains the centre and is JSON-serialisable.** For any object at any in-bounds pose, each
  coordinate of `pos` lies within `[bbox.min, bbox.max]`, and `state()` round-trips through
  `json.dumps`. *(existing)*
- **P7 — Bbox is consistent with size.** For any object, `bbox.max - bbox.min` is at least the smallest
  object extent and at most the body diagonal; the box is non-degenerate (`bbox.max >= bbox.min`).
- **P8 — State shape is contract-exact.** For any scene, `state()` has keys `scene_id`, `mode`, `tote`,
  `objects` (a list), and every entry has exactly `id`, `type`, `size`, `pos`, `bbox{min,max}`, `yaw`,
  `shape`.
- **P9 — Yaw round-trips.** For any lone object loaded with a given yaw, the `yaw` reported by `state()`
  matches the input (within tolerance).
- **P10 — Settled objects stay in the tote.** After `load`, every object's centre is within the inner
  tote bounds in x-y and its z is at or above the floor (within tolerance).
- **P11 — State monotonicity.** A grasp never increases the number of in-tote objects; the set of object
  ids after a grasp is a subset of the set before it.

**Push properties**
- **P12 — Push always returns ok and respects direction.** For any item and direction, `push` returns
  `"ok"`; when `last_push_blocked` is `False`, the item's planar displacement has a positive component
  along the (normalised) push direction.
- **P13 — Push keeps the item in the tote.** After any `push`, the item's centre is still within the
  inner tote bounds (motion is clipped to the bounds).

**Determinism / invariance properties**
- **P14 — Determinism.** Loading and running the identical scene and action sequence twice yields
  identical outcomes (the simulator carries no hidden random state).
- **P15 — Translation invariance of isolated picks.** Translating a lone object within the in-bounds
  region does not change its grasp outcome.
- **P16 — Rotation invariance of isolated picks.** Rotating a lone object about z (yaw) does not change
  its grasp outcome.
- **P17 — Clock monotonicity.** `clock()` is non-decreasing across a sequence of grasps and pushes and
  strictly increases over any executed move.

**Run-contract properties**
- **P18 — Run schema.** For any scene and policy, `run_scene` output has exactly the contracted top-level
  keys, step keys, and summary keys (`clean`, `double_grabs`, `stalls`, `moves`, `misses`, `items`); the
  written JSON equals the returned dict. *(partly existing)*
- **P19 — Summary accounting.** For any run, `summary["moves"]` equals the number of executed (non-stall)
  steps, `summary["items"]` equals the scene's object count, and `misses` equals the count of `miss`
  grasp outcomes.

## Design for remaining features

### 1. Highlight the picked item in renders
- **Goal:** In rendered frames/videos, visually mark the object a grasp is acting on.
- **Approach:** Record the target id (plus any clung/ridden ids) on the `Sim` at the start of a grasp;
  in the render path, tint the matching geoms' render RGBA (or overlay a marker) for the duration of the
  grasp, then clear it. Prefer adjusting the renderer's scene RGBA over editing the model so `state()`
  and physics stay untouched.
- **Contract impact:** None. `state()`, outcomes, and the run schema are unchanged; this is render-only.
- **Verification:** A test that renders one frame mid-grasp and asserts the target geom's colour differs
  from its baseline; property P4/P5 must still hold with highlighting on.

### 2. `--camera angled` option in `run_batch` for demo videos
- **Goal:** Let a batch produce demo videos from the existing `angled` camera instead of `overhead`.
- **Approach:** `Sim.__init__` already accepts `camera` and `render` accepts a camera override, and the
  MJCF already defines an `angled` camera via `camera_poses`. Add a `--camera {overhead,angled}` argparse
  option (default `overhead`), thread it through `run_batch` → `run_scene` → `Sim(camera=...)`. Keep
  `overhead` the default so existing runs are unchanged. Note: `_remove` parks picked items out of the
  overhead view; confirm they are also out of frame (or acceptably placed) for the angled camera.
- **Contract impact:** Additive CLI flag and optional kwarg; default preserves current behaviour and the
  run-file schema is unchanged.
- **Verification:** A test that runs a one-scene batch with `camera="angled"` and asserts an MP4 is
  written; confirm `overhead` remains the default when the flag is omitted.

### 3. Physics-based suction instead of attach-on-touch
- **Goal:** Replace the deterministic "attach everything under the cup" rule (`_plan_suction` + `_cling`
  + `follow`-pinning) with a force/constraint model (e.g. a weld/equality constraint or a suction force
  applied while the cup is in contact), so multi-picks emerge from physics.
- **Approach (phased, behind the same `grasp` signature):**
  1. Detect contact between the cup geom and the top object via MuJoCo contacts rather than
     `_dist_point_poly`.
  2. Create a dynamic weld/equality constraint (or apply a pull force) between the cup body and the
     contacted object, enable it during the lift, and disable it on release.
  3. Let cling/riders fall out of real contact forces; remove the `_cling` special case only once the
     property tests (P1, P2) and the flat-stack examples still pass.
- **Contract impact:** `grasp` signature and the `clean`/`double`/`miss` strings stay the same. This is
  the riskiest change; the outcome rule (count objects risen > `LIFT_CHECK`) is the stable interface and
  should be preserved. Introduce behind a flag/constant first and flip the default only when P1, P2, and
  the flat-stack examples are green.
- **Verification:** Reuse P1–P5 as the acceptance gate. Add a test comparing the old and new models on
  the known flat-stack scenes to confirm outcomes match before switching the default.

### 4. Optional YCB mesh support
- **Goal:** Render real YCB meshes (keyed off the object's `type`) instead of primitive box/cylinder
  geoms, when meshes are available.
- **Approach:** Add an optional asset path/config. When an object's `type` maps to a mesh file that is
  present, emit a `<mesh>` asset and a `type="mesh"` geom in `build_mjcf`; otherwise fall back to the
  current primitive from `obj_shape`. Keep collision geometry as the primitive (or a convex hull) for
  stable, fast contacts while using the mesh for visuals. Guard behind a flag so default scenes keep
  using primitives.
- **Contract impact:** `size`/`shape` semantics for collision are preserved; meshes are visual-only by
  default. `state()` still reports `size`, `shape`, and `type` from the scene.
- **Verification:** A test that builds MJCF for a scene with a present mesh and asserts a mesh asset is
  emitted, plus a test that a missing mesh falls back to the primitive without error. Re-run P1–P16 to
  confirm physics outcomes are unchanged when meshes are visual-only.

## Testing strategy

- Fast suite: `python -m pytest -q -m "not slow"` (example-based behaviour tests).
- Full suite incl. properties: `python -m pytest -q`.
- New behaviour: write/extend a test in `tests/` before changing `sim/` (steering rule). For any
  "for any scene/object" requirement, add a Hypothesis property from the list above rather than a single
  example.
- Scene validation: `python -m sim.validate_scenes` before batch runs.
- Rendering is the slow part; keep property tests on small frames (`64x48`) and don't start recording.
