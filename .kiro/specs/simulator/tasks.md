# Tasks — ClearPick Simulator (`sim/`)

Scope: only the four remaining features. The existing API (contract 3), outcome strings, run schema
(contract 4), and current behaviour are already implemented and tested; do not change the public surface
without telling P3. Steering rules apply: write or update a test in `tests/` before changing `sim/`,
run `python -m pytest -q -m "not slow"` for a quick check and `python -m pytest -q` before committing,
and add a Hypothesis property (not a single example) for any "for any scene/object" claim.

- [x] 1. Highlight the item being picked in renders
  - [x] 1.1 Record the grasp target (and any clung/ridden ids) on `Sim` at the start of `grasp`, and
    clear it once the move ends, so the render path knows which geoms to mark.
  - [x] 1.2 In `render`, tint the matching geoms' render RGBA (or overlay a marker) while a grasp is in
    progress, adjusting the renderer scene only — not the model, `state()`, or physics.
  - [x] 1.3 Add `tests/` coverage: render one frame mid-grasp and assert the target geom's colour differs
    from its baseline, and confirm the grasp outcome is unchanged with highlighting on.
  - _Design:_ "Design for remaining features > 1". _Requirements:_ 8.4 (render), plus no change to 2, 3,
    5 (highlight is render-only).

- [x] 2. `--camera angled` option for demo videos in `run_batch`
  - [x] 2.1 Add a `--camera {overhead,angled}` argparse option to `run_batch` (default `overhead`) and
    thread it through `run_batch` → `run_scene` → `Sim(camera=...)`.
  - [x] 2.2 Confirm picked items parked by `_remove` are out of frame (or acceptably placed) for the
    angled camera; adjust the park offset only if they intrude, keeping the overhead view unchanged.
  - [x] 2.3 Add `tests/` coverage: a one-scene batch with `camera="angled"` writes a valid `.mp4`, and
    `overhead` remains the default when the flag is omitted.
  - _Design:_ "Design for remaining features > 2". _Requirements:_ 8.4 (render camera), 10.6 (recording
    in `run_scene`); default preserves 10.1–10.2 schema.

- [ ] 3. Physics-based suction instead of attach-on-touch
  - [x] 3.1 Behind a flag/constant, detect cup–object contact via MuJoCo contacts instead of
    `_dist_point_poly` in `_plan_suction`.
  - [ ] 3.2 Hold the contacted object with a dynamic weld/equality constraint (or a suction force) during
    the lift and release it afterward, replacing the `follow`-pinning path for suction.
  - [ ] 3.3 Let cling/riders emerge from contact forces; remove the `_cling` special case only after the
    flat-stack behaviour still holds under the new model.
  - [ ] 3.4 Add a comparison test on the known flat-stack scenes (old vs new model agree), then extend the
    Hypothesis properties P1 (isolated → clean) and P2 (flat stack → double) to run against the physics
    model.
  - [ ] 3.5 Flip the default to the physics model only once P1, P2, P4, P5 and the existing suction
    example tests pass; keep `clean`/`double`/`miss` and the `LIFT_CHECK` outcome rule unchanged.
  - _Design:_ "Design for remaining features > 3". _Requirements:_ 3.1–3.6 (suction outcomes), 7.1–7.2
    (validation unchanged). Highest risk — gate on the property suite.

- [ ] 4. Optional YCB mesh support
  - [ ] 4.1 Add an optional asset path/config and a flag to enable meshes; map an object's `type` to a
    mesh file.
  - [ ] 4.2 In `build_mjcf`, when a mesh for the object's `type` is present, emit a `<mesh>` asset and a
    mesh visual geom; otherwise fall back to the current primitive from `obj_shape`.
  - [ ] 4.3 Keep collision geometry as the primitive (or convex hull) so grasp/push stay deterministic;
    meshes are visual-only by default.
  - [ ] 4.4 Add `tests/` coverage: MJCF for a scene with a present mesh emits a mesh asset, and a missing
    mesh falls back to the primitive without error; re-run the property suite to confirm physics outcomes
    are unchanged.
  - _Design:_ "Design for remaining features > 4". _Requirements:_ 1.4 (shape selection), 2.2 (`state`
    still reports `type`/`shape`/`size`); physics outcomes unchanged (P1–P16).
