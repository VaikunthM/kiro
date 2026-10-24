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

- [ ] 3. Physics-based suction, opt-in behind `Sim(suction="physics")`
  - Default stays `suction="attach"` (the agreed, tested model P3's scorer mirrors); never flip it.
  - Acceptance tests are already written: `tests/test_physics_suction.py` (red until done). Prototype
    findings, incl. working weld code: `.kiro/specs/simulator/notes-physics-suction.md`.
  - [ ] 3.1 Add a `suction` constructor argument (`"attach"` default, `"physics"`; anything else raises
    `ValueError`) and expose it as `sim.suction`. In physics mode only, `build_mjcf` makes the cup geom
    collide and adds one inactive `weld` equality (cup -> object) per object.
  - [ ] 3.2 In physics mode, lower the colliding cup until MuJoCo's contact list (`data.contact`) reports a
    contact with the cup geom, and take the touched bodies from those contacts instead of
    `_dist_point_poly`. Keep the wall-seal rule (b) check before descending.
  - [ ] 3.3 Hold the touched objects (plus `_cling` and rule (a) neighbours, which stay as rules: a weld
    never drags the box underneath physically) by activating their welds with the current relative pose,
    lift with physics instead of `follow`-pinning, and release the welds before `_remove`.
  - [ ] 3.4 Make `tests/test_physics_suction.py` pass, plus the full suite; outcomes in physics mode must
    match attach mode on the failure-rule scenes (P2, P20, P21) and riders must still ride.
  - _Design:_ "Design for remaining features > 3". _Requirements:_ 3.1-3.8 (suction outcomes, rules a/b),
    7.1-7.2 (validation unchanged). Contract outputs and the default model are unchanged.

- [ ] 4. Optional YCB mesh visuals, opt-in behind `Sim(meshes=True)`
  - Assets are in the repo: `assets/ycb/<type>/textured.obj` + `texture_map.png` for sugar_box,
    tomato_soup_can, gelatin_box, potted_meat_can, foam_brick, and `assets/ycb/manifest.json` with the
    per-type `scale`, `quat` and `pos` that fit each mesh exactly into the primitive's size (computed by
    `assets/ycb/make_manifest.py`). flat_box has no mesh and stays a primitive.
  - Acceptance tests are already written: `tests/test_meshes.py` (red until done).
  - [ ] 4.1 Add `meshes=False` and `asset_dir=None` constructor arguments (default folder: `assets/ycb` at
    the repo root) and a loader that reads the manifest, keeping only entries whose files exist; a missing
    folder means no meshes and no error.
  - [ ] 4.2 In `build_mjcf`, for each object whose `type` has a mesh: emit the `<mesh scale=...>`,
    `<texture>` and `<material>` assets once per type, add a visual geom named `<id>_visual` with the
    manifest `pos`/`quat`, `contype="0" conaffinity="0" mass="0"`, and set the collision primitive's alpha
    to 0. Collision geoms, physics, `state()` and outcomes stay identical to `meshes=False`.
  - [ ] 4.3 Highlight must tint the `<id>_visual` geom too (the primitive is invisible with meshes on).
  - [ ] 4.4 Add `--meshes` to `run_batch` (passed through to `Sim`), make `tests/test_meshes.py` pass, plus
    the full suite.
  - _Design:_ "Design for remaining features > 4". _Requirements:_ 1.4 (shape selection), 2.2 (`state`
    still reports `type`/`shape`/`size`); physics outcomes unchanged (P1-P16).
