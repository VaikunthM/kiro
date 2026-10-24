# Requirements — ClearPick Simulator (`sim/`)

## Introduction

The ClearPick simulator (`sim/sim.py`) builds a tote + objects + overhead camera scene in MuJoCo from a
scene JSON and exposes a small, deterministic API that P3's robot policies call and the Robot page reads.
It implements **contract 3 (Simulator API)** of the shared contracts steering doc; scenes follow
contract 2. This document captures the behaviour the simulator already contracts for, derived from the
`sim/sim.py` module docstring and the behaviour pinned by `tests/`. It is written in EARS (Easy Approach
to Requirements Syntax) so each requirement maps to an observable behaviour.

Units are metres and kilograms. The tote frame has its origin at the middle of the tote floor with z
pointing up; the tote is centred at the origin and the inner floor surface is z = 0. A resting object's
centre is at z = size z / 2. The default tote size is `0.40 x 0.30 x 0.15`.

The public surface (`load`, `state`, `grasp`, `push`, `record`, `clock`, plus `last_push_blocked`, the
grasp types, and the outcome strings) is a team contract and must not change without telling P3.

## Glossary

- **Scene JSON**: a dict (or path to a `.json`) with `scene_id`, optional `mode`, optional `seed`,
  optional `tote`, and an `objects` list. Each object has `id`, `type`, `size` (full extents), `pos`
  (centre in tote frame), and optional `yaw`, `shape`, `quat` (overrides `yaw`), `mass`, `rgba`.
- **Grasp type**: one of `suction_center`, `suction_edge`, `side_pinch` (`GRASP_TYPES`).
- **Grasp outcome**: `clean` (exactly one object lifted), `double` (two or more lifted), `miss` (none).
- **Push outcome**: always `"ok"`; `sim.last_push_blocked` is `True` when the pusher had no room or the
  item did not move.
- **Lifted**: an object whose body rose more than `LIFT_CHECK` during the move.
- **Picked / removed**: an object that was lifted; it leaves the tote and disappears from `state()`.
- **Cling (flat-stack multi-pick)**: a flat attached item drags up a flat item it rests on when their
  footprints overlap by at least `CLING_MIN_OVERLAP`, both thinner than `CLING_MAX_THICKNESS`.

---

## Requirement 1: Load a scene and settle it

**User story:** As a policy author, I want to load a scene from a dict or a path and have objects settle
under physics, so that `state()` reflects a stable starting configuration.

#### Acceptance criteria
1. WHEN `load` is called with a scene dict THEN the simulator SHALL build the MJCF model and return the
   result of `state()`.
2. WHEN `load` is called with a `str` or `Path` THEN the simulator SHALL read and parse that file as the
   scene JSON before building the model.
3. WHEN a scene omits the `tote` field THEN the simulator SHALL use `TOTE_DEFAULT` (`0.40 x 0.30 x 0.15`,
   wall `0.01`).
4. WHEN an object's `type` is in `CYLINDER_TYPES` (or ends in `_can`, except `potted_meat_can`) and no
   explicit `shape` is given THEN the simulator SHALL load it as a cylinder; otherwise as a box.
5. WHEN an object omits `mass` THEN the simulator SHALL derive mass from `DENSITY` and the object's
   volume (with a 0.01 kg floor).
6. WHEN an object provides `yaw` but no `quat` THEN the simulator SHALL orient it by that yaw about z;
   WHEN `quat` is present THEN it SHALL override `yaw`.
7. WHEN an object omits `rgba` THEN the simulator SHALL assign a colour from `PALETTE` by index.
8. WHEN `load` completes THEN the simulator SHALL have advanced physics by the settle period, SHALL reset
   the simulated clock origin, and SHALL return control with objects at rest.
9. WHEN `load` is called more than once on the same `Sim` instance THEN the simulator SHALL rebuild the
   model and reset recorded frames and recording state so the new scene is independent of the previous.

## Requirement 2: Report state

**User story:** As a policy author, I want `state()` to return each in-tote object as a list entry with
pose and bounding box in a JSON-serialisable form, so that a `decide(state)` function can choose actions.

#### Acceptance criteria
1. WHEN `state` is called THEN the simulator SHALL return a dict with keys `scene_id`, `mode`, `tote`
   (with `size`), and `objects`, where `objects` is a list.
2. FOR EACH in-tote object, WHEN `state` is called THEN its list entry SHALL include `id`, `type`,
   `size`, `pos`, `bbox` (with `min` and `max`), `yaw`, and `shape`.
3. WHERE an object has been picked (removed), WHEN `state` is called THEN the simulator SHALL omit that
   object from `objects`.
4. WHEN `state` is called THEN each entry's `bbox.min`/`bbox.max` SHALL be the world-frame axis-aligned
   bounds of the object's current pose and SHALL contain the object's `pos`.
5. WHEN `state` is called THEN each entry's `yaw` SHALL be the current orientation about z.
6. WHEN `state` is called THEN the returned dict SHALL be JSON-serialisable.

## Requirement 3: Suction grasp

**User story:** As a policy author, I want a suction grasp that behaves like the ARMBench "attach on
touch" model, so that single picks and flat-stack multi-picks are reproducible.

#### Acceptance criteria
1. WHEN `grasp(item, "suction_center")` targets an isolated object THEN the simulator SHALL return
   `clean`.
2. WHEN `grasp(item, "suction_edge")` is called THEN the simulator SHALL place the cup toward the side of
   the item with the most clearance from neighbouring objects.
3. WHEN the suction cup lands on a surface AND another object's top is within `TOUCH_TOL` of that surface
   and overlaps the cup THEN the simulator SHALL attach that object as well.
4. WHEN an attached flat item (thinner than `CLING_MAX_THICKNESS`) rests on another flat item whose
   footprint overlaps by at least `CLING_MIN_OVERLAP` THEN the simulator SHALL also lift the lower item
   (cling / flat-stack multi-pick).
5. WHEN a grasp lifts an object that itself supports another object THEN the supported object SHALL ride
   along and count toward the outcome.
6. WHEN exactly one object rises more than `LIFT_CHECK` THEN the outcome SHALL be `clean`; WHEN two or
   more rise THEN the outcome SHALL be `double`; WHEN none rise THEN the outcome SHALL be `miss`.

## Requirement 4: Side-pinch grasp

**User story:** As a policy author, I want a side-pinch grasp whose fingers close across the item's
shorter horizontal side, so that pinching respects walls and neighbours.

#### Acceptance criteria
1. WHEN `grasp(item, "side_pinch")` targets an isolated object THEN the simulator SHALL return `clean`.
2. WHEN a finger cannot descend because a wall or another object blocks it THEN the grasp SHALL be a
   `miss`.
3. WHEN pinching the top item of a stack THEN the simulator SHALL lift only the items between the fingers
   at the item's height and SHALL NOT take a lower item outside the jaw span.
4. WHEN the fingers close THEN the simulator SHALL lift every object between the fingers at the item's
   height.

## Requirement 5: Picked items leave the tote

**User story:** As a policy author, I want picked items to disappear from the scene, so that subsequent
`state()` calls reflect an emptier tote.

#### Acceptance criteria
1. WHEN a grasp returns `clean` THEN exactly one item SHALL be absent from the next `state()`.
2. WHEN a grasp returns `double` THEN two or more items SHALL be absent from the next `state()`.
3. WHEN an object is removed THEN the simulator SHALL relocate it outside the overhead camera's view so
   it does not appear in overhead renders.

## Requirement 6: Push

**User story:** As a policy author, I want to push an item along a 2D direction, so that policies can
separate stacks or clear space before grasping.

#### Acceptance criteria
1. WHEN `push(item, [dx, dy])` is called THEN the simulator SHALL always return `"ok"`.
2. WHEN the pusher has room behind the item AND the item moves more than 1 cm THEN `last_push_blocked`
   SHALL be `False`.
3. WHEN the pusher cannot get behind the item (a wall or another object blocks its start) OR the item
   does not move more than 1 cm THEN `last_push_blocked` SHALL be `True`.
4. WHEN a push separates a flat stack THEN a subsequent centre suction on the previously-top item SHALL
   be able to return `clean`.
5. WHEN the push target would exit the tote THEN the simulator SHALL clip the motion to stay within the
   inner bounds.

## Requirement 7: Input validation

**User story:** As a policy author, I want invalid calls to fail loudly, so that bugs surface
immediately rather than producing silent wrong outcomes.

#### Acceptance criteria
1. WHEN `grasp` is called with a grasp type not in `GRASP_TYPES` THEN the simulator SHALL raise
   `ValueError`.
2. WHEN `grasp` or `push` targets an item not in the tote THEN the simulator SHALL raise `ValueError`.

## Requirement 8: Recording, rendering, and clock

**User story:** As a reviewer, I want to record a run to an MP4 and read the simulated clock, so that the
Robot page and demos can replay what happened and timestamp each step.

#### Acceptance criteria
1. WHEN `record(path)` is called with a non-`None` path THEN the simulator SHALL start capturing frames
   at `FPS` for an `.mp4` at that path and SHALL return `0`.
2. WHEN `record(None)` is called while recording THEN the simulator SHALL stop, write the `.mp4` (creating
   parent directories as needed), and return the number of frames saved.
3. WHEN `record(None)` is called while not recording THEN the simulator SHALL do nothing and return `0`.
4. WHEN `render(path, camera)` is called THEN the simulator SHALL render from the named camera
   (`overhead` or `angled`, defaulting to the instance camera) and, if a path is given, write the image.
5. WHEN `clock()` is called THEN the simulator SHALL return the simulated seconds elapsed since `load`
   finished settling, and this value SHALL increase as grasps and pushes execute.

## Requirement 9: Scene validation

**User story:** As a batch operator, I want to validate scenes before a run, so that overlapping,
missing-field, or out-of-tote scenes are reported rather than silently misbehaving.

#### Acceptance criteria
1. WHEN an object is missing any of `id`, `type`, `size`, `pos` THEN `check_scene` SHALL report the
   missing field and SHALL NOT raise.
2. WHEN an object's centre lies outside the tote footprint THEN `check_scene` SHALL report it as outside.
3. WHEN objects overlap such that one moves more than `MAX_SETTLE_MOVE` while settling THEN `check_scene`
   SHALL report that object as moved.
4. WHEN all scenes pass THEN `main` SHALL exit `0`; WHEN any scene has a problem THEN `main` SHALL exit
   `1`.

## Requirement 10: Batch runner and run contract

**User story:** As a batch operator, I want every scene run with every policy and written in a fixed
schema (contract 4), so that downstream tooling can read results consistently.

#### Acceptance criteria
1. WHEN `run_scene` runs a policy on a scene THEN it SHALL write `runs/<scene_id>_<policy>.json` whose
   content equals the returned run dict.
2. WHEN a run is written THEN it SHALL contain keys `scene_id`, `mode`, `policy`, `steps`, `summary`,
   `video`; each step SHALL contain `t`, `action`, `item`, `grasp`, `direction`, `risk`, `reason`,
   `outcome`; and `summary` SHALL contain exactly `clean`, `double_grabs`, `stalls`, `moves`, `misses`,
   `items`.
3. WHEN a policy's `decide(state)` returns `None` THEN the run SHALL stop with no further steps.
4. WHEN a step carries `stall: True` THEN the runner SHALL execute no robot action, count it as a stall,
   and record the step with `outcome` `None`.
5. WHEN a grasp returns `miss`, OR a push leaves `last_push_blocked` `True` THEN the runner SHALL treat
   the step as no-progress, AND WHEN `MAX_STALLS_IN_A_ROW` no-progress steps occur consecutively THEN the
   run SHALL end.
6. WHEN recording is enabled THEN `run_scene` SHALL call `record(path)` after `load` and `record(None)`
   before writing the run JSON, and SHALL set `video` to the run's `.mp4` path.
7. WHEN `run_batch` runs THEN it SHALL run each policy on each scene and write `runs/index.json`.
8. WHEN a requested policy module `policies/<name>.py` is absent but a matching function exists in
   `sim.reference_policies` THEN the runner SHALL fall back to the reference policy.
