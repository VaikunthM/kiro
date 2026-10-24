# Requirements — ClearPick Simulator (`sim/`)

## Introduction

The ClearPick simulator (`sim/sim.py`) builds a tote + objects + overhead camera scene in MuJoCo
from a scene JSON and exposes a small, deterministic API that P3's robot policies call and the Robot
page reads. This document captures the behaviour the simulator already contracts for, derived from the
module docstring in `sim/sim.py` and the behaviour pinned by `tests/`. It is written in EARS
(Easy Approach to Requirements Syntax) so each requirement maps to an observable behaviour.

Units are metres and kilograms throughout. The tote frame has its origin at the middle of the tote
floor with z pointing up. The public surface (`load`, `state`, `grasp`, `push`, `record`, plus the
grasp types and outcome strings) is a team contract and must not change without telling P3.

## Glossary

- **Scene JSON**: a dict (or path to a `.json`) with `scene_id`, optional `mode`, optional `tote`, and
  an `objects` list. Each object has `id`, `size` (full extents), `pos` (centre in tote frame), and
  optional `shape` (`box`|`cylinder`), `quat`, `mass`, `rgba`, `ycb`.
- **Grasp type**: one of `suction_center`, `suction_offcenter`, `side_pinch` (`GRASP_TYPES`).
- **Grasp outcome**: `clean` (1 object lifted), `double_grab` (2+ lifted), `miss` (0 lifted).
- **Push outcome**: `ok` (item moved) or `blocked`.
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
3. WHEN a scene omits the `tote` field THEN the simulator SHALL use `TOTE_DEFAULT` for tote size and wall.
4. WHEN a scene object omits `shape` THEN the simulator SHALL treat it as a `box`.
5. WHEN `load` completes THEN the simulator SHALL have advanced physics by the settle period so objects
   rest before control is returned.
6. WHEN an object omits `rgba` THEN the simulator SHALL assign a colour from `PALETTE` by index.
7. WHEN `load` is called more than once on the same `Sim` instance THEN the simulator SHALL rebuild the
   model and reset recorded frames so the new scene is independent of the previous one.

## Requirement 2: Report state

**User story:** As a policy author, I want `state()` to return each in-tote object's pose and bounding
box in a JSON-serialisable form, so that a `decide(state)` function can choose actions.

#### Acceptance criteria
1. WHEN `state` is called THEN the simulator SHALL return a dict with keys `scene_id`, `tote` (with
   `size`), and `objects`.
2. FOR EACH in-tote object, WHEN `state` is called THEN the simulator SHALL include `pos`, `quat`,
   `size`, `shape`, `bbox_min`, and `bbox_max` for that object.
3. WHERE an object has been picked (removed), WHEN `state` is called THEN the simulator SHALL omit that
   object from `objects`.
4. WHEN `state` is called THEN the returned `bbox_min` and `bbox_max` SHALL be the world-frame
   axis-aligned bounds of the object's current pose and SHALL contain the object's `pos`.
5. WHEN `state` is called THEN the returned dict SHALL be JSON-serialisable.

## Requirement 3: Suction grasp

**User story:** As a policy author, I want a suction grasp that behaves like the ARMBench "attach on
touch" model, so that single picks and flat-stack multi-picks are reproducible.

#### Acceptance criteria
1. WHEN `grasp(item, "suction_center")` targets an isolated object THEN the simulator SHALL return
   `clean`.
2. WHEN `grasp(item, "suction_offcenter")` is called THEN the simulator SHALL place the cup toward the
   side of the item with the most clearance from neighbouring objects.
3. WHEN the suction cup lands on a surface AND another object's top is within `TOUCH_TOL` of that
   surface and overlaps the cup THEN the simulator SHALL attach that object as well.
4. WHEN an attached flat item (thinner than `CLING_MAX_THICKNESS`) rests on another flat item whose
   footprint overlaps by at least `CLING_MIN_OVERLAP` THEN the simulator SHALL also lift the lower item
   (cling / flat-stack multi-pick).
5. WHEN a grasp lifts an object that itself supports another object THEN the supported object SHALL ride
   along and count toward the outcome.
6. WHEN exactly one object rises more than `LIFT_CHECK` THEN the outcome SHALL be `clean`; WHEN two or
   more rise THEN the outcome SHALL be `double_grab`; WHEN none rise THEN the outcome SHALL be `miss`.

## Requirement 4: Side-pinch grasp

**User story:** As a policy author, I want a side-pinch grasp whose fingers close across the item's
shorter horizontal side, so that pinching respects walls and neighbours.

#### Acceptance criteria
1. WHEN `grasp(item, "side_pinch")` targets an isolated object THEN the simulator SHALL return `clean`.
2. WHEN a finger cannot descend because a wall or another object blocks it THEN the grasp SHALL be a
   `miss`.
3. WHEN pinching the top item of a stack THEN the simulator SHALL lift only the items that lie between
   the fingers at the item's height and SHALL NOT take a lower item outside the jaw span.
4. WHEN the fingers close THEN the simulator SHALL lift every object between the fingers at the item's
   height.

## Requirement 5: Picked items leave the tote

**User story:** As a policy author, I want picked items to disappear from the scene, so that subsequent
`state()` calls reflect an emptier tote.

#### Acceptance criteria
1. WHEN a grasp returns `clean` THEN the picked item SHALL be absent from the next `state()`.
2. WHEN a grasp returns `double_grab` THEN both picked items SHALL be absent from the next `state()`.
3. WHEN an object is removed THEN the simulator SHALL relocate it outside the overhead camera's view so
   it does not appear in overhead renders.

## Requirement 6: Push

**User story:** As a policy author, I want to push an item along a 2D direction, so that policies can
separate stacks or clear space before grasping.

#### Acceptance criteria
1. WHEN `push(item, [dx, dy])` has room behind the item THEN the simulator SHALL move the item along the
   normalised direction and return `ok` when it moves more than 1 cm.
2. WHEN the pusher cannot get behind the item (a wall or another object blocks its start position) THEN
   the simulator SHALL return `blocked`.
3. WHEN a push separates a flat stack THEN a subsequent centre suction on the previously-top item SHALL
   be able to return `clean`.
4. WHEN the push target would exit the tote THEN the simulator SHALL clip the motion to stay within the
   inner bounds.

## Requirement 7: Input validation

**User story:** As a policy author, I want invalid calls to fail loudly, so that bugs surface
immediately rather than producing silent wrong outcomes.

#### Acceptance criteria
1. WHEN `grasp` is called with a grasp type not in `GRASP_TYPES` THEN the simulator SHALL raise
   `ValueError`.
2. WHEN `grasp` or `push` targets an item not in the tote THEN the simulator SHALL raise `ValueError`.

## Requirement 8: Recording and rendering

**User story:** As a reviewer, I want each run recorded to an MP4, so that the Robot page and demos can
replay what happened.

#### Acceptance criteria
1. WHEN `Sim` is constructed with `video=True` THEN the simulator SHALL capture frames at `FPS` from
   `load` onward.
2. WHEN `record(path)` is called THEN the simulator SHALL write an `.mp4` of the frames since `load`,
   create parent directories as needed, and return the number of frames written.
3. WHEN `record(path)` is called and no frames were captured THEN the simulator SHALL write at least a
   single rendered frame rather than an empty file.
4. WHEN `render(path, camera)` is called THEN the simulator SHALL render from the named camera and, if a
   path is given, write the image to it.

## Requirement 9: Scene validation

**User story:** As a batch operator, I want to validate scenes before a run, so that overlapping,
missing-field, or out-of-tote scenes are reported rather than silently misbehaving.

#### Acceptance criteria
1. WHEN an object is missing any of `id`, `size`, `pos` THEN `check_scene` SHALL report the missing
   field and SHALL NOT raise.
2. WHEN an object's centre lies outside the tote footprint THEN `check_scene` SHALL report it as outside.
3. WHEN objects overlap such that one moves more than `MAX_SETTLE_MOVE` while settling THEN `check_scene`
   SHALL report that object as moved.
4. WHEN all scenes pass THEN `main` SHALL exit `0`; WHEN any scene has a problem THEN `main` SHALL exit
   `1`.

## Requirement 10: Batch runner and run contract

**User story:** As a batch operator, I want every scene run with every policy and written in a fixed
schema, so that downstream tooling can read results consistently.

#### Acceptance criteria
1. WHEN `run_scene` runs a policy on a scene THEN it SHALL write `runs/<scene_id>_<policy>.json` whose
   content equals the returned run dict.
2. WHEN a run is written THEN it SHALL contain keys `scene_id`, `mode`, `policy`, `steps`, `summary`,
   `video`, each step SHALL contain `t`, `action`, `item`, `grasp`, `direction`, `risk`, `reason`,
   `outcome`, and `summary` SHALL contain exactly `clean`, `double_grabs`, `stalls`, `moves`.
3. WHEN a policy's `decide(state)` returns `None` THEN the run SHALL stop with no further steps.
4. WHEN a grasp or push returns `miss` or `blocked` THEN the run SHALL count it as a stall, AND WHEN
   `MAX_STALLS_IN_A_ROW` stalls occur consecutively THEN the run SHALL end.
5. WHEN `run_batch` runs THEN it SHALL run each policy on each scene and write `runs/index.json`.
6. WHEN a requested policy module `policies/<name>.py` is absent but a matching function exists in
   `sim.reference_policies` THEN the runner SHALL fall back to the reference policy.
