---
inclusion: fileMatch
fileMatchPattern: ["sim/**/*.py", "tests/**/*.py", "scenes/**/*.json", "runs/**/*.json", ".kiro/specs/simulator/**"]
---

# Simulator lane (Person 2)

You are working on ClearPick's MuJoCo simulator. P3's robot policies call it and the Robot page plays
its videos. The shared contracts steering doc is the source of truth for every interface: scenes
(contract 2), the simulator API (contract 3) and run files (contract 4). If code and contract disagree,
the contract wins; raise it rather than changing the contract.

## Where to look (don't go hunting)
- The simulator API and mechanics are in the module docstring: #[[file:sim/sim.py:1-34]]
- Behaviour is pinned down by `tests/`. Planning documents live in `docs/` and are not needed for code
  tasks; never try to parse `.docx` files.

## How the physics is meant to work
- Attach-on-touch suction and pinch are deliberate simplifications. Outcomes come from counting objects
  that actually rose more than LIFT_CHECK, so riders on a lifted item count.
- The flat-stack multi-pick comes from the cling rule (CLING_MAX_THICKNESS, CLING_MIN_OVERLAP). Tune
  those constants instead of adding special cases for particular scenes.
- `push` always returns "ok" (contract); `sim.last_push_blocked` says whether the pusher had room.
- Never feed simulator renders to the ARMBench image model (team design rule).

## Working rules
- Change a test in `tests/` before changing behaviour in `sim/`. Quick check:
  `python -m pytest -q -m "not slow"`; full suite before committing: `python -m pytest -q`.
- Property-based tests use Hypothesis and are marked `@pytest.mark.slow`. When a requirement says
  "for any scene/object", write a Hypothesis property rather than one example.
- Recording is off unless `sim.record(path)` was called, which keeps tests fast.
- Validate generated scenes with `python -m sim.validate_scenes` before batch runs.
