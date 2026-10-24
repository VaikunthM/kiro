---
inclusion: fileMatch
fileMatchPattern: ["sim/**/*.py", "tests/**/*.py", "scenes/**/*.json", "runs/**/*.json"]
---

# Simulator lane (Person 2)

You are working on ClearPick's MuJoCo simulator. P3 calls this code from the robot policies and the
Robot page reads its output, so the public interface below is a team contract.

## Contract: do not change without telling P3
- `Sim.load(scene)`, `Sim.state()`, `Sim.grasp(item_id, grasp_type)`, `Sim.push(item_id, [dx, dy])`,
  `Sim.record(path)`. Signatures and return values are documented in the module docstring:
  #[[file:sim/sim.py:1-38]]
- Grasp types: `suction_center`, `suction_offcenter`, `side_pinch`. Outcomes: `clean`, `double_grab`,
  `miss` (grasp) and `ok`, `blocked` (push).
- Run files: `runs/<scene_id>_<policy>.json` with keys scene_id, mode, policy, steps, summary, video.
  Each step has t, action, item, grasp, direction, risk, reason, outcome. Summary keys: clean,
  double_grabs, stalls, moves. A stall is a `miss` or `blocked` outcome.
- Policies live in `policies/<name>.py` (P3) and expose `decide(state) -> step dict | None`.
  See `sim/reference_policies.py`.

## How the physics is meant to work
- Attach-on-touch suction and pinch are deliberate simplifications, not bugs. Outcomes come from
  counting objects that actually rose more than LIFT_CHECK, so riders on a lifted item count.
- The flat-stack multi-pick comes from the cling rule (CLING_MAX_THICKNESS, CLING_MIN_OVERLAP).
  Tune those constants instead of adding special cases per scene.
- Tote frame: origin at the middle of the tote floor, z up, units metres. Resting object z = size z / 2.
- Never feed simulator renders to the ARMBench image model (team design rule).

## Working rules
- Write or update a test in `tests/` before changing behaviour in `sim/`. Run `python -m pytest -q -m "not slow"`
  for a quick check and `python -m pytest -q` before committing.
- Property-based tests use Hypothesis and are marked `@pytest.mark.slow`. When a requirement says
  "for any scene/object", add a Hypothesis property rather than a single example.
- Pass `video=False` to `Sim` in tests; rendering is the slow part.
- Shell is Windows PowerShell, Python env is the conda env `hackachusetts`. Don't write .sh scripts.
- Validate generated scenes with `python -m sim.validate_scenes` before batch runs.
