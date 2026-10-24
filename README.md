# ClearPick

Hackachusetts 2026, Robotics & hardware track.

Warehouse picking robots fail in a few repeating ways: a suction cup grabs two items at once, drags a neighbour along, or can't seal against a wall. ClearPick shows (1) which failures happen, (2) a simulated tote where they can be reproduced, and (3) a robot brain that avoids them by scoring each grasp before trying it.

| Part | Owner | Where |
|---|---|---|
| Failure atlas | P1 | `p1/` |
| MuJoCo simulator, scenes, videos | P2 | `sim/`, `scenes/`, `runs/` |
| Robot brain, scoreboard, Robot page | P3 | `brain/`, `policies/`, `site/robot.html` |

## Quick start

```
python3 -m venv .venv && source .venv/bin/activate
pip install "mujoco<3.3" numpy imageio imageio-ffmpeg pytest hypothesis
python tests/test_brain.py                          # brain tests, no dependencies
python -m sim.run_batch --limit 20 --no-video       # run both robots on every scene
python -m brain.evaluate                            # scoreboard table -> scoreboard.json
python -m brain.build_site                          # bundle runs into site/data.js
open site/robot.html                                # Robot page, opens straight from disk
```

## The robot brain (P3)

Two robots run on the same scenes in the same simulator.

**Grab-only** (`policies/grab_only.py`) is the baseline. Every turn it suctions the highest item at its centre.

**ClearPick** (`policies/clearpick.py`) looks before it grabs:

1. For every item and every grasp type (`suction_center`, `suction_edge`, `side_pinch`) it computes a risk from 0 to 1 using `score_grasp`.
2. If the safest option is at or below the cutoff (0.35), it grasps.
3. Otherwise it pushes the most crowded free item to make room (at most twice per item), then re-scores.
4. If a grasp misses, it remembers that (item, grasp) pair and tries something different.
5. Every step writes a one-line reason, which the Robot page shows next to the step.

`score_grasp(state, item_id, grasp_type)` in `brain/score.py` uses geometry only: what is on top of the item, what it sits on, how close its neighbours and the tote walls are, and whether a suction cup could seal. Risk is a weighted sum passed through a sigmoid. It never sees the simulator's outcome, so it can't cheat.

### Files

```
brain/geometry.py     reads the simulator state (list or dict) into one internal form
brain/score.py        score_grasp: risk 0..1 for one item and one grasp
brain/strategy.py     GrabOnly, ClearPick, push-direction search, retry memory
brain/scene_gen.py    seeded scene recipes -> scenes/<mode>/1..20.json
brain/evaluate.py     per-mode scoreboard from runs/*.json
brain/diagnose.py     explain one scene: per-item risks and every step
brain/build_site.py   bundle runs, scenes and scoreboard into site/data.js
policies/*.py         decide(state) wrappers that the simulator loads
site/robot.html       Robot page: both robots side by side per scene
tests/test_brain.py   plain-function tests, run with `python tests/test_brain.py`
```

### Scene modes

| Mode | What makes it hard |
|---|---|
| `flat_stacked` | flat boxes stacked on each other; grabbing the bottom one takes the top one along |
| `small_beside_big` | a small item within about 1 cm of a big one, so a centre grab lifts both |
| `against_wall` | items flush to the tote wall, where a suction cup can't seal |

Regenerate scenes with `python -m brain.scene_gen`. Scenes are seeded, so everyone gets the same ones.

### Debugging one scene

```
python -m brain.diagnose scenes/against_wall/3.json clearpick
```

This prints each item's risk for each grasp, then each step with its outcome and the items that came along.

### Reading the results

`python -m brain.evaluate` prints one row per mode and robot:

- `clean`: items picked one at a time.
- `double`: grabs that took more than one item.
- `miss`: grasps that failed to pick anything.
- `push`: pushes made to clear space.
- `dbl rate`: double grabs divided by all grasps.

A lower double-grab rate for ClearPick than Grab-only is the headline result. Mock and test scenes (`*_000`, `*_t01`) are excluded.

## Failure atlas (P1)

_P1: describe the dataset, the failure clusters, and how to open `p1/site/atlas.html`._

## Simulator (P2)

_P2: describe the MuJoCo setup, the failure rules, and how videos are recorded._

## Demo and write-up

_Links to the hosted pages, the demo video, and the Kiro write-up go here._
