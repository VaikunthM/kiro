# P3 robot brain: design

```
scenes/<mode>/<n>.json  ->  sim (P2)  ->  state()  ->  policies/<name>.py decide(state)  ->  runs/<scene>_<policy>.json
                                                     \-> brain/score.py  score_grasp(state, item, grasp)
runs/*.json -> brain/evaluate.py (scoreboard) and brain/build_site.py (site/data.js) -> site/robot.html
```

- `brain/geometry.py` turns the simulator state into one internal form (item boxes, tote size).
- `brain/score.py` computes geometric features (what is above and below, neighbours, wall distance, cup seal) and turns them into a 0 to 1 risk with a sigmoid. It imports the simulator's rule constants so the two cannot drift apart.
- `brain/strategy.py` has `GrabOnly` and `ClearPick`. ClearPick scores every item and grasp, grasps the safest if risk is at most 0.35, otherwise pushes the most crowded free item (at most twice per item), and remembers failed grasps.
- `brain/scene_gen.py` makes seeded scenes for the three modes.

## Decisions
- Weights are hand-set per grasp kind and scaled per mode, not learned: 20 scenes is too few to train on and hand-set weights stay explainable.
- The scorer never reads simulator outcomes.
- Final tuning was made against the simulator with a 10 cm push.
