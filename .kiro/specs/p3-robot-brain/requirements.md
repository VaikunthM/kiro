# P3 robot brain: requirements

ClearPick must pick items out of a simulated tote with fewer failed picks than a naive robot, and say why for every step.

## User stories and acceptance criteria

1. As a warehouse engineer, I want the robot to avoid grasps likely to lift two items, so that fewer picks fail.
   - WHEN a scene has stacked flat items THEN ClearPick SHALL clear space or choose a safer grasp before picking, and double grabs SHALL be lower than Grab-only on the same 20 scenes.
   - WHEN a small item sits within 1 cm of a large one THEN ClearPick SHALL avoid a centre grasp that would lift both.
   - WHEN an item is close enough to a wall that a suction cup cannot seal THEN ClearPick SHALL choose a grasp that can seal, or move the item first.
2. As a reviewer, I want every step to explain itself, so that I can trust the choices.
   - EVERY step SHALL carry the action, the item, the risk, and a one-line reason.
3. As a developer, I want `score_grasp` to be a pure function of geometry, so that it cannot cheat on simulator outcomes.
   - Risk SHALL always be between 0 and 1.
   - An isolated item SHALL score lower than the same item stacked under another.
4. As a judge, I want a side-by-side comparison, so that I can see the difference.
   - The Robot page SHALL show both robots per scene with video, steps, risk, and a per-mode scoreboard that counts doubles and misses.
5. Scenes SHALL be reproducible from a seed, 20 per mode, for the modes `flat_stacked`, `small_beside_big` and `against_wall`.
