# Notes for task 3 (physics-based suction) and task 4 (mesh visuals)

Findings from prototypes run before the tasks were written. Use them instead of re-exploring.

## Task 3: what works and what doesn't

- **MuJoCo's `adhesion` actuator does not work here.** On the mocap cup it reports full force (60 N on a
  0.1 kg box) but the box does not rise when the cup lifts. Don't spend time on it.
- **A weld equality activated at contact works.** Lowering a colliding cup onto a two-box stack, reading the
  touched body from `data.contact`, then activating a cup-to-object weld with the current relative pose
  lifted the top box 20 cm while the box underneath stayed on the floor.
- Consequence: the flat-stack multi-pick does **not** emerge from physics (a weld only holds what it welds),
  so `_cling` stays as a rule, and rules (a) neighbour drag and (b) wall seal stay too. Physics mode changes
  *how* touched objects are detected and held, not which outcomes the agreed rules produce.
- Riders (an item resting on a lifted item) still ride, by physics, as in attach mode.

### MJCF changes (physics mode only)

```xml
<!-- cup geom: drop contype="0" conaffinity="0" so it collides; give it a name -->
<geom name="cup" type="cylinder" size="0.02 0.004" pos="0 0 0.004" rgba="0.95 0.95 0.95 1"/>

<!-- after </worldbody>: one inactive weld per object -->
<equality>
  <weld name="hold_box_1" body1="cup" body2="box_1" active="false" solref="0.01 1"/>
</equality>
```

### Reading cup contacts and welding (verified in the prototype)

```python
def _cup_contacts(self):
    gid = self.model.geom("cup").id
    out = set()
    for c in self.data.contact[: self.data.ncon]:
        if gid in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == gid else c.geom1
            out.add(self.model.body(self.model.geom_bodyid[other]).name)
    return sorted(out)

def _weld(self, oid, on):
    eq = self.model.equality(f"hold_{oid}").id
    if on:  # eq_data for a weld: anchor(3), relative pos(3), relative quat(4), torquescale(1)
        cup, obj = self.data.body("cup"), self.data.body(oid)
        rel_p = cup.xmat.reshape(3, 3).T @ (obj.xpos - cup.xpos)
        qinv, rel_q = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(qinv, cup.xquat)
        mujoco.mju_mulQuat(rel_q, qinv, obj.xquat)
        self.model.eq_data[eq][:11] = [0, 0, 0, *rel_p, *rel_q, 1]
    self.data.eq_active[eq] = 1 if on else 0
```

Descend the cup a step at a time (TOOL_SPEED x timestep per step) and stop at the first cup contact, or
3 mm below the expected top surface. Weld the touched objects plus `_cling` and rule (a) neighbours, lift
with `follow=None` (physics carries them), hold, then release every weld **before** `_remove` teleports the
picked objects, or the weld drags them back.

## Task 4: meshes

- `assets/ycb/manifest.json` gives, per type: `mesh`, `texture` (paths relative to `assets/ycb/`),
  `scale` (applied in `<mesh scale=...>`), and the visual geom's `quat` and `pos` in the object frame.
  With those, each mesh's bounding box matches the primitive's `size` exactly; verified visually on
  P3's scenes with the collision boxes overlaid.
- Keep collision as the primitive (MuJoCo skill guidance: don't use visual meshes as the grasp collision
  basis). Hide the primitive with alpha 0 and add a `<id>_visual` mesh geom with contype/conaffinity 0 and
  mass 0, so physics and outcomes are untouched.
- Use absolute paths (`Path(...).as_posix()`) in `file=` attributes: `from_xml_string` doesn't know the
  repo folder.
- The highlight tints scene geoms by id and keeps their alpha, so with meshes on it must also tint
  `<id>_visual`, or the highlight disappears.
