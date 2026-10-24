"""ClearPick simulator: tote + objects + overhead camera in MuJoCo, built from a scene JSON.
Implements contract 3 (Simulator API) of the shared contracts steering doc; scenes follow contract 2.

Scene JSON (scenes/<mode>/<n>.json), units metres:
  {"scene_id": "flat_stacked_007", "mode": "flat_stacked", "seed": 7,
   "tote": {"size": [0.40, 0.30, 0.15]},                 # optional, this is the default
   "objects": [{"id": "box_1", "type": "flat_box", "size": [0.14, 0.10, 0.015],
                "pos": [0.02, -0.03, 0.0075], "yaw": 0.2}]}
  size = full extents x, y, z; pos = object CENTRE (tote floor at z = 0, tote centred at origin);
  resting z = size z / 2. Types in CYLINDER_TYPES load as cylinders (size = diameter, diameter, height),
  everything else as a box. Optional extras: "shape", "quat" (w x y z, overrides yaw), "mass", "rgba".

API:
  sim.load(scene)                 -> state()        scene dict or path to a scene JSON
  sim.state()  -> {"scene_id", "mode", "tote": {"size"},
                   "objects": [{"id", "type", "size", "pos", "bbox": {"min", "max"}, "yaw", "shape"}]}
                   only objects still in the tote
  sim.grasp(item_id, grasp_type)  -> "clean" | "double" | "miss"; grasp_type in GRASP_TYPES
  sim.push(item_id, [dx, dy])     -> "ok"  (sim.last_push_blocked is True when the pusher had no room)
  sim.record(path)                start recording to an .mp4; sim.record(None) stops, saves, returns frames
  sim.clock()                     -> simulated seconds since load
"clean" removes exactly one object from the tote, "double" two or more, "miss" none.

Grasp mechanics (attach-on-touch, deliberately simple and deterministic):
  suction  The cup (radius CUP_RADIUS) comes down at the pick point and stops on the highest surface
           under it. Every object whose top is within TOUCH_TOL of that surface and that overlaps the cup
           attaches. A flat attached item (thinner than CLING_MAX_THICKNESS) also drags up a flat item it
           rests on when they overlap by CLING_MIN_OVERLAP or more: the ARMBench "flat items stacked"
           multi-pick. Anything resting on a lifted item rides along by physics.
  pinch    Fingers close across the item's shorter horizontal side. If a finger can't get down (wall or
           another object in the way) it is a miss; every object between the fingers at the item's
           height is lifted.
Outcome = number of objects that rose more than LIFT_CHECK.
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

TOTE_DEFAULT = {"size": [0.40, 0.30, 0.15], "wall": 0.01}
CYLINDER_TYPES = {"tomato_soup_can", "master_chef_can", "tuna_fish_can", "chips_can", "coffee_can",
                  "soup_can", "can", "cylinder", "jar", "cup", "mug"}
DENSITY = 400.0           # kg/m^3 for objects without a mass, roughly a packed retail box
PALETTE = [
    [0.90, 0.35, 0.30, 1], [0.95, 0.75, 0.25, 1], [0.35, 0.70, 0.40, 1], [0.30, 0.55, 0.90, 1],
    [0.65, 0.40, 0.85, 1], [0.95, 0.55, 0.20, 1], [0.40, 0.80, 0.80, 1], [0.85, 0.45, 0.65, 1],
]

GRASP_TYPES = ("suction_center", "suction_edge", "side_pinch")
CUP_RADIUS = 0.02
TOUCH_TOL = 0.006
CLING_MAX_THICKNESS = 0.045
CLING_MIN_OVERLAP = 0.5
FINGER_T, FINGER_W, JAW_CLEAR = 0.012, 0.03, 0.006
LIFT_ABOVE_WALL, LIFT_CHECK = 0.10, 0.10  # tools lift to wall height + 10 cm
TOOL_SPEED = 0.4           # m/s for approach and lift
PUSH_DIST, PUSH_SPEED = 0.10, 0.15
PUSHER_HALF = (0.008, 0.035, 0.010)   # thickness along push, width across, half-height
PARK = np.array([1.0, 1.0, 0.6])
FPS = 30


def _v(values):
    # snap denormals/tiny floats to 0: MuJoCo's XML parser rejects values like 2.2e-309
    return " ".join(f"{(0.0 if abs(x) < 1e-12 else x):.6g}" for x in values)


# ---------------------------------------------------------------- 2D geometry helpers

def _hull(points):
    pts = sorted(map(tuple, np.round(points, 9)))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _rect(center, axis_a, half_a, half_b):
    a = np.asarray(axis_a, float) / np.linalg.norm(axis_a)
    b = np.array([-a[1], a[0]])
    c = np.asarray(center, float)[:2]
    return np.array([c + sa * half_a * a + sb * half_b * b for sa, sb in ((1, 1), (-1, 1), (-1, -1), (1, -1))])


def _polys_overlap(p, q, margin=0.0):
    """Separating-axis test for two convex polygons; margin > 0 shrinks the test (needs real overlap)."""
    for poly in (p, q):
        for i in range(len(poly)):
            edge = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-edge[1], edge[0]])
            if not n.any():
                continue
            n /= np.linalg.norm(n)
            pp, qq = p @ n, q @ n
            if pp.max() <= qq.min() + margin or qq.max() <= pp.min() + margin:
                return False
    return True


def _point_in_poly(pt, poly):
    sign = None
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        c = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        s = c >= 0
        if sign is None:
            sign = s
        elif s != sign and abs(c) > 1e-12:
            return False
    return True


def _dist_point_poly(pt, poly):
    if _point_in_poly(pt, poly):
        return 0.0
    best = np.inf
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        ab = b - a
        t = np.clip(np.dot(pt - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
        best = min(best, np.linalg.norm(pt - (a + t * ab)))
    return best


def _overlap_fraction(small, big, n=12):
    """Fraction of polygon `small`'s area that lies inside `big` (grid sampling)."""
    lo, hi = small.min(0), small.max(0)
    xs, ys = np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n)
    inside = total = 0
    for x in xs:
        for y in ys:
            p = np.array([x, y])
            if _point_in_poly(p, small):
                total += 1
                inside += _point_in_poly(p, big)
    return inside / max(total, 1)


def _area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# ---------------------------------------------------------------- scene fields

def obj_shape(obj):
    if "shape" in obj:
        return obj["shape"]
    t = str(obj.get("type", "")).lower()
    return "cylinder" if t in CYLINDER_TYPES or t.endswith("_can") and t != "potted_meat_can" else "box"


def obj_quat(obj):
    if "quat" in obj:
        return list(obj["quat"])
    yaw = float(obj.get("yaw", 0.0))
    return [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]


def obj_mass(obj):
    if "mass" in obj:
        return obj["mass"]
    sx, sy, sz = obj["size"]
    vol = np.pi * (sx / 2) ** 2 * sz if obj_shape(obj) == "cylinder" else sx * sy * sz
    return max(0.01, DENSITY * vol)


def camera_poses(tote_size):
    """Overhead camera high enough to frame the tote at 4:3 with fovy 45; angled camera scaled to match."""
    L, W, H = tote_size
    tan = np.tan(np.radians(22.5))
    h = H + max((W / 2 + 0.04) / tan, (L / 2 + 0.04) / (tan * 4 / 3))  # frame the wall tops, not the floor
    s = max(L / 0.6, W / 0.4)
    return h, (0.0, -0.55 * s, 0.95 * s)


# ---------------------------------------------------------------- MJCF

def build_mjcf(scene: dict) -> str:
    """Return MJCF XML text for a scene dict."""
    tote = {**TOTE_DEFAULT, **scene.get("tote", {})}
    L, W, H = tote["size"]
    t = tote["wall"]

    walls = [  # (name, half-size, centre); inner floor surface is z = 0
        ("tote_floor", [L / 2 + t, W / 2 + t, t / 2], [0, 0, -t / 2]),
        ("tote_wall_xpos", [t / 2, W / 2 + t, H / 2], [L / 2 + t / 2, 0, H / 2]),
        ("tote_wall_xneg", [t / 2, W / 2 + t, H / 2], [-L / 2 - t / 2, 0, H / 2]),
        ("tote_wall_ypos", [L / 2, t / 2, H / 2], [0, W / 2 + t / 2, H / 2]),
        ("tote_wall_yneg", [L / 2, t / 2, H / 2], [0, -W / 2 - t / 2, H / 2]),
    ]
    wall_xml = "\n      ".join(
        f'<geom name="{n}" type="box" size="{_v(s)}" pos="{_v(p)}" material="tote"/>' for n, s, p in walls
    )

    body_xml = []
    for i, obj in enumerate(scene["objects"]):
        oid = obj["id"]
        sx, sy, sz = obj["size"]
        if obj_shape(obj) == "cylinder":
            geom = f'type="cylinder" size="{_v([sx / 2, sz / 2])}"'
        else:
            geom = f'type="box" size="{_v([sx / 2, sy / 2, sz / 2])}"'
        rgba = obj.get("rgba", PALETTE[i % len(PALETTE)])
        mass = obj_mass(obj)
        quat = obj_quat(obj)
        body_xml.append(
            f'<body name="{oid}" pos="{_v(obj["pos"])}" quat="{_v(quat)}">\n'
            f'      <freejoint name="{oid}_joint"/>\n'
            f'      <geom name="{oid}" {geom} mass="{mass:.5g}" rgba="{_v(rgba)}"/>\n'
            f"    </body>"
        )

    ph = PUSHER_HALF
    cam_h, cam_angled = camera_poses(tote["size"])
    return f"""<mujoco model="{scene.get('scene_id', 'scene')}">
  <option timestep="0.002" cone="elliptic" impratio="10"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.25 0.25 0.25" diffuse="0.35 0.35 0.35" specular="0 0 0"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.82 0.80" rgb2="0.74 0.74 0.72" width="512" height="512"/>
    <material name="ground" texture="grid" texrepeat="10 10"/>
    <material name="tote" rgba="0.18 0.32 0.62 1"/>
    <material name="tool" rgba="0.25 0.25 0.28 1"/>
  </asset>
  <default>
    <geom friction="0.8 0.01 0.001" solref="0.004 1"/>
  </default>
  <worldbody>
    <light pos="0.3 -0.3 1.5" dir="-0.2 0.2 -1" diffuse="0.45 0.45 0.45" specular="0.1 0.1 0.1" castshadow="true"/>
    <geom name="ground" type="plane" size="3 3 0.1" pos="0 0 {-t:.6g}" material="ground"/>
    <body name="tote">
      {wall_xml}
    </body>
    <camera name="overhead" pos="0 0 {cam_h:.4f}" xyaxes="1 0 0 0 1 0" fovy="45"/>
    <camera name="angled" pos="{_v(cam_angled)}" xyaxes="1 0 0 0 0.866 0.5" fovy="45"/>
    <body name="cup" mocap="true" pos="{_v(PARK)}">
      <geom type="cylinder" size="{CUP_RADIUS} 0.004" pos="0 0 0.004" rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.008 0.06" pos="0 0 0.068" material="tool" contype="0" conaffinity="0"/>
    </body>
    <body name="finger_a" mocap="true" pos="{_v(PARK)}">
      <geom type="box" size="{FINGER_T / 2} {FINGER_W / 2} 0.08" pos="0 0 0.08" material="tool" contype="0" conaffinity="0"/>
    </body>
    <body name="finger_b" mocap="true" pos="{_v(PARK)}">
      <geom type="box" size="{FINGER_T / 2} {FINGER_W / 2} 0.08" pos="0 0 0.08" material="tool" contype="0" conaffinity="0"/>
    </body>
    <body name="pusher" mocap="true" pos="{_v(PARK + [0.3, 0, 0])}">
      <geom name="pusher" type="box" size="{_v(ph)}" material="tool" friction="0.3 0.01 0.001"/>
      <geom type="cylinder" size="0.006 0.06" pos="0 0 0.07" material="tool" contype="0" conaffinity="0"/>
    </body>
    {chr(10).join('    ' + b for b in body_xml).strip()}
  </worldbody>
</mujoco>
"""


# ---------------------------------------------------------------- simulator

class Sim:
    def __init__(self, width: int = 640, height: int = 480, camera: str = "overhead"):
        self.width, self.height, self.camera = width, height, camera
        self.model = self.data = self.scene = None
        self._renderer = None
        self.frames, self._record_path, self._recording = [], None, False
        self.last_push_blocked = False

    # ---- load / step / state ----
    def load(self, scene, settle_s: float = 1.0):
        """Load a scene dict or a path to a scene JSON, then let objects settle."""
        if isinstance(scene, (str, Path)):
            scene = json.loads(Path(scene).read_text())
        self.scene = scene
        self.tote = {**TOTE_DEFAULT, **scene.get("tote", {})}
        self.objs = {o["id"]: o for o in scene["objects"]}
        self.removed = []
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(scene))
        self.data = mujoco.MjData(self.model)
        self._qadr = {i: self.model.joint(f"{i}_joint").qposadr[0] for i in self.objs}
        self._vadr = {i: self.model.joint(f"{i}_joint").dofadr[0] for i in self.objs}
        self._mocap = {n: self.model.body(n).mocapid[0] for n in ("cup", "finger_a", "finger_b", "pusher")}
        self._step_count = 0
        self._frame_every = max(1, round(1 / (FPS * self.model.opt.timestep)))
        mujoco.mj_forward(self.model, self.data)
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = mujoco.Renderer(self.model, self.height, self.width)
        self.frames, self._record_path, self._recording = [], None, False
        self.last_push_blocked = False
        self.step(settle_s)
        self._t0 = float(self.data.time)
        self._lift_z = self.tote["size"][2] + LIFT_ABOVE_WALL
        return self.state()

    def clock(self) -> float:
        """Simulated seconds since load() finished settling."""
        return round(float(self.data.time) - self._t0, 3)

    def step(self, seconds: float, follow=None):
        """Advance physics. `follow` = (tool_body, {obj_id: offset}) keeps attached objects on the tool."""
        dt = self.model.opt.timestep
        for _ in range(max(1, int(round(seconds / dt)))):
            self._physics_step(follow)

    def _physics_step(self, follow=None, tool_vel=np.zeros(3)):
        if follow:
            tool, attached = follow
            tpos = self.data.mocap_pos[self._mocap[tool]]
            for oid, off in attached.items():
                a, v = self._qadr[oid], self._vadr[oid]
                self.data.qpos[a:a + 3] = tpos + off
                self.data.qvel[v:v + 3] = tool_vel
                self.data.qvel[v + 3:v + 6] = 0
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        if self._recording and self._step_count % self._frame_every == 0:
            self.frames.append(self.render())

    def state(self) -> dict:
        """Each object's pose and world-frame axis-aligned bounding box (picked objects are left out)."""
        objects = []
        for oid, obj in self.objs.items():
            if oid in self.removed:
                continue
            corners = self._corners(oid)
            body = self.data.body(oid)
            R = body.xmat.reshape(3, 3)
            objects.append({
                "id": oid,
                "type": obj.get("type", obj_shape(obj)),
                "size": list(obj["size"]),
                "pos": body.xpos.round(4).tolist(),
                "bbox": {"min": corners.min(0).round(4).tolist(), "max": corners.max(0).round(4).tolist()},
                "yaw": round(float(np.arctan2(R[1, 0], R[0, 0])), 4),
                "shape": obj_shape(obj),
            })
        return {"scene_id": self.scene.get("scene_id"), "mode": self.scene.get("mode"),
                "tote": {"size": list(self.tote["size"])}, "objects": objects}

    # ---- geometry of current poses ----
    def _corners(self, oid):
        body = self.data.body(oid)
        half = np.array(self.objs[oid]["size"]) / 2
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        return (signs * half) @ body.xmat.reshape(3, 3).T + body.xpos

    def _footprint(self, oid):
        return _hull(self._corners(oid)[:, :2])

    def _zrange(self, oid):
        c = self._corners(oid)[:, 2]
        return c.min(), c.max()

    def _in_tote(self, oid):
        return oid not in self.removed

    def _others(self, oid):
        return [o for o in self.objs if o != oid and self._in_tote(o)]

    def _horizontal_axes(self, oid):
        """Object's local x and y axes projected to the table, with their full extents."""
        R = self.data.body(oid).xmat.reshape(3, 3)
        sx, sy, _ = self.objs[oid]["size"]
        ax = R[:2, 0] / max(np.linalg.norm(R[:2, 0]), 1e-9)
        ay = R[:2, 1] / max(np.linalg.norm(R[:2, 1]), 1e-9)
        return (ax, sx), (ay, sy)

    def _inner_bounds(self):
        L, W, _ = self.tote["size"]
        return np.array([L / 2, W / 2])

    def _rect_hits_wall(self, rect):
        return bool((np.abs(rect) > self._inner_bounds() + 1e-4).any())

    # ---- tool motion ----
    def _move(self, tool, target, speed=TOOL_SPEED, follow=None):
        mid = self._mocap[tool]
        start = self.data.mocap_pos[mid].copy()
        target = np.asarray(target, float)
        dist = np.linalg.norm(target - start)
        n = max(1, int(dist / (speed * self.model.opt.timestep)))
        vel = (target - start) / (n * self.model.opt.timestep)
        for k in range(1, n + 1):
            self.data.mocap_pos[mid] = start + (target - start) * k / n
            self._physics_step(follow, vel)

    def _teleport(self, tool, pos):
        self.data.mocap_pos[self._mocap[tool]] = pos

    def _park(self):
        for name, off in (("cup", 0), ("finger_a", 0.1), ("finger_b", 0.2), ("pusher", 0.3)):
            self._teleport(name, PARK + [off, 0, 0])

    # ---- grasp ----
    def grasp(self, item_id: str, grasp_type: str) -> str:
        if grasp_type not in GRASP_TYPES:
            raise ValueError(f"grasp_type must be one of {GRASP_TYPES}, got {grasp_type!r}")
        if not self._in_tote(item_id):
            raise ValueError(f"{item_id} is not in the tote")
        z_before = {o: self.data.body(o).xpos[2] for o in self.objs if self._in_tote(o)}

        if grasp_type == "side_pinch":
            attached, tool, touch_z = self._plan_pinch(item_id)
        else:
            attached, tool, touch_z = self._plan_suction(item_id, grasp_type)

        if attached:
            tpos = self.data.mocap_pos[self._mocap[tool]].copy()
            offsets = {o: self.data.body(o).xpos.copy() - tpos for o in attached}
            for o in attached:  # fingers / cup hold, so the attached group moves with the tool
                self.data.qvel[self._vadr[o]:self._vadr[o] + 6] = 0
            self._lift(tool, offsets)
        else:
            self._lift(tool, {})

        self.step(0.4, follow=(tool, {o: self.data.body(o).xpos.copy() - self.data.mocap_pos[self._mocap[tool]]
                                      for o in attached}) if attached else None)
        lifted = [o for o, z in z_before.items() if self.data.body(o).xpos[2] - z > LIFT_CHECK]
        for o in lifted:
            self._remove(o)
        self._park()
        self.step(0.3)
        if len(lifted) == 0:
            return "miss"
        return "clean" if len(lifted) == 1 else "double"

    def _lift(self, tool, offsets):
        follow = (tool, offsets) if offsets else None
        if tool == "finger_a":  # move both fingers together; objects follow finger_a
            a0 = self.data.mocap_pos[self._mocap["finger_a"]].copy()
            b0 = self.data.mocap_pos[self._mocap["finger_b"]].copy()
            dz = self._lift_z - a0[2]
            n = max(1, int(abs(dz) / (TOOL_SPEED * self.model.opt.timestep)))
            vel = np.array([0, 0, dz / (n * self.model.opt.timestep)])
            for k in range(1, n + 1):
                self.data.mocap_pos[self._mocap["finger_a"]] = a0 + [0, 0, dz * k / n]
                self.data.mocap_pos[self._mocap["finger_b"]] = b0 + [0, 0, dz * k / n]
                self._physics_step(follow, vel)
        else:
            p = self.data.mocap_pos[self._mocap[tool]].copy()
            self._move(tool, [p[0], p[1], self._lift_z], follow=follow)

    def _suction_point(self, item_id, grasp_type):
        pos = self.data.body(item_id).xpos[:2].copy()
        if grasp_type == "suction_center":
            return pos
        # off-centre: pick the side of the item with the most clearance from its neighbours
        (ax, sx), (ay, sy) = self._horizontal_axes(item_id)
        cands = [pos + s * 0.35 * e * a for a, e in ((ax, sx), (ay, sy)) for s in (1, -1)]
        others = [self._footprint(o) for o in self._others(item_id)]
        if not others:
            return cands[0]
        return max(cands, key=lambda p: min(_dist_point_poly(p, f) for f in others))

    def _plan_suction(self, item_id, grasp_type):
        pt = self._suction_point(item_id, grasp_type)
        touched = []
        for o in self.objs:
            if self._in_tote(o) and _dist_point_poly(pt, self._footprint(o)) <= CUP_RADIUS:
                touched.append((self._zrange(o)[1], o))
        self._teleport("cup", [pt[0], pt[1], self._lift_z])
        self._step_recorded(0.1)
        if not touched:
            self._move("cup", [pt[0], pt[1], 0.0])
            return [], "cup", 0.0
        touch_z = max(z for z, _ in touched)
        attached = [o for z, o in touched if z >= touch_z - TOUCH_TOL]
        attached += self._cling(attached)
        self._move("cup", [pt[0], pt[1], touch_z])
        return attached, "cup", touch_z

    def _cling(self, attached):
        extra = []
        for a in attached:
            if min(self.objs[a]["size"]) > CLING_MAX_THICKNESS:
                continue
            a_bottom = self._zrange(a)[0]
            fa = self._footprint(a)
            for o in self._others(a):
                if o in attached or o in extra or min(self.objs[o]["size"]) > CLING_MAX_THICKNESS:
                    continue
                if abs(self._zrange(o)[1] - a_bottom) > TOUCH_TOL:
                    continue
                fo = self._footprint(o)
                small, big = (fa, fo) if _area(fa) <= _area(fo) else (fo, fa)
                if _overlap_fraction(small, big) >= CLING_MIN_OVERLAP:
                    extra.append(o)
        return extra

    def _plan_pinch(self, item_id):
        (ax, sx), (ay, sy) = self._horizontal_axes(item_id)
        (jaw, e_jaw), (_, e_side) = ((ax, sx), (ay, sy)) if sx <= sy else ((ay, sy), (ax, sx))
        c = self.data.body(item_id).xpos.copy()
        z_lo, z_hi = self._zrange(item_id)
        grip_z = z_lo + 0.005
        off = e_jaw / 2 + JAW_CLEAR + FINGER_T / 2
        fingers = [_rect(c + s * off * np.append(jaw, 0), jaw, FINGER_T / 2, FINGER_W / 2) for s in (1, -1)]
        yaw = np.arctan2(jaw[1], jaw[0])
        quat = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        for name, s in (("finger_a", 1), ("finger_b", -1)):
            p = c + s * off * np.append(jaw, 0)
            self.data.mocap_quat[self._mocap[name]] = quat
            self._teleport(name, [p[0], p[1], self._lift_z])
        self._step_recorded(0.1)

        blocked = any(self._rect_hits_wall(f) for f in fingers) or any(
            _polys_overlap(f, self._footprint(o), margin=1e-4) and self._zrange(o)[1] > grip_z
            for f in fingers for o in self._others(item_id)
        )
        if blocked:  # fingers come down and stop on the obstacle
            self._move_pair(grip_z + 0.06)
            return [], "finger_a", grip_z
        self._move_pair(grip_z)
        between = _rect(c, jaw, e_jaw / 2 + JAW_CLEAR, FINGER_W / 2)
        attached = [item_id] + [
            o for o in self._others(item_id)
            if _polys_overlap(between, self._footprint(o), margin=1e-4)
            and self._zrange(o)[1] > grip_z + 0.005 and self._zrange(o)[0] < z_hi - 0.005
        ]
        return attached, "finger_a", grip_z

    def _move_pair(self, z):
        a0 = self.data.mocap_pos[self._mocap["finger_a"]].copy()
        b0 = self.data.mocap_pos[self._mocap["finger_b"]].copy()
        dz = z - a0[2]
        n = max(1, int(abs(dz) / (TOOL_SPEED * self.model.opt.timestep)))
        for k in range(1, n + 1):
            self.data.mocap_pos[self._mocap["finger_a"]] = a0 + [0, 0, dz * k / n]
            self.data.mocap_pos[self._mocap["finger_b"]] = b0 + [0, 0, dz * k / n]
            self._physics_step()

    def _step_recorded(self, seconds):
        self.step(seconds)

    def _remove(self, oid):
        """Put a picked object on the floor outside the tote, out of the overhead camera's view."""
        k = len(self.removed)
        self.removed.append(oid)
        a, v = self._qadr[oid], self._vadr[oid]
        sz = self.objs[oid]["size"][2]
        self.data.qpos[a:a + 7] = [1.2 + 0.2 * (k % 6), 1.2 + 0.2 * (k // 6), -self.tote["wall"] + sz / 2 + 0.002, 1, 0, 0, 0]
        self.data.qvel[v:v + 6] = 0

    # ---- push ----
    def push(self, item_id: str, direction) -> str:
        self.last_push_blocked = False
        if not self._in_tote(item_id):
            raise ValueError(f"{item_id} is not in the tote")
        d = np.asarray(direction, float)[:2]
        d /= np.linalg.norm(d)
        c = self.data.body(item_id).xpos.copy()
        z_lo, z_hi = self._zrange(item_id)
        h = PUSHER_HALF[2]
        z = z_lo + min(h + 0.002, (z_hi - z_lo) / 2)
        back = np.max(c[:2] @ d - self._footprint(item_id) @ d)
        start = c[:2] - d * (back + PUSHER_HALF[0] + 0.012)
        start_rect = _rect(start, d, PUSHER_HALF[0], PUSHER_HALF[1])

        yaw = np.arctan2(d[1], d[0])
        self.data.mocap_quat[self._mocap["pusher"]] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self._teleport("pusher", [start[0], start[1], self._lift_z])
        self.step(0.1)

        blocked = self._rect_hits_wall(start_rect) or any(
            _polys_overlap(start_rect, self._footprint(o), margin=1e-4) and self._zrange(o)[1] > z - h
            for o in self._others(item_id)
        )
        if blocked:
            self._move("pusher", [start[0], start[1], z + 0.08])
            self._move("pusher", [start[0], start[1], self._lift_z])
            self._park()
            self.step(0.2)
            self.last_push_blocked = True
            return "ok"

        before = c[:2].copy()
        self._move("pusher", [start[0], start[1], z])
        end = start + d * PUSH_DIST
        lim = self._inner_bounds() - PUSHER_HALF[1]
        end = np.clip(end, -lim, lim)
        self._move("pusher", [end[0], end[1], z], speed=PUSH_SPEED)
        self._move("pusher", [end[0], end[1], self._lift_z])
        self._park()
        self.step(0.4)
        moved = np.linalg.norm(self.data.body(item_id).xpos[:2] - before)
        self.last_push_blocked = bool(moved <= 0.01)
        return "ok"

    # ---- rendering / video ----
    def render(self, path=None, camera: str = None) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=camera or self.camera)
        img = self._renderer.render()
        if path:
            import imageio.v3 as iio
            iio.imwrite(path, img)
        return img

    def record(self, path) -> int:
        """record(path) starts capturing frames for an .mp4 at path; record(None) stops and saves it.
        Returns the number of frames saved (0 when starting, or when nothing was being recorded)."""
        if path is not None:
            self._record_path, self.frames, self._recording = Path(path), [self.render()], True
            return 0
        if not self._recording:
            return 0
        import imageio.v2 as iio2
        self.step(0.3)  # short tail so the last move is visible
        out, frames = self._record_path, self.frames
        self._recording, self._record_path, self.frames = False, None, []
        out.parent.mkdir(parents=True, exist_ok=True)
        iio2.mimwrite(out, frames, fps=FPS, codec="libx264", macro_block_size=8,
                      pixelformat="yuv420p", ffmpeg_params=["-preset", "veryfast", "-crf", "23"])
        return len(frames)


if __name__ == "__main__":
    # python -m sim.sim scenes/mock/flat_stacked_000.json
    scene_path = Path(sys.argv[1])
    out = Path("out")
    out.mkdir(exist_ok=True)
    scene = json.loads(scene_path.read_text())
    (out / f"{scene_path.stem}.xml").write_text(build_mjcf(scene))
    sim = Sim()
    st = sim.load(scene)
    sim.render(out / f"{scene_path.stem}_overhead.png")
    sim.render(out / f"{scene_path.stem}_angled.png", camera="angled")
    for o in st["objects"]:
        print(f"{o['id']:22s} {o['shape']:8s} pos={o['pos']}  bbox_z=[{o['bbox']['min'][2]:.3f}, {o['bbox']['max'][2]:.3f}]")
    print(f"wrote out/{scene_path.stem}.xml and two PNGs")
