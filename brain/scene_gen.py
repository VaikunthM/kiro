"""Scene generator in P2's format: scenes/<mode>/<n>.json, 20 seeded scenes per failure mode.

Usage (repo root):  python -m brain.scene_gen            # writes scenes/<mode>/1..20.json
Object types and sizes follow scenes/mock/flat_stacked_000.json. Only stacked flat boxes get a small yaw;
everything else is yaw 0, so axis-aligned footprints are exact and objects never overlap by accident.
"""
import argparse
import json
import random
from pathlib import Path

TOTE = [0.40, 0.30, 0.15]
MODES = ("flat_stacked", "small_beside_big", "against_wall")
CLEARANCE = 0.0005  # gap between stacked items so they do not start interpenetrating

SIZES = {   # full extents (x, y, z) as in the mock scene
    "flat_box": (0.14, 0.10, 0.015),
    "sugar_box": (0.175, 0.089, 0.038),
    "tomato_soup_can": (0.066, 0.066, 0.101),
    "gelatin_box": (0.085, 0.073, 0.028),
    "potted_meat_can": (0.097, 0.082, 0.050),
    "foam_brick": (0.075, 0.050, 0.050),
}
BIG = ("sugar_box", "potted_meat_can")
SMALL = ("foam_brick", "gelatin_box", "tomato_soup_can")
FILLERS = ("sugar_box", "tomato_soup_can", "gelatin_box", "potted_meat_can", "foam_brick")


def _obj(typ, counter, size, pos, yaw=0.0):
    counter[typ] = counter.get(typ, 0) + 1
    return {"id": f"{typ}_{counter[typ]}", "type": typ, "size": [round(v, 4) for v in size],
            "pos": [round(v, 4) for v in pos], "yaw": round(yaw, 3)}


def _gap(a_pos, a_size, b):
    dx = abs(a_pos[0] - b["pos"][0]) - (a_size[0] + b["size"][0]) / 2
    dy = abs(a_pos[1] - b["pos"][1]) - (a_size[1] + b["size"][1]) / 2
    return max(dx, dy)


def _free(rng, size, placed, margin=0.01, gap=0.012, tries=300):
    hx = TOTE[0] / 2 - size[0] / 2 - margin
    hy = TOTE[1] / 2 - size[1] / 2 - margin
    for _ in range(tries):
        pos = [rng.uniform(-hx, hx), rng.uniform(-hy, hy), size[2] / 2]
        if all(_gap(pos, size, o) >= gap for o in placed):
            return pos
    return None


def _fill(rng, objs, counter, n):
    for _ in range(n):
        typ = rng.choice(FILLERS)
        pos = _free(rng, SIZES[typ], objs)
        if pos:
            objs.append(_obj(typ, counter, SIZES[typ], pos))


def _flat_stacked(rng):
    objs, counter = [], {}
    for _ in range(rng.randint(1, 2)):
        base_size = [SIZES["flat_box"][0] * rng.uniform(0.95, 1.05), SIZES["flat_box"][1] * rng.uniform(0.95, 1.05), 0.015]
        base = _free(rng, base_size, objs, margin=0.03, gap=0.02)
        if not base:
            continue
        z = 0.0
        for k in range(rng.randint(2, 3)):
            shrink = 1.0 if k == 0 else rng.uniform(0.78, 0.95)
            size = [base_size[0] * shrink, base_size[1] * shrink, 0.015]
            off = (0, 0) if k == 0 else (rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01))
            objs.append(_obj("flat_box", counter, size,
                             [base[0] + off[0], base[1] + off[1], z + size[2] / 2 + (CLEARANCE if k else 0)],
                             rng.uniform(-0.1, 0.1) if k else 0.0))
            z += size[2] + (CLEARANCE if k else 0)
    _fill(rng, objs, counter, rng.randint(3, 4))
    return objs


def _small_beside_big(rng):
    objs, counter = [], {}
    for _ in range(rng.randint(2, 3)):
        typ = rng.choice(BIG)
        size = SIZES[typ]
        pos = _free(rng, size, objs, margin=0.04, gap=0.05, tries=400)
        if not pos:
            continue
        objs.append(_obj(typ, counter, size, pos))
        for _ in range(rng.randint(1, 2)):
            styp = rng.choice(SMALL)
            ssz = SIZES[styp]
            side = rng.choice([-1, 1])
            gap = rng.uniform(0.005, 0.015)
            cand = [pos[0] + side * (size[0] / 2 + gap + ssz[0] / 2),
                    pos[1] + rng.uniform(-size[1] / 3, size[1] / 3), ssz[2] / 2]
            inside = abs(cand[0]) + ssz[0] / 2 < TOTE[0] / 2 - 0.005 and abs(cand[1]) + ssz[1] / 2 < TOTE[1] / 2 - 0.005
            if inside and all(_gap(cand, ssz, o) >= 0.003 for o in objs):
                objs.append(_obj(styp, counter, ssz, cand))
    _fill(rng, objs, counter, 2)
    return objs


def _against_wall(rng):
    objs, counter = [], {}
    hx, hy = TOTE[0] / 2, TOTE[1] / 2
    for _ in range(rng.randint(3, 4)):
        typ = rng.choice(("flat_box", "gelatin_box", "potted_meat_can", "sugar_box", "foam_brick"))
        size = SIZES[typ]
        gap = rng.uniform(0.0, 0.012)
        wall = rng.choice(("x-", "x+", "y-", "y+"))
        if wall in ("x-", "x+"):
            sx = -1 if wall == "x-" else 1
            pos = [sx * (hx - size[0] / 2 - gap), rng.uniform(-hy + size[1], hy - size[1]), size[2] / 2]
        else:
            sy = -1 if wall == "y-" else 1
            pos = [rng.uniform(-hx + size[0], hx - size[0]), sy * (hy - size[1] / 2 - gap), size[2] / 2]
        if all(_gap(pos, size, o) >= 0.008 for o in objs):
            objs.append(_obj(typ, counter, size, pos))
    _fill(rng, objs, counter, 2)
    return objs


RECIPES = {"flat_stacked": _flat_stacked, "small_beside_big": _small_beside_big, "against_wall": _against_wall}


def make_scene(mode, n, base_seed=1000):
    seed = base_seed + n
    for attempt in range(30):  # retry (deterministically) until the tote is reasonably full
        objs = RECIPES[mode](random.Random(f"{mode}-{seed}-{attempt}"))
        if len(objs) >= 5:
            break
    return {"scene_id": f"{mode}_{n:03d}", "mode": mode, "seed": seed,
            "tote": {"size": list(TOTE)}, "objects": objs}


def write_all(out_dir="scenes", per_mode=20, modes=MODES):
    out, paths = Path(out_dir), []
    for mode in modes:
        (out / mode).mkdir(parents=True, exist_ok=True)
        for n in range(1, per_mode + 1):
            p = out / mode / f"{n}.json"
            p.write_text(json.dumps(make_scene(mode, n), indent=1))
            paths.append(p)
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scenes")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--modes", nargs="*", default=list(MODES))
    a = ap.parse_args()
    print(f"wrote {len(write_all(a.out, a.n, a.modes))} scenes to {a.out}/")
