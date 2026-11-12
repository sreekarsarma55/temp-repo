"""Make fake Air Ink training recordings in the real JSONL format (for smoke tests only).

The cursor is driven through block letters; orientation angles are the inverse of the trace()
mapping plus noise. Includes clears, undos and occasional recenters. Accuracy on this data says
nothing about the real exam; it only proves the code runs end to end.

    python tests/make_synthetic.py --n 150 --out data/raw/synthetic.jsonl
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from airink import ALPHABET  # noqa: E402

T = {
    "A": [[(0, 1), (0.5, 0), (1, 1)], [(0.22, 0.6), (0.78, 0.6)]],
    "C": [[(0.95, 0.15), (0.6, 0), (0.25, 0.05), (0.02, 0.5), (0.25, 0.95), (0.6, 1), (0.95, 0.85)]],
    "E": [[(1, 0), (0, 0), (0, 1), (1, 1)], [(0, 0.5), (0.75, 0.5)]],
    "F": [[(1, 0), (0, 0), (0, 1)], [(0, 0.5), (0.75, 0.5)]],
    "H": [[(0, 0), (0, 1)], [(1, 0), (1, 1)], [(0, 0.5), (1, 0.5)]],
    "J": [[(0.1, 0), (1, 0)], [(0.65, 0), (0.65, 0.8), (0.45, 1), (0.15, 0.95), (0, 0.75)]],
    "K": [[(0, 0), (0, 1)], [(0.95, 0), (0.05, 0.55), (1, 1)]],
    "M": [[(0, 1), (0, 0), (0.5, 0.6), (1, 0), (1, 1)]],
    "N": [[(0, 1), (0, 0), (1, 1), (1, 0)]],
    "P": [[(0, 1), (0, 0), (0.75, 0.03), (0.95, 0.25), (0.75, 0.48), (0, 0.5)]],
    "R": [[(0, 1), (0, 0), (0.75, 0.03), (0.95, 0.25), (0.75, 0.48), (0, 0.5)], [(0.35, 0.5), (1, 1)]],
    "T": [[(0, 0), (1, 0)], [(0.5, 0), (0.5, 1)]],
    "U": [[(0, 0), (0, 0.75), (0.25, 1), (0.75, 1), (1, 0.75), (1, 0)]],
    "V": [[(0, 0), (0.5, 1), (1, 0)]],
    "W": [[(0, 0), (0.25, 1), (0.5, 0.35), (0.75, 1), (1, 0)]],
    "X": [[(0, 0), (1, 1)], [(1, 0), (0, 1)]],
    "Y": [[(0, 0), (0.5, 0.5), (1, 0)], [(0.5, 0.5), (0.5, 1)]],
}
assert set(T) == set(ALPHABET)


def densify(poly, speed, rng, dt_range=(30, 60)):
    pts = [np.array(poly[0], float)]
    out = []
    for a, b in zip(poly[:-1], poly[1:]):
        a, b = np.array(a, float), np.array(b, float)
        d = np.linalg.norm(b - a)
        n = max(1, int(d / speed / 0.045))
        for i in range(1, n + 1):
            out.append(a + (b - a) * i / n)
    return [pts[0]] + out


def make_record(rng):
    k = int(rng.choice([3, 4, 5]))
    code = "".join(rng.choice(list(ALPHABET), size=k))
    w = int(rng.choice([360, 384, 393, 412]))
    canvas = {"width": w, "height": 300, "screen": "portrait-primary"}
    left = rng.uniform(0.1, 0.2)
    cell = (0.8 - left) / k
    lw, lh = cell * rng.uniform(0.6, 0.85), rng.uniform(0.3, 0.45)
    top = rng.uniform(0.25, 0.35)
    slant = rng.uniform(-0.15, 0.15)
    v_down, v_up = rng.uniform(0.25, 0.45), rng.uniform(0.5, 0.9)
    t = 1000.0
    pos = np.array([0.5, 0.5])
    samples, strokes, controls = [], [], []

    def emit(p, pen):
        nonlocal t
        gx = (p[0] - 0.5) * 29 / 0.46
        gy = (p[1] - 0.5) * 34 / 0.46
        samples.append([round(t, 1), round(10 + gy + rng.normal(0, 0.25), 3),
                        round(gx + rng.normal(0, 0.25), 3), int(pen)])
        t += rng.uniform(30, 60)

    def move(to, pen, speed):
        nonlocal pos
        for p in densify([pos, to], speed, rng)[1:]:
            emit(p, pen)
        pos = np.array(to, float)

    emit(pos, 0)
    if rng.rand() < 0.15:  # scribble, then clear
        for _ in range(3):
            move(rng.uniform(0.2, 0.8, 2), 1, v_down)
        controls.append({"t": t, "type": "clear"})
        t += 400
    for li, ch in enumerate(code):
        x0 = left + li * cell
        for poly in T[ch]:
            p_canvas = [(x0 + (px + slant * (1 - py)) * lw, top + py * lh) for px, py in poly]
            move(p_canvas[0], 0, v_up)
            stroke = [list(map(float, pos))]
            for p in densify(p_canvas, v_down, rng)[1:]:
                emit(p, 1)
                stroke.append([float(p[0]), float(p[1])])
            pos = np.array(p_canvas[-1], float)
            strokes.append(stroke)
            if rng.rand() < 0.06:  # pen up, junk stroke, then undo
                emit(pos, 0)
                move(pos + rng.uniform(-0.05, 0.05, 2), 1, v_down)
                emit(pos, 0)
                controls.append({"t": t, "type": "undo"})
                t += 300
    move([0.5, 0.6], 0, v_up)
    rid = "".join(rng.choice(list("0123456789abcdef"), size=12))
    return {"id": rid, "length": k, "code": code, "canvas": canvas, "samples": samples,
            "controls": controls, "strokes": strokes}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/raw/synthetic.jsonl")
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for _ in range(a.n):
            f.write(json.dumps(make_record(rng)) + "\n")
    print(f"wrote {a.n} synthetic recordings to {a.out}")
