"""Checks that the vectorised v15 descriptor equals the original v14 loop version."""
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import airink as A  # noqa: E402


def v14_descriptor(segments_list, weights_list=None):
    segs = np.array(segments_list, dtype=np.float32)
    weights = np.ones(len(segs), dtype=np.float32) if weights_list is None else np.asarray(weights_list, dtype=np.float32)
    if len(segs) >= 12 and np.max(weights) > 0.42:
        keep = weights >= 0.30
        if np.sum(keep) >= 8:
            segs = segs[keep]
            weights = weights[keep]
    pts = segs.reshape(-1, 2)
    xmin, ymin = np.min(pts, axis=0)
    xmax, ymax = np.max(pts, axis=0)
    w = max(0.02, xmax - xmin)
    h = max(0.02, ymax - ymin)
    scale = max(w, h)
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    norm_segs = np.empty_like(segs)
    norm_segs[:, :, 0] = 0.5 + (segs[:, :, 0] - cx) / (scale * 1.15)
    norm_segs[:, :, 1] = 0.5 + (segs[:, :, 1] - cy) / (scale * 1.15)
    norm_segs = np.clip(norm_segs, 0.0, 0.999)
    grid_size = 12
    raster = np.zeros((grid_size, grid_size), dtype=np.float32)
    hog = np.zeros((3, 3, 6), dtype=np.float32)
    v_strips = np.zeros((3, 8), dtype=np.float32)
    raw_dy = segs[:, 1, 1] - segs[:, 0, 1]
    dx = norm_segs[:, 1, 0] - norm_segs[:, 0, 0]
    dy = norm_segs[:, 1, 1] - norm_segs[:, 0, 1]
    lengths = np.hypot(dx, dy)
    orientations = (np.arctan2(dy, dx) % np.pi) / np.pi * 6.0
    box_x0 = np.clip((segs[:, 0, 0] - xmin) / w, 0.0, 0.999)
    box_y0 = np.clip((segs[:, 0, 1] - ymin) / h, 0.0, 0.999)
    for idx in range(len(norm_segs)):
        w_i = weights[idx] * (lengths[idx] + 0.005)
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            px = norm_segs[idx, 0, 0] + alpha * dx[idx]
            py = norm_segs[idx, 0, 1] + alpha * dy[idx]
            gx = min(grid_size - 1, max(0, int(px * grid_size)))
            gy = min(grid_size - 1, max(0, int(py * grid_size)))
            raster[gy, gx] += w_i
        mx = 0.5 * (norm_segs[idx, 0, 0] + norm_segs[idx, 1, 0])
        my = 0.5 * (norm_segs[idx, 0, 1] + norm_segs[idx, 1, 1])
        hog[min(2, max(0, int(my * 3))), min(2, max(0, int(mx * 3))), min(5, max(0, int(orientations[idx])))] += w_i
        v_strips[min(2, int(box_y0[idx] * 3)), min(7, int(box_x0[idx] * 8))] += weights[idx] * abs(raw_dy[idx])
    raster = gaussian_filter(raster, sigma=0.65)
    if np.max(raster) > 0:
        raster /= np.max(raster)
    if np.linalg.norm(hog) > 0:
        hog /= (np.linalg.norm(hog) + 1e-5)
    if np.max(v_strips) > 0:
        v_strips /= np.max(v_strips)
    return np.concatenate([raster.ravel(), hog.ravel(), v_strips.ravel()])


rng = np.random.RandomState(0)
worst = 0.0
for trial in range(200):
    n = rng.randint(2, 120)
    start = rng.rand(2) * 0.5
    walk = start + np.cumsum(rng.randn(n + 1, 2) * 0.02, axis=0)
    segs = np.stack([walk[:-1], walk[1:]], axis=1)
    wts = rng.rand(n) if trial % 2 else None
    new = A.extract_letter_descriptor(segs, wts)[:222]
    old = v14_descriptor(segs, wts)
    worst = max(worst, float(np.max(np.abs(new - old))))
print(f"max abs difference over 200 random letters (raster+HOG+strips): {worst:.2e}")
assert worst < 1e-4, "vectorised descriptor differs from v14"
print("OK")
