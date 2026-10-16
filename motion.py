"""J / Z motion model: data, features, augmentation, network.

A recording (from the page's Collect panel) is a list of frames with a time stamp (ms) and 21 hand points. A *window* (last W ms) is resampled by time
to T steps, so the result does not depend on the camera frame rate. Per step: hand shape (wrist-origin, hand-size scaled, 63) + path (position
relative to the window's mean wrist position, in hand sizes, 63) + a 'hand present' flag = 127 numbers. Classes: J, Z, other.
"""
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["J", "Z", "other"]
T, WINDOW_MS, MAX_GAP_MS = 24, 2000.0, 350.0


def window_features(t, P, t_end, W=WINDOW_MS, steps=T, max_gap=MAX_GAP_MS):
    """t: (n,) ms of frames that HAVE a hand, sorted; P: (n,21,3) pixel-like coords. Returns (steps,127) float32 or None."""
    t = np.asarray(t, float)
    if len(t) < 3:
        return None
    grid = np.linspace(t_end - W, t_end, steps)
    flat = np.asarray(P, float).reshape(len(t), 63)
    Q = np.stack([np.interp(grid, t, flat[:, j]) for j in range(63)], 1).reshape(steps, 21, 3)
    nearest = np.abs(grid[:, None] - t[None, :]).min(1)
    present = (nearest <= max_gap).astype(np.float32)
    if present.sum() < 3:
        return None
    size = np.linalg.norm(Q[:, 9, :2] - Q[:, 0, :2], axis=1)
    med = float(np.median(size[present > 0])) or 1.0
    size = np.where(size < 1e-6, med, size)
    shape = (Q - Q[:, :1]) / size[:, None, None]
    ref = Q[present > 0, 0].mean(0)
    path = (Q - ref) / med
    f = np.concatenate([shape.reshape(steps, 63), path.reshape(steps, 63), present[:, None]], 1).astype(np.float32)
    f[present == 0] = 0
    f[:, -1] = present
    return f


def load_recordings(root=None):
    """-> list of dict(label, name, t (ms, hand frames only), P (n,21,3), order)"""
    root = Path(root) if root else ROOT / "data" / "collected"
    recs = []
    for f in sorted(glob.glob(str(root / "*" / "*" / "frames.json"))):
        d = json.load(open(f, encoding="utf-8"))
        w, h = d["video"] or [640, 480]
        fr = [x for x in d["frames"] if x.get("lm")]
        if len(fr) < 5:
            continue
        P = np.array([x["lm"] for x in fr], float).reshape(-1, 21, 3) * np.array([w, h, w])
        recs.append({"label": d["label"] if d["label"] in ("J", "Z") else "other", "name": Path(f).parent.name,
                     "t": np.array([x["t"] for x in fr], float), "P": P})
    return recs


def synthetic_others(n, seed=0):
    """Hand held still / changing shape / moving as a rigid block: everything that is NOT a J or Z, built from public landmark rows."""
    import pandas as pd
    rng = np.random.RandomState(seed)
    df = pd.read_csv(ROOT / "data" / "processed" / "landmarks_asl_alphabet.csv", usecols=[f"{a}{i}" for i in range(21) for a in "xyz"] + ["letter"])
    df = df[~df.letter.isin(["J", "Z"])]
    H = df[[f"{a}{i}" for i in range(21) for a in "xyz"]].to_numpy().reshape(-1, 21, 3)
    out = []
    for _ in range(n):
        a, b = H[rng.randint(len(H))], H[rng.randint(len(H))]
        fps = rng.uniform(5, 15)
        t = np.arange(0, 2000, 1000 / fps) + rng.uniform(0, 30, 1)
        s = np.linspace(0, 1, len(t))
        k = np.clip((s - rng.uniform(0.2, 0.6)) / rng.uniform(0.1, 0.4), 0, 1)[:, None, None] if rng.rand() < 0.4 else np.zeros((len(t), 1, 1))
        P = (1 - k) * a[None] + k * b[None]
        size = np.linalg.norm(P[:, 9, :2] - P[:, 0, :2], axis=1).mean()
        amp, ang = rng.uniform(0, 2.5) * size * (rng.rand() < 0.6), rng.uniform(0, 2 * np.pi)
        prof = np.sin(s * np.pi * rng.uniform(0.5, 2))[:, None]
        P = P + prof[:, :, None] * amp * np.array([np.cos(ang), np.sin(ang), 0])
        out.append({"label": "other", "name": "synthetic", "t": t, "P": P + rng.normal(0, 0.004 * size, P.shape)})
    return out


def augment(rec, rng):
    """Random crop of the start/end, lower frame rate, noise, small rotation, mirror -> (steps,127) features."""
    t, P = rec["t"], rec["P"].copy()
    keep, last = [], -1e9
    gap = 1000 / rng.uniform(5, 15)
    for i, ti in enumerate(t):
        if ti - last >= gap * rng.uniform(0.6, 1.0) and rng.rand() > 0.08:
            keep.append(i)
            last = ti
    if len(keep) < 3:
        keep = list(range(len(t)))
    t, P = t[keep], P[keep]
    t_end = t[-1] - rng.uniform(0, 250) if rec["name"] != "synthetic" else 2000.0
    size = np.linalg.norm(P[:, 9, :2] - P[:, 0, :2], axis=1).mean() + 1e-6
    P = P + rng.normal(0, 0.012 * size, P.shape)
    th = rng.uniform(-0.2, 0.2)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    P = P @ R.T
    if rng.rand() < 0.5:
        P[..., 0] = -P[..., 0]
    return window_features(t, P, max(t_end, t[2]), W=WINDOW_MS * rng.uniform(0.75, 1.0))


def full_features(rec):
    return window_features(rec["t"], rec["P"], rec["t"][-1])


def build_model():
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv1d(127, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 96, 5, padding=4, dilation=2), nn.BatchNorm1d(96), nn.ReLU(), nn.Dropout(0.2),
                nn.Conv1d(96, 96, 3, padding=1), nn.BatchNorm1d(96), nn.ReLU())
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(192, len(CLASSES)))

        def forward(self, x):            # x: (B, T, 127)
            h = self.body(x.transpose(1, 2))
            return self.head(torch_cat([h.mean(2), h.amax(2)]))

    import torch
    torch_cat = lambda xs: torch.cat(xs, 1)
    return Net()
