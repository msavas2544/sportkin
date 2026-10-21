"""Synthetic side-view CMJ used across tests.

A stick figure performs: quiet standing -> countermovement to a known knee
flexion -> ballistic flight of known duration -> landing flexion. Because every
quantity is known, the analysis can be checked against ground truth.
"""

import numpy as np
import pytest

from sportkin.jump import G
from sportkin.pose import LANDMARKS, PoseSeries

THIGH = SHANK = 200.0
TRUNK = 260.0
PX_PER_M = 500.0
GROUND_Y = 900.0
ANKLE_X = 400.0


def _smooth(a, b, n):
    """Cosine ramp from a to b over n frames."""
    t = np.linspace(0, 1, n)
    return a + (b - a) * (1 - np.cos(np.pi * t)) / 2


def make_cmj(fps=240, flight=0.5, cm_depth=90.0, land_depth=40.0,
             right_visibility=0.6, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    n_stand, n_down, n_up = int(0.5 * fps), int(0.4 * fps), int(0.25 * fps)
    n_air, n_land, n_rec = int(round(flight * fps)), int(0.15 * fps), int(0.4 * fps)
    knee = np.concatenate([
        np.zeros(n_stand), _smooth(0, cm_depth, n_down), _smooth(cm_depth, 0, n_up),
        np.zeros(n_air), _smooth(0, land_depth, n_land), _smooth(land_depth, 10, n_rec),
    ])
    t_air = np.arange(n_air) / fps
    v0 = G * flight / 2
    lift = np.concatenate([
        np.zeros(n_stand + n_down + n_up),
        (v0 * t_air - G * t_air ** 2 / 2) * PX_PER_M,
        np.zeros(n_land + n_rec),
    ])
    n = knee.size
    th = np.radians(knee) / 2
    ankle = np.column_stack([np.full(n, ANKLE_X), GROUND_Y - lift])
    knee_pt = ankle + np.column_stack([SHANK * np.sin(th), -SHANK * np.cos(th)])
    hip = knee_pt + np.column_stack([-THIGH * np.sin(th), -THIGH * np.cos(th)])
    shoulder = hip + np.column_stack([TRUNK * np.sin(th), -TRUNK * np.cos(th)])
    heel = ankle + np.array([-25.0, 10.0])
    toe = ankle + np.array([60.0, 15.0])
    head = shoulder + np.array([10.0, -70.0])

    xy = np.full((n, len(LANDMARKS), 2), np.nan)
    vis = np.zeros((n, len(LANDMARKS)))
    parts = {"shoulder": shoulder, "hip": hip, "knee": knee_pt, "ankle": ankle,
             "heel": heel, "foot_index": toe, "elbow": shoulder + [5, 120],
             "wrist": shoulder + [10, 220]}
    for side, offset, v in (("left", 0.0, 0.95), ("right", 4.0, right_visibility)):
        for name, p in parts.items():
            j = LANDMARKS.index(f"{side}_{name}")
            xy[:, j] = p + offset
            vis[:, j] = v
    for j in range(11):  # head landmarks
        xy[:, j] = head + [(j % 4) * 6.0, (j // 4) * 6.0]
        vis[:, j] = 0.9
    xy += rng.normal(0, noise, xy.shape)
    truth = {
        "takeoff": n_stand + n_down + n_up,
        "landing": n_stand + n_down + n_up + n_air,
        "flight": n_air / fps,
        "cm_depth": cm_depth,
        "land_depth": land_depth,
    }
    return PoseSeries(xy, vis, float(fps), 800, 1000), truth


@pytest.fixture
def cmj_pose():
    return make_cmj()
