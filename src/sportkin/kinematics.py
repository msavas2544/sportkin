"""Joint angles, signal filtering and angular derivatives.

All functions are pure NumPy/SciPy so they can be used (and tested) without
any video or pose-estimation dependency.

Conventions
-----------
* Points are arrays of shape ``(n_frames, 2)`` or ``(n_frames, 3)``.
* Image coordinates are assumed: x to the right, y **downwards**.
* Angles are returned in degrees.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

__all__ = [
    "joint_angle",
    "segment_angle",
    "interpolate_gaps",
    "butter_lowpass",
    "residual_analysis",
    "angular_velocity",
]


def _as_2d(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError("points must have shape (n, 2) or (n, 3)")
    return arr


def joint_angle(proximal, joint, distal) -> np.ndarray:
    """Included angle at ``joint`` formed by ``proximal`` and ``distal`` points.

    Example: knee angle = ``joint_angle(hip, knee, ankle)``.
    A fully extended knee gives ~180 degrees; flexion reduces the value.
    Use ``180 - joint_angle(...)`` to express it as flexion (0 = straight).

    Frames with missing (NaN) coordinates return NaN.
    """
    a, b, c = _as_2d(proximal), _as_2d(joint), _as_2d(distal)
    if not (a.shape == b.shape == c.shape):
        raise ValueError("proximal, joint and distal must have the same shape")
    v1 = a - b
    v2 = c - b
    dot = np.sum(v1 * v2, axis=1)
    norm = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.clip(dot / norm, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    ang[norm == 0] = np.nan
    return ang


def segment_angle(start, end) -> np.ndarray:
    """Absolute 2-D segment angle relative to the horizontal (degrees).

    Uses a y-up convention (image y is flipped), so a segment pointing
    straight up returns +90 and pointing right returns 0. Range (-180, 180].
    """
    s, e = _as_2d(start), _as_2d(end)
    dx = e[:, 0] - s[:, 0]
    dy = -(e[:, 1] - s[:, 1])
    return np.degrees(np.arctan2(dy, dx))


def interpolate_gaps(signal, max_gap: int | None = None) -> np.ndarray:
    """Linearly interpolate NaN gaps (needed before filtering).

    Gaps longer than ``max_gap`` frames are left as NaN. Leading/trailing NaNs
    are filled with the nearest valid value.
    """
    y = np.asarray(signal, dtype=float).copy()
    nan = np.isnan(y)
    if nan.all() or not nan.any():
        return y
    idx = np.arange(len(y))
    filled = np.interp(idx, idx[~nan], y[~nan])
    if max_gap is not None:
        # restore NaN for gaps longer than max_gap
        run_start = None
        for i, missing in enumerate(np.append(nan, False)):
            if missing and run_start is None:
                run_start = i
            elif not missing and run_start is not None:
                if i - run_start > max_gap:
                    filled[run_start:i] = np.nan
                run_start = None
    return filled


def butter_lowpass(signal, fs: float, cutoff: float = 6.0, order: int = 2) -> np.ndarray:
    """Zero-lag Butterworth low-pass filter (``filtfilt``).

    ``order=2`` applied forward and backward gives an effective 4th-order,
    zero-phase filter, the usual choice in biomechanics (Winter, 2009).
    NaN gaps are linearly interpolated before filtering and restored after.
    """
    if cutoff <= 0 or cutoff >= fs / 2:
        raise ValueError(f"cutoff must be between 0 and fs/2 ({fs / 2:.1f} Hz)")
    y = np.asarray(signal, dtype=float)
    nan = np.isnan(y)
    if nan.all():
        return y.copy()
    work = interpolate_gaps(y)
    b, a = butter(order, cutoff / (fs / 2), btype="low")
    padlen = min(3 * max(len(a), len(b)), len(work) - 1)
    out = filtfilt(b, a, work, padlen=padlen)
    out[nan] = np.nan
    return out


def residual_analysis(signal, fs: float, cutoffs=None, order: int = 2):
    """Winter's residual analysis for choosing a filter cut-off frequency.

    Returns ``(cutoffs, residuals, suggested_cutoff)``. The suggestion is the
    lowest cut-off whose residual falls within the noise level estimated by a
    straight-line fit to the high-frequency end of the residual curve.
    """
    y = interpolate_gaps(np.asarray(signal, dtype=float))
    if cutoffs is None:
        cutoffs = np.arange(1.0, min(fs / 2 - 1, 25.0) + 0.5, 0.5)
    cutoffs = np.asarray(cutoffs, dtype=float)
    res = np.array([
        np.sqrt(np.mean((y - butter_lowpass(y, fs, fc, order)) ** 2)) for fc in cutoffs
    ])
    # fit a line to the upper third of the curve (noise region)
    tail = max(3, len(cutoffs) // 3)
    slope, intercept = np.polyfit(cutoffs[-tail:], res[-tail:], 1)
    noise = intercept  # residual extrapolated to 0 Hz
    above = np.where(res <= noise)[0]
    suggested = float(cutoffs[above[0]]) if above.size else float(cutoffs[-1])
    return cutoffs, res, suggested


def angular_velocity(angle_deg, fs: float) -> np.ndarray:
    """First derivative of an angle series in deg/s (central differences)."""
    a = np.asarray(angle_deg, dtype=float)
    return np.gradient(a, 1.0 / fs)
