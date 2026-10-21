"""Vertical jump height from flight time.

The flight-time method only needs the take-off and landing frames, so no
distance calibration is required:

    h = g * t_f**2 / 8

where ``t_f`` is the flight time (s) and ``g = 9.81 m/s**2`` (Bosco et al.,
1983). Take-off is the first frame with both feet off the ground, landing is
the first frame of renewed ground contact; this is the same convention used by
manual frame-selection apps such as My Jump (Balsalobre-Fernández et al., 2015).

Automatic detection works on the vertical image coordinate of a foot landmark
(heel, ankle or toe). Because detection can be off by a frame, results can be
corrected with :func:`jump_from_frames`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .kinematics import interpolate_gaps

G = 9.81

__all__ = [
    "G",
    "JumpResult",
    "height_from_flight_time",
    "frame_resolution_cm",
    "jump_from_frames",
    "detect_jumps",
]


@dataclass
class JumpResult:
    takeoff_frame: int
    landing_frame: int
    fps: float
    flight_time_s: float
    height_cm: float
    method: str = "auto"

    def to_dict(self) -> dict:
        return asdict(self)


def height_from_flight_time(flight_time_s):
    """Jump height in centimetres from flight time in seconds (scalar or array)."""
    t = np.asarray(flight_time_s, dtype=float)
    if np.any(t < 0):
        raise ValueError("flight time must be non-negative")
    h = G * t ** 2 / 8 * 100
    return float(h) if h.ndim == 0 else h


def frame_resolution_cm(flight_time_s: float, fps: float) -> float:
    """Height change (cm) caused by a one-frame error at a given frame rate.

    Useful for the methods/limitations section, e.g. at 240 fps and a
    0.5 s flight a single-frame error changes the height by ~0.5 cm.
    """
    dt = 1.0 / fps
    return height_from_flight_time(flight_time_s + dt) - height_from_flight_time(flight_time_s)


def _flight_time(takeoff: int, landing: int, fps: float, times=None) -> float:
    if times is not None:
        return float(times[landing] - times[takeoff])
    return (landing - takeoff) / fps


def jump_from_frames(takeoff_frame: int, landing_frame: int, fps: float,
                     method: str = "manual", times=None) -> JumpResult:
    """Compute a jump from manually selected (or corrected) frames.

    If per-frame ``times`` (seconds) are given they are used instead of
    ``fps``, which is safer for variable-frame-rate phone videos.
    """
    if landing_frame <= takeoff_frame:
        raise ValueError("landing_frame must be after takeoff_frame")
    if fps <= 0:
        raise ValueError("fps must be positive")
    t = _flight_time(takeoff_frame, landing_frame, fps, times)
    return JumpResult(int(takeoff_frame), int(landing_frame), float(fps), t,
                      height_from_flight_time(t), method)


def _robust_noise(y: np.ndarray) -> float:
    """Robust SD of frame-to-frame tracking noise (MAD of first differences)."""
    d = np.diff(y)
    d = d[np.isfinite(d)]
    if d.size < 3:
        return 0.0
    return float(np.median(np.abs(d - np.median(d))) * 1.4826 / np.sqrt(2))


def _impact_frame(y: np.ndarray, start: int, stop: int, floor: float, rise: float,
                  step: int, frac: float = 0.3, lookback: int = 3) -> int | None:
    """First frame in [start, stop] where a falling foot suddenly stops.

    During the last part of the descent the foot moves fast (about the take-off
    velocity); at ground contact the downward motion drops abruptly. Contact is
    the first frame, within the lowest 20 % of the rise, after which the
    displacement over ``step`` frames is below ``frac`` x the incoming speed.
    The incoming speed is the fastest descent over the last ``lookback`` steps,
    because pose trackers smooth landmarks in time and spread the impact over
    2-3 frames. This is robust to the landmark sliding further down after
    contact (foot flattening, knee flexion, perspective).
    """
    n = len(y)
    stop = min(stop, n - 1 - step)
    if stop <= start + step:
        return None
    for j in range(max(start, step * lookback), stop + 1):
        if y[j] < floor - 0.2 * rise:
            continue  # still clearly in the air
        incoming = max(y[j - i * step + step] - y[j - i * step] for i in range(1, lookback + 1))
        outgoing = y[j + step] - y[j]
        if incoming > 0 and outgoing < frac * incoming:
            return j
    return None


def detect_jumps(foot_y, fps: float, *, times=None, threshold_px: float | None = None,
                 min_rise_px: float | None = None, min_flight_s: float = 0.15,
                 max_flight_s: float = 1.2, min_ground_s: float = 0.05) -> list[JumpResult]:
    """Detect jumps from the vertical image coordinate of the feet.

    Method: candidate flights are found where the feet rise clearly above the
    surrounding floor level. For each candidate the floor is then measured
    locally and separately just before take-off and just after landing (90th
    percentile of a short window), and take-off/landing are the frames where
    the foot leaves / returns to within tracking noise of that local floor.
    This handles standing jumps, run-up jumps and perspective drift of the
    floor line in the image.

    Parameters
    ----------
    foot_y : array (n_frames,)
        Image y (pixels, growing downwards) of the **lower** foot, i.e. the
        per-frame maximum of the left and right toe y, so that "airborne"
        requires both feet off the ground. NaNs are interpolated.
    fps : float
        Capture frame rate (used when ``times`` is not given).
    times : array (n_frames,), optional
        Per-frame timestamps in seconds; preferred for variable frame rates.
    threshold_px : float, optional
        Rise above the local floor that marks a candidate flight.
        Default: max(3 px, 6 x robust tracking noise).
    min_rise_px : float, optional
        Minimum apex rise for an event to count as a jump. Default:
        3 x ``threshold_px``; :func:`sportkin.cmj.analyze_cmj` sets it from the
        athlete's leg length in pixels.
    min_flight_s, max_flight_s : float
        Plausible flight-time window; events outside are discarded.
    min_ground_s : float
        Contacts shorter than this inside a candidate are treated as noise.
    """
    from scipy.ndimage import maximum_filter1d

    y = interpolate_gaps(np.asarray(foot_y, dtype=float))
    if np.isnan(y).all():
        return []
    n = len(y)
    noise = _robust_noise(y)
    if threshold_px is None:
        threshold_px = max(3.0, 6.0 * noise)
    if min_rise_px is None:
        min_rise_px = 3.0 * threshold_px
    tol = max(1.0, 2.0 * noise)  # distance from the floor regarded as contact

    w = int(round(max_flight_s * fps))
    ground_ref = maximum_filter1d(y, size=2 * w + 1, mode="nearest")
    # The candidate threshold must sit above slow ground jitter of the
    # landmark (a few px), which the frame-to-frame noise underestimates.
    core = (ground_ref - y) > max(threshold_px, 0.5 * min_rise_px)

    min_ground = max(1, int(round(min_ground_s * fps)))
    i = 0
    while i < n:  # bridge very short "contacts" (tracking glitches)
        if not core[i]:
            j = i
            while j < n and not core[j]:
                j += 1
            if i > 0 and j < n and (j - i) < min_ground:
                core[i:j] = True
            i = j
        else:
            i += 1

    jumps: list[JumpResult] = []
    seen = set()
    i = 0
    while i < n:
        if not core[i]:
            i += 1
            continue
        start = i
        while i < n and core[i]:
            i += 1
        end = i
        if start == 0 or end >= n:
            continue  # event cut by the start/end of the recording
        apex = start + int(np.argmin(y[start:end]))
        if apex in seen:
            continue
        ya = y[apex]
        ref_pre = np.max(y[max(0, apex - w):apex + 1])
        ref_post = np.max(y[apex:min(n, apex + w + 1)])
        # 1) locate the fast part of the flight: where the foot is a quarter of
        #    the rise below the apex (time from contact to there is ~T/4)
        k0 = apex
        while k0 > 0 and y[k0] < ya + 0.25 * (ref_pre - ya):
            k0 -= 1
        m0 = apex
        while m0 < n - 1 and y[m0] < ya + 0.25 * (ref_post - ya):
            m0 += 1
        # 2) floor level measured *locally*, separately before take-off and
        #    after landing (robust to run-ups and perspective drift)
        reach = (m0 - k0) // 2 + int(round(0.15 * fps)) + 1
        pre = y[max(0, k0 - reach):k0 + 1]
        post = y[m0:min(n, m0 + reach)]
        if pre.size < 3 or post.size < 3:
            continue
        g_pre, g_post = np.percentile(pre, 90), np.percentile(post, 90)
        # 3) contact = first frame within noise of the local floor
        k = k0
        while k > 0 and y[k - 1] < g_pre - tol:
            k -= 1
        m = m0
        while m < n - 1 and y[m] < g_post - tol:
            m += 1
        if k == 0 or m >= n - 1:
            continue
        rise = min(g_pre, g_post) - ya
        # 4) refine with the impact signature (sudden stop of the foot): the
        #    landmark may keep sliding down after contact, so the floor-level
        #    rule alone can be late. Take-off is the same problem in reverse.
        step = max(1, int(round(fps / 120)))
        lookback = max(2, int(round(0.05 * fps / step)))  # ~50 ms of incoming motion
        land_hit = _impact_frame(y, m0, m, g_post, g_post - ya, step, lookback=lookback)
        yr = y[::-1]
        take_hit = _impact_frame(yr, n - 1 - k0, n - 1 - (k - 1), g_pre, g_pre - ya, step,
                                 lookback=lookback)
        landing = land_hit if land_hit is not None else m
        takeoff = (n - 1 - take_hit) + 1 if take_hit is not None else k
        if rise < min_rise_px:
            continue
        seen.add(apex)
        t = _flight_time(takeoff, landing, fps, times)
        if min_flight_s <= t <= max_flight_s:
            jumps.append(JumpResult(takeoff, landing, float(fps), t,
                                    height_from_flight_time(t), "auto"))
    return jumps
