"""Countermovement jump (CMJ) analysis from a sagittal-plane video.

Camera set-up assumed: side view, camera perpendicular to the plane of motion,
whole body in frame, static camera. Angles are 2-D projections and are only
valid under this set-up (state it in the methods section).

Angle definitions (degrees)
---------------------------
* knee flexion = 180 - angle(hip, knee, ankle)            (0 = fully extended)
* hip flexion  = 180 - angle(shoulder, hip, knee)          (0 = upright)
* ankle angle  = angle(knee, ankle, foot_index)            (included angle)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .jump import JumpResult, detect_jumps, frame_resolution_cm, jump_from_frames
from .kinematics import angular_velocity, butter_lowpass, joint_angle
from .pose import PoseSeries

__all__ = ["JumpMetrics", "CMJAnalysis", "choose_side", "analyze_cmj"]


@dataclass
class JumpMetrics:
    jump_index: int
    side: str
    takeoff_frame: int
    landing_frame: int
    flight_time_s: float
    height_cm: float
    height_resolution_cm: float
    peak_knee_flexion_cm_deg: float      # deepest point of the countermovement
    peak_hip_flexion_cm_deg: float
    knee_flexion_takeoff_deg: float
    peak_knee_ext_velocity_deg_s: float  # concentric phase, absolute value
    peak_knee_flexion_landing_deg: float
    method: str = "auto"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CMJAnalysis:
    fps: float
    side: str
    cutoff_hz: float
    frames: dict                 # column name -> array (per-frame table)
    jumps: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def summary_rows(self) -> list:
        return [j.to_dict() for j in self.jumps]


def choose_side(pose: PoseSeries) -> str:
    """Pick the body side facing the camera (higher mean landmark visibility)."""
    def score(side):
        names = [f"{side}_{p}" for p in ("hip", "knee", "ankle", "foot_index")]
        idx = [pose.names.index(n) for n in names]
        return float(np.nanmean(pose.visibility[:, idx]))
    return "left" if score("left") >= score("right") else "right"


def _angles(pose: PoseSeries, side: str, fs: float, cutoff: float, min_vis: float):
    p = {k: pose.point(f"{side}_{k}", min_vis)
         for k in ("shoulder", "hip", "knee", "ankle", "foot_index")}
    knee = 180 - joint_angle(p["hip"], p["knee"], p["ankle"])
    hip = 180 - joint_angle(p["shoulder"], p["hip"], p["knee"])
    ankle = joint_angle(p["knee"], p["ankle"], p["foot_index"])
    filt = {}
    for name, sig in (("knee_flexion", knee), ("hip_flexion", hip), ("ankle_angle", ankle)):
        filt[name] = butter_lowpass(sig, fs, cutoff) if np.isfinite(sig).sum() > 12 else sig
    leg = np.linalg.norm(p["hip"] - p["ankle"], axis=1)
    return filt, p, float(np.nanmedian(leg)) if np.isfinite(leg).any() else float("nan")


def _lower_foot_y(pose: PoseSeries, min_vis: float) -> np.ndarray:
    """Per-frame y of the lower of the two feet (toe landmarks).

    Flight = *both* feet off the ground, so the lower foot is the one that
    decides. If one foot is not visible, the other is used.
    """
    ys = np.column_stack([pose.point(f"{s}_foot_index", min_vis)[:, 1] for s in ("left", "right")])
    with np.errstate(all="ignore"):
        out = np.nanmax(ys, axis=1)
    return out


def analyze_cmj(pose: PoseSeries, *, side: str = "auto", cutoff_hz: float = 6.0,
                min_visibility: float = 0.5, manual_frames=None,
                **detect_kwargs) -> CMJAnalysis:
    """Analyse one or more CMJs in a pose series.

    Parameters
    ----------
    pose : PoseSeries
        Landmarks with the **capture** frame rate in ``pose.fps``.
    side : {"auto", "left", "right"}
        Body side used for angles and foot tracking. "auto" picks the side
        facing the camera.
    cutoff_hz : float
        Butterworth cut-off for angles (6 Hz is common for jumping; check
        with :func:`sportkin.kinematics.residual_analysis`).
    manual_frames : list of (takeoff, landing), optional
        Override automatic detection with hand-picked frames.
    """
    fs = float(pose.fps)
    warnings = []
    if side == "auto":
        side = choose_side(pose)
    if side not in ("left", "right"):
        raise ValueError("side must be 'auto', 'left' or 'right'")
    if fs < 100:
        warnings.append(
            f"Frame rate {fs:.0f} fps: one frame changes jump height by about "
            f"{frame_resolution_cm(0.5, fs):.1f} cm. Use >=120 fps (ideally 240) for research."
        )

    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        ang, pts, leg_px = _angles(pose, side, fs, cutoff_hz, min_visibility)
        foot_y = _lower_foot_y(pose, min_visibility)
    missing = float(np.isnan(foot_y).mean())
    if missing > 0.2:
        warnings.append(f"{missing:.0%} of frames have no reliable foot landmark.")

    times = pose.times
    if manual_frames:
        events: list[JumpResult] = [jump_from_frames(a, b, fs, times=times)
                                    for a, b in manual_frames]
    else:
        if np.isfinite(leg_px) and "min_rise_px" not in detect_kwargs:
            detect_kwargs["min_rise_px"] = 0.1 * leg_px  # ignore running-step flight phases
        events = detect_jumps(foot_y, fs, times=times, **detect_kwargs)
        if not events:
            warnings.append("No jump detected. Check fps, framing and foot visibility, "
                            "or supply manual_frames.")

    knee, hip = ang["knee_flexion"], ang["hip_flexion"]
    knee_vel = angular_velocity(knee, fs)
    jumps = []
    for k, ev in enumerate(events, start=1):
        to, ld = ev.takeoff_frame, ev.landing_frame
        pre = slice(max(0, to - int(1.5 * fs)), to)          # countermovement window
        post = slice(ld, min(len(knee), ld + int(0.5 * fs)))  # landing window
        bottom = pre.start + int(np.nanargmax(knee[pre])) if np.isfinite(knee[pre]).any() else to
        concentric = slice(bottom, to)

        def _max(x, s):
            seg = x[s]
            return float(np.nanmax(seg)) if np.isfinite(seg).any() else float("nan")

        jumps.append(JumpMetrics(
            jump_index=k, side=side, takeoff_frame=to, landing_frame=ld,
            flight_time_s=ev.flight_time_s, height_cm=ev.height_cm,
            height_resolution_cm=frame_resolution_cm(ev.flight_time_s, fs),
            peak_knee_flexion_cm_deg=_max(knee, pre),
            peak_hip_flexion_cm_deg=_max(hip, pre),
            knee_flexion_takeoff_deg=float(knee[to]) if to < len(knee) else float("nan"),
            # knee extension = flexion decreasing -> negative velocity
            peak_knee_ext_velocity_deg_s=_max(-knee_vel, concentric),
            peak_knee_flexion_landing_deg=_max(knee, post),
            method=ev.method,
        ))

    frames = {
        "frame": np.arange(pose.n_frames),
        "time_s": pose.time,
        f"{side}_knee_flexion_deg": knee,
        f"{side}_hip_flexion_deg": hip,
        f"{side}_ankle_angle_deg": ang["ankle_angle"],
        f"{side}_knee_velocity_deg_s": knee_vel,
        "lower_foot_y_px": foot_y,
    }
    return CMJAnalysis(fs, side, cutoff_hz, frames, jumps, warnings)
