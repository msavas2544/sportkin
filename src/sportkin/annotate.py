"""Annotated video export: skeleton overlay, event markers and face blurring.

Face blurring uses the head landmarks (nose, eyes, ears, mouth), so it needs no
extra model and runs offline. Use it whenever annotated clips are shared
(theses, presentations, supplementary material) to protect participants.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .pose import LANDMARKS, PoseSeries

__all__ = ["SKELETON", "write_annotated_video"]

SKELETON = [
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"), ("left_heel", "left_foot_index"),
    ("left_ankle", "left_foot_index"),
    ("right_ankle", "right_heel"), ("right_heel", "right_foot_index"),
    ("right_ankle", "right_foot_index"),
]
_HEAD = [LANDMARKS.index(n) for n in LANDMARKS[:11]]


def _blur_face(frame, pts, vis, cv2):
    head = pts[_HEAD][vis[_HEAD] > 0.3]
    head = head[~np.isnan(head).any(axis=1)]
    if len(head) < 2:
        return
    (x0, y0), (x1, y1) = head.min(axis=0), head.max(axis=0)
    size = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r = size * 1.3 + 10
    h, w = frame.shape[:2]
    a, b = int(max(0, cx - r)), int(max(0, cy - r))
    c, d = int(min(w, cx + r)), int(min(h, cy + r))
    if c - a > 2 and d - b > 2:
        roi = frame[b:d, a:c]
        k = max(15, (min(c - a, d - b) // 3) | 1)
        frame[b:d, a:c] = cv2.GaussianBlur(roi, (k, k), 0)


def write_annotated_video(frames, pose: PoseSeries, out_path, *, blur_face: bool = False,
                          events=None, labels=None, out_fps: float = 30.0,
                          min_visibility: float = 0.5) -> Path:
    """Draw the skeleton on ``frames`` and write an MP4.

    events : dict frame -> text (e.g. {412: "take-off", 530: "landing"})
    labels : dict column -> per-frame values drawn in the corner (e.g. knee angle)
    out_fps : playback rate of the output file (slow motion stays slow).
    """
    import cv2

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    events = events or {}
    labels = labels or {}
    idx = {n: i for i, n in enumerate(pose.names)}
    try:
        for i, frame in enumerate(frames):
            if i >= pose.n_frames:
                break
            frame = frame.copy()
            pts, vis = pose.xy[i], pose.visibility[i]
            if blur_face:
                _blur_face(frame, pts, vis, cv2)
            for a, b in SKELETON:
                ia, ib = idx[a], idx[b]
                if vis[ia] >= min_visibility and vis[ib] >= min_visibility and \
                        np.isfinite(pts[[ia, ib]]).all():
                    cv2.line(frame, tuple(int(v) for v in pts[ia]), tuple(int(v) for v in pts[ib]),
                             (0, 220, 255), 2, cv2.LINE_AA)
            for j in range(len(pose.names)):
                if j in _HEAD and blur_face:
                    continue
                if vis[j] >= min_visibility and np.isfinite(pts[j]).all():
                    cv2.circle(frame, tuple(int(v) for v in pts[j]), 3, (0, 80, 255), -1, cv2.LINE_AA)
            y = 24
            cv2.putText(frame, f"frame {i}  t={i / pose.fps:.3f}s", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            for name, values in labels.items():
                y += 22
                v = values[i]
                txt = f"{name}: {'-' if not np.isfinite(v) else f'{v:.1f}'}"
                cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2, cv2.LINE_AA)
            if i in events:
                cv2.putText(frame, events[i], (10, frame.shape[0] - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         out_fps, (w, h))
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()
    return out_path
