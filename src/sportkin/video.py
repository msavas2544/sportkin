"""Video reading helpers (OpenCV).

Phone slow-motion clips are frequently saved with a 30 fps *playback* rate even
though they were captured at 120 or 240 fps. The container value is reported
as ``container_fps``; analyses must use the true capture rate, which the user
supplies with ``--fps``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["VideoInfo", "probe", "frame_times", "iter_frames"]


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install the video extras: pip install 'sportkin[video]'") from exc
    return cv2


@dataclass
class VideoInfo:
    path: Path
    container_fps: float
    n_frames: int
    width: int
    height: int
    times: np.ndarray | None = None  # decoded per-frame timestamps (s)

    @property
    def measured_fps(self) -> float:
        """Frame rate from the decoded timestamps (median frame interval)."""
        if self.times is None or len(self.times) < 3:
            return float("nan")
        dt = np.diff(self.times)
        dt = dt[dt > 0]
        return float(1.0 / np.median(dt)) if dt.size else float("nan")

    @property
    def variable_frame_rate(self) -> bool:
        if self.times is None or len(self.times) < 3:
            return False
        dt = np.diff(self.times)
        return bool(np.max(dt) > 1.5 * np.median(dt))

    @property
    def duration_s(self) -> float:
        if self.times is not None and len(self.times) > 1:
            return float(self.times[-1] - self.times[0])
        return self.n_frames / self.container_fps if self.container_fps else float("nan")


def frame_times(path) -> np.ndarray:
    """Decoded timestamp (s) of every frame, read without converting pixels."""
    cv2 = _cv2()
    cap = cv2.VideoCapture(str(path))
    out = []
    try:
        while cap.grab():
            out.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    finally:
        cap.release()
    return np.asarray(out)


def probe(path, measure: bool = True) -> VideoInfo:
    """Video metadata. With ``measure=True`` the real frame timestamps are read,
    because container metadata (fps, frame count) is often wrong for phone videos.
    """
    cv2 = _cv2()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video: {path}")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # honour phone rotation metadata
    ok, frame = cap.read()
    info = VideoInfo(
        path=path,
        container_fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        width=int(frame.shape[1]) if ok else int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(frame.shape[0]) if ok else int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    if measure:
        info.times = frame_times(path)
        info.n_frames = len(info.times)
    return info


def iter_frames(path, start: int = 0, stop: int | None = None):
    """Yield BGR frames (rotation corrected) from ``start`` up to ``stop``."""
    cv2 = _cv2()
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    i = start
    try:
        while stop is None or i < stop:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
            i += 1
    finally:
        cap.release()
