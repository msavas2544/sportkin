"""Pose landmark series and the MediaPipe backend.

:class:`PoseSeries` is backend-independent: it stores pixel coordinates for
every frame and landmark, so analyses can be tested with synthetic data and
other estimators (YOLO-Pose, OpenPose, manual digitising) can be plugged in.

The MediaPipe model file (``.task``) is downloaded once into a local cache
and reused offline afterwards. In fully offline environments, copy the file
manually and pass ``model_path`` or set ``SPORTKIN_MODEL``.
"""

from __future__ import annotations

import csv
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "LANDMARKS",
    "PoseSeries",
    "model_cache_dir",
    "resolve_model",
    "download_model",
    "MediaPipeBackend",
]

# BlazePose / MediaPipe Pose landmark order (33 points)
LANDMARKS = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]
_INDEX = {name: i for i, name in enumerate(LANDMARKS)}

MODEL_URLS = {
    v: ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        f"pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task")
    for v in ("lite", "full", "heavy")
}


@dataclass
class PoseSeries:
    """Landmark trajectories for one person.

    xy : (n_frames, n_landmarks, 2) pixel coordinates (NaN when missing)
    visibility : (n_frames, n_landmarks) confidence in [0, 1]
    """

    xy: np.ndarray
    visibility: np.ndarray
    fps: float
    width: int = 0
    height: int = 0
    names: list = field(default_factory=lambda: list(LANDMARKS))
    times: np.ndarray | None = None  # per-frame timestamps (s); None -> index / fps

    def __post_init__(self):
        self.xy = np.asarray(self.xy, dtype=float)
        self.visibility = np.asarray(self.visibility, dtype=float)
        if self.xy.ndim != 3 or self.xy.shape[2] != 2:
            raise ValueError("xy must have shape (n_frames, n_landmarks, 2)")
        if self.visibility.shape != self.xy.shape[:2]:
            raise ValueError("visibility must have shape (n_frames, n_landmarks)")
        if self.times is not None:
            self.times = np.asarray(self.times, dtype=float)
            if self.times.shape != (self.xy.shape[0],):
                raise ValueError("times must have one value per frame")

    @property
    def n_frames(self) -> int:
        return self.xy.shape[0]

    @property
    def time(self) -> np.ndarray:
        if self.times is not None:
            return self.times
        return np.arange(self.n_frames) / self.fps

    def point(self, name: str, min_visibility: float = 0.5) -> np.ndarray:
        """(n_frames, 2) coordinates of one landmark; low-confidence frames -> NaN."""
        idx = self.names.index(name) if name in self.names else _INDEX[name]
        p = self.xy[:, idx, :].copy()
        p[self.visibility[:, idx] < min_visibility] = np.nan
        return p

    def to_csv(self, path) -> Path:
        """Wide CSV: frame, time, <name>_x, <name>_y, <name>_vis for all landmarks."""
        path = Path(path)
        header = ["frame", "time_s"]
        for n in self.names:
            header += [f"{n}_x", f"{n}_y", f"{n}_vis"]
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            t = self.time
            for i in range(self.n_frames):
                row = [i, f"{t[i]:.6f}"]
                for j in range(len(self.names)):
                    x, y = self.xy[i, j]
                    row += ["" if np.isnan(x) else f"{x:.2f}",
                            "" if np.isnan(y) else f"{y:.2f}",
                            f"{self.visibility[i, j]:.3f}"]
                w.writerow(row)
        return path

    @classmethod
    def from_csv(cls, path, fps: float | None = None) -> "PoseSeries":
        """Read a CSV written by :meth:`to_csv` (or a compatible digitising tool)."""
        with Path(path).open(newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        header, body = rows[0], rows[1:]
        names = [h[:-2] for h in header if h.endswith("_x")]
        n = len(body)
        xy = np.full((n, len(names), 2), np.nan)
        vis = np.zeros((n, len(names)))
        col = {h: i for i, h in enumerate(header)}
        for i, r in enumerate(body):
            for j, name in enumerate(names):
                for k, ax in enumerate("xy"):
                    v = r[col[f"{name}_{ax}"]]
                    xy[i, j, k] = float(v) if v else np.nan
                vkey = f"{name}_vis"
                vis[i, j] = float(r[col[vkey]]) if vkey in col and r[col[vkey]] else 1.0
        times = None
        if "time_s" in col and n > 1:
            times = np.array([float(r[col["time_s"]]) for r in body])
        if fps is None:
            if "time_s" not in col or n < 2:
                raise ValueError("fps could not be inferred; pass fps=")
            span = float(body[-1][col["time_s"]]) - float(body[0][col["time_s"]])
            if span <= 0:
                raise ValueError("fps could not be inferred; pass fps=")
            fps = round((n - 1) / span, 2)
        else:
            times = None  # explicit capture fps overrides stored times
        return cls(xy, vis, float(fps), names=names, times=times)


def model_cache_dir() -> Path:
    base = os.environ.get("SPORTKIN_CACHE") or os.path.join(Path.home(), ".cache", "sportkin")
    return Path(base)


def resolve_model(model_path=None, variant: str = "full") -> Path:
    """Find a pose model file without touching the network."""
    candidates = [model_path, os.environ.get("SPORTKIN_MODEL"),
                  model_cache_dir() / f"pose_landmarker_{variant}.task"]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    raise FileNotFoundError(
        "Pose model not found. Run `sportkin download-model` once (needs internet), "
        "or download pose_landmarker_full.task manually and pass --model PATH."
    )


def download_model(variant: str = "full", dest=None) -> Path:
    """Download a MediaPipe pose model once into the local cache."""
    if variant not in MODEL_URLS:
        raise ValueError(f"variant must be one of {list(MODEL_URLS)}")
    dest = Path(dest) if dest else model_cache_dir() / f"pose_landmarker_{variant}.task"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(MODEL_URLS[variant], tmp)
    tmp.replace(dest)
    return dest


class MediaPipeBackend:
    """Single-person pose estimation with MediaPipe Pose Landmarker (VIDEO mode)."""

    def __init__(self, model_path=None, variant: str = "full",
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ImportError("Install the video extras: pip install 'sportkin[video]'") from exc
        self._mp = mp
        self.model_path = resolve_model(model_path, variant)
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def process(self, frames, fps: float, width: int, height: int,
                n_frames: int | None = None, progress=None, times=None) -> PoseSeries:
        """Run pose estimation on an iterable of BGR frames (OpenCV order).

        ``times`` (seconds per frame) are stored in the result when given;
        otherwise frame index / ``fps`` is used.
        """
        import cv2

        xy_list, vis_list = [], []
        for i, frame in enumerate(frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            # tracking clock: must strictly increase; capture fps drives it
            res = self._landmarker.detect_for_video(image, int(round(i * 1000.0 / fps)))
            xy = np.full((len(LANDMARKS), 2), np.nan)
            vis = np.zeros(len(LANDMARKS))
            if res.pose_landmarks:
                for j, lm in enumerate(res.pose_landmarks[0]):
                    xy[j] = (lm.x * width, lm.y * height)
                    vis[j] = lm.visibility if lm.visibility is not None else 1.0
            xy_list.append(xy)
            vis_list.append(vis)
            if progress:
                progress(i + 1, n_frames)
        return PoseSeries(np.array(xy_list).reshape(-1, len(LANDMARKS), 2),
                          np.array(vis_list).reshape(-1, len(LANDMARKS)),
                          fps, width, height,
                          times=None if times is None else np.asarray(times)[:len(xy_list)])

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
