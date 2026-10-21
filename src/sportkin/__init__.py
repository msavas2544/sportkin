"""sportkin: offline, camera-based kinematics and jump analysis for sport science.

Core modules (NumPy/SciPy only, fully offline):

* :mod:`sportkin.kinematics` - joint/segment angles, Butterworth filtering,
  residual analysis, angular velocity
* :mod:`sportkin.jump` - jump height from flight time (auto + manual frames)
* :mod:`sportkin.stats` - ICC, Bland-Altman, typical error, effect sizes
* :mod:`sportkin.cmj` - countermovement jump analysis from pose landmarks
* :mod:`sportkin.io` - CSV export for SPSS / R / Turkish Excel

Optional extras: :mod:`sportkin.pose` and :mod:`sportkin.video` (``[video]``),
:mod:`sportkin.report` (``[report]``), :mod:`sportkin.annotate` (``[video]``).
"""

from .cmj import CMJAnalysis, analyze_cmj
from .jump import JumpResult, detect_jumps, frame_resolution_cm, height_from_flight_time, jump_from_frames
from .kinematics import (
    angular_velocity,
    butter_lowpass,
    interpolate_gaps,
    joint_angle,
    residual_analysis,
    segment_angle,
)
from .stats import agreement, bland_altman, icc, interpret_icc, reliability, typical_error

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "joint_angle",
    "segment_angle",
    "interpolate_gaps",
    "butter_lowpass",
    "residual_analysis",
    "angular_velocity",
    "JumpResult",
    "height_from_flight_time",
    "frame_resolution_cm",
    "jump_from_frames",
    "detect_jumps",
    "icc",
    "interpret_icc",
    "bland_altman",
    "typical_error",
    "agreement",
    "reliability",
    "analyze_cmj",
    "CMJAnalysis",
]
