import numpy as np
import pytest

from sportkin.kinematics import (
    angular_velocity,
    butter_lowpass,
    interpolate_gaps,
    joint_angle,
    residual_analysis,
    segment_angle,
)


def test_right_angle():
    hip = np.array([[0.0, 0.0]])
    knee = np.array([[0.0, 1.0]])
    ankle = np.array([[1.0, 1.0]])
    assert joint_angle(hip, knee, ankle)[0] == pytest.approx(90.0)


def test_straight_limb_is_180():
    pts = [np.array([[0.0, y]]) for y in (0.0, 1.0, 2.0)]
    assert joint_angle(*pts)[0] == pytest.approx(180.0)


def test_known_angles_3d_and_vectorised():
    angles = np.radians([30, 60, 120, 150])
    joint = np.zeros((4, 3))
    prox = np.tile([1.0, 0.0, 0.0], (4, 1))
    dist = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(4)])
    np.testing.assert_allclose(joint_angle(prox, joint, dist), [30, 60, 120, 150], atol=1e-9)


def test_nan_and_degenerate_frames():
    a = np.array([[0.0, 0.0], [np.nan, 0.0], [0.0, 0.0]])
    b = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 0.0]])  # last frame: zero-length vector
    c = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    out = joint_angle(a, b, c)
    assert out[0] == pytest.approx(90.0)
    assert np.isnan(out[1]) and np.isnan(out[2])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        joint_angle(np.zeros((3, 2)), np.zeros((2, 2)), np.zeros((3, 2)))


def test_segment_angle_image_coordinates():
    start = np.array([[0.0, 0.0]])
    assert segment_angle(start, np.array([[1.0, 0.0]]))[0] == pytest.approx(0.0)
    # image y grows downwards: a point *above* is a smaller y
    assert segment_angle(start, np.array([[0.0, -1.0]]))[0] == pytest.approx(90.0)


def test_interpolate_gaps():
    y = np.array([0.0, np.nan, 2.0, np.nan, np.nan, np.nan, 6.0])
    np.testing.assert_allclose(interpolate_gaps(y), [0, 1, 2, 3, 4, 5, 6])
    limited = interpolate_gaps(y, max_gap=2)
    assert limited[1] == pytest.approx(1.0)
    assert np.isnan(limited[3:6]).all()


def test_lowpass_keeps_signal_removes_noise():
    fs = 240.0
    t = np.arange(0, 2, 1 / fs)
    clean = 30 * np.sin(2 * np.pi * 1.5 * t)
    noisy = clean + 3 * np.sin(2 * np.pi * 40 * t)
    out = butter_lowpass(noisy, fs, cutoff=6)
    core = slice(50, -50)
    assert np.max(np.abs(out[core] - clean[core])) < 0.5


def test_lowpass_zero_lag():
    fs = 100.0
    t = np.arange(0, 3, 1 / fs)
    x = np.sin(2 * np.pi * 1.0 * t)
    y = butter_lowpass(x, fs, cutoff=10)
    assert np.argmax(x[50:200]) == np.argmax(y[50:200])


def test_lowpass_preserves_nan_positions():
    fs = 100.0
    x = np.sin(np.linspace(0, 6, 300))
    x[100:105] = np.nan
    y = butter_lowpass(x, fs, cutoff=6)
    assert np.isnan(y[100:105]).all() and not np.isnan(y[:100]).any()


def test_lowpass_bad_cutoff():
    with pytest.raises(ValueError):
        butter_lowpass(np.zeros(50), fs=30, cutoff=20)


def test_residual_analysis_suggests_reasonable_cutoff():
    rng = np.random.default_rng(0)
    fs = 200.0
    t = np.arange(0, 3, 1 / fs)
    x = 20 * np.sin(2 * np.pi * 2 * t) + rng.normal(0, 0.8, t.size)
    _, res, fc = residual_analysis(x, fs)
    assert np.all(np.diff(res) <= 1e-9)  # residual falls as cut-off rises
    assert 2.5 <= fc <= 12


def test_angular_velocity_of_linear_angle():
    fs = 120.0
    ang = 90 + 30 * np.arange(120) / fs  # 30 deg/s
    np.testing.assert_allclose(angular_velocity(ang, fs), 30.0)
