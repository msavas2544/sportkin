import numpy as np
import pytest

from sportkin.jump import (
    G,
    detect_jumps,
    frame_resolution_cm,
    height_from_flight_time,
    jump_from_frames,
)


def test_height_formula():
    # h = g t^2 / 8 ; 0.5 s -> 30.66 cm
    assert height_from_flight_time(0.5) == pytest.approx(G * 0.25 / 8 * 100)
    assert height_from_flight_time(0.5) == pytest.approx(30.656, abs=1e-3)


def test_negative_flight_time_raises():
    with pytest.raises(ValueError):
        height_from_flight_time(-0.1)


def test_manual_frames():
    r = jump_from_frames(100, 220, fps=240)
    assert r.flight_time_s == pytest.approx(0.5)
    assert r.height_cm == pytest.approx(30.656, abs=1e-3)
    assert r.method == "manual"
    with pytest.raises(ValueError):
        jump_from_frames(220, 100, fps=240)


def test_frame_resolution_shrinks_with_fps():
    assert frame_resolution_cm(0.5, 240) < frame_resolution_cm(0.5, 60)
    # dh/dt = g t / 4 -> 1.226 m/s x (1/240 s) = 0.51 cm
    assert frame_resolution_cm(0.5, 240) == pytest.approx(0.51, abs=0.01)


def _synthetic_jump(fps, flight, ground_px=800.0, scale_px_per_m=500.0,
                    pre_s=1.0, post_s=0.6, noise=0.0, seed=0):
    """Foot y trajectory (image coords) for one ballistic jump."""
    rng = np.random.default_rng(seed)
    n_pre, n_air, n_post = int(pre_s * fps), int(round(flight * fps)), int(post_s * fps)
    v0 = G * flight / 2
    t = np.arange(n_air) / fps
    height_m = v0 * t - G * t ** 2 / 2
    y = np.concatenate([
        np.full(n_pre, ground_px),
        ground_px - height_m * scale_px_per_m,
        np.full(n_post, ground_px),
    ])
    return y + rng.normal(0, noise, y.size), n_pre, n_pre + n_air


@pytest.mark.parametrize("fps", [60, 120, 240])
def test_detects_single_jump(fps):
    flight = 0.5
    y, takeoff, landing = _synthetic_jump(fps, flight)
    jumps = detect_jumps(y, fps)
    assert len(jumps) == 1
    j = jumps[0]
    assert abs(j.takeoff_frame - takeoff) <= 1
    assert abs(j.landing_frame - landing) <= 1
    # within one frame of the true flight time
    assert abs(j.height_cm - height_from_flight_time(flight)) <= frame_resolution_cm(flight, fps) + 1e-6


def test_detects_with_tracking_noise():
    fps = 240
    y, _, _ = _synthetic_jump(fps, 0.45, noise=1.0, seed=3)
    jumps = detect_jumps(y, fps)
    assert len(jumps) == 1
    assert jumps[0].height_cm == pytest.approx(height_from_flight_time(0.45), abs=1.5)


def test_detects_repeated_jumps_and_ignores_edges():
    fps = 120
    a, _, _ = _synthetic_jump(fps, 0.5, pre_s=0.8, post_s=0.5)
    b, _, _ = _synthetic_jump(fps, 0.4, pre_s=0.5, post_s=0.8)
    jumps = detect_jumps(np.concatenate([a, b]), fps)
    assert len(jumps) == 2
    assert jumps[0].height_cm > jumps[1].height_cm


def test_approach_jump_with_drifting_ground():
    """Run-up jump: floor line moves in the image (perspective) before and after."""
    fps = 30
    flight = 0.6
    y, takeoff, landing = _synthetic_jump(fps, flight, pre_s=1.5, post_s=1.0)
    drift = np.concatenate([np.linspace(-90, 0, takeoff), np.zeros(landing - takeoff),
                            np.linspace(-25, -25, len(y) - landing)])
    y = y + drift  # pre-take-off ground rises 90 px, landing ground 25 px higher
    jumps = detect_jumps(y, fps)
    assert len(jumps) == 1
    assert abs(jumps[0].flight_time_s - flight) <= 1.5 / fps


def test_timestamps_override_fps():
    y, takeoff, landing = _synthetic_jump(120, 0.5)
    times = np.arange(len(y)) / 100.0  # real clock is 100 fps
    j = detect_jumps(y, 120, times=times)[0]
    assert j.flight_time_s == pytest.approx((landing - takeoff) / 100.0, abs=0.02)


def test_no_jump_in_flat_signal():
    assert detect_jumps(np.full(500, 700.0), 120) == []


def test_handles_missing_frames():
    fps = 240
    y, _, _ = _synthetic_jump(fps, 0.5)
    y[300:303] = np.nan
    assert len(detect_jumps(y, fps)) == 1


def test_landing_with_landmark_sliding_after_contact():
    """After touchdown the toe landmark keeps moving down (knee flexion,
    perspective). Landing must be the impact frame, not the end of the slide."""
    fps = 30
    y, takeoff, landing = _synthetic_jump(fps, 0.6, pre_s=1.0, post_s=1.0)
    slide = np.zeros(len(y))
    slide[landing:landing + 6] = np.linspace(14, 25, 6)
    slide[landing + 6:] = 25
    y = y + slide
    j = detect_jumps(y, fps)[0]
    assert j.landing_frame == landing
    assert abs(j.takeoff_frame - takeoff) <= 1


def test_height_vectorised():
    np.testing.assert_allclose(height_from_flight_time(np.array([0.5, 0.6])),
                               [height_from_flight_time(0.5), height_from_flight_time(0.6)])
