import numpy as np
import pytest

from conftest import make_cmj
from sportkin.cmj import analyze_cmj, choose_side
from sportkin.jump import frame_resolution_cm, height_from_flight_time
from sportkin.pose import PoseSeries


def test_choose_side_prefers_visible_side(cmj_pose):
    pose, _ = cmj_pose
    assert choose_side(pose) == "left"


def test_cmj_against_ground_truth(cmj_pose):
    pose, truth = cmj_pose
    res = analyze_cmj(pose)
    assert res.side == "left"
    assert len(res.jumps) == 1
    j = res.jumps[0]
    assert abs(j.takeoff_frame - truth["takeoff"]) <= 1
    assert abs(j.landing_frame - truth["landing"]) <= 1
    true_h = height_from_flight_time(truth["flight"])
    assert abs(j.height_cm - true_h) <= frame_resolution_cm(truth["flight"], pose.fps) + 1e-6
    assert j.peak_knee_flexion_cm_deg == pytest.approx(truth["cm_depth"], abs=3)
    assert j.knee_flexion_takeoff_deg < 10
    assert j.peak_knee_flexion_landing_deg == pytest.approx(truth["land_depth"], abs=4)
    assert j.peak_knee_ext_velocity_deg_s > 300  # 90 deg extended in 0.25 s
    assert not res.warnings


def test_cmj_with_landmark_noise():
    pose, truth = make_cmj(noise=1.0, seed=2)
    res = analyze_cmj(pose)
    assert len(res.jumps) == 1
    assert res.jumps[0].height_cm == pytest.approx(height_from_flight_time(truth["flight"]), abs=1.5)
    assert res.jumps[0].peak_knee_flexion_cm_deg == pytest.approx(truth["cm_depth"], abs=5)


def test_manual_frames_override(cmj_pose):
    pose, truth = cmj_pose
    res = analyze_cmj(pose, manual_frames=[(truth["takeoff"], truth["landing"])])
    assert res.jumps[0].method == "manual"
    assert res.jumps[0].height_cm == pytest.approx(height_from_flight_time(truth["flight"]))


def test_low_fps_warning():
    pose, _ = make_cmj(fps=60)
    res = analyze_cmj(pose)
    assert any("fps" in w for w in res.warnings)


def test_frames_table_shape(cmj_pose):
    pose, _ = cmj_pose
    res = analyze_cmj(pose, side="right")
    assert res.side == "right"
    for col in res.frames.values():
        assert len(col) == pose.n_frames


def test_pose_csv_roundtrip(tmp_path, cmj_pose):
    pose, _ = cmj_pose
    path = pose.to_csv(tmp_path / "p.csv")
    back = PoseSeries.from_csv(path)
    assert back.fps == pytest.approx(240, abs=0.1)
    np.testing.assert_allclose(back.xy, pose.xy, atol=0.01, equal_nan=True)
    # times are stored with 6 decimals -> sub-micrometre differences only
    assert analyze_cmj(back).jumps[0].height_cm == pytest.approx(
        analyze_cmj(pose).jumps[0].height_cm, abs=1e-3)


def test_single_foot_lift_is_not_a_jump():
    """A step (one toe up, other on the ground) must not count as flight."""
    pose, truth = make_cmj()
    j = pose.names.index("right_foot_index")
    pose.xy[100:180, j, 1] -= 120  # right toe lifted for 0.33 s, left stays down
    res = analyze_cmj(pose)
    assert len(res.jumps) == 1
    assert abs(res.jumps[0].takeoff_frame - truth["takeoff"]) <= 1


def test_variable_frame_rate_uses_timestamps():
    pose, truth = make_cmj(fps=240)
    # same frames, but the clock runs 10 % slower (e.g. wrong container fps)
    pose.times = np.arange(pose.n_frames) / 240 * 1.1
    res = analyze_cmj(pose)
    assert res.jumps[0].flight_time_s == pytest.approx(truth["flight"] * 1.1, abs=2 / 240)
