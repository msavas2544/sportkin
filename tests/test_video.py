import numpy as np
import pytest

from conftest import make_cmj

cv2 = pytest.importorskip("cv2")

from sportkin.annotate import write_annotated_video  # noqa: E402
from sportkin.video import iter_frames, probe  # noqa: E402


def _video(path, n=40, w=800, h=1000, fps=30):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(0)
    for _ in range(n):
        wr.write(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
    wr.release()
    return path


def test_probe_and_iter(tmp_path):
    p = _video(tmp_path / "v.mp4")
    info = probe(p)
    assert (info.width, info.height) == (800, 1000)
    assert info.container_fps == pytest.approx(30, abs=0.5)
    assert sum(1 for _ in iter_frames(p, start=5, stop=15)) == 10


def test_probe_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe(tmp_path / "nope.mp4")


def test_annotated_video_blurs_face(tmp_path):
    pose, _ = make_cmj()
    src = _video(tmp_path / "v.mp4", n=20)
    out = write_annotated_video(iter_frames(src), pose, tmp_path / "a.mp4", blur_face=True,
                                events={3: "take-off"}, labels={"knee": np.zeros(pose.n_frames)})
    assert out.exists()
    frames = list(iter_frames(out))
    assert len(frames) == 20
    # region around the head must be much smoother than the original noise
    hx, hy = pose.xy[0, 0].astype(int)
    orig = next(iter_frames(src))[hy - 15:hy + 15, hx - 15:hx + 15].astype(float)
    blur = frames[0][hy - 15:hy + 15, hx - 15:hx + 15].astype(float)
    assert blur.std() < orig.std() * 0.5
