from pathlib import Path

import cv2
import numpy as np

from src.video.sampling import iter_sampled_frames


def _make_tiny_video(path: Path, fps: int = 10, seconds: int = 3, w: int = 160, h: int = 120) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    assert out.isOpened()
    total = fps * seconds
    for i in range(total):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(frame, str(i), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    out.release()


def test_iter_sampled_frames_max_frames(tmp_path: Path) -> None:
    vid = tmp_path / "tiny.avi"
    _make_tiny_video(vid, fps=10, seconds=5)

    frames = list(iter_sampled_frames(str(vid), sample_fps=1, max_frames=3))
    assert len(frames) == 3
    assert frames[0].shape[2] == 3


def test_iter_sampled_frames_end_of_video(tmp_path: Path) -> None:
    vid = tmp_path / "tiny.avi"
    _make_tiny_video(vid, fps=10, seconds=2)

    frames = list(iter_sampled_frames(str(vid), sample_fps=5, max_frames=999))
    assert 1 <= len(frames) <= 50
