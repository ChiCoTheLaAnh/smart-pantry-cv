from __future__ import annotations

from typing import Iterator
import cv2


def iter_sampled_frames(path: str, sample_fps: float = 1, max_frames: int = 30) -> Iterator["cv2.Mat"]:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be > 0")
    if max_frames <= 0:
        return
        yield

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 0 else 0.0

    step_sec = 1.0 / float(sample_fps)
    next_t = 0.0
    yielded = 0
    last_t = -1.0
    frame_idx = -1

    try:
        while yielded < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            t = cap.get(cv2.CAP_PROP_POS_MSEC)
            t = (float(t) / 1000.0) if t and t >= 0 else -1.0

            if t <= last_t + 1e-9:
                t = (frame_idx / fps) if fps > 0 else (last_t + step_sec)

            last_t = t

            if t + 1e-6 >= next_t:
                yield frame
                yielded += 1
                next_t += step_sec
    finally:
        cap.release()
