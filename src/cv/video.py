import cv2
import numpy as np

def iter_sampled_frames(video_path: str, every_n_frames: int = 30, max_frames: int = 30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {video_path}")

    idx = 0
    yielded = 0
    try:
        while yielded < max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            if idx % every_n_frames == 0:
                if frame is not None and isinstance(frame, np.ndarray):
                    yield frame
                    yielded += 1
            idx += 1
    finally:
        cap.release()
