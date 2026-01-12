from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
import tempfile
import os
from typing import Optional

from src.cv.detect import Detector
from src.cv.video import iter_sampled_frames

app = FastAPI(title="Smart Pantry CV API", version="0.1.0")

detector: Optional[Detector] = None


@app.on_event("startup")
def _startup():
    global detector
    detector = Detector(model_name=os.getenv("YOLO_MODEL", "yolov8n.pt"))


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/detect/video")
async def detect_video(
    video: UploadFile = File(...),
    every_n_frames: int = Query(30, ge=1),
    max_frames: int = Query(30, ge=1),
    conf: float = Query(0.25, ge=0.0, le=1.0),
):
    if detector is None:
        return JSONResponse(status_code=500, content={"ok": False, "error": "detector_not_ready"})

    suffix = os.path.splitext(video.filename or "")[1] or ".mp4"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            tmp_path = f.name
            f.write(await video.read())

        labels = []
        counts: dict[str, int] = {}
        sampled = 0

        for frame in iter_sampled_frames(tmp_path, every_n_frames=every_n_frames, max_frames=max_frames):
            sampled += 1
            frame_labels = detector.detect(frame, conf=conf)
            for lab in frame_labels:
                counts[lab] = counts.get(lab, 0) + 1

        labels = sorted(counts.keys())
        return {
            "ok": True,
            "labels": labels,
            "unique_labels": labels,
            "frame_hits": counts,
            "sampled_frames": sampled,
            "every_n_frames": every_n_frames,
            "max_frames": max_frames,
            "conf": conf,
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
