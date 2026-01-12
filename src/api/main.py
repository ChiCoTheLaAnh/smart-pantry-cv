import os
import cv2
import tempfile
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse

from src.cv.detector import detect_on_frame, get_model
from src.video.sampling import iter_sampled_frames

app = FastAPI(title="Smart Pantry CV API", version="0.1.0")


@app.on_event("startup")
def _startup():
    # Preload YOLO model once
    get_model()


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
    try:
        get_model()
    except Exception as exc:  # pragma: no cover - defensive guard
        return JSONResponse(status_code=500, content={"ok": False, "error": "detector_not_ready", "detail": str(exc)})

    suffix = os.path.splitext(video.filename or "")[1] or ".mp4"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            tmp_path = f.name
            f.write(await video.read())

        labels = []
        counts: dict[str, int] = {}
        detections_by_frame: list[list[dict]] = []
        sampled = 0

        cap = cv2.VideoCapture(tmp_path)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = float(fps) if fps and fps > 0 else None
        finally:
            cap.release()

        if fps:
            sample_fps = fps / float(every_n_frames)
        else:
            sample_fps = 1.0

        frames = iter_sampled_frames(tmp_path, sample_fps=sample_fps, max_frames=max_frames)
        for frame_bgr in frames:
            sampled += 1
            frame_detections = detect_on_frame(frame_bgr, conf=conf)
            detections_by_frame.append(frame_detections)

            for det in frame_detections:
                label = str(det.get("label", ""))
                counts[label] = counts.get(label, 0) + 1

        labels = sorted(counts.keys())
        return {
            "ok": True,
            "labels": labels,
            "unique_labels": labels,
            "frame_hits": counts,
            "detections": detections_by_frame,
            "sampled_frames": sampled,
            "sample_fps": sample_fps,
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
