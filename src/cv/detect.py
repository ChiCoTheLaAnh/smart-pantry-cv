import numpy as np

from src.cv.detector import MODEL_PATH, detect_on_frame, get_model


class Detector:
    """
    Backward-compatible wrapper that reuses the global YOLO singleton.
    """

    def __init__(self, model_name: str = MODEL_PATH):
        # Singleton is configured via YOLO_MODEL/YOLO_DEVICE env vars.
        self.model_name = model_name
        get_model()

    def detect(self, frame_bgr: np.ndarray, conf: float = 0.25) -> list[str]:
        """
        Input: frame BGR (OpenCV)
        Output: list label strings (unique per frame)
        """
        detections = detect_on_frame(frame_bgr, conf=conf)
        labels = {str(det.get("label", "")) for det in detections}
        labels.discard("")
        return sorted(labels)
