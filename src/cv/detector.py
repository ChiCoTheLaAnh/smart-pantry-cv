from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
from ultralytics import YOLO

# ====== CONFIG ======
MODEL_PATH = os.getenv("YOLO_MODEL", "yolov8n.pt")
CONF_THRES = float(os.getenv("YOLO_CONF", "0.25"))
DEVICE = os.getenv("YOLO_DEVICE", "cpu")  # e.g., "cpu" or "cuda"

# ====== GLOBAL SINGLETON ======
_model: Optional[YOLO] = None
_class_names: Optional[Dict[int, str]] = None


def get_model() -> YOLO:
    """
    Load YOLO model once (singleton).
    """
    global _model, _class_names
    if _model is None:
        _model = YOLO(MODEL_PATH)
        if DEVICE:
            _model.to(DEVICE)
        _class_names = _model.names  # dict: {class_id: class_name}
    return _model


def detect_on_frame(frame: np.ndarray, conf: Optional[float] = None) -> List[Dict]:
    """
    Args:
        frame: np.ndarray (H, W, 3), BGR or RGB đều được YOLO xử lý
        conf: optional confidence threshold override
    Returns:
        List of detections:
        {
            "label": str,
            "conf": float,
            "box": [x1, y1, x2, y2]
        }
    """
    model = get_model()

    conf_thres = float(conf) if conf is not None else CONF_THRES
    results = model.predict(source=frame, conf=conf_thres, verbose=False)

    detections: List[Dict] = []

    if not results:
        return detections

    r0 = results[0]
    boxes = getattr(r0, "boxes", None)
    if boxes is None:
        return detections

    for box in boxes:
        cls_id = int(box.cls.item())
        conf_score = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        label_map = _class_names or {}
        label = label_map.get(cls_id, str(cls_id)) if isinstance(label_map, dict) else str(cls_id)

        detections.append({
            "label": str(label),
            "conf": round(conf_score, 4),
            "box": [int(x1), int(y1), int(x2), int(y2)],
        })

    return detections
