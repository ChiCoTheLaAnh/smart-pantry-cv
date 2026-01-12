from ultralytics import YOLO
import numpy as np

class Detector:
    def __init__(self, model_name: str = "yolov8n.pt"):
        # model_name có thể là "yolov8n.pt" hoặc đường dẫn weights
        self.model = YOLO(model_name)

    def detect(self, frame_bgr: np.ndarray, conf: float = 0.25) -> list[str]:
        """
        Input: frame BGR (OpenCV)
        Output: list label strings (unique per frame)
        """
        # Ultralytics nhận numpy array OK (BGR/RGB đều chạy, nhưng label không đổi)
        results = self.model.predict(frame_bgr, conf=conf, verbose=False)
        r0 = results[0]

        labels: set[str] = set()
        if r0.boxes is None:
            return []

        for cls_id in r0.boxes.cls.tolist():
            name = self.model.names.get(int(cls_id), str(int(cls_id)))
            labels.add(str(name))

        return sorted(labels)
