from __future__ import annotations

import numpy as np
import pytest

from src.cv import detector


class _FakeBox:
    def __init__(self, cls_id: int, conf_score: float, xyxy: tuple[int, int, int, int]):
        self.cls = np.array([cls_id], dtype=float)
        self.conf = np.array([conf_score], dtype=float)
        self.xyxy = np.array([xyxy], dtype=float)


class _FakeResult:
    def __init__(self, boxes: list[tuple[int, float, tuple[int, int, int, int]]]):
        self.boxes = [_FakeBox(*b) for b in boxes]


class _FakeModel:
    init_count = 0

    def __init__(self, boxes: list[tuple[int, float, tuple[int, int, int, int]]] | None = None):
        _FakeModel.init_count += 1
        self.names = {0: "foo", 1: "bar"}
        self._boxes = boxes or []
        self.to_device = None
        self.last_conf = None

    def to(self, device: str):
        self.to_device = device
        return self

    def predict(self, source, conf: float, verbose: bool = False):
        self.last_conf = conf
        return [_FakeResult(self._boxes)]


@pytest.fixture(autouse=True)
def _reset_singleton(monkeypatch):
    monkeypatch.setattr(detector, "_model", None)
    monkeypatch.setattr(detector, "_class_names", None)
    _FakeModel.init_count = 0
    yield
    monkeypatch.setattr(detector, "_model", None)
    monkeypatch.setattr(detector, "_class_names", None)


def test_get_model_is_singleton(monkeypatch):
    monkeypatch.setattr(detector, "YOLO", lambda path: _FakeModel())
    m1 = detector.get_model()
    m2 = detector.get_model()

    assert m1 is m2
    assert _FakeModel.init_count == 1
    assert detector._class_names == {0: "foo", 1: "bar"}
    assert m1.to_device is not None  # DEVICE is applied


def test_detect_on_frame_returns_detections(monkeypatch):
    boxes = [
        (0, 0.55, (10, 20, 30, 40)),
        (1, 0.42, (5, 5, 15, 15)),
    ]
    monkeypatch.setattr(detector, "YOLO", lambda path: _FakeModel(boxes))

    detections = detector.detect_on_frame(np.zeros((4, 4, 3), dtype=np.uint8), conf=0.33)

    assert len(detections) == 2
    assert detections[0]["label"] == "foo"
    assert detections[1]["label"] == "bar"
    assert detections[0]["box"] == [10, 20, 30, 40]
    assert detections[1]["box"] == [5, 5, 15, 15]
    # Ensure conf override passes through
    assert detector._model.last_conf == 0.33
