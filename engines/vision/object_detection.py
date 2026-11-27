"""Object detection adapters for Ultralytics and Transformers models."""

from pathlib import Path

from ._image import load_pil_image, model_id_or_error
from .interface import BoundingBox, Detection, ImageInput, ObjectDetectionProvider


class UltralyticsObjectDetector(ObjectDetectionProvider):
    def __init__(self, model_path: str, confidence: float = 0.25, device: str | None = None):
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self._model = None

    def load(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("Install Ultralytics with: pip install ultralytics") from exc
            self._model = YOLO(model_id_or_error(self.model_path, "object detection"))
        return self._model

    async def detect(self, image: ImageInput) -> list[Detection]:
        pil_image = load_pil_image(image)
        options = {"source": pil_image, "conf": self.confidence, "verbose": False}
        if self.device is not None:
            options["device"] = self.device
        results = self.load().predict(**options)
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            names = result.names
            for box in result.boxes:
                coordinates = box.xyxy[0].tolist()
                class_id = int(box.cls[0].item())
                detections.append(Detection(
                    label=str(names[class_id]),
                    confidence=float(box.conf[0].item()),
                    box=BoundingBox(*map(float, coordinates)),
                ))
        return detections


class TransformersObjectDetector(ObjectDetectionProvider):
    def __init__(self, model_id: str, threshold: float = 0.25, device: int = -1):
        self.model_id = model_id
        self.threshold = threshold
        self.device = device
        self._pipeline = None

    def load(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError("Install Transformers with: pip install transformers torch") from exc
            self._pipeline = pipeline(
                "object-detection",
                model=model_id_or_error(self.model_id, "object detection"),
                device=self.device,
            )
        return self._pipeline

    async def detect(self, image: ImageInput) -> list[Detection]:
        results = self.load()(load_pil_image(image), threshold=self.threshold)
        detections = []
        for result in results:
            box = result["box"]
            detections.append(Detection(
                label=str(result["label"]),
                confidence=float(result["score"]),
                box=BoundingBox(
                    float(box["xmin"]), float(box["ymin"]),
                    float(box["xmax"]), float(box["ymax"]),
                ),
            ))
        return detections
