"""Face detection adapters."""

from pathlib import Path

from ._image import load_pil_image, model_id_or_error
from .interface import BoundingBox, Detection, FaceDetectionProvider, ImageInput


class OpenCVYuNetFaceDetector(FaceDetectionProvider):
    def __init__(self, model_path: str, score_threshold: float = 0.8, nms_threshold: float = 0.3):
        self.model_path = Path(model_path)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self._detector = None

    def _create_detector(self, image_size: tuple[int, int]):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Install OpenCV with: pip install opencv-contrib-python") from exc
        if not self.model_path.is_file():
            raise FileNotFoundError(f"YuNet ONNX weights not found: {self.model_path}")
        return cv2.FaceDetectorYN.create(
            str(self.model_path), "", image_size,
            self.score_threshold, self.nms_threshold, 5000,
        )

    async def detect(self, image: ImageInput) -> list[Detection]:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Install OpenCV with: pip install opencv-contrib-python") from exc

        pil_image = load_pil_image(image)
        rgb = np.asarray(pil_image)
        height, width = rgb.shape[:2]
        if self._detector is None or self._detector.getInputSize() != (width, height):
            self._detector = self._create_detector((width, height))
        _, faces = self._detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if faces is None:
            return []

        detections = []
        for face in faces:
            x, y, box_width, box_height = map(float, face[:4])
            detections.append(Detection(
                label="face",
                confidence=float(face[14]),
                box=BoundingBox(max(0.0, x), max(0.0, y), x + box_width, y + box_height),
            ))
        return detections


class UltralyticsFaceDetector(FaceDetectionProvider):
    """Use a face-trained Ultralytics checkpoint supplied by the user."""

    def __init__(self, model_path: str, confidence: float = 0.25, device: str | None = None):
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self._detector = None

    def load(self):
        if self._detector is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("Install Ultralytics with: pip install ultralytics") from exc
            self._detector = YOLO(model_id_or_error(self.model_path, "face detection"))
        return self._detector

    async def detect(self, image: ImageInput) -> list[Detection]:
        options = {"source": load_pil_image(image), "conf": self.confidence, "verbose": False}
        if self.device is not None:
            options["device"] = self.device
        results = self.load().predict(**options)
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                detections.append(Detection(
                    label="face",
                    confidence=float(box.conf[0].item()),
                    box=BoundingBox(x1, y1, x2, y2),
                ))
        return detections
