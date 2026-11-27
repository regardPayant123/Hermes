"""Optional Transformers and Ultralytics providers for additional CV tasks."""

from ._image import load_pil_image, model_id_or_error
from .interface import (
    BoundingBox,
    Classification,
    DepthEstimationProvider,
    DepthResult,
    ImageClassificationProvider,
    ImageInput,
    Pose,
    PoseEstimationProvider,
    SegmentationProvider,
    SegmentationResult,
)


class TransformersImageClassifier(ImageClassificationProvider):
    def __init__(self, model_id: str, device: int = -1, top_k: int = 5):
        self.model_id = model_id
        self.device = device
        self.top_k = top_k
        self._pipeline = None

    def load(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError("Install Transformers with: pip install transformers torch") from exc
            self._pipeline = pipeline("image-classification", model=model_id_or_error(self.model_id, "image classification"), device=self.device)
        return self._pipeline

    async def classify(self, image: ImageInput) -> list[Classification]:
        results = self.load()(load_pil_image(image), top_k=self.top_k)
        return [Classification(str(item["label"]), float(item["score"])) for item in results]


class TransformersSegmenter(SegmentationProvider):
    def __init__(self, model_id: str, device: int = -1):
        self.model_id = model_id
        self.device = device
        self._pipeline = None

    def load(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError("Install Transformers with: pip install transformers torch") from exc
            self._pipeline = pipeline("image-segmentation", model=model_id_or_error(self.model_id, "image segmentation"), device=self.device)
        return self._pipeline

    async def segment(self, image: ImageInput) -> SegmentationResult:
        results = self.load()(load_pil_image(image))
        segments = tuple({
            "label": str(item.get("label", "")),
            "score": float(item.get("score", 0.0)),
            "mask": item.get("mask"),
        } for item in results)
        return SegmentationResult(segments)


class TransformersDepthEstimator(DepthEstimationProvider):
    def __init__(self, model_id: str, device: int = -1):
        self.model_id = model_id
        self.device = device
        self._pipeline = None

    def load(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError("Install Transformers with: pip install transformers torch") from exc
            self._pipeline = pipeline("depth-estimation", model=model_id_or_error(self.model_id, "depth estimation"), device=self.device)
        return self._pipeline

    async def estimate(self, image: ImageInput) -> DepthResult:
        result = self.load()(load_pil_image(image))
        depth = result["depth"]
        return DepthResult(width=depth.width, height=depth.height, depth=depth)


class UltralyticsPoseEstimator(PoseEstimationProvider):
    def __init__(self, model_path: str, confidence: float = 0.25, device: str | None = None):
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
            self._model = YOLO(model_id_or_error(self.model_path, "pose estimation"))
        return self._model

    async def estimate(self, image: ImageInput) -> list[Pose]:
        options = {"source": load_pil_image(image), "conf": self.confidence, "verbose": False}
        if self.device is not None:
            options["device"] = self.device
        results = self.load().predict(**options)
        poses = []
        for result in results:
            keypoints = result.keypoints
            if keypoints is None:
                continue
            values = keypoints.data.cpu().tolist()
            boxes = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else [None] * len(values)
            names = getattr(keypoints, "has_visible", None)
            for index, person_points in enumerate(values):
                points = tuple({"x": float(point[0]), "y": float(point[1]), "confidence": float(point[2])} for point in person_points)
                raw_box = boxes[index]
                box = BoundingBox(*map(float, raw_box)) if raw_box is not None else None
                poses.append(Pose(box=box, keypoints=points))
        return poses
