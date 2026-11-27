"""JSON configuration and provider factory for the vision engine."""

import json
from pathlib import Path
from typing import Any


class VisionFactory:
    @staticmethod
    def load_config(path: str | Path) -> dict[str, Any]:
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Vision configuration not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
        if not isinstance(config, dict):
            raise ValueError("Vision configuration must contain a JSON object")
        return config

    @staticmethod
    def create_provider(capability: str, config: dict[str, Any]):
        provider = str(config.get("provider", "")).lower()
        options = config.get("options", {})
        if not isinstance(options, dict):
            raise ValueError(f"{capability}.options must contain a JSON object")
        options = dict(options)

        if capability == "object_detection":
            if provider == "ultralytics":
                from .object_detection import UltralyticsObjectDetector
                return UltralyticsObjectDetector(**options)
            if provider == "transformers":
                from .object_detection import TransformersObjectDetector
                return TransformersObjectDetector(**options)
        elif capability == "face_detection":
            if provider == "opencv_yunet":
                from .face_detection import OpenCVYuNetFaceDetector
                return OpenCVYuNetFaceDetector(**options)
            if provider == "ultralytics":
                from .face_detection import UltralyticsFaceDetector
                return UltralyticsFaceDetector(**options)
        elif capability == "vlm" and provider == "transformers":
            from .vlm import TransformersVLM
            return TransformersVLM(**options)
        elif capability == "ocr":
            from .ocr import EasyOCRProvider, PaddleOCRProvider, TesseractOCR
            providers = {
                "pytesseract": TesseractOCR,
                "easyocr": EasyOCRProvider,
                "paddleocr": PaddleOCRProvider,
            }
            if provider in providers:
                return providers[provider](**options)
        elif capability == "image_classification" and provider == "transformers":
            from .extras import TransformersImageClassifier
            return TransformersImageClassifier(**options)
        elif capability == "segmentation":
            if provider == "transformers":
                from .extras import TransformersSegmenter
                return TransformersSegmenter(**options)
        elif capability == "pose_estimation" and provider == "ultralytics":
            from .extras import UltralyticsPoseEstimator
            return UltralyticsPoseEstimator(**options)
        elif capability == "depth_estimation" and provider == "transformers":
            from .extras import TransformersDepthEstimator
            return TransformersDepthEstimator(**options)

        raise ValueError(f"Unsupported vision provider '{provider}' for capability '{capability}'")

    @classmethod
    def from_json(cls, path: str | Path) -> dict[str, Any]:
        config = cls.load_config(path)
        providers = {}
        for capability, section in config.items():
            if not isinstance(section, dict):
                raise ValueError(f"{capability} configuration must be a JSON object")
            if not section.get("enabled", True):
                continue
            if capability == "stream":
                from .stream import VisionStream

                source = section.get("source")
                if source is None:
                    raise ValueError("stream.source must be a camera index, video path, or stream URL")
                options = section.get("options", {})
                if not isinstance(options, dict):
                    raise ValueError("stream.options must contain a JSON object")
                providers[capability] = VisionStream(source, **options)
                continue
            providers[capability] = cls.create_provider(capability, section)
        return providers
