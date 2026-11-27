"""Shared contracts and result types for computer vision providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ImageInput = str | Path | bytes | bytearray | memoryview


@dataclass(frozen=True, slots=True)
class BoundingBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def __post_init__(self) -> None:
        if self.xmin < 0 or self.ymin < 0 or self.xmax < self.xmin or self.ymax < self.ymin:
            raise ValueError("bounding box coordinates must be non-negative and ordered")


@dataclass(frozen=True, slots=True)
class Detection:
    label: str
    confidence: float
    box: BoundingBox
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("detection label cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("detection confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class Classification:
    label: str
    confidence: float


@dataclass(frozen=True, slots=True)
class OCRResult:
    text: str
    confidence: float | None = None
    boxes: tuple[Detection, ...] = ()


@dataclass(frozen=True, slots=True)
class VLMResponse:
    text: str
    model: str


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    segments: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class Pose:
    box: BoundingBox | None
    keypoints: tuple[dict[str, float | str], ...]
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class DepthResult:
    width: int
    height: int
    depth: Any


class ObjectDetectionProvider(ABC):
    @abstractmethod
    async def detect(self, image: ImageInput) -> list[Detection]:
        raise NotImplementedError


class FaceDetectionProvider(ABC):
    @abstractmethod
    async def detect(self, image: ImageInput) -> list[Detection]:
        raise NotImplementedError


class VLMProvider(ABC):
    @abstractmethod
    async def ask(self, image: ImageInput, prompt: str) -> VLMResponse:
        raise NotImplementedError


class OCRProvider(ABC):
    @abstractmethod
    async def extract_text(self, image: ImageInput) -> OCRResult:
        raise NotImplementedError


class ImageClassificationProvider(ABC):
    @abstractmethod
    async def classify(self, image: ImageInput) -> list[Classification]:
        raise NotImplementedError


class SegmentationProvider(ABC):
    @abstractmethod
    async def segment(self, image: ImageInput) -> SegmentationResult:
        raise NotImplementedError


class PoseEstimationProvider(ABC):
    @abstractmethod
    async def estimate(self, image: ImageInput) -> list[Pose]:
        raise NotImplementedError


class DepthEstimationProvider(ABC):
    @abstractmethod
    async def estimate(self, image: ImageInput) -> DepthResult:
        raise NotImplementedError
