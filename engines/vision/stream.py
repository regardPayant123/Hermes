"""Asynchronous image streams from cameras, videos, and application sources."""

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterable, AsyncIterator, Awaitable, Callable, Generic, TypeVar

from .interface import ImageInput


StreamSource = int | str | Path | AsyncIterable[ImageInput]
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class StreamFrame:
    index: int
    timestamp: float
    image: ImageInput


@dataclass(frozen=True, slots=True)
class StreamOutput(Generic[T]):
    frame_index: int
    timestamp: float
    result: T


class VisionStream:
    """Read images continuously and optionally run an async vision operation.

    Integer sources select camera devices (usually 0); strings and Paths can
    identify video files or network streams such as RTSP URLs. An application
    may alternatively pass an async iterable of supported image inputs.
    """

    def __init__(
        self,
        source: StreamSource,
        *,
        jpeg_quality: int = 85,
    ):
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        self.source = source
        self.jpeg_quality = jpeg_quality

    async def frames(self, *, max_frames: int | None = None) -> AsyncIterator[StreamFrame]:
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be greater than zero")

        if hasattr(self.source, "__aiter__"):
            index = 0
            async for image in self.source:
                if max_frames is not None and index >= max_frames:
                    break
                yield StreamFrame(index=index, timestamp=time.monotonic(), image=image)
                index += 1
            return

        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Install OpenCV for camera/video streams: pip install opencv-python") from exc

        source = str(self.source) if isinstance(self.source, Path) else self.source
        capture = await asyncio.to_thread(cv2.VideoCapture, source)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to open image stream: {source}")
        await asyncio.to_thread(capture.set, cv2.CAP_PROP_BUFFERSIZE, 1)

        index = 0
        try:
            while max_frames is None or index < max_frames:
                success, frame = await asyncio.to_thread(capture.read)
                if not success:
                    break
                encoded, buffer = await asyncio.to_thread(
                    cv2.imencode,
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                )
                if not encoded:
                    continue
                yield StreamFrame(
                    index=index,
                    timestamp=time.monotonic(),
                    image=buffer.tobytes(),
                )
                index += 1
        finally:
            await asyncio.to_thread(capture.release)

    async def process(
        self,
        operation: Callable[[ImageInput], Awaitable[T]],
        *,
        max_frames: int | None = None,
        every_n_frames: int = 1,
    ) -> AsyncIterator[StreamOutput[T]]:
        if every_n_frames <= 0:
            raise ValueError("every_n_frames must be greater than zero")

        async for frame in self.frames(max_frames=max_frames):
            if frame.index % every_n_frames:
                continue
            result = await operation(frame.image)
            yield StreamOutput(
                frame_index=frame.index,
                timestamp=frame.timestamp,
                result=result,
            )
