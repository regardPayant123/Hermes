# engines/voice/stt/interface.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


AudioInput = Path | str | bytes | bytearray | memoryview
TranscriptTask = Literal["transcribe", "translate"]


@dataclass(frozen=True, slots=True)
class Transcript:
    """The text and metadata produced by a speech-to-text provider."""

    text: str
    language: str | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("transcript text cannot be empty")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")


class STTProvider(ABC):
    """Common interface for speech-to-text providers."""

    @abstractmethod
    async def transcribe(
        self,
        audio: AudioInput,
        *,
        language: str | None = None,
        task: TranscriptTask = "transcribe",
    ) -> Transcript:
        """Transcribe or translate audio into a structured transcript.

        Args:
            audio: A file path or in-memory audio bytes.
            language: Optional provider language code, for example ``"fr"``.
            task: Whether to transcribe in the original language or translate
                the result to English.

        Raises:
            TypeError: If the audio input type is unsupported.
            ValueError: If the audio or options are invalid.
            RuntimeError: If the provider fails to process the audio.
        """
        raise NotImplementedError