"""faster-whisper implementation of the STT provider interface."""

import tempfile
from pathlib import Path

from .interface import STTProvider, Transcript, AudioInput


class WhisperSTT(STTProvider):
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None

    def load(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("Install faster-whisper with: pip install faster-whisper") from exc

        if not str(self.model_path) or not self.model_path.exists():
            raise FileNotFoundError(f"Whisper model not found: {self.model_path}")
        self._model = WhisperModel(
            str(self.model_path),
            device=self.device,
            compute_type=self.compute_type,
        )
        return self._model

    def unload(self) -> None:
        self._model = None

    async def transcribe(
        self,
        audio: AudioInput,
        *,
        language: str | None = None,
        task="transcribe",
    ) -> Transcript:
        model = self.load()
        temporary_path: Path | None = None

        if isinstance(audio, (bytes, bytearray, memoryview)):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as file:
                file.write(bytes(audio))
                temporary_path = Path(file.name)
            audio_path = str(temporary_path)
        else:
            audio_path = str(audio)

        try:
            segments, info = model.transcribe(
                audio_path,
                language=language or self.language,
                task=task,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
            if not text:
                raise RuntimeError("Whisper returned an empty transcript")
            return Transcript(
                text=text,
                language=info.language or language or self.language,
                confidence=getattr(info, "language_probability", None),
            )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
