"""Vosk implementation of the STT provider interface."""

import json
import wave
from pathlib import Path

from .interface import STTProvider, Transcript, AudioInput


class VoskSTT(STTProvider):
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self._model = None

    def load(self):
        if self._model is not None:
            return self._model
        try:
            from vosk import Model
        except ImportError as exc:
            raise RuntimeError("Install Vosk with: pip install vosk") from exc
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Vosk model directory not found: {self.model_path}")
        self._model = Model(str(self.model_path))
        return self._model

    async def transcribe(self, audio: AudioInput, *, language: str | None = None, task="transcribe") -> Transcript:
        if task != "transcribe":
            raise ValueError("Vosk does not support translation")
        if not isinstance(audio, (str, Path)):
            raise TypeError("Vosk currently requires a WAV file path")

        try:
            from vosk import KaldiRecognizer
        except ImportError as exc:
            raise RuntimeError("Install Vosk with: pip install vosk") from exc

        model = self.load()
        with wave.open(str(audio), "rb") as wav:
            if wav.getnchannels() != 1:
                raise ValueError("Vosk requires mono PCM WAV audio")
            recognizer = KaldiRecognizer(model, wav.getframerate())
            result_text = []
            while True:
                data = wav.readframes(4000)
                if not data:
                    break
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    result_text.append(result.get("text", ""))
            final = json.loads(recognizer.FinalResult())
            result_text.append(final.get("text", ""))

        text = " ".join(part.strip() for part in result_text if part.strip())
        if not text:
            raise RuntimeError("Vosk returned an empty transcript")
        return Transcript(text=text, language=language)
