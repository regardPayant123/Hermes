"""Live PulseAudio -> OpenWakeWord -> VAD -> STT -> brain command pipeline."""

import asyncio
import inspect
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from engines.brain.llm.interface import LLMResponse
from engines.voice.stt.interface import Transcript
from engines.voice.wake_word.interface import WakeWordProvider


class PCMFrameSource(Protocol):
    async def __aenter__(self): ...
    async def __aexit__(self, exc_type, exc_value, traceback): ...
    async def read_frame(self, frame_samples: int) -> bytes: ...


@dataclass(frozen=True, slots=True)
class VoiceCommandResult:
    wake_word: str
    transcript: Transcript
    response: LLMResponse


class LiveVoiceCommandPipeline:
    def __init__(
        self,
        *,
        audio_source: PCMFrameSource,
        wake_word: WakeWordProvider,
        stt: Any,
        brain: Any,
        vad: Any | None = None,
        vad_factory: Callable[[], Awaitable[Any]] | None = None,
        sample_rate: int = 16000,
        frame_samples: int = 1280,
        end_silence_seconds: float = 0.8,
        speech_start_timeout_seconds: float = 5.0,
        max_utterance_seconds: float = 20.0,
    ):
        if sample_rate != 16000:
            raise ValueError("The configured wake-word and voice models require 16000 Hz")
        if frame_samples <= 0 or frame_samples % 1280:
            raise ValueError("frame_samples must be a positive multiple of 1280 samples")
        if end_silence_seconds <= 0 or speech_start_timeout_seconds <= 0 or max_utterance_seconds <= 0:
            raise ValueError("speech timeouts and utterance limit must be positive")
        self.audio_source = audio_source
        self.wake_word = wake_word
        self.vad = vad
        self.vad_factory = vad_factory
        if self.vad is None and self.vad_factory is None:
            raise ValueError("Provide a VAD instance or a lazy vad_factory")
        self.stt = stt
        self.brain = brain
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.end_silence_frames = max(1, round(end_silence_seconds * sample_rate / frame_samples))
        self.speech_start_frames = max(1, round(speech_start_timeout_seconds * sample_rate / frame_samples))
        self.max_utterance_frames = max(1, round(max_utterance_seconds * sample_rate / frame_samples))

    async def listen_once(self) -> VoiceCommandResult | None:
        async with self.audio_source as source:
            while True:
                wake_frame = await source.read_frame(self.wake_word.frame_length)
                wake_result = await self.wake_word.detect(wake_frame, sample_rate=self.sample_rate)
                if not wake_result.detected:
                    continue

                audio = await self._record_utterance(source)
                if audio is None:
                    continue
                break

        path = self._write_temporary_wav(audio)
        try:
            transcript = await self.stt.transcribe(path)
        finally:
            path.unlink(missing_ok=True)

        response = await self.brain.run(transcript.text)
        return VoiceCommandResult(
            wake_word=wake_result.keyword,
            transcript=transcript,
            response=response,
        )

    async def run_forever(self, on_command=None) -> None:
        while True:
            result = await self.listen_once()
            if result is not None and on_command is not None:
                callback_result = on_command(result)
                if inspect.isawaitable(callback_result):
                    await callback_result

    async def _record_utterance(self, source: PCMFrameSource) -> bytes | None:
        vad = self.vad
        if vad is None:
            vad = await self.vad_factory()
            self.vad = vad

        reset_stream = getattr(vad, "reset_stream", None)
        if reset_stream is not None:
            await asyncio.to_thread(reset_stream)

        frames = []
        speech_started = False
        silent_frames = 0
        for frame_index in range(self.max_utterance_frames):
            frame = await source.read_frame(self.frame_samples)
            if len(frame) != self.frame_samples * 2:
                raise ValueError("PulseAudio source returned an incomplete int16 frame")
            frames.append(frame)
            detector = getattr(vad, "detect_stream", None)
            if detector is None:
                detector = vad.detect
            activity = await detector(frame, sample_rate=self.sample_rate)
            if activity.is_speech:
                speech_started = True
                silent_frames = 0
            elif speech_started:
                silent_frames += 1
                if silent_frames >= self.end_silence_frames:
                    break
            elif frame_index + 1 >= self.speech_start_frames:
                return None

        if not speech_started:
            return None
        return b"".join(frames)

    def _write_temporary_wav(self, audio: bytes) -> Path:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary_file:
            path = Path(temporary_file.name)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio)
        return path
