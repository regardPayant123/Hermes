"""WebRTC implementation of the VAD provider interface."""

from pathlib import Path

from .interface import AudioInput, VADProvider, VoiceActivity


class WebRTCVAD(VADProvider):
    def __init__(self, aggressiveness: int = 2, frame_duration_ms: int = 30):
        if aggressiveness not in range(4):
            raise ValueError("aggressiveness must be between 0 and 3")
        if frame_duration_ms not in {10, 20, 30}:
            raise ValueError("frame_duration_ms must be 10, 20, or 30")
        self.aggressiveness = aggressiveness
        self.frame_duration_ms = frame_duration_ms

    async def detect(self, audio: AudioInput, *, sample_rate: int = 16000) -> VoiceActivity:
        self.validate_request(audio, sample_rate)
        if sample_rate not in {8000, 16000, 32000, 48000}:
            raise ValueError("WebRTC supports 8000, 16000, 32000, or 48000 Hz audio")

        try:
            import webrtcvad
        except ImportError as exc:
            raise RuntimeError("Install WebRTC VAD with: pip install webrtcvad") from exc

        vad = webrtcvad.Vad(self.aggressiveness)
        data = bytes(audio)
        frame_size = int(sample_rate * self.frame_duration_ms / 1000) * 2
        speech_frames = 0
        total_frames = 0
        first_speech_ms = None

        for offset in range(0, len(data) - frame_size + 1, frame_size):
            frame = data[offset:offset + frame_size]
            is_speech = vad.is_speech(frame, sample_rate)
            timestamp_ms = offset // frame_size * self.frame_duration_ms
            if is_speech:
                speech_frames += 1
                if first_speech_ms is None:
                    first_speech_ms = timestamp_ms
            total_frames += 1

        confidence = speech_frames / total_frames if total_frames else 0.0
        is_speech = speech_frames > 0
        duration_ms = total_frames * self.frame_duration_ms
        return VoiceActivity(
            is_speech=is_speech,
            confidence=confidence,
            start_ms=first_speech_ms,
            end_ms=duration_ms if is_speech else None,
        )
