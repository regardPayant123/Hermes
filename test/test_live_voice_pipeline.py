import asyncio
import wave

from engines.brain.llm.interface import LLMResponse
from engines.voice.audio.command_pipeline import LiveVoiceCommandPipeline
from engines.voice.stt.interface import Transcript
from engines.voice.vad.interface import VoiceActivity
from engines.voice.wake_word.interface import WakeWordResult


class FakeAudioSource:
    def __init__(self):
        self.frames = [b"wake-frame"] + [bytes(1280 * 2) for _ in range(5)]
        self.open = False

    async def __aenter__(self):
        self.open = True
        return self

    async def __aexit__(self, *_args):
        self.open = False
        return False

    async def read_frame(self, _frame_samples):
        return self.frames.pop(0)


class FakeWakeWord:
    frame_length = 5
    sample_rate = 16000

    async def detect(self, _frame, *, sample_rate):
        assert sample_rate == 16000
        return WakeWordResult(True, "Hey Jarvis", 1.0)


class FakeVAD:
    def __init__(self):
        self.frame_count = 0
        self.was_reset = False

    def reset_stream(self):
        self.was_reset = True

    async def detect_stream(self, _frame, *, sample_rate):
        assert sample_rate == 16000
        self.frame_count += 1
        is_speech = self.frame_count <= 2
        return VoiceActivity(is_speech, 0.9 if is_speech else 0.0)


class FakeSTT:
    def __init__(self, source):
        self.source = source

    async def transcribe(self, audio_path):
        assert not self.source.open
        with wave.open(str(audio_path), "rb") as wav_file:
            assert wav_file.getframerate() == 16000
            assert wav_file.getnchannels() == 1
            assert wav_file.getnframes() == 5 * 1280
        return Transcript("ouvre la porte", language="fr")


class FakeBrain:
    async def run(self, prompt):
        assert prompt == "ouvre la porte"
        return LLMResponse("C'est fait", "fake-brain")


def test_live_pipeline_wake_vad_temp_wav_stt_then_brain():
    async def run():
        source = FakeAudioSource()
        vad = FakeVAD()
        pipeline = LiveVoiceCommandPipeline(
            audio_source=source,
            wake_word=FakeWakeWord(),
            stt=FakeSTT(source),
            brain=FakeBrain(),
            vad=vad,
            end_silence_seconds=0.24,
        )

        result = await pipeline.listen_once()

        assert result.wake_word == "Hey Jarvis"
        assert result.transcript.text == "ouvre la porte"
        assert result.response.text == "C'est fait"
        assert vad.was_reset
        assert vad.frame_count == 5
        assert not source.open

    asyncio.run(run())