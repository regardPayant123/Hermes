import asyncio

import numpy as np
import pytest

from engines.voice.wake_word.openwakeword import OpenWakeWordProvider


class FakeWakeWordModel:
    def __init__(self):
        self.frames = 0

    def predict(self, frame):
        self.frames += 1
        return {"hey_jarvis_v0.1": 0.9}


def test_wake_word_buffers_partial_frames_and_applies_cooldown():
    async def run():
        provider = OpenWakeWordProvider(
            "wake.onnx",
            "melspectrogram.onnx",
            "embedding_model.onnx",
            cooldown_seconds=10,
        )
        fake_model = FakeWakeWordModel()
        provider._model = fake_model

        first = await provider.detect(np.zeros(1000, dtype=np.int16).tobytes())
        assert not first.detected
        assert fake_model.frames == 0

        completion = await provider.detect(np.zeros(280, dtype=np.int16).tobytes())
        assert completion.detected
        assert completion.confidence == 0.9
        assert fake_model.frames == 1

        cooldown = await provider.detect(np.zeros(1280, dtype=np.int16).tobytes())
        assert not cooldown.detected
        assert cooldown.confidence == 0.9
        assert fake_model.frames == 2

    asyncio.run(run())


@pytest.mark.parametrize(
    ("audio", "sample_rate"),
    [(b"\x00", 16000), (b"\x00\x00", 8000)],
)
def test_wake_word_rejects_malformed_audio(audio, sample_rate):
    async def run():
        provider = OpenWakeWordProvider("wake.onnx", "mel.onnx", "embed.onnx")
        with pytest.raises(ValueError):
            await provider.detect(audio, sample_rate=sample_rate)

    asyncio.run(run())