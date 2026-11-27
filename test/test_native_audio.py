import math

import numpy as np
import pytest

from engines.native import audio_metrics


def test_audio_metrics_for_pcm16_samples():
    samples = np.array([0, 32767, -32768], dtype=np.int16)

    metrics = audio_metrics(samples, 16000)

    expected_rms = math.sqrt((0.0 + (32767 / 32768) ** 2 + 1.0) / 3)
    assert math.isclose(metrics.rms, expected_rms, rel_tol=1e-12)
    assert metrics.peak == 1.0
    assert metrics.duration_ms == 0.1875
    assert metrics.sample_count == 3


@pytest.mark.parametrize(
    ("samples", "sample_rate"),
    [
        (np.array([], dtype=np.int16), 16000),
        (np.zeros((2, 2), dtype=np.int16), 16000),
        (np.array([0], dtype=np.int16), 0),
    ],
)
def test_audio_metrics_rejects_invalid_input(samples, sample_rate):
    with pytest.raises(ValueError):
        audio_metrics(samples, sample_rate)
