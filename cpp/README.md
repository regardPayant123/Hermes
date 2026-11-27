# Hermes Native C++

This directory contains C++ operations exposed to Python through pybind11. Python remains responsible for orchestration, configuration, and model selection; native code is reserved for measured compute-heavy loops.

## Build

From the repository root, install the build tools once:

```bash
python3 -m pip install -r requirements-cpp.txt
```

Configure, compile, and install the extension into `engines/native`:

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD"
cmake --build cpp/build --parallel
cmake --install cpp/build
```

Re-run the install step after each rebuild so Python receives the updated extension. Generated build files and shared libraries are ignored by Git.

## Audio primitive

`engines.native.audio_metrics(samples, sample_rate)` accepts a one-dimensional NumPy-compatible array of mono PCM signed 16-bit samples and returns normalized RMS, peak amplitude, duration in milliseconds, and sample count. The native sample loop releases the Python GIL.

```python
import numpy as np
from engines.native import audio_metrics

metrics = audio_metrics(np.frombuffer(pcm_bytes, dtype=np.int16), 16000)
print(metrics.rms, metrics.peak, metrics.duration_ms)
```

This first primitive validates the extension boundary; it is not evidence of a speedup by itself. Benchmark representative buffers against the current Python/NumPy pipeline before moving additional work into C++.
