# Wake-word detection

Hermes uses openWakeWord's `hey_jarvis` ONNX model at 16 kHz, with 80 ms PCM frames. The model path and shared mel/embedding ONNX files are configured under `wake_word.options` in `configs/voice.json`. The provider is loaded at backend startup and returns a positive result only after crossing its threshold and cooldown.

Install the normal voice dependencies with `pip install -r requirements-voice.txt`. On Linux/Python 3.13, upstream openWakeWord declares `tflite-runtime`, for which no compatible wheel is currently available. Hermes uses its ONNX backend, so install the package without that unused optional dependency:

```bash
python3 -m pip install --no-deps openwakeword==0.6.0
```

Download the wake-word classifier and shared feature models from the official openWakeWord v0.5.1 release into the paths in `configs/voice.json`:

```bash
mkdir -p models/voice/wake_word
curl -fL https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.onnx -o models/voice/wake_word/hey_jarvis_v0.1.onnx
curl -fL https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx -o models/voice/wake_word/melspectrogram.onnx
curl -fL https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx -o models/voice/wake_word/embedding_model.onnx
```

The upstream pre-trained wake-word weights are CC BY-NC-SA 4.0, and the included model is trained for the English phrase “Hey Jarvis”. Confirm that the model license and English phrase suit the intended deployment before shipping it.