# Live Voice Input

The live command path is:

```text
PulseAudio/pipewire-pulse (parec, PCM s16le mono 16 kHz)
  -> OpenWakeWord Hey Jarvis classifier
  -> Silero VAD endpointing
  -> temporary WAV
  -> faster-whisper
  -> brain.run(transcript)
```

Install Python dependencies with `pip install -r requirements-voice.txt`. Install `parec` from the OS package manager (Debian/Ubuntu: `sudo apt install pulseaudio-utils`) and ensure PulseAudio or `pipewire-pulse` is running. Use `pactl list short sources` to select an input source; set `audio_input.options.device` in `configs/voice.json`, or leave it `null` for the default source.

Wake-word ONNX weights and the shared feature models must exist at the paths configured under `wake_word.options` in `configs/voice.json`. The bundled Hey Jarvis model is English-only. Its pre-trained weights are CC BY-NC-SA 4.0, so review that license before distribution or commercial use.

After the backend runtime is started:

```python
result = await runtime.run_live_voice_once()
print(result.transcript.text, result.response.text)
```

For always-listening mode, call `await runtime.run_live_voice_forever(on_command=handler)`. VAD loads only after OpenWakeWord detects the wake phrase. PulseAudio capture is closed before transcription and brain inference; the temporary WAV is deleted after STT.