"""Python access to Hermes' optional native C++ extension."""

try:
    from ._hermes_native import AudioMetrics, audio_metrics
except ImportError as exc:
    raise ImportError(
        "Hermes native extension is not built. Follow cpp/README.md to build and install it."
    ) from exc

__all__ = ["AudioMetrics", "audio_metrics"]
