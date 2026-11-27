#include "audio/audio_features.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_hermes_native, module) {
    module.doc() = "Native performance-critical primitives for Hermes";

    py::class_<hermes::audio::AudioMetrics>(module, "AudioMetrics")
        .def_readonly("rms", &hermes::audio::AudioMetrics::rms)
        .def_readonly("peak", &hermes::audio::AudioMetrics::peak)
        .def_readonly("duration_ms", &hermes::audio::AudioMetrics::duration_ms)
        .def_readonly("sample_count", &hermes::audio::AudioMetrics::sample_count);

    module.def(
        "audio_metrics",
        &hermes::audio::calculate_metrics,
        py::arg("samples"),
        py::arg("sample_rate"),
        "Calculate normalized RMS, peak amplitude, and duration for mono PCM int16 samples."
    );
}
