#include "audio_features.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <pybind11/pybind11.h>

namespace hermes::audio {

AudioMetrics calculate_metrics(
    pybind11::array_t<std::int16_t, pybind11::array::c_style | pybind11::array::forcecast> samples,
    int sample_rate
) {
    if (sample_rate <= 0) {
        throw std::invalid_argument("sample_rate must be greater than zero");
    }

    const auto buffer = samples.request();
    if (buffer.ndim != 1) {
        throw std::invalid_argument("samples must be a one-dimensional array");
    }
    if (buffer.size == 0) {
        throw std::invalid_argument("samples cannot be empty");
    }

    const auto* values = static_cast<const std::int16_t*>(buffer.ptr);
    const auto count = static_cast<std::size_t>(buffer.size);
    double sum_squares = 0.0;
    std::int32_t peak = 0;

    {
        pybind11::gil_scoped_release release;
        for (std::size_t index = 0; index < count; ++index) {
            const auto sample = static_cast<std::int32_t>(values[index]);
            const auto magnitude = std::abs(sample);
            peak = std::max(peak, magnitude);
            const double normalized = static_cast<double>(sample) / 32768.0;
            sum_squares += normalized * normalized;
        }
    }

    return {
        std::sqrt(sum_squares / static_cast<double>(count)),
        static_cast<double>(peak) / 32768.0,
        static_cast<double>(count) * 1000.0 / static_cast<double>(sample_rate),
        count,
    };
}

}  // namespace hermes::audio
