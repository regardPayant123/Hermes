#pragma once

#include <cstddef>
#include <cstdint>

#include <pybind11/numpy.h>

namespace hermes::audio {

struct AudioMetrics {
    double rms;
    double peak;
    double duration_ms;
    std::size_t sample_count;
};

AudioMetrics calculate_metrics(
    pybind11::array_t<std::int16_t, pybind11::array::c_style | pybind11::array::forcecast> samples,
    int sample_rate
);

}  // namespace hermes::audio
