cmake_minimum_required(VERSION 3.18)
project(hermes_native LANGUAGES CXX)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_CMAKE_RESULT
)
if(NOT PYBIND11_CMAKE_RESULT EQUAL 0)
    message(FATAL_ERROR "pybind11 is required. Install it with: python -m pip install pybind11")
endif()
list(PREPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_hermes_native
    bindings/module.cpp
    audio/audio_features.cpp
)
target_compile_features(_hermes_native PRIVATE cxx_std_17)
target_include_directories(_hermes_native PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(_hermes_native PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

install(TARGETS _hermes_native LIBRARY DESTINATION engines/native)
