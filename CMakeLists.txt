cmake_minimum_required(VERSION 3.31)

project(awannacu
    VERSION 1.0
    DESCRIPTION "Viewshed computation using Bresenham's Line Algorithm. Includes serial implementation and shared/distributed memort CPU/GPU implementations."
    LANGUAGES CXX)

add_executable(awannacu-serial
    src/serial.cpp
    src/common/file_io.cpp
    src/common/file_io.hpp
    src/common/bresenham.cpp
    src/common/bresenham.hpp
    src/common/viewshed.cpp
    src/common/viewshed.hpp)

target_compile_features(awannacu-serial PRIVATE cxx_std_20)

include_directories(src/common)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_compile_options(awannacu-serial PRIVATE /W4)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_options(awannacu-serial PRIVATE -Wall -Wextra -Wpedantic)
endif()