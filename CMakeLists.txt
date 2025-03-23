cmake_minimum_required(VERSION 3.31)

project(awannacu
    VERSION 1.0
    DESCRIPTION "Viewshed computation using Bresenham's Line Algorithm. Includes serial implementation and shared/distributed memort CPU/GPU implementations."
    LANGUAGES CXX)

add_executable(awannacu-serial
    src/serial.cpp
    src/readfile.cpp
    src/readfile.hpp)

target_compile_features(awannacu-serial PRIVATE cxx_std_20)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_compile_options(awannacu-serial PRIVATE /W4)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_options(awannacu-serial PRIVATE -Wall -Wextra -Wpedantic)
endif()