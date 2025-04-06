cmake_minimum_required(VERSION 3.31)

project(awannacu
    VERSION 1.0
    DESCRIPTION "Viewshed computation using Bresenham's Line Algorithm. Includes serial implementation and shared/distributed memory CPU/GPU implementations."
    LANGUAGES CXX)

include_directories(src/common)


# SERIAL IMPLEMENTATION

add_executable(awannacu-serial
    src/serial.cpp
    src/common/constants.hpp
    src/common/file_io.cpp
    src/common/file_io.hpp
    src/common/viewshed.cpp
    src/common/viewshed.hpp)

target_compile_features(awannacu-serial PRIVATE cxx_std_20)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_compile_options(awannacu-serial PRIVATE /W4)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_options(awannacu-serial PRIVATE -Wall -Wextra -Wpedantic)
endif()


# CPU - SHARED MEMORY IMPLEMENTATION - OpenMP

add_executable(awannacu-cpu-shared
    src/cpu_shared.cpp
    src/common/constants.hpp
    src/common/file_io.cpp
    src/common/file_io.hpp
    src/common/viewshed.cpp
    src/common/viewshed.hpp)

find_package(OpenMP REQUIRED)

target_link_libraries(awannacu-cpu-shared PRIVATE OpenMP::OpenMP_CXX)

target_compile_features(awannacu-cpu-shared PRIVATE cxx_std_20)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_compile_options(awannacu-cpu-shared PRIVATE /W4)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_options(awannacu-cpu-shared PRIVATE -Wall -Wextra -Wpedantic)
endif()