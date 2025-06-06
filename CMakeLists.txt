cmake_minimum_required(VERSION 3.26)

project(awannacu
    VERSION 1.0
    DESCRIPTION "Viewshed computation using Bresenham's Line Algorithm. Includes serial implementation and shared/distributed memory CPU/GPU implementations."
    LANGUAGES CXX CUDA)

include_directories(src/common)


# SERIAL IMPLEMENTATION

add_executable(awannacu-serial
    src/serial.cpp
    src/common/constants.hpp
    src/common/file_io.cpp
    src/common/file_io.hpp
    src/common/viewshed.cuh)

target_compile_features(awannacu-serial PRIVATE cxx_std_20)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_compile_options(awannacu-serial PRIVATE /W4)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_options(awannacu-serial PRIVATE -Wall -Wextra -Wpedantic)
endif()

# SERIAL VISUAL IMPLEMENTATION

add_executable(awannacu-serial-visual
    src/serial_visual.cpp
    src/common/constants.hpp
    src/common/file_io.cpp
    src/common/file_io.hpp
    src/common/viewshed.cuh)

target_compile_features(awannacu-serial-visual PRIVATE cxx_std_20)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_compile_options(awannacu-serial-visual PRIVATE /W4)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_options(awannacu-serial-visual PRIVATE -Wall -Wextra -Wpedantic)
endif()

# VALIDATOR

add_executable(awannacu-validate
    src/validator.cpp
    src/common/constants.hpp
    src/common/file_io.cpp
    src/common/file_io.hpp)

target_compile_features(awannacu-validate PRIVATE cxx_std_20)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_compile_options(awannacu-validate PRIVATE /W4)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_options(awannacu-validate PRIVATE -Wall -Wextra -Wpedantic)
endif()

# CPU - SHARED MEMORY IMPLEMENTATION - OpenMP

add_executable(awannacu-cpu-shared
    src/cpu_shared.cpp
    src/common/constants.hpp
    src/common/file_io.cpp
    src/common/file_io.hpp
    src/common/viewshed.cuh)

find_package(OpenMP REQUIRED)

target_link_libraries(awannacu-cpu-shared PRIVATE OpenMP::OpenMP_CXX)

target_compile_features(awannacu-cpu-shared PRIVATE cxx_std_20)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_compile_options(awannacu-cpu-shared PRIVATE /W4)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_options(awannacu-cpu-shared PRIVATE -Wall -Wextra -Wpedantic)
endif()

# CPU - DISTRIBUTED MEMORY IMPLEMENTATION - MPI

add_executable(awannacu-cpu-distributed
    src/cpu_distributed.cpp
    src/common/constants.hpp
    src/common/file_io.cpp
    src/common/file_io.hpp
    src/common/viewshed.cuh)

find_package(MPI REQUIRED)

target_link_libraries(awannacu-cpu-distributed PRIVATE MPI::MPI_CXX)

target_compile_features(awannacu-cpu-distributed PRIVATE cxx_std_20)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_compile_options(awannacu-cpu-distributed PRIVATE /W4)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_options(awannacu-cpu-distributed PRIVATE -Wall -Wextra -Wpedantic)
endif()

# GPU IMPLEMENTATION
include(CheckLanguage)
check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    message(STATUS "CUDA found. Building GPU implementations.")

    add_executable(awannacu-gpu
        src/gpu.cu
        src/common/constants.hpp
        src/common/file_io.cpp
        src/common/file_io.hpp
        src/common/viewshed.cuh)
        
        add_executable(awannacu-gpu-distributed
        src/gpu_distributed.cu
        src/common/constants.hpp
        src/common/file_io.cpp
        src/common/file_io.hpp
        src/common/viewshed.cuh)

        target_link_libraries(awannacu-gpu-distributed PRIVATE MPI::MPI_CXX)

# Suppresses warnings because there are a lot, could fix warnings if needed tho
    target_compile_options(awannacu-gpu PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w>
    )

    target_compile_options(awannacu-gpu-distributed PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w>
    )

    if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
        target_compile_options(awannacu-gpu PRIVATE /W4)
    elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_compile_options(awannacu-gpu PRIVATE -Wall -Wextra -Wpedantic)
    endif()

else()
    message(WARNING "CUDA not found. Skipping GPU build.")
endif()
