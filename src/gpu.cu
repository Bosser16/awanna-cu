#include "constants.hpp"
#include "file_io.hpp"
#include "viewshed.cuh"

#include <cuda.h>
#include <iostream>
#include <string>

// File path is relative from .cpp file location?
const std::string FILEPATH = "../";
const std::string FILENAME = "srtm_14_04_6000x6000_short16.raw";
const std::string OUTPUT = "awannacu_serial.raw";

__global__ void kernel_viewshed(int16_t* data, int32_t* visible_counts, int portion) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < portion) {
        visible_counts[idx] = VIEWSHED_CUH::get_visible_count(data, idx);
    }
}

__global__ void hello_kernel(){
    int idx = threadIdx.x;
    printf("Hello World from : %d",idx);
}


int main() {

    hello_kernel<<<1,1>>>();
    cudaDeviceSynchronize();


    return 0;
}
