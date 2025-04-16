#include "constants.hpp"
#include "file_io.hpp"
#include "viewshed.cuh"

#include <cuda.h>
#include <iostream>
#include <string>

// File path is relative from .cpp file location?
const std::string FILEPATH = "../";
const std::string FILENAME = "srtm_14_04_6000x6000_short16.raw";
const std::string OUTPUT = "awannacu_gpu.raw";

__global__ void kernel_viewshed(int16_t* data, int32_t* visible_counts, int portion) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * WIDTH + x;
    if (idx < portion) {
        visible_counts[idx] = VIEWSHED_CUH::get_visible_count(data, idx);
    }
}

// temp for testing
const int PORTION = 8000;

int main() {

    // Read data from specified file
    int16_t* data_h = FILEIO_H::read_file_to_array(FILEPATH + FILENAME, SIZE);

    // Create int32_t array to store visibility count
    int32_t* visible_counts_h = new int32_t[PORTION];

    // Device variables
    int16_t* data_d;
    int32_t* visible_counts_d;
    cudaMalloc((void **)&data_d, SIZE * sizeof(int16_t));
    cudaMalloc((void **)&visible_counts_d, PORTION * sizeof(int32_t));

    cudaMemcpy(data_d, data_h, SIZE * sizeof(int16_t), cudaMemcpyHostToDevice);


    cudaEvent_t start_g, stop_g;
	cudaEventCreate(&start_g);
	cudaEventCreate(&stop_g);
	float global_time;

    dim3 DimGrid(WIDTH, HEIGHT);
    dim3 DimBlock(8,8);

    cudaEventRecord(start_g, 0);
    
    kernel_viewshed<<<DimGrid, DimBlock>>>(data_d, visible_counts_d, PORTION);

	cudaEventRecord(stop_g, 0);

	cudaEventSynchronize(stop_g);
    cudaEventElapsedTime(&global_time, start_g, stop_g);
    
    cudaMemcpy(visible_counts_h, visible_counts_d, PORTION * sizeof(int32_t), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    std::cout << "Finished " << PORTION << " pixels in " << global_time << " ms" << std::endl;

    FILEIO_H::write_array_to_file(FILEPATH + OUTPUT, visible_counts_h, PORTION);

    delete(data_h);
    delete(visible_counts_h);
    cudaFree(data_d);
    cudaFree(visible_counts_d);


    return 0;
}
