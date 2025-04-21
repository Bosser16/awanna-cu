#include "constants.hpp"
#include "file_io.hpp"
#include "viewshed.cuh"

#include <mpi.h>

#include <cuda.h>
#include <iostream>
#include <string>

// File path is relative from .cpp file location?
const std::string FILEPATH = "../";
const std::string FILENAME = "srtm_14_04_6000x6000_short16.raw";
const std::string OUTPUT = "awannacu_gpu.raw";

__global__ void kernel_viewshed(int16_t* data, int32_t* visible_counts, int start_idx, int end_idx) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * WIDTH + x;
    if (idx >= start_idx && idx <= end_idx) {
        visible_counts[idx] = VIEWSHED_CUH::get_visible_count(data, idx);
    }
}

// temp for testing
const int PORTION = 8000;

int main() {

    // Start MPI
    MPI_Init(NULL, NULL);

    // Get MPI rank and size
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    // Set variable for host data
    int16_t* data_h = nullptr;

    // Have process 0 read data from specified file
    if (rank == 0) {
        data_h = FILEIO_H::read_file_to_array(FILEPATH + FILENAME, SIZE);
    }

    // Broadcast DEM data to all ranks
    if (rank != 0) {
        data_h = new int16_t[SIZE];
    }
    MPI_Bcast(data_h, SIZE, MPI_SHORT, 0, MPI_COMM_WORLD);

    // Calculate start and end for each node
    int chunk_size = PORTION / size;
    int start_idx = rank * chunk_size;
    int end_idx = (rank == size - 1) ? PORTION : start_idx + chunk_size;

    // Calculate local size and allocate space
    int local_size = end_idx - start_idx;
    int32_t* visible_counts_h = new int32_t[local_size];

    // Device variables
    int16_t* data_d;
    int32_t* visible_counts_d;
    cudaMalloc((void **)&data_d, SIZE * sizeof(int16_t));   // All of the data
    cudaMalloc((void **)&visible_counts_d, local_size * sizeof(int32_t));   // Visible counts just for this node

    // Copy host image data to device
    cudaMemcpy(data_d, data_h, SIZE * sizeof(int16_t), cudaMemcpyHostToDevice);

    // Create timing event
    cudaEvent_t start_g, stop_g;
	cudaEventCreate(&start_g);
	cudaEventCreate(&stop_g);
	float my_time;
    float global_time;

    // Determine dimensions for the grid and blocks
    int n = 8;
    dim3 DimGrid(ceil(WIDTH / n), ceil(HEIGHT / n));
    dim3 DimBlock(n,n);

    // Time the kernel
    cudaEventRecord(start_g, 0);
    kernel_viewshed<<<DimGrid, DimBlock>>>(data_d, visible_counts_d, start_idx, end_idx);
	cudaEventRecord(stop_g, 0);

    // Calculate elapsed time
	cudaEventSynchronize(stop_g);
    cudaEventElapsedTime(&my_time, start_g, stop_g);

    // Find the maximum time spent in the kernel via MPI_MAX
    MPI_Reduce(&my_time, &global_time, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // Copy local results from device to host
    cudaMemcpy(visible_counts_h, visible_counts_d, PORTION * sizeof(int32_t), cudaMemcpyDeviceToHost);


    cudaDeviceSynchronize();

    // Process 0 calculates how much data it will receive from each other process
    int* recvcounts = nullptr;
    int* displs = nullptr;
    if (rank == 0) {
        recvcounts = new int[size];
        displs = new int[size];
        for (int i = 0; i < size; i++) {
            recvcounts[i] = (i == size - 1) ? PORTION - i * chunk_size : chunk_size;
            displs[i] = i * chunk_size;
        }
    }

    // Have process 0 gather visibility counts into int32_t array
    int32_t* visible_counts = nullptr;
    if (rank == 0) {
        visible_counts = new int32_t[PORTION];
    }
    MPI_Gatherv(visible_counts_h, local_size, MPI_INT,
        visible_counts, recvcounts, displs, MPI_INT,
        0, MPI_COMM_WORLD);

    // Process 0 prints out the maximum time spent then writes the visible counts to a file
    if (rank == 0) {
        std::cout << "Finished " << PORTION << " pixels in " << global_time << " ms" << std::endl;
        FILEIO_H::write_array_to_file(FILEPATH + OUTPUT, visible_counts, PORTION);
    }

    
    // Cleanup
    delete(data_h);
    delete(visible_counts_h);
    cudaFree(data_d);
    cudaFree(visible_counts_d);


    return 0;
}
