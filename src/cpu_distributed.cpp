#include "constants.hpp"
#include "file_io.hpp"
#include "viewshed.hpp"

#include <mpi.h>

#include <iostream>
#include <string>
#include <chrono>

// File path is relative from .cpp file location?
const std::string FILEPATH = "../";
const std::string FILENAME = "srtm_14_04_6000x6000_short16.raw";
const std::string OUTPUT = "awannacu_cpu_distributed.raw";

// temp for testing
const int PORTION = 2000 * 4;


int main() {
    MPI_Init(NULL, NULL);

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int16_t* data = nullptr;

    // Have process 0 read data from specified file
    if (rank == 0) {
        data = FILEIO_H::read_file_to_array(FILEPATH + FILENAME, SIZE);
    }

    // Broadcast DEM data to all ranks (I don't actually think I want to do this lol)
    if (rank != 0) {
        data = new int16_t[SIZE];
    }
    MPI_Bcast(data, SIZE, MPI_SHORT, 0, MPI_COMM_WORLD);

    int chunk_size = PORTION / size;
    int start_idx = rank * chunk_size;
    int end_idx = (rank == size - 1) ? PORTION : start_idx + chunk_size;

    int local_size = end_idx - start_idx;
    int32_t* local_counts = new int32_t[local_size];

    // track start time on rank 0
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    if (rank == 0) {
        start = std::chrono::high_resolution_clock::now();
    }

    for (int i = 0; i < local_size; i++) {
        local_counts[i] = VIEWSHED_H::get_visible_count(data, start_idx + i);

            /*
            // For testing
            if (i % 100 == 0) {
                std::cout << i << std::endl;
            }
            */
    }



    // Have process 0 gather visibility counts into int32_t array
    int32_t* visible_counts = nullptr;
    if (rank == 0) {
        visible_counts = new int32_t[PORTION];
    }

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

    MPI_Gatherv(local_counts, local_size, MPI_INT,
        visible_counts, recvcounts, displs, MPI_INT,
        0, MPI_COMM_WORLD);

    // Get end time and calculate duration
    if (rank == 0) {
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = stop - start;

        std::cout << "Finished " << PORTION << " pixels in " << duration.count() << " ms" << std::endl;

        FILEIO_H::write_array_to_file(FILEPATH + OUTPUT, visible_counts, PORTION);

        delete[] visible_counts;
        delete[] recvcounts;
        delete[] displs;
    }

    delete[] data;
    delete[] local_counts;

    MPI_Finalize();
    return 0;
}