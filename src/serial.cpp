#include "constants.hpp"
#include "file_io.hpp"
#include "viewshed.cuh"

#include <iostream>
#include <string>
#include <chrono>

// File path is relative from .cpp file location?
const std::string FILEPATH = "../";
const std::string FILENAME = "srtm_14_04_6000x6000_short16.raw";
const std::string OUTPUT = "awannacu_serial.raw";

// Adjust PORTION to test different image sizes
const int PORTION = SIZE;


int main() {
    // Read data from specified file
    int16_t* data = FILEIO_H::read_file_to_array(FILEPATH + FILENAME, SIZE);

    // Create int32_t array to store visibility count
    int32_t* visible_counts = new int32_t[PORTION];

    // Get start time
    const std::chrono::time_point start = std::chrono::high_resolution_clock::now();

    // Iterate through each pixel and find the number of visible pixels in its viewshed
    for (int i = 0; i < PORTION; i++) {
        visible_counts[i] = VIEWSHED_CUH::get_visible_count(data, i);

        
        // For testing
        if (i % 100 == 0) {
            std::cout << i << std::endl;
        }
        
    }

    // Get end time and calculate duration
    const std::chrono::time_point stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop - start;

    std::cout << "Finished " << PORTION << " pixels in " << duration.count() << " ms" << std::endl;

    FILEIO_H::write_array_to_file(FILEPATH + OUTPUT, visible_counts, PORTION);

    delete(data);
    delete(visible_counts);

    return 0;
}
