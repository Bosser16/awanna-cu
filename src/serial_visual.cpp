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

// Starting pixel
const int start_x = 0;
const int start_y = 0;
// Visualization size
const int square_size = 200;

int main() {
    // Read data from specified file
    int16_t* data = FILEIO_H::read_file_to_array(FILEPATH + FILENAME, SIZE);

    // Create int32_t array to store visibility count
    int32_t* visible_counts = new int32_t[square_size * square_size];

    // Get start time
    const std::chrono::time_point start = std::chrono::high_resolution_clock::now();

    // Iterate through each pixel in a square to find the number of visible pixels in its viewshed
    for(int y = 0; y < square_size; y++){
        for (int x = 0; x < square_size; x++) {
            // Calculate the global pixel index
            int global_x = start_x + x;
            int global_y = start_y + y;

            // Calculate the pixel index in the data array
            int pixel_index = global_y * WIDTH + global_x;

            int flat_index = y * square_size + x;
            visible_counts[flat_index] = VIEWSHED_CUH::get_visible_count(data, pixel_index);
    
            // Log progress
            if (flat_index % 100 == 0) {
                std::cout << flat_index << std::endl;
            }
            
        }
    }

    // Get end time and calculate duration
    const std::chrono::time_point stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop - start;

    std::cout << "Finished " << square_size*square_size << " pixels in " << duration.count() << " ms" << std::endl;

    FILEIO_H::write_array_to_file(FILEPATH + OUTPUT, visible_counts, square_size * square_size);

    delete(data);
    delete(visible_counts);

    return 0;
}
