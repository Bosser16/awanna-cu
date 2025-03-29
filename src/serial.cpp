#include "file_io.hpp"
#include "viewshed.hpp"

#include <iostream>
#include <string>
#include <chrono>

// File path is relative from .cpp file location?
const std::string FILEPATH = "../";
const std::string FILENAME = "srtm_14_04_6000x6000_short16.raw";
const int WIDTH = 6000;
const int HEIGHT = 6000;
const int RADIUS = 100;
const int SIZE = WIDTH * HEIGHT;


int main() {
    // Read data from specified file
    int16_t* data = FILEIO_H::read_file_to_array(FILEPATH + FILENAME, SIZE);

    // Create int32_t array to store visibility count
    int32_t* visible_counts = new int32_t[SIZE];

    // Get start time
    const std::chrono::time_point start = std::chrono::high_resolution_clock::now();

    // Iterate through each pixel and find the number of visible pixels in its viewshed
    for (int i = 0; i < SIZE; i++) {
        visible_counts[i] = VIEWSHED_H::get_visible_count(data, WIDTH, HEIGHT, RADIUS, i);
    }

    // Get end time and calculate duration
    const std::chrono::time_point stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop - start;

    std::cout << "Finished " << SIZE << " pixels in " << duration.count() << " ms" << std::endl;

    return 0;
}