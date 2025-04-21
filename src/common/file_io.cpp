#include "file_io.hpp"

#include <iostream>
#include <fstream>


// Takes the name of an SRTM file and the size
// Returns an elevation data array of int16_t (short)
int16_t* read_file_to_array(std::string filename, int size) {
    // Create an array to store the data
    int16_t* data = new int16_t[size];

    // Open the file in binary mode
    std::ifstream file(filename, std::ios::binary);
    
    // Handle invalid or unopenable files
    if (!file.good()) {
        std::cerr << "Error opening file '" << filename << "'!" << std::endl;
        return data;
    }

    // Read entire file into int16_t array
    file.read(reinterpret_cast<char*>(data), size * sizeof(int16_t));

    // Close file and return the array
    file.close();
    return data;
}


// Takes the name of an output file, an offset, an array of data, and a size
// Writes the data to the file at the given offset and returns
void write_array_to_file(std::string filename, int32_t* data, int size) {
    // Open the file in binary mode
    std::ofstream file(filename, std::ios::binary);

    // Handle invalid or unopenable files
    if (!file.good()) {
        std::cerr << "Error opening file '" << filename << "'!" << std::endl;
        return;
    }

    // Move the pointer to the selected offset (from the beginning)
    //file.seekp(offset * sizeof(int32_t));

    // Write the array to the file
    file.write(reinterpret_cast<char*>(data), size * sizeof(int32_t));

    // Close file and return
    file.close();
    return;
}