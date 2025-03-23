#include <iostream>
#include <fstream>

#include "readfile.hpp"

int16_t* read_file_to_array (std::string filename, int width, int height) {
    long size = width * height;
    int16_t* data = new int16_t[size];
    std::ifstream file(filename, std::ios::binary);
    
    // Handle invalid or unopenable files
    if (!file.is_open()) {
        std::cerr << "Error opening file '" << filename << "'!" << std::endl;
        return data;
    }

    file.read(reinterpret_cast<char*>(data), size * sizeof(int16_t));

    file.close();

    return data;
}