#include "constants.hpp"
#include "file_io.hpp"

#include <iostream>
#include <string>
#include <chrono>

// File path is relative from .cpp file location?
const std::string FILEPATH = "../";
const std::string FILENAME1 = "awannacu_serial.raw";
const std::string FILENAME2 = "awannacu_cpu_shared.raw";

bool isValid(int16_t* data1, int16_t* data2, int height, int width){
    int x,y;

    for(x = 0; x < height; x++){
        for(y = 0; y < width; y++){
            int index = x * width + y;
            if(data1[index] != data2[index])
                return false;
        }
    }

    return true;
}

int main() {
    // Read data from specified file
    int16_t* data1 = FILEIO_H::read_file_to_array(FILEPATH + FILENAME1, SIZE);
    int16_t* data2 = FILEIO_H::read_file_to_array(FILEPATH + FILENAME2, SIZE);

    if(isValid(data1, data2, HEIGHT, WIDTH))
        std::cout << "VALID\n" << FILENAME1 << " and " << FILENAME2 << " contain the same data" << std::endl;
    else
        std::cout << "INVALID\n" << FILENAME1 << " and " << FILENAME2 << " contain the different data" << std::endl;


    delete(data1);
    delete(data2);

    return 0;
}
