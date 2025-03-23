#include <iostream>
#include <string>

#include "readfile.hpp"


std::string FILENAME = "srtm_14_04_6000x6000_short16.raw";
int WIDTH = 6000;
int HEIGHT = 6000;


int main() {
    std::cout << "From main file" << std::endl;

    int16_t* data = READFILE_H::read_file_to_array(FILENAME, WIDTH, HEIGHT);

    std::cout << "First elevation is " << data[0] << std::endl;

    return 0;
}