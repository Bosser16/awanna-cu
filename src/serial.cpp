#include <iostream>
#include <string>

#include "readfile.hpp"
#include "viewshed.hpp"

// File path is relative from .cpp file location?
std::string FILEPATH = "../";
std::string FILENAME = "srtm_14_04_6000x6000_short16.raw";
int WIDTH = 6000;
int HEIGHT = 6000;
int RADIUS = 100;


int main() {
    std::cout << "From main file" << std::endl;

    int16_t* data = READFILE_H::read_file_to_array(FILEPATH + FILENAME, WIDTH, HEIGHT);

    int x1 = 0;
    int y1 = 0;

    int32_t visible = VIEWSHED_H::get_visible_count(data, WIDTH, HEIGHT, RADIUS, 0);
    std::cout << visible << " visible at (" << x1 << ", " << y1 << ")" << std::endl;

    return 0;
}