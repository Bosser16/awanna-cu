#include <iostream>
#include <string>

#include "readfile.hpp"
#include "bresenham.hpp"

// File path is relative from .cpp file location?
std::string FILEPATH = "../";
std::string FILENAME = "srtm_14_04_6000x6000_short16.raw";
int WIDTH = 6000;
int HEIGHT = 6000;


int main() {
    std::cout << "From main file" << std::endl;

    int16_t* data = READFILE_H::read_file_to_array(FILEPATH + FILENAME, WIDTH, HEIGHT);

    std::cout << "First elevation is " << data[0] << std::endl;

    // Starting: (0, 0)
	int x1 = 0;
	int y1 = 0;

	// Stopping: (7, 6)
	int x2 = 7;
	int y2 = 5;

    int length = x2 - x1 + 1;

	// Up and to the right
	std::cout << "up-right: Starting at (" << x1 << ", " << y1 << ") and stopping at (" << x2 << ", " << y2 << ")" << std::endl;
	std::tuple<int, int>* line = BRESENHAM_H::plot_line(x1, y1, x2, y2);

    for (int i = 0; i < length; i++) {
        std::cout << "(" << std::get<0>(line[i]) << "," << std::get<1>(line[i]) << ")" << std::endl;
    }

	// Up and to the left
	std::cout << "\nup-left: Starting at (" << x1 << ", " << y1 << ") and stopping at (" << -x2 << ", " << y2 << ")" << std::endl;
	line = BRESENHAM_H::plot_line(x1, y1, -x2, y2);

    for (int i = 0; i < length; i++) {
        std::cout << "(" << std::get<0>(line[i]) << "," << std::get<1>(line[i]) << ")" << std::endl;
    }

	// Down and to the left
	std::cout << "\ndown-left: Starting at (" << x1 << ", " << y1 << ") and stopping at (" << -x2 << ", " << -y2 << ")" << std::endl;
	line = BRESENHAM_H::plot_line(x1, y1, -x2, -y2);

    for (int i = 0; i < length; i++) {
        std::cout << "(" << std::get<0>(line[i]) << "," << std::get<1>(line[i]) << ")" << std::endl;
    }

	// Down and to the left
	std::cout << "\ndown-right: Starting at (" << x1 << ", " << y1 << ") and stopping at (" << x2 << ", " << -y2 << ")" << std::endl;
	line = BRESENHAM_H::plot_line(x1, y1, x2, -y2);

    for (int i = 0; i < length; i++) {
        std::cout << "(" << std::get<0>(line[i]) << "," << std::get<1>(line[i]) << ")" << std::endl;
    }

    return 0;
}