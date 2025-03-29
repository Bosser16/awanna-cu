#include "viewshed.hpp"
#include "bresenham.hpp"

#include <cmath>
#include <iostream>


// Takes data, width, and a set of coordinates
// Returns elevation at the coordinates
int16_t get_elevation(int16_t* data, int width, int x, int y) {
    return data[width * y + x];
}


// Takes test coordinates, data dimensions, radius, and main coordinates
// Returns if given pixel is a valid coordinate and within a radius of the main pixel
bool is_valid_point(int x2, int y2, int width, int height, int radius, int x, int y) {
    // Test pixel and main pixel must be different
    if (x2 == x && y2 == x) return false;

    // Pixel must be within the width of the data
    if (x2 < 0 || x2 >= width) return false;
    
    // Pixel must be within the height of the data
    if (y2 < 0 || y2 >= height) return false;

    // Pixel must be within a radius of the main pixel
    int dx = x - x2;
    int dy = y - y2;

    if (radius < std::sqrt((dx * dx) + (dy * dy))) return false;

    // Otherwise, pixel is valid
    return true;
};


// Takes data, width, and the coordinates of two points
// Returns if an uninterrupted line can be drawn between the points
bool is_visible(int16_t* data, int width, int x1, int y1, int x2, int y2) {
    int x, y;
    double limit;

    // Calculate differences
    int dx = x2 - x1;
    int dy = y2 - y1;
    
    // Calculate line array length (number of pixels)
    int abs_dx = abs(dx);
    int abs_dy = abs(dy);
    int length;
    if (abs_dx > abs_dy) {
        length = abs_dx + 1;
    } else {
        length = abs_dy + 1;
    }

    // Calculate initial elevation and slope (change in elevation per pixel distance)
    int16_t elevation = data[width * y1 + x1];
    double distance = std::sqrt((dx * dx) + (dy * dy));
    double slope = (data[width * y2 + x2] - elevation) / distance;

    // Generate line from p1 to p2
    std::tuple<int, int>* coordinates = BRESENHAM_H::plot_line(x1, y1, x2, y2);

    // Iterate through points in the line
    for (int i = 0; i < length; i++) {
        // Get coordinates of point
        x = std::get<0>(coordinates[i]);
        y = std::get<1>(coordinates[i]);

        // Only check if the coordinates are not the same as p1 and p2
        if (!(x == x1 && y == y1) && !(x == x2 && y == y2)) {
            // Calculate elevation limit
            dx = x - x1;
            dy = y - y1;
            distance = std::sqrt((dx * dx) + (dy * dy));
            limit = slope * distance + elevation;

            // Return false if current elevation is higher than limit
            if (data[width * y + x] > limit) return false;
        }
    }

    return true;
}


int32_t get_visible_count(int16_t* data, int width, int height, int radius, int pixel) {
    int32_t visible = 0;

    // Verify that the pixel is in bounds
    if (pixel >= width * height) {
        std::cerr << "Pixel " << pixel << " is out of bounds of the data!" << std::endl;
        return visible;
    }

    int x = pixel % width;
    int y = pixel / height;
    
    // Iterate through a square with length 2 * radius around the main pixel
    for (int x2 = x - radius; x2 <= x + radius; x2++) {
        for (int y2 = y - radius; y2 <= y + radius; y2++) {
            // Check if the point is valid
            if (is_valid_point(x2, y2, width, height, radius, x, y)) {
                if (is_visible(data, width, x, y, x2, y2)) visible++;
            }
        }
    }

    return visible;
}