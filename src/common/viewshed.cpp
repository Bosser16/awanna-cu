#include "constants.hpp"
#include "viewshed.hpp"

#include <cmath>
#include <iostream>


// Takes test coordinates, data dimensions, radius, and main coordinates
// Returns if given pixel is a valid coordinate and within a radius of the main pixel
bool is_valid_point(int x2, int y2, int x, int y) {
    // Test pixel and main pixel must be different
    if (x2 == x && y2 == y) return false;

    // Pixel must be within the width of the data
    if (x2 < 0 || x2 >= WIDTH) return false;
    
    // Pixel must be within the height of the data
    if (y2 < 0 || y2 >= HEIGHT) return false;

    // Pixel must be within a radius of the main pixel
    int dx = x - x2;
    int dy = y - y2;

    if (RADIUS < std::sqrt((dx * dx) + (dy * dy))) return false;

    // Otherwise, pixel is valid
    return true;
};


// Takes data, width, and the coordinates of two points
// Uses Bresenham's Line Algorithm: https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
// Based on implementation given by Professor Petruzza
// Returns if an uninterrupted line can be drawn between the points
bool is_visible(int16_t* data, int x1, int y1, int x2, int y2) {
    double limit;

    // Compute differences between start and end points
    const int dx = x2 - x1;
    const int dy = y2 - y1;
    int tdx = dx;
    int tdy = dy;
    
    // Compute absolute values of the change in x and y
    const int abs_dx = abs(dx);
    const int abs_dy = abs(dy);

    // Set initial point
    int x = x1;
    int y = y1;

    // Calculate initial elevation and slope (change in elevation per pixel distance)
    int16_t elevation = data[WIDTH * y1 + x1];
    double distance = std::sqrt((dx * dx) + (dy * dy));
    double slope = (data[WIDTH * y2 + x2] - elevation) / distance;

    // Proceed on the absolute differences to support all octants
    if (abs_dx > abs_dy) {
        // If the line is moving to the left, set dx accordingly
        int dx_update;
        if (dx > 0) {
            dx_update = 1;
        } else {
            dx_update = -1;
        }

        // Calculate the initial decision parameter
        int p = 2 * abs_dy - abs_dx;

        // Plot the line for the x-major case
        for (int i = 0; i <= abs_dx; i++) {
            // Only check if the coordinates are not the same as p1 and p2
            if (!(x == x1 && y == y1) && !(x == x2 && y == y2)) {
                // Calculate elevation limit
                tdx = x - x1;
                tdy = y - y1;
                distance = std::sqrt((tdx * tdx) + (tdy * tdy));
                limit = slope * distance + elevation;

                // Return false if current elevation is higher than limit
                if (data[WIDTH * y + x] > limit) {
                    return false;
                }
            }

            // Threshold for deciding whether or not to update y
            if (p < 0) {
                p += 2 * abs_dy;
            } else {
                // Update y
                if (dy >= 0) {
                    y++;
                } else {
                    y--;
                }

                p += 2 * abs_dy - 2 * abs_dx;
            }

            // Always update x
            x += dx_update;
        }
    } else {
        // If the line is moving downwards, set dy accordingly
        int dy_update;
        if (dy > 0) {
            dy_update = 1;
        } else {
            dy_update = -1;
        }

        // Calculate the initial decision parameter
		int p = 2 * abs_dx - abs_dy;

        // Plot the line for the y-major case
        for (int i = 0; i <= abs_dy; i++) {
            // Only check if the coordinates are not the same as p1 and p2
            if (!(x == x1 && y == y1) && !(x == x2 && y == y2)) {
                // Calculate elevation limit
                tdx = x - x1;
                tdy = y - y1;
                distance = std::sqrt((tdx * tdx) + (tdy * tdy));
                limit = slope * distance + elevation;

                // Return false if current elevation is higher than limit
                if (data[WIDTH * y + x] > limit) {
                    return false;
                }
            }

            // Threshold for deciding whether or not to update x
            if (p < 0) {
                p += 2 * abs_dx;
            } else {
                // Update x
                if (dx >= 0) {
                    x++;
                } else {
                    x--;
                }

                p += 2 * abs_dx - 2 * abs_dy;
            }

            // Always update y
            y += dy_update;
        }
    }

    return true;
}


int32_t get_visible_count(int16_t* data, int pixel) {
    int32_t visible = 0;

    // Verify that the pixel is in bounds
    if (pixel >= SIZE) {
        std::cerr << "Pixel " << pixel << " is out of bounds of the data!" << std::endl;
        return visible;
    }

    int x = pixel % WIDTH;
    int y = pixel / HEIGHT;
    
    // Iterate through a square with length 2 * radius around the main pixel
    for (int x2 = x - RADIUS; x2 <= x + RADIUS; x2++) {
        for (int y2 = y - RADIUS; y2 <= y + RADIUS; y2++) {
            // Check if the point is valid
            if (is_valid_point(x2, y2, x, y)) {
                if (is_visible(data, x, y, x2, y2)) visible++;
            }
        }
    }

    return visible;
}