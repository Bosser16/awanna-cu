#include <cmath>
#include <tuple>

// Bresenham's Line Algorithm: https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
// Based on implementation given by Professor Petruzza

// Plots a line between points (x1, y1) and (x2, y2) and returns a tuple array containing the intermediate coordinates
std::tuple<int, int>* plot_line(int x1, int y1, int x2, int y2) {
    std::tuple<int, int>* coordinates;

    // Compute differences between start and end points
    int dx = x2 - x1;
    int dy = y2 - y1;

    // Compute absolute values of the change in x and y
    const int abs_dx = abs(dx);
    const int abs_dy = abs(dy);

    // Set initial point
    int x = x1;
    int y = y1;

    // Proceed on the absolute differences to support all octants
    if (abs_dx > abs_dy) {

        // Allocate space for tuple array of coordinates of length abs_dx
        coordinates = new std::tuple<int, int>[abs_dx + 1];

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
            // Add the current coordinate
            coordinates[i] = std::tuple<int, int>(x, y);

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

        // Allocate space for tuple array of coordinates of length abs_dy
        coordinates = new std::tuple<int, int>[abs_dy + 1];

        // If the line is moving downwards, set dy accordingly
        int dy_update;
        if (dy > 0) {
            dy_update = 1;
        } else {
            dy_update = -1;
        }

        // Calculate the initial decision parameter
		int p = 2 * abs_dx - abs_dy;

		// Draw the line for the y-major case
        for (int i = 0; i <= abs_dy; i++) {
            // Add the current coordinate
            coordinates[i] = std::tuple<int, int>(x, y);

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

    return coordinates;
}