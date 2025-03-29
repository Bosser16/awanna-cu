#ifndef BRESENHAM_H
#define BRESENHAM_H

#include <tuple>

// Takes two sets of integer coordinates
// Returns a list of coordinates forming a line between the given coordinates
std::tuple<int, int>* plot_line(int x1, int y1, int x2, int y2);

#endif