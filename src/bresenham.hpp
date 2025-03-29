#ifndef BRESENHAM_H
#define BRESENHAM_H

#include <tuple>

// Takes integer coordinates and returns a list of coordinates forming a line between them
std::tuple<int, int>* plot_line(int x1, int y1, int x2, int y2);

#endif