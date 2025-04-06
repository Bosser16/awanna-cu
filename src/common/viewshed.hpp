#ifndef VIEWSHED_H
#define VIEWSHED_H

#include <cstdint>

// Takes elevation data, data dimensions, a radius, and coordinates of a pixel
// Returns the number of other pixels visible within the radius given
int32_t get_visible_count(int16_t* data, int pixel);

#endif