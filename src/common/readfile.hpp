#ifndef READFILE_H
#define READFILE_H

#include <string>

// Takes the name of an SRTM file and the data dimensions
// Returns an elevation data array of int16_t (short)
int16_t* read_file_to_array (std::string filename, int width, int height);

#endif