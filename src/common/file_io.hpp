#ifndef FILEIO_H
#define FILEIO_H

#include <string>

// Takes the name of an SRTM file and the size
// Returns an elevation data array of int16_t (short)
int16_t* read_file_to_array (std::string filename, int size);

#endif