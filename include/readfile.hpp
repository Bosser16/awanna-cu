#ifndef READFILE_H
#define READFILE_H

#include <string>

// Function reads an SRTM file and returns an array of short
int16_t* read_file_to_array (std::string filename, int width, int height);

#endif