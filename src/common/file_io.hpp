#ifndef FILEIO_H
#define FILEIO_H

#include <string>


// Takes the name of an SRTM file and the size
// Returns an elevation data array of int16_t (short)
int16_t* read_file_to_array(std::string filename, int size);


// Takes the name of an output file, an offset, an array of data, and a size
// Writes the data to the file at the given offset and returns
void write_array_to_file(std::string filename, int32_t* data, int size);


#endif