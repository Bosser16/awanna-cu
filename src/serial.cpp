#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <utility>

#include "readfile.hpp"

const std::string FILENAME = "srtm_14_04_6000x6000_short16.raw";
const int WIDTH = 6000;
const int HEIGHT = 6000;
const uint8_t RADIUS = 100;


// Based off of provided bresenham_serial.cpp, but returns vector of points instead of printing them
// Traces a line between points (x1, y1) and (x2, y2) and prints the intermediate coordinates
std::vector<std::pair<int, int>> plot_line(int x1, int y1, int x2, int y2)
{
    std::vector<std::pair<int, int>> out_vector;
    
    // Compute the differences between start and end points
    int dx = x2 - x1;
	int dy = y2 - y1;
    
	// Absolute values of the change in x and y
	const int abs_dx = abs(dx);
	const int abs_dy = abs(dy);
    
	// Initial point
	int x = x1;
	int y = y1;
    
    // Proceed based on the absolute differences to support all octants
	if (abs_dx > abs_dy)
	{
        // If the line is moving to the left, set dx accordingly
		int dx_update;
		if (dx > 0)
		{
            dx_update = 1;
		}
		else
		{
            dx_update = -1;
		}
        
		// Calculate the initial decision parameter
		int p = 2 * abs_dy - abs_dx;
        
		// Draw the line for the x-major case
		for (int i = 0; i <= abs_dx; i++)
		{
            // Add current coordinate to vector
            out_vector.emplace_back(std::pair<int, int>(x, y));
            
			// Threshold for deciding whether or not to update y
			if (p < 0)
			{
				p = p + 2 * abs_dy;
			}
			else
			{
				// Update y
				if (dy >= 0)
				{
					y += 1;
				}
				else
				{
					y += -1;
				}
                
				p = p + 2 * abs_dy - 2 * abs_dx;
			}
            
			// Always update x
			x += dx_update;
		}
	}
	else
	{
		// If the line is moving downwards, set dy accordingly
		int dy_update;
		if (dy > 0)
		{
			dy_update = 1;
		}
		else
		{
			dy_update = -1;
		}
        
		// Calculate the initial decision parameter
		int p = 2 * abs_dx - abs_dy;
        
		// Draw the line for the y-major case
		for (int i = 0; i <= abs_dy; i++)
		{
			// Add current coordinate to vector
            out_vector.emplace_back(std::pair<int, int>(x, y));
            
			// Threshold for deciding whether or not to update x
			if (p < 0)
			{
                p = p + 2 * abs_dx;
			}
			else
			{
                // Update x
				if (dx >= 0)
				{
                    x += 1;
				}
				else
				{
					x += -1;
				}
                
				p = p + 2 * abs_dx - 2 * abs_dy;
			}
            
			// Always update y
			y += dy_update;
		}
	}
	
	return out_vector;
}


std::vector<double> find_altitudes(int altitude1, int altitude2, int steps)
{
	std::vector<double> altitudes;
	double altitude_diff = altitude1 - altitude2;
	double step = altitude_diff / steps;
	auto current_altitude = altitude1;
	for (int i = 0; i < steps; i ++)
	{
		altitudes.emplace_back(current_altitude);
		current_altitude += step;
	}

	return altitudes;
}


bool isVisible(
	uint16_t x1,
    uint16_t x2,
    uint16_t y1,
    uint16_t y2,
    int16_t* data,
    int width,
    int height
) {
	int16_t elevation1 = data[y1 * width + x1];
    int16_t elevation2 = data[y2 * width + x2];
	

    std::vector<std::pair<int, int>> pixel_coords = plot_line(x1, y1, x2, y2);
    std::vector<double> pixel_elevations = find_altitudes(elevation1, elevation2, pixel_coords.size());
	 
    for(int i = 1; i < pixel_coords.size()-1; i++)
    {
		if(pixel_elevations[i] <= data[pixel_coords[i].second * width + pixel_coords[i].first])
        {
			return false;
        } 
    }
	return true;
}


bool isValidPoint(
	uint16_t x,
    uint16_t y,
	int width,
	int height
)
{
	return (x >= 0 && x < width && y >= 0 && y < height);
}


int32_t getVisibilityCount(
	uint16_t x,
    uint16_t y,
	uint8_t radius,
	int16_t* data,
	int width,
	int height
) {
	int32_t visible = 0;
	for (int x_offset = 0 - radius; x_offset <= radius; x_offset++)
	{
		for (int y_offset = abs(x_offset) - radius; y_offset <= 0 - (abs(x_offset) - radius); y_offset++)
		{
			if (isValidPoint(x + x_offset, y + y_offset, width, height))
			{
				if (isVisible(x, x + x_offset, y, y + y_offset, data, width, height))
				{
					visible++;
				}
			}
		}
	}
	return visible;
}


int main() {
    auto data = READFILE_H::read_file_to_array(FILENAME, WIDTH, HEIGHT);

	std::ofstream outFile("serial_visability.bin", std::ios::binary);
	if (!outFile)
	{
		std::cerr << "Error opening file for writing.\n";
		return 1;
	}

	int32_t val;
	for (int x = 0; x < WIDTH; x++)
	{
		for (int y = 0; y < HEIGHT; y++)
		{
			val = getVisibilityCount(x, y, RADIUS, data, WIDTH, HEIGHT);
			// std::cout << "(" << x << ", " << y << ") = " << val << std::endl;
			outFile.write(reinterpret_cast<const char*>(&val), sizeof(val));
		}
	}
	outFile.close();
	return 0;
}