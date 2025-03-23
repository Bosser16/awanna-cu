# awanna-cu
## Viewshed Computation using Bresenham's Line Algorithm

### Directions
 - Download SRTM file
 - Create "build/" directory
 - CD to "build/"
 - Run "cmake .."
 - Run "cmake --build ." for every change
 - Run the executable (for Windows: "./Debug/<executable>")

### Includes
 - Serial - "awannacu-serial"
 - Shared Memory CPU using OpenMP - "awannacu-shared-cpu"
 - Distributed Memory CPU using MPI - "awannacu-distributed-cpu"
 - Shared Memory GPU using CUDA - "awannacu-shared-gpu"
 - Distributed Memory GPU using CUDA and MPI? - "awannacu-distributed-gpu"