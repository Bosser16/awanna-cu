# awanna-cu
## Viewshed Computation using Bresenham's Line Algorithm

### Directions
 - Download SRTM file
 - Create "build/" directory
 - CD to "build/"
 - Run "cmake .."
 - Run "cmake --build ." for every change
 - Run the executable (for Windows: "./Debug/<executable>")

### Executables
 - Serial - "awannacu-serial"
 - Shared Memory CPU using OpenMP - "awannacu-shared-cpu"
 - Distributed Memory CPU using MPI - "awannacu-distributed-cpu"
 - Shared Memory GPU using CUDA - "awannacu-gpu"
 - Distributed Memory GPU using CUDA and MPI - "awannacu-distributed-gpu"
 - Serial (For visualization) - "awannacu-serial-visual"
    - src/common/awannacu_visual.ipynb contains python code to show visual
 - Validator - "awannacu-validate"
    - change variables in src/common/validator.cpp to match output file names
    - size (number of pixels computed) must match or will output INVALID