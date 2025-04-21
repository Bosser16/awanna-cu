# awanna-cu
## Viewshed Computation using Bresenham's Line Algorithm

### Set Up on CHPC
 - module load cuda
 - module load cmake
 - module load mpich

### Directions
 - Download SRTM file
 - Create "build/" directory
 - CD to "build/"
 - Run "cmake .."
 - Run "cmake --build ." for every change
 - Run the executables (for Windows: "./Debug/<executable>")

### Executables
 - Serial - "awannacu-serial"
 - Shared Memory CPU using OpenMP - "awannacu-cpu-shared"
 - Distributed Memory CPU using MPI - "awannacu-cpu-distributed"
 - Shared Memory GPU using CUDA - "awannacu-gpu"
 - Distributed Memory GPU using CUDA and MPI - "awannacu-gpu-distributed"
 - Serial (For small scale visualization) - "awannacu-serial-visual"
    - src/awannacu_visual.ipynb contains python code to show visual in a jupyter notebook
    - uses awannacu_serial_visual.raw for input
    - can use any .raw output, must modify height/width of image in notebook
 - Validator - "awannacu-validate"
    - change variables in src/validator.cpp to match output file names
    - size (number of pixels computed) must match or will output INVALID

## Specific Tasks Completed by Members
### Boston Musgrave
 - Serial implementation
 - CUDA GPU implementation
 - Serial Visualization
 - Validator
 - CUDA GPU Scaling Study
### Calvin Clark
- Serial implementation
- Viewshed functions
- Distributed CPU implementaion
- Distributed CPU/GPU scaling study
### Ryan Van Gieson
- Serial implementation
- Viewshed functions
- Shared CPU implementation
- Distributed GPU implementation
- Serial CPU/Parallel CPU scaling study