#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH -o gpu-dist-%j.out-%N
#SBATCH -e gpu-dist-%j.err-%N
#SBATCH --account=owner-gpu-guest
#SBATCH --partition=notchpeak-gpu-guest


### Load necessary modules
module load mpi cuda

### Set up scratch directory
SCRDIR=/scratch/general/vast/$USER/$SLURM_JOBID
mkdir -p $SCRDIR

### Move input files into scratch directory
cp ../srtm_14_04_6000x6000_short16.raw $SCRDIR/.
cp ../build/awannacu-gpu-distributed $SCRDIR/.
cd $SCRDIR

### Run multiple versions of the program, with different numbers of threads
srun --ntasks=1 --cpus-per-task=1 --gpus-per-task=1 --exclusive mpirun -n 1 ./awannacu-gpu-distributed
srun --ntasks=1 --cpus-per-task=2 --gpus-per-task=2 --exclusive mpirun -n 2 ./awannacu-gpu-distributed
srun --ntasks=1 --cpus-per-task=3 --gpus-per-task=3 --exclusive mpirun -n 3 ./awannacu-gpu-distributed
srun --ntasks=1 --cpus-per-task=4 --gpus-per-task=4 --exclusive mpirun -n 4 ./awannacu-gpu-distributed


### Move files out of working directory and clean up
cd $HOME
rm -rf $SCRDIR