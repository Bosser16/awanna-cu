#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=7
#SBATCH -o cputest-kp-%j.out-%N
#SBATCH -e cputest-kp-%j.err-%N
#SBATCH --account=owner-guest
#SBATCH --partition=kingspeak-guest


### Load necessary modules


### Set up scratch directory
SCRDIR=/scratch/general/vast/$USER/$SLURM_JOBID
mkdir -p $SCRDIR

### Move input files into scratch directory
cp ../srtm_14_04_6000x6000_short16.raw $SCRDIR/.
cp ../build/awannacu-cpu-shared $SCRDIR/.
cd $SCRDIR

### Run multiple versions of the program, with different numbers of threads
srun --ntasks=1 --cpus-per-task=1 --exclusive ./awannacu-cpu-shared 1 100000 &
srun --ntasks=1 --cpus-per-task=2 --exclusive ./awannacu-cpu-shared 2 100000 &
srun --ntasks=1 --cpus-per-task=3 --exclusive ./awannacu-cpu-shared 3 100000 &
srun --ntasks=1 --cpus-per-task=4 --exclusive ./awannacu-cpu-shared 4 100000 &
srun --ntasks=1 --cpus-per-task=5 --exclusive ./awannacu-cpu-shared 5 100000 &
srun --ntasks=1 --cpus-per-task=8 --exclusive ./awannacu-cpu-shared 8 100000 &
srun --ntasks=1 --cpus-per-task=16 --exclusive ./awannacu-cpu-shared 16 100000 &
wait

### Move files out of working directory and clean up
cd $HOME
rm -rf $SCRDIR