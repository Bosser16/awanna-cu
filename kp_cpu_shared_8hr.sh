#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH -o cpu8hr-kp-%j.out-%N
#SBATCH -e cpu8hr-kp-%j.err-%N
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
for i in {1..16}
do
    srun --ntasks=$i --cpus-per-task=1 --exclusive ./awannacu-cpu-shared $i 4000000 &
done
wait

### Move files out of working directory and clean up
cd $HOME
rm -rf $SCRDIR