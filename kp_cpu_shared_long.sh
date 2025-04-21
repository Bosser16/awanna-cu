#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem-per-cpu=1G
#SBATCH -o new-cpu3hr-kp-%j.out-%N
#SBATCH -e new-cpu3hr-kp-%j.err-%N
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
    srun --ntasks=1 --cpus-per-task=$i --mem-per-cpu=1G --exclusive ./awannacu-cpu-shared $i 1200000 &
done
wait

### Move files out of working directory and clean up
cd $HOME
rm -rf $SCRDIR