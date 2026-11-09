#!/bin/bash
#SBATCH --job-name=matmul
#SBATCH --partition=short-40core
#SBATCH --time=01:00:00
#SBATCH --output=matmul_%j.out

# Number of processes comes from sbatch --ntasks (see submitAll.sh).
module load julia
cd $SLURM_SUBMIT_DIR

./mpiexecjl --project=. -n $SLURM_NTASKS julia matmul.jl
