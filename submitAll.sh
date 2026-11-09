#!/bin/bash
# Submit one job per core count. short-40core nodes have 40 cores,
# so P = 64 is spread over 2 nodes with 32 processes each.
module load slurm
sbatch --nodes=1 --ntasks=4 run.slurm
sbatch --nodes=1 --ntasks=16 run.slurm
sbatch --nodes=2 --ntasks-per-node=32 run.slurm
