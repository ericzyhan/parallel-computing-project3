#!/bin/bash
# One-time setup on SeaWulf (run on the login node from this directory).
module load julia
julia --project=. -e 'using Pkg; Pkg.instantiate(); using MPI; MPI.install_mpiexecjl(destdir=".")'
