#!/bin/bash
# One-time setup on SeaWulf (run on the login node from this directory).
module load julia

# Install packages in the home directory. The trailing ":" keeps
# Julia's built-in (read-only) depots available after it.
export JULIA_DEPOT_PATH="$HOME/.julia:"
julia --project=. -e 'using Pkg; Pkg.instantiate(); using MPI; MPI.install_mpiexecjl(destdir=".")'
