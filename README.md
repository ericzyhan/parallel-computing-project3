# Project 3.1 – Fox's and Strassen's Parallel Matrix Multiplication (Julia + MPI)

## Files
| File | Purpose |
|---|---|
| `matmul.jl` | Serial baseline, Fox, and Strassen+Fox; correctness check; writes `results.csv` |
| `run.slurm` | Slurm job script (P comes from `--ntasks`) |
| `submitAll.sh` | Submits the P = 4, 16, 64 jobs |
| `setup.sh` | One-time install of MPI.jl and `mpiexecjl` |
| `plotSpeedup.jl` | Makes `speedup.png` from `results.csv` |

## Running on SeaWulf
```bash
./setup.sh          # once, on the login node
./submitAll.sh      # 3 jobs: P = 4, 16, 64; each runs N = 256, 1024, 4096
# when the jobs finish (results.csv has 9 rows), copy results.csv back and run:
julia plotSpeedup.jl
```
If `module load julia` fails, check the available version with `module avail julia`.
To test locally: `./mpiexecjl --project=. -n 4 julia matmul.jl 64 256`.

## Algorithms

**Fox's algorithm.** The P processes form a q × q grid (q = √P), and each one holds an
(N/q) × (N/q) block of A, B and C. There are q stages. In stage s, the process in row i and
column (i + s) mod q broadcasts its A block to the rest of row i. Every process multiplies
the received A block by its current B block and adds the result to its C block. Then every
B block moves up one row (cyclically). After q stages, C<sub>ij</sub> = Σ<sub>k</sub> A<sub>ik</sub>B<sub>kj</sub>.
Each stage costs one row broadcast and one column shift.

**Strassen's algorithm.** Split A and B into 2 × 2 quadrants. Strassen's method then needs 7 products
of quadrants instead of 8:
```
M1 = (A11+A22)(B11+B22)   M5 = (A11+A12)B22
M2 = (A21+A22)B11         M6 = (A21-A11)(B11+B12)
M3 = A11(B12-B22)         M7 = (A12-A22)(B21+B22)
M4 = A22(B21-B11)
C11 = M1+M4-M5+M7   C12 = M3+M5   C21 = M2+M4   C22 = M1-M2+M3+M6
```

## How Strassen is mapped onto the P cores
I use one level of recursion, and each of the 7 products is computed with Fox's method on all P cores.
Each quadrant (N/2 × N/2) is split into q × q blocks of size N/(2q), and process (i, j) holds
block (i, j) of **every** quadrant of A and B. Because all processes hold matching pieces,
all the Strassen additions and subtractions (forming the operands and combining M1…M7 into C) are
purely local and need no communication. The only communication is in the 7 Fox multiplications.

## Measurement details
- Elements are `1 - 2*rand()`, which lies in (-1, 1].
- **Serial baseline:** a plain triple loop (the same kernel each process uses on its blocks)
  multiplying the full N × N matrices on one core (rank 0).
- **Parallel time:** from a barrier before the multiplication to a barrier after it. This excludes
  sending the matrices out from rank 0 and collecting the results.
- A small warm-up run comes first, so Julia's compile time is left out of the timings.
- **Correctness:** rank 0 collects both parallel results and reports the max |C<sub>par</sub> − C<sub>serial</sub>|
  (expected ≈ 1e-13 because of floating-point rounding).
