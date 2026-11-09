# Project 3.1: parallel matrix multiplication with Fox's and Strassen's methods.
#
# Run with P = q*q MPI processes, e.g.
#     mpiexecjl --project=. -n 16 julia matmul.jl
# Optional arguments override the matrix sizes:
#     mpiexecjl --project=. -n 4 julia matmul.jl 64 128
#
# The processes form a q x q grid. Process (myRow, myCol) has rank myRow*q + myCol.
# Rank 0 builds A and B, sends the blocks out, collects the results and checks
# them against a serial product.

using MPI
using Random
using Printf

const matrixSizes = [256, 1024, 4096]
const resultFile = "results.csv"

# Random N x N matrix with elements in (-1, 1].
# rand() is in [0, 1), so 1 - 2*rand() is in (-1, 1].
function randomMatrix(n)
    m = zeros(n, n)
    for j in 1:n
        for i in 1:n
            m[i, j] = 1.0 - 2.0 * rand()
        end
    end
    return m
end

# c = c + a * b with a plain triple loop.
# Julia stores matrices column by column, so the loop order j, k, i
# walks through memory in order.
function multiplyAdd!(c, a, b)
    n = size(a, 1)
    @inbounds for j in 1:n
        for k in 1:n
            bValue = b[k, j]
            for i in 1:n
                c[i, j] = c[i, j] + a[i, k] * bValue
            end
        end
    end
end

# Copy out block (blockRow, blockCol) of size blockSize. Block indices start at 0.
function getBlock(m, blockRow, blockCol, blockSize)
    rowStart = blockRow * blockSize + 1
    colStart = blockCol * blockSize + 1
    rowEnd = rowStart + blockSize - 1
    colEnd = colStart + blockSize - 1
    return m[rowStart:rowEnd, colStart:colEnd]
end

function setBlock!(m, block, blockRow, blockCol, blockSize)
    rowStart = blockRow * blockSize + 1
    colStart = blockCol * blockSize + 1
    rowEnd = rowStart + blockSize - 1
    colEnd = colStart + blockSize - 1
    m[rowStart:rowEnd, colStart:colEnd] = block
end

# Rank 0 sends block (rowOffset + row, colOffset + col) of fullMatrix to the
# process at grid position (row, col). Every process returns its own block.
function scatterMatrix(fullMatrix, rowOffset, colOffset, blockSize, gridSize, comm)
    rank = MPI.Comm_rank(comm)
    if rank == 0
        for otherRank in 1:gridSize*gridSize-1
            row = div(otherRank, gridSize)
            col = mod(otherRank, gridSize)
            block = getBlock(fullMatrix, rowOffset + row, colOffset + col, blockSize)
            MPI.Send(block, comm; dest=otherRank, tag=0)
        end
        return getBlock(fullMatrix, rowOffset, colOffset, blockSize)
    else
        block = zeros(blockSize, blockSize)
        MPI.Recv!(block, comm; source=0, tag=0)
        return block
    end
end

# Reverse of scatterMatrix: every process sends its block to rank 0,
# which writes it into fullMatrix.
function gatherMatrix!(fullMatrix, localBlock, rowOffset, colOffset, blockSize, gridSize, comm)
    rank = MPI.Comm_rank(comm)
    if rank == 0
        setBlock!(fullMatrix, localBlock, rowOffset, colOffset, blockSize)
        block = zeros(blockSize, blockSize)
        for otherRank in 1:gridSize*gridSize-1
            row = div(otherRank, gridSize)
            col = mod(otherRank, gridSize)
            MPI.Recv!(block, comm; source=otherRank, tag=1)
            setBlock!(fullMatrix, block, rowOffset + row, colOffset + col, blockSize)
        end
    else
        MPI.Send(localBlock, comm; dest=0, tag=1)
    end
end

# Fox's algorithm on a gridSize x gridSize process grid.
# In stage s, the process in column (myRow + s) mod q broadcasts its A block
# along its row. Every process multiplies that block by its current B block,
# then B blocks are shifted up one row (cyclically).
function foxMultiply(localA, localB, gridSize, myRow, rowComm, colComm)
    blockSize = size(localA, 1)
    localC = zeros(blockSize, blockSize)
    broadcastA = zeros(blockSize, blockSize)
    currentB = copy(localB)
    receivedB = zeros(blockSize, blockSize)

    # In colComm the rank of a process equals its grid row.
    upNeighbor = mod(myRow - 1, gridSize)
    downNeighbor = mod(myRow + 1, gridSize)

    for stage in 0:gridSize-1
        # In rowComm the rank of a process equals its grid column.
        rootCol = mod(myRow + stage, gridSize)
        if MPI.Comm_rank(rowComm) == rootCol
            copyto!(broadcastA, localA)
        end
        MPI.Bcast!(broadcastA, rowComm; root=rootCol)

        multiplyAdd!(localC, broadcastA, currentB)

        MPI.Sendrecv!(currentB, receivedB, colComm;
                      dest=upNeighbor, sendtag=2, source=downNeighbor, recvtag=2)
        copyto!(currentB, receivedB)
    end
    return localC
end

# One level of Strassen. Each process holds block (myRow, myCol) of every
# quadrant of A and B. The additions are done locally, and each of the 7
# products is done with Fox's method on the whole process grid.
function strassenMultiply(a11, a12, a21, a22, b11, b12, b21, b22,
                          gridSize, myRow, rowComm, colComm)
    m1 = foxMultiply(a11 + a22, b11 + b22, gridSize, myRow, rowComm, colComm)
    m2 = foxMultiply(a21 + a22, b11, gridSize, myRow, rowComm, colComm)
    m3 = foxMultiply(a11, b12 - b22, gridSize, myRow, rowComm, colComm)
    m4 = foxMultiply(a22, b21 - b11, gridSize, myRow, rowComm, colComm)
    m5 = foxMultiply(a11 + a12, b22, gridSize, myRow, rowComm, colComm)
    m6 = foxMultiply(a21 - a11, b11 + b12, gridSize, myRow, rowComm, colComm)
    m7 = foxMultiply(a12 - a22, b21 + b22, gridSize, myRow, rowComm, colComm)

    c11 = m1 + m4 - m5 + m7
    c12 = m3 + m5
    c21 = m2 + m4
    c22 = m1 - m2 + m3 + m6
    return c11, c12, c21, c22
end

function runOneSize(n, gridSize, myRow, comm, rowComm, colComm, printResults)
    rank = MPI.Comm_rank(comm)
    numProcs = MPI.Comm_size(comm)

    # Rank 0 makes the matrices and the serial reference product.
    a = zeros(0, 0)
    b = zeros(0, 0)
    serialC = zeros(0, 0)
    serialTime = 0.0
    if rank == 0
        Random.seed!(n)
        a = randomMatrix(n)
        b = randomMatrix(n)
        serialC = zeros(n, n)
        startTime = MPI.Wtime()
        multiplyAdd!(serialC, a, b)
        serialTime = MPI.Wtime() - startTime
    end

    # ---------- Fox ----------
    blockSize = div(n, gridSize)
    localA = scatterMatrix(a, 0, 0, blockSize, gridSize, comm)
    localB = scatterMatrix(b, 0, 0, blockSize, gridSize, comm)

    MPI.Barrier(comm)
    startTime = MPI.Wtime()
    localC = foxMultiply(localA, localB, gridSize, myRow, rowComm, colComm)
    MPI.Barrier(comm)
    foxTime = MPI.Wtime() - startTime

    foxC = zeros(0, 0)
    if rank == 0
        foxC = zeros(n, n)
    end
    gatherMatrix!(foxC, localC, 0, 0, blockSize, gridSize, comm)

    # ---------- Strassen ----------
    # Each quadrant is n/2 x n/2 and is split into q x q blocks of size n/(2q).
    # Quadrant (1,1) starts at block (0,0), quadrant (1,2) at block (0,q), etc.
    halfBlock = div(n, 2 * gridSize)
    q = gridSize
    a11 = scatterMatrix(a, 0, 0, halfBlock, gridSize, comm)
    a12 = scatterMatrix(a, 0, q, halfBlock, gridSize, comm)
    a21 = scatterMatrix(a, q, 0, halfBlock, gridSize, comm)
    a22 = scatterMatrix(a, q, q, halfBlock, gridSize, comm)
    b11 = scatterMatrix(b, 0, 0, halfBlock, gridSize, comm)
    b12 = scatterMatrix(b, 0, q, halfBlock, gridSize, comm)
    b21 = scatterMatrix(b, q, 0, halfBlock, gridSize, comm)
    b22 = scatterMatrix(b, q, q, halfBlock, gridSize, comm)

    MPI.Barrier(comm)
    startTime = MPI.Wtime()
    c11, c12, c21, c22 = strassenMultiply(a11, a12, a21, a22, b11, b12, b21, b22,
                                          gridSize, myRow, rowComm, colComm)
    MPI.Barrier(comm)
    strassenTime = MPI.Wtime() - startTime

    strassenC = zeros(0, 0)
    if rank == 0
        strassenC = zeros(n, n)
    end
    gatherMatrix!(strassenC, c11, 0, 0, halfBlock, gridSize, comm)
    gatherMatrix!(strassenC, c12, 0, q, halfBlock, gridSize, comm)
    gatherMatrix!(strassenC, c21, q, 0, halfBlock, gridSize, comm)
    gatherMatrix!(strassenC, c22, q, q, halfBlock, gridSize, comm)

    # ---------- check and report ----------
    if rank == 0 && printResults
        foxError = maximum(abs.(foxC - serialC))
        strassenError = maximum(abs.(strassenC - serialC))
        foxSpeedup = serialTime / foxTime
        strassenSpeedup = serialTime / strassenTime

        @printf("P=%d N=%d serial=%.4fs fox=%.4fs (speedup %.2f, error %.2e) strassen=%.4fs (speedup %.2f, error %.2e)\n",
                numProcs, n, serialTime, foxTime, foxSpeedup, foxError,
                strassenTime, strassenSpeedup, strassenError)

        writeHeader = !isfile(resultFile)
        open(resultFile, "a") do file
            if writeHeader
                println(file, "P,N,serialTime,foxTime,strassenTime,foxError,strassenError")
            end
            println(file, "$numProcs,$n,$serialTime,$foxTime,$strassenTime,$foxError,$strassenError")
        end
    end
end

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    numProcs = MPI.Comm_size(comm)

    gridSize = round(Int, sqrt(numProcs))
    if gridSize * gridSize != numProcs
        if rank == 0
            println("Number of processes must be a perfect square.")
        end
        MPI.Abort(comm, 1)
    end

    myRow = div(rank, gridSize)
    myCol = mod(rank, gridSize)
    # Processes in the same grid row / column; ranks inside are col / row.
    rowComm = MPI.Comm_split(comm, myRow, myCol)
    colComm = MPI.Comm_split(comm, myCol, myRow)

    # Small warm-up run so Julia's compile time is not part of the timings.
    runOneSize(4 * gridSize, gridSize, myRow, comm, rowComm, colComm, false)

    # Sizes can be given on the command line, e.g. "julia matmul.jl 64 128".
    sizes = matrixSizes
    if length(ARGS) > 0
        sizes = [parse(Int, arg) for arg in ARGS]
    end

    for n in sizes
        runOneSize(n, gridSize, myRow, comm, rowComm, colComm, true)
    end

    MPI.Finalize()
end

main()
