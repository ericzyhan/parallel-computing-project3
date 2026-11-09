# Read results.csv and plot speedup vs. number of cores.
# Speedup = serial time / parallel time, where the serial time is the plain
# triple-loop product of the full matrix on one core (measured in the same run).
#
# Needs Plots.jl:  julia -e 'using Pkg; Pkg.add("Plots")'
# Run:             julia plotSpeedup.jl

using Plots

lines = readlines("results.csv")
coreCounts = Int[]
sizes = Int[]
foxSpeedups = Float64[]
strassenSpeedups = Float64[]

# Skip the header line.
for line in lines[2:end]
    fields = split(line, ",")
    numProcs = parse(Int, fields[1])
    n = parse(Int, fields[2])
    serialTime = parse(Float64, fields[3])
    foxTime = parse(Float64, fields[4])
    strassenTime = parse(Float64, fields[5])

    push!(coreCounts, numProcs)
    push!(sizes, n)
    push!(foxSpeedups, serialTime / foxTime)
    push!(strassenSpeedups, serialTime / strassenTime)
end

foxPlot = plot(title="Fox", xlabel="cores P", ylabel="speedup", xscale=:log2, yscale=:log2)
strassenPlot = plot(title="Strassen (1 level) + Fox", xlabel="cores P", ylabel="speedup",
                    xscale=:log2, yscale=:log2)

for n in sort(unique(sizes))
    rows = findall(sizes .== n)
    order = sortperm(coreCounts[rows])
    rows = rows[order]
    plot!(foxPlot, coreCounts[rows], foxSpeedups[rows], marker=:circle, label="N=$n")
    plot!(strassenPlot, coreCounts[rows], strassenSpeedups[rows], marker=:circle, label="N=$n")
end

# Ideal linear speedup for reference.
idealCores = [4, 16, 64]
plot!(foxPlot, idealCores, idealCores, linestyle=:dash, color=:black, label="ideal")
plot!(strassenPlot, idealCores, idealCores, linestyle=:dash, color=:black, label="ideal")

combined = plot(foxPlot, strassenPlot, layout=(1, 2), size=(1000, 400))
savefig(combined, "speedup.png")
println("Saved speedup.png")
