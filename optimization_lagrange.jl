using Revise

using JuMP
using Gurobi
using Printf

using MetaGraphs
using Plots

using Serialization
using Memoize

using LinearAlgebra

using Serialization
using Statistics
using HiGHS
using Gurobi


includet("src/simulation.jl")
using Main.Simulation

# load params
n_graph = load_graph("data/weights.intra")
n_vertices = get_vertices(n_graph)
n_edges = get_edges(n_graph)
n_links_used = links_used(n_graph);

# gets all vertices that are using this edge 
@memoize function get_connected_edges(links_used, idx)
    [ (I[1], I[2]) for I in CartesianIndices(links_used) if idx in P[I[1], I[2]]]
end

df = load_videos("data/videos.txt",200)
df[!,:count] = df[!,:prob]*500;
df



# model vars
n_vho = length(n_vertices)
n_links = length(n_edges)
n_videos = length(df[!,:count])
n_slices = 1

V = 1:n_vho
L = 1:n_links
M = 1:n_videos
T = 1:n_slices

D = [20 for i in V] # disk space
s = [1 for i in M] # size of the video
P = n_links_used # possible links 
B = [5000 for i in L] # capacity (Mbps)
r = [10 for i in M] # bitrate
a = [df[i,:count] for i in M, j in V] # calls for video m at vho j
f = [a[i,j] for i in M, j in V, w in T] # calls for video m at vho j at time t
c = [length(n_links_used[i,j])*0.09 for i in V, j in V]; # transfer cost

# Model

function solve_model(l_mul, mip=false)
    model = Model(Gurobi.Optimizer)
    # MOI.set(model, MOI.Silent(), true)

    if (!mip)
        @variable(model, 0 <= y[1:n_videos, 1:n_vho] <= 1)
    else
        @variable(model, y[1:n_videos, 1:n_vho], Bin)
    end
    @variable(model, x[1:n_videos, 1:n_vho, 1:n_vho] >= 0)

    # lagrangian relaxation
    l_diskspace = [ sum(s[m]*y[m,i] for m in M) - D[i] for i in V]
    l_bandwidth = [[  sum(sum(sum(r[m]*f[m,j,t]*x[m,i,j] for (i,j) in get_connected_edges(P, l)) for m in M)) - B[l] for l in L] for t in T]
    l_val = vcat(l_diskspace, vcat(l_bandwidth...))

    @objective(model, Min, sum(sum(sum(s[m]*a[m,j]*c[i,j]*x[m,i,j] for j in V) for i in V) for m in M) + (transpose(l_mul)*l_val))

    @constraint(model, [m in M, j in V], sum(x[m,i,j] for i in V) == 1)
    @constraint(model, [i in V, j in V, m in M], x[m,i,j] <= y[m,i]) # (5)

    #@constraint(model, [i in V], sum(s[m]*y[m,i] for m in M) <= D[i])

    optimize!(model)

    #@info "solved model"

    return objective_value(model), value.(y), value.(x)
end

# positive -> violation
function calc_subgradient(x,y)
    l_diskspace = [ sum(s[m]*y[m,i] for m in M) - D[i] for i in V]
    l_bandwidth = [[  sum(sum(sum(r[m]*f[m,j,t]*x[m,i,j] for (i,j) in get_connected_edges(P, l)) for m in M)) - B[l] for l in L] for t in T]
    l_val = vcat(l_diskspace, vcat(l_bandwidth...))
    #l_val = vcat(vcat(l_bandwidth...))
    l_val
end

function calc_bandwidth(x)
    t = 1
    bandwidth = [begin 
    connected_edges = get_connected_edges(P, l)
    xc = x
    sum(sum(sum(r[m]*f[m,j,t]*xc[m,i,j] for (i,j) in connected_edges) for m in M))
    end
    for l in L];

    return bandwidth
end

function eval_model(l_mul)
    obj, y, x = solve_model(l_mul)
    @info "model stats: ($obj) $(sum(y)) $(mean(calc_bandwidth(x))) $(maximum(calc_bandwidth(x))) $(sum(l_val .> 0))"
end

#  solve

l_size = length(V) + (length(T)*length(L))
#l_size = (length(T)*length(L))
l_mul = ones(Float64, l_size)*0;
l_mul_mod = l_mul
l_val = 0
l_log = []


for i in 1:100
    obj, y, x = solve_model(l_mul)

    l_val = calc_subgradient(x,y)
    l_val[1:length(V)] *= 100000000
    l_val = l_val / norm(l_val)
    l_mul = clamp.(l_mul + l_val*0.1, 0, 10^16)
    #println(l_mul)
    if i % 5 == 0
        @info "iteration $i ($obj) $(sum(y)) $(mean(calc_bandwidth(x))) $(maximum(calc_bandwidth(x))) $(sum(y, dims=1)) $(sum(l_val .> 0))"
    end
    push!(l_log, obj)
end


# generate plot
plotd = plot(l_log, label="Relaxed Model", size=(600,400))
hline!([4431.71450], label="Initial Model")
ylims!((0, 6000))
xlabel!("Iteration")
ylabel!("Objective Value")
#savefig(plotd,"plots/lagrange2.pdf")

#serialize("temp/lmul.dat", l_mul)
#serialize("temp/lmul_stats.dat", l_log)


#plot(l_log, label="Relaxed Model", size=(600,400))
#plotd = histogram(["Used Disk Space (GB)"],[transpose(sum(y, dims=1))], 
#label="Relaxed Model", fillalpha = 0.4, bins=10,
#xlabel="Distribution of Used Disk Space (GB)", ylabel="Number of VHOs")
#savefig(plotd,"plots/histogram.pdf")


#serialize("temp/y_lag.dat", y |> x -> convert(Matrix{Int},x))
#serialize("temp/x_lag.dat", x)


