#=
in this section we load all need libraries.
=#

using JuMP
using Gurobi

using MetaGraphs
using Plots

using Serialization
using Memoize
using LinearAlgebra
using Statistics
using Printf

includet("src/simulation.jl")
using Main.Simulation


#=
parameters
=#
top_n = 200
hits_per_vho = 500
subgrad_iterations = 500


#=
here we load the network graph and the video data.
=#
n_graph = load_graph("data/weights.intra")
n_vertices = get_vertices(n_graph)
n_edges = get_edges(n_graph)

# this is a matrix where each entry (i,j) is the list of links of the shortest path between those two vertices. (calc. w/ Dijkstra)
n_links_used = links_used(n_graph)

# here we load the video data
df = load_videos("data/videos.txt", top_n)
df[!,:count] = df[!,:prob]*hits_per_vho;


#=
model (variable names are as in the bachelor thesis)
=#
n_vho = length(n_vertices)
n_links = length(n_edges)
n_videos = length(df[!,:count])
n_slices = 1

V = 1:n_vho                             # vhos
L = 1:n_links                           # links
M = 1:n_videos                          # videos
T = 1:n_slices                          # time slices

D = [20 for i in V]                     # disk space
s = [1 for i in M]                      # size of the video
P = n_links_used                        # possible links 
B = [5000 for i in L]                   # capacity (Mbps)
r = [10 for i in M]                     # bitrate
a = [df[i,:count] for i in M, j in V]   # calls for video m at vho j
f = [a[i,j] for i in M, j in V, w in T] # calls for video m at vho j at time t
c = [length(n_links_used[i,j])*0.09 for i in V, j in V] # transfer cost

# helper function to get all vertices (vhos) that are using this edge in their route to another vertex (vho)
@memoize function get_connected_vertices(links_used, idx)
    [ (I[1], I[2]) for I in CartesianIndices(links_used) if idx in P[I[1], I[2]]]
end

#=
okay, now this is where it gets more inresting. Here we define the model in JuMP
=#
function solve_model(l_mul, mip=false)
    model = Model(Gurobi.Optimizer)

    # here we define y to be a variable (which is constrained to be an integer if mip is true)
    if (!mip)
        @variable(model, 0 <= y[1:n_videos, 1:n_vho] <= 1)
    else
        @variable(model, y[1:n_videos, 1:n_vho], Bin)
    end
    @variable(model, x[1:n_videos, 1:n_vho, 1:n_vho] >= 0)

    #=
    lagrangian relaxation
    =#
    # violations of the diskspace constraints
    l_diskspace = [sum(s[m]*y[m,i] for m in M) - D[i] for i in V]
    # violations of the bandwidth constraints
    l_bandwidth = [
            [sum(sum(sum(r[m]*f[m,j,t]*x[m,i,j] for (i,j) in get_connected_vertices(P, l)) for m in M)) - B[l] 
            for l in L] 
        for t in T]
    # concationation of the violations above
    l_val = vcat(l_diskspace, vcat(l_bandwidth...))

    # our objective funtion where we now weight the violations with the lagrange multiplicators
    @objective(model, Min, sum(sum(sum(s[m]*a[m,j]*c[i,j]*x[m,i,j] for j in V) for i in V) for m in M) 
        + (transpose(l_mul)*l_val))

    # constraints
    @constraint(model, [m in M, j in V], sum(x[m,i,j] for i in V) == 1)
    @constraint(model, [i in V, j in V, m in M], x[m,i,j] <= y[m,i]) # (5)

    optimize!(model)

    # return our decision variables and current objective value
    return objective_value(model), value.(y), value.(x)
end

# gradient with regard to the lagrange multiplicators
function calc_subgradient(x,y)
    l_diskspace = [sum(s[m]*y[m,i] for m in M) - D[i] for i in V]
    l_bandwidth = [
            [sum(sum(sum(r[m]*f[m,j,t]*x[m,i,j] for (i,j) in get_connected_vertices(P, l)) for m in M)) - B[l] 
            for l in L] 
        for t in T]
    l_val = vcat(l_diskspace, vcat(l_bandwidth...))
    return l_val
end

#=
debugging info functions
=#

# calculates the whole used bandwidth at time t=1 (only useful for debugging)
function calc_bandwidth(x)
    t = 1
    bandwidth = [begin 
        connected_edges = get_connected_vertices(P, l)
        xc = x
        sum(sum(sum(r[m]*f[m,j,t]*xc[m,i,j] for (i,j) in connected_edges) for m in M))
        end
        for l in L
    ];
    return bandwidth
end

#=
subgradient method
=#

# here we initalize the lagrange multiplier with zeros
l_size = length(V) + (length(T)*length(L)) # no. of diskspace + bandwidth constrains
l_mul = ones(Float64, l_size)*0;
l_mul_mod = l_mul
# the amount by which we change the lag. mul. in each iteration
l_val = :nothing

# logging variables
l_log = []


for i in 1:subgrad_iterations
    # here we make a copy of currently used multiplier and weight the diskspace constraints higher
    l_mul_mod = copy(l_mul)
    l_mul_mod[1:length(V)] *= 10^8 # 1:length(V) -> part of the lagr. mul. that is used for diskspace constraints

    obj, y, x = solve_model(l_mul_mod)

    # here calculate the subgradient and update the lagrange multiplier
    l_val = calc_subgradient(x,y)
    # normalize the lagrange multiplier
    l_val = l_val / norm(l_val) 
    # update with learning rate and clamp so that we don't get negative values
    l_mul = clamp.(l_mul + l_val*10^(-7), 0, 10^16)

    # logging
    if i % 5 == 0
        @info "iteration $i ($obj) $(sum(y)) $(mean(calc_bandwidth(x))) $(maximum(calc_bandwidth(x))) $(sum(y, dims=1)) $(sum(l_val .> 0))"
    end

    push!(l_log, obj)
end

for i in 1:subgrad_iterations
    # solve the model and get objectiva and decision variables
    obj, y, x = solve_model(l_mul)

    # calcultate the subgradient
    l_val = calc_subgradient(x,y)

    # weight the diskspace constraints higher (before normalization)
    l_val[1:length(V)] *= 10^8
     # normalize the lagrange multiplier
    l_val = l_val / norm(l_val)
    # update with learning rate and clamp so that we don't get negative values
    l_mul = clamp.(l_mul + l_val*10^-1, 0, 10^16)
    #println(l_mul)
    if i % 5 == 0
        @info "iteration $i ($obj) $(sum(y)) $(mean(calc_bandwidth(x))) $(maximum(calc_bandwidth(x))) $(sum(y, dims=1)) $(sum(l_val .> 0))"
    end
    push!(l_log, obj)
end



# in this step we re-introduce the integer constraint to get binary values if we should save a video at a vho
obj, y, x = solve_model(l_mul, mip=true)

# save the results
serialize("temp/y_lag.dat", y |> x -> convert(Matrix{Int},x))
serialize("temp/x_lag.dat", x)