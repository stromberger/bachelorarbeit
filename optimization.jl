using Revise

using JuMP
using Gurobi
using Printf

using GraphPlot
using MetaGraphs

using Serialization
using Memoize
using HiGHS


includet("src/simulation.jl")
using Main.Simulation


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
df;

# model vars
n_vho = length(n_vertices)
n_links = length(n_edges)
n_videos = size(df)[1]
n_slices = 1

V = 1:n_vho
L = 1:n_links
M = 1:n_videos
T = 1:n_slices

D = [20 for i in V] # disk space
s = [1 for i in M] # size of the video
P = n_links_used # possible links 
B = [10000 for i in L] # capacity (Mbps)
r = [10 for i in M] # bitrate
a = [df[i,:count] for i in M, j in V] # calls for video m at vho j
f = [a[i,j] for i in M, j in V, w in T] # calls for video m at vho j at time t
c = [length(n_links_used[i,j])*0.09 for i in V, j in V]; # transfer cost
#c = [1*0.02 for i in V, j in V]; # transfer cost


# model
model = Model(Gurobi.Optimizer)
#set_optimizer_attribute(model, "BestBdStop", 0.058)
#set_optimizer_attribute(model, "MIPGap", 0.04)

@variable(model, y[1:n_videos, 1:n_vho], Bin)
#@variable(model, 0 <= y[1:n_videos, 1:n_vho] <= 1)
@variable(model, x[1:n_videos, 1:n_vho, 1:n_vho] >= 0)

@objective(model, Min, sum(sum(sum(s[m]*a[m,j]*c[i,j]*x[m,i,j] for j in V) for i in V) for m in M))

@constraint(model, [m in M, j in V], sum(x[m,i,j] for i in V) == 1)
@constraint(model, [i in V, j in V, m in M], x[m,i,j] <= y[m,i])

@constraint(model, [i in V], sum(s[m]*y[m,i] for m in M) <= D[i])

for t in T
    for l in L
        connected_edges = get_connected_edges(P, l)
        @constraint(model, sum(sum(sum(r[m]*f[m,j,t]*x[m,i,j] for (i,j) in connected_edges) for m in M)) <= B[l])
    end
end


# solve
optimize!(model)

#serialize("temp/y_100.dat", value.(y) |> x -> convert(Matrix{Int},x))
#serialize("temp/x_100.dat", value.(x))


t = 1
bandwidth = [begin 
    connected_edges = get_connected_edges(P, l)
    xc = value.(x)
    sum(sum(sum(r[m]*f[m,j,t]*xc[m,i,j] for (i,j) in connected_edges) for m in M))
end for l in L];