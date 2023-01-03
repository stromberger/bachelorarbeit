module Simulation

using DataFrames
using Graphs, MetaGraphs, GraphPlot
using CSV
using EzXML
using Interpolations

export load_graph, get_traffic_stats, get_vertices, get_edges ,links_used, load_videos, get_videos_from_array, generate_oracle

function load_videos(filename, first_n=10000)
    df = DataFrames.DataFrame(CSV.File(filename, header=1))
    df = first(sort(df, order(:views,rev=true)), first_n)
    df[!,:prob] = df[!,:views] / sum(df[!,:views])
    df

end

function get_edges(graph)
    [x for x in edges(graph)]
end

function get_vertices(graph)
    [x for x in vertices(graph)]
end

function links_used(graph)
    n_vertices = get_vertices(graph)
    [(a_star(graph,i,j) |> f -> map(e -> get_prop(graph, e.src, e.dst, :id),f) ) for i in n_vertices, j in n_vertices]
end

function load_graph(filename)
    network_df = DataFrames.DataFrame(CSV.File(filename, delim=' ', header=0))
    rename!(network_df, :Column1 => :from, :Column2 => :to)

    g = SimpleDiGraph(0)
    mg = MetaDiGraph(g)
    set_indexing_prop!(mg, :name)

    for (idx, row) in network_df |> eachrow |> enumerate
        name_from = row[:from]
        name_to = row[:to]
        g_keys = keys(mg[:name])

        if !(name_from in g_keys)
            add_vertex!(mg, :name, name_from)
        end

        if !(name_to in g_keys)
            add_vertex!(mg, :name, name_to)
        end

        add_edge!(mg, mg[name_from, :name], mg[name_to, :name], :id, idx)

    end
    mg
end


function get_traffic_stats(steps_per_hour=10)
    traffic_stats = read("./data/stats.xml", String) |> parsexml |> root

    # starts as 17:10
    offset = (7*12)-2
    traffic_start = [parse(Float32,nodes(x)[1].content) for x in traffic_stats |> eachelement ][begin:offset-1]
    traffic = [parse(Float32,nodes(x)[1].content) for x in traffic_stats |> eachelement ][offset-1:end]
    traffic = vcat(traffic_start, traffic)
    step_size = 1/2016
    # interpolate for smoothness

    traffic_intp = LinearInterpolation(0:step_size:1, traffic)

    pdf = [traffic_intp(i) for i in 0:(1/(24*steps_per_hour*7)):0.9999]
    pdf/sum(pdf)
end

function generate_oracle(dc_network, network, df)
    videos = df[!,:url]

    video_oracle = zeros(Int32, length(videos), length(vertices(network)), length(vertices(network)))
    for vertex in vertices(network)
        datacenters = [(d,vho) for (vho,d) in enumerate(dijkstra_shortest_paths(network, vertex).dists)] |> sort |> (l -> map(x->x[2],l))

        for (idx, video) in enumerate(videos)
            target_vertex = -1

            for dc in datacenters
                if video in dc_network[dc].videos
                    target_vertex = dc
                    break
                end
            end

            video_oracle[idx, vertex, target_vertex] = 1
        end
    end
    video_oracle
end

function get_videos_from_array(network, arr, df)
    videos_per_dc = Dict()

    for vertex in vertices(network)
        videos = arr[:,vertex]
        videos_per_dc[vertex] = Set([df[idx,:url] for (idx,video) in enumerate(videos) if video > 0.99])
    end
    
    videos_per_dc
end


end