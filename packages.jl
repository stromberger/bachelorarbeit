using Pkg

dependencies = [
 "Revise",
 "JuMP",
 "Gurobi",
 "Printf",
 "GraphPlot",
 "MetaGraphs",
 "Plots",
 "Memoize",
 "HiGHS",
 "Dataframes",
 "Graphs",
 "CSV",
 "EzXML",
 "Interpolations"
]

Pkg.add(dependencies)