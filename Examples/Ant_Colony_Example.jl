cd(@__DIR__)
using Revise, PathOptimization, Distributions, Random, Plots
using Fitness
Random.seed!(5214)
n_obj = 2
n_nodes = 25
cost = [rand(Uniform(0, 50), n_nodes, n_nodes) for _ in 1:n_obj] 
iterations = 2000
method = AntColony(n_ants=100, n_nodes=n_nodes, β=4.0)
options = (parallel = true, progress = false)
result = optimize(method, cost, iterations; options...)

frontier = get_best_cost(result.frontier)
pyplot()
scatter(frontier, grid=false, leg=false, ylims=(0,800), xlims=(0,800),
    size=(600,400), markersize=9, markerstrokewidth=2, color=:purple, xlabel="Cost 1",
     ylabel="Cost 2", xaxis=font(12), yaxis=font(12))