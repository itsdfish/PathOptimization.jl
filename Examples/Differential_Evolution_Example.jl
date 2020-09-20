cd(@__DIR__)
using Revise, PathOptimization, Distributions, Random, Plots
using Fitness
Random.seed!(8512)
n_obj = 2
n_nodes = 20
cost = [rand(Uniform(0, 10), n_nodes, n_nodes) for _ in 1:n_obj] 
iterations = 10_000
method = DE(n_nodes=n_nodes, start_node=3, end_node=8)
options = (parallel = false, progress = false)
@elapsed result = optimize(method, cost, iterations; options...)

frontier = get_best_cost(result.frontier)
pyplot()
scatter(frontier, grid=false, ylims=(0,120), xlims=(0,120), label="DE",
    color=:grey)