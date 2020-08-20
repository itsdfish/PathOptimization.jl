cd(@__DIR__)
using Revise, AntColonyOptimization, Distributions, Random, Plots
Random.seed!(51474)
n_nodes = 50
cost_matrix = [rand(Uniform(0, 50), n_nodes, n_nodes) for _ in 1:2] 
iterations = 1000
method = AntColony(n_ants=100, n_nodes=n_nodes, β=3.0)
options = (trace = true, parallel = true, progress = false)
@elapsed result = optimize!(method, cost_matrix, iterations; options...)

frontier = get_best_cost(result.frontier)
scatter(frontier)