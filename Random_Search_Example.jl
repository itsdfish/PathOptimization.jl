cd(@__DIR__)
using Revise, AntColonyOptimization, Distributions, Random, Plots
using Fitness
Random.seed!(514074)
n_obj = 2
n_nodes = 50
cost = [rand(Uniform(0, 50), n_nodes, n_nodes) for _ in 1:n_obj] 
iterations = 10000
method = RandomSearch(n_nodes=n_nodes)
options = (trace = true, parallel = true, progress = false)
@elapsed result = optimize!(method, cost, iterations; options...)

frontier = get_best_cost(result.frontier)
scatter(frontier)