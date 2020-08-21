cd(@__DIR__)
using Revise, AntColonyOptimization, Distributions, Random, Plots
using Fitness
Random.seed!(514074)
n_obj = 2
n_nodes = 50
cost = [rand(Uniform(0, 50), n_nodes, n_nodes) for _ in 1:n_obj] 
###############################################################################
# 
###############################################################################
iterations = 1000
method = AntColony(n_ants=100, n_nodes=n_nodes, β=4.0)
options = (trace = true, parallel = true, progress = false)
result = optimize!(method, cost, iterations; options...)

frontier = get_best_cost(result.frontier)
pyplot()
scatter(frontier, grid=false, ylims=(0,1500), xlims=(0,1500), label="Ant Colony")
###############################################################################
# 
###############################################################################
iterations = 10_000
method = RandomSearch(n_nodes=n_nodes)
options = (trace = true, parallel = true, progress = false)
result = optimize!(method, cost, iterations; options...)

frontier = get_best_cost(result.frontier)
pyplot()
scatter!(frontier, grid=false, ylims=(0,1500), xlims=(0,1500), label="Random")
###############################################################################
# 
###############################################################################
iterations = 10_000
method = NearestNeighbor(n_nodes=n_nodes)
options = (trace = true, parallel = true, progress = false)
result = optimize!(method, cost, iterations; options...)

frontier = get_best_cost(result.frontier)
pyplot()
scatter!(frontier, grid=false, ylims=(0,1500), xlims=(0,1500), label="Nearest Neighbor")