cd(@__DIR__)
using Revise, PathOptimization, Distributions, Random, Plots
using Fitness
Random.seed!(81512)
n_obj = 2
n_nodes = 20
cost = [rand(Uniform(0, 10), n_nodes, n_nodes) for _ in 1:n_obj]
###############################################################################
# 
###############################################################################
iterations = 1000
method = AntColony(n_ants=100, n_nodes=n_nodes, β=4.0, ρ=.10,
    τmin=0.0, τmax=10.0, use2opt=true)
options = (parallel = true, progress = false)
result = optimize(method, cost, iterations; options...)

frontier = get_best_cost(result.frontier)
pyplot()
scatter(frontier, grid=false, ylims=(0,120), xlims=(0,120), label="Ant Colony",
    color=:grey)
###############################################################################
# 
###############################################################################
iterations = 10_000
method = RandomSearch(n_nodes=n_nodes)
options = (parallel = true, progress = false)
result = optimize(method, cost, iterations; options...)

frontier = get_best_cost(result.frontier)
pyplot()
scatter!(frontier, grid=false, ylims=(0,120), xlims=(0,120), label="Random")
###############################################################################
# 
###############################################################################
iterations = 10_000
method = NearestNeighbor(n_nodes=n_nodes, use2opt=false)
options = (parallel = true, progress = false)
result = optimize(method, cost, iterations; options...)

frontier = get_best_cost(result.frontier)
pyplot()
scatter!(frontier, grid=false, ylims=(0,120), xlims=(0,120), label="Nearest Neighbor",
    color=:purple)
###############################################################################
# 
###############################################################################
iterations = 1_000
fun(de, state, Pt, rng) = deepcopy(Pt)
method = DE(n_nodes=n_nodes, start_node=3, end_node=8, cross_over_fun=cross_over_ensemble,
    Θneighbor = 0.1)
options = (parallel = false, progress = false)
result = optimize(method, cost, iterations; options...)

frontier = get_best_cost(result.frontier)
pyplot()
scatter!(frontier, grid=false, ylims=(0,120), xlims=(0,120), label="DE",
    color=:darkred)