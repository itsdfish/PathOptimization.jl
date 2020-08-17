cd(@__DIR__)
using Revise, AntColonyOptimization, Distributions, Random
Random.seed!(51474)
n_nodes = 200
cost_matrix = rand(Uniform(0, 50), n_nodes, n_nodes)
iterations = 200
method = AntColony(n_ants=50, n_best=3, n_nodes=n_nodes, β=2.0, 
    retain_solutions=true)
options = (trace = true, parallel = true, progress = false)
result = optimize!(method, cost_matrix, iterations; options...);