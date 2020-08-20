cd(@__DIR__)
using Revise, AntColonyOptimization, Distributions, Random
Random.seed!(514074)
n_nodes = 200
cost_matrix = rand(Uniform(0, 50), n_nodes, n_nodes)
iterations = 200
method = AntColony(n_ants=50, n_best=3, n_nodes=n_nodes, β=2.0, 
    retain_solutions=false)
options = (trace = false, parallel = true, progress = false)
@elapsed result = optimize!(method, cost_matrix, iterations; options...)