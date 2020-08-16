cd(@__DIR__)
using Revise, AntColonyOptimization, Distributions, Random
Random.seed!(51474)
n_nodes = 50
iterations = 100
method = AntColony(n_ants=50, n_best=3, n_nodes=n_nodes, β=2.0, 
    retain_solutions=true)
cost_matrix = rand(Uniform(0, 50), n_nodes, n_nodes)

result = optimize!(method, cost_matrix, iterations; trace=true);

function path_cost(path, cost)
    c = 0.0
    for i in 1:(length(path) - 1)
        s0 = path[i]
        s1 = path[i + 1]
        c += cost[s0,s1]
    end
    return c
end