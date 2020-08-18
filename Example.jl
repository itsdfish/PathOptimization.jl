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

scheme = Scheme{2}(0.1, is_minimizing=true)
a = EpsBoxArchive(scheme)

x = [2.0]
add_candidate!(a, (3.21, 1.12), [1.0, 5.0,1.0], 2)
add_candidate!(a, (30.21, 10.12), [1.0, 5.0,1.0], 2)
add_candidate!(a, (3.21, 10.12), [1.0, 5.0,1.0], 2)
add_candidate!(a, (1.21, 10.12), [1.0, 5.0,1.0], 2)
add_candidate!(a, (-1.21, -10.12), x, 2)

scheme = Scheme{1}(0.1, is_minimizing=true)

a = EpsBoxArchive(scheme)

x = [2.0]
add_candidate!(a, (3.21,), [1.0, 5.0,1.0], 2)
add_candidate!(a, (30.21,), [1.0, 5.0,1.0], 2)
add_candidate!(a, (3.21, ), [1.0, 5.0,1.0], 2)
add_candidate!(a, (1.21,), [1.0, 5.0,1.0], 2)
add_candidate!(a, (-1.21,), x, 2)
