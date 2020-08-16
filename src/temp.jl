using Revise, AntColonyOptimization, Distributions, Random
Random.seed!(5454)
n_nodes = 100
iterations = 100
method = AntColony(n_ants=20, n_best=3, n_nodes=n_nodes, β=2.0, 
    retain_solutions=true)
cost_matrix = rand(n_nodes, n_nodes)

result = optimize!(method, cost_matrix, iterations; trace=true)

# state = initialize(cost_matrix, method)
# ants = [Ant(method.n_nodes) for _ in 1:method.n_ants]
# for ant in ants 
#     find_path!(ant, method, state)
# end
# best_ants = select_best_ants(ants, method)
# set_pheremones!(state, best_ants)