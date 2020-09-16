using SafeTestsets

@safetestset "ant colony find path" begin 
    using PathOptimization, Distributions, Random
    using Test
    Random.seed!(95590)
    n_obj = 2
    n_nodes = 10
    cost = [rand(Uniform(0, 50), n_nodes, n_nodes) for _ in 1:n_obj] 
    method = AntColony(n_nodes=n_nodes)
    state = initialize(method, cost)
    ant = method.ants[1]
    find_path!(method, state, ant)
    @test length(ant.fitness) == n_obj
    @test length(unique(ant.path)) == n_nodes
    @test ant.path[1] == 1
    @test ant.path[end] == n_nodes

    n_obj = 1
    n_nodes = 4
    cost = [rand(Uniform(0, 50), n_nodes, n_nodes) for _ in 1:n_obj] 
    method = AntColony(n_nodes=n_nodes)
    state = initialize(method, cost)
    ant = method.ants[1]
    find_path!(method, state, ant)
    path = ant.path
    path_cost = sum(i -> cost[1][path[i],path[i + 1]], 1:length(path) - 1)
    @test ant.fitness[1] == path_cost    
end

@safetestset "nearest neighbor find path" begin 
    using PathOptimization, Distributions, Random
    using Test
    Random.seed!(95590)
    n_obj = 2
    n_nodes = 10
    cost = [rand(Uniform(0, 50), n_nodes, n_nodes) for _ in 1:n_obj] 
    method = method = NearestNeighbor(n_nodes=n_nodes)
    state = initialize(method, cost)
    find_path!(method, state)
    @test length(state.fitness) == n_obj
    @test length(unique(state.path)) == n_nodes
    @test state.path[1] == 1
    @test state.path[end] == n_nodes

    n_obj = 1
    n_nodes = 4
    cost = [rand(Uniform(0, 50), n_nodes, n_nodes) for _ in 1:n_obj] 
    method = NearestNeighbor(n_nodes=n_nodes)
    state = initialize(method, cost)
    find_path!(method, state)
    path = state.path
    path_cost = sum(i -> cost[1][path[i],path[i + 1]], 1:length(path) - 1)
    @test state.fitness[1] == path_cost    
end

@safetestset "random find path" begin 
    using PathOptimization, Distributions, Random
    using Test
    Random.seed!(95590)
    n_obj = 2
    n_nodes = 10
    cost = [rand(Uniform(0, 50), n_nodes, n_nodes) for _ in 1:n_obj] 
    method = method = RandomSearch(n_nodes=n_nodes)
    state = initialize(method, cost)
    find_path!(method, state)
    @test length(state.fitness) == n_obj
    @test length(unique(state.path)) == n_nodes
    @test state.path[1] == 1
    @test state.path[end] == n_nodes

    n_obj = 1
    n_nodes = 4
    cost = [rand(Uniform(0, 50), n_nodes, n_nodes) for _ in 1:n_obj] 
    method = RandomSearch(n_nodes=n_nodes)
    state = initialize(method, cost)
    find_path!(method, state)
    path = state.path
    path_cost = sum(i -> cost[1][path[i],path[i + 1]], 1:length(path) - 1)
    @test state.fitness[1] == path_cost    
end

@safetestset "Ant Colony Run" begin
    using PathOptimization, Distributions, Random
    using Test
    Random.seed!(514074)
    n_obj = 2
    n_nodes = 10
    cost = [rand(Uniform(0, 50), n_nodes, n_nodes) for _ in 1:n_obj] 
    iterations = 1000
    method = AntColony(n_ants=100, n_nodes=n_nodes, β=4.0)
    options = (parallel = true, progress = false)
    result = optimize(method, cost, iterations; options...)
    @test true
end

@safetestset "Random Search Run" begin
    using PathOptimization, Distributions, Random
    using Test
    Random.seed!(514074)
    n_obj = 2
    n_nodes = 10
    cost = [rand(Uniform(0, 50), n_nodes, n_nodes) for _ in 1:n_obj] 
    iterations = 10_000
    method = RandomSearch(n_nodes=n_nodes)
    options = (parallel = true, progress = false)
    result = optimize(method, cost, iterations; options...)
    @test true
end

@safetestset "Nearest Neighbor Run" begin
    using PathOptimization, Distributions, Random
    using Test
    Random.seed!(514074)
    n_obj = 2
    n_nodes = 10
    cost = [rand(Uniform(0, 50), n_nodes, n_nodes) for _ in 1:n_obj] 
    iterations = 10_000
    method = NearestNeighbor(n_nodes=n_nodes)
    options = (parallel = true, progress = false)
    result = optimize(method, cost, iterations; options...)
    @test true
end

@safetestset "best ant" begin 
    using PathOptimization, Distributions, Random
    using Test
    Random.seed!(95590)
    n_obj = 2
    n_nodes = 10
    cost = [rand(Uniform(0, 50), n_nodes, n_nodes) for _ in 1:n_obj] 
    method = AntColony(n_nodes=n_nodes, n_ants=3)
    state = initialize(method, cost)
    ants = method.ants
    ants[1].fitness = [2.0,1.0]
    ants[2].fitness = [1.0,0.0]
    ants[3].fitness = [0.0,1.0]
    store_solutions!(method, state)
    best_ants = get_best_ants(method, state)
    @test length(best_ants.current) == 2
    @test length(best_ants.all) == 2
    @test ants[2] in best_ants.current
    @test ants[3] in best_ants.current
    @test ants[2] in best_ants.all
    @test ants[3] in best_ants.all  
end

@safetestset "pheremone" begin 
    using PathOptimization, Distributions, Random
    using Test
    Random.seed!(95590)
    n_obj = 1
    n_nodes = 5
    cost = [rand(Uniform(0, 50), n_nodes, n_nodes) for _ in 1:n_obj] 
    method = AntColony(n_nodes=n_nodes, n_ants=1)
    state = initialize(method, cost)
    path = [1,3,2,4,5]
    current_fit = 2.0
    best_fit = 1.0
    τ = state.τ[1]
    τ .= 2.0
    set_pheremones!(τ, method.ρ, method.τmin, method.τmax, current_fit, best_fit, path)
    for i in 1:4
        @test τ[path[i],path[i + 1]] == 2.3
    end
    @test count(x -> x == 1.8, τ) == 21
    τ .= 12.0
    set_pheremones!(τ, method.ρ, method.τmin, method.τmax, current_fit, best_fit, path)
    @test all(x -> x == 10.0, τ)
    τ .= -2.0
    set_pheremones!(τ, method.ρ, method.τmin, method.τmax, current_fit, best_fit, path)
    @test all(x -> x == 1.0, τ)
end

@safetestset "find min" begin 
    using PathOptimization, Distributions, Random
    using Test
    n_nodes = 10
    method = AntColony(n_nodes=n_nodes, n_ants=3)
    ants = method.ants
    ants[1].fitness = [2.0,1.0]
    ants[2].fitness = [1.0,0.0]
    ants[3].fitness = [0.0,1.0]
    min_ant,min_idx = findmin(x->x.fitness[1], ants)
    @test min_idx == 3
    @test min_ant == ants[3]
end

@safetestset "2-opt" begin 
    using PathOptimization, Random
    import PathOptimization: two_opt, compute_path_cost
    using Test
    Random.seed!(584410)
    costs = [rand(10,10)]
    path = [1:10;]
    path1 = two_opt(path, costs)
    path2 = two_opt(path, costs)
    path_cost = compute_path_cost(path, costs)
    path1_costs = compute_path_cost(path1, costs)
    @test path1 == path2
    @test path_cost > path1_costs
end

@safetestset "relative cost" begin 
    using PathOptimization, Random
    import PathOptimization: relative_cost, compute_path_cost
    using Test
    Random.seed!(6540)
    n_nodes = 10
    costs = [rand(n_nodes, n_nodes)]
    for r in 2:(n_nodes-1), c in (r+1):(n_nodes-1)
        path = [1:10;]
        rel_cost = relative_cost(path, costs[1], r, c)
        cost1 = compute_path_cost(path, costs)[1]
        reverse!(path, r, c)
        cost2 = compute_path_cost(path, costs)[1]
        diff_cost = cost2 - cost1
        @test diff_cost ≈ rel_cost rtol = .0001
    end
end

@safetestset "rank_order!" begin 
    using PathOptimization, Random, Distributions
    import PathOptimization: rank_order!
    using Test
    Random.seed!(6540)
    n_obj = 2
    n_nodes = 10
    start_node = 3
    end_node = 8
    cost = [rand(Uniform(0, 50), n_nodes, n_nodes) for _ in 1:n_obj] 
    method = DE(n_nodes=n_nodes, start_node=start_node, end_node=end_node)
    state = initialize(method, cost)
    particle = Particle(n_nodes, n_obj, start_node, end_node)
    shuffle!(@view particle.Θ[2:end-1])
    rank_order!(method, state, particle)
    correct_order = [3,9,5,7,2,10,4,6,1,8]
    @test particle.path == correct_order
end

@safetestset "swap_nodes!" begin 
    using PathOptimization, Random, Distributions
    import PathOptimization: swap_nodes!
    using Test
    Random.seed!(6540)
    n = 6
    path = [6,10,2,8,4,3,5,7,1,9]
    best_path = [1,10,2,8,4,7,3,5,9,6]
    correct_path = [6,10,2,8,4,7,5,3,1,9]
    swap_nodes!(path, best_path, n)
    @test path == correct_path
end