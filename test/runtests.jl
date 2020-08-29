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