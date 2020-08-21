using SafeTestsets

@safetestset "nearest neighbor find path" begin 
    using AntColonyOptimization, Distributions, Random
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
    method = method = NearestNeighbor(n_nodes=n_nodes)
    state = initialize(method, cost)
    find_path!(method, state)
    path = state.path
    path_cost = sum(i -> cost[1][path[i],path[i + 1]], 1:length(path) - 1)
    @test state.fitness[1] == path_cost    
end

@safetestset "random find path" begin 
    using AntColonyOptimization, Distributions, Random
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
    method = method = RandomSearch(n_nodes=n_nodes)
    state = initialize(method, cost)
    find_path!(method, state)
    path = state.path
    path_cost = sum(i -> cost[1][path[i],path[i + 1]], 1:length(path) - 1)
    @test state.fitness[1] == path_cost    
end

@safetestset "Ant Colony Run" begin
    using AntColonyOptimization, Distributions, Random
    using Test
    Random.seed!(514074)
    n_obj = 2
    n_nodes = 10
    cost = [rand(Uniform(0, 50), n_nodes, n_nodes) for _ in 1:n_obj] 
    iterations = 1000
    method = AntColony(n_ants=100, n_nodes=n_nodes, β=4.0)
    options = (trace = true, parallel = true, progress = false)
    result = optimize!(method, cost, iterations; options...)
    @test true
end

@safetestset "Random Search Run" begin
    using AntColonyOptimization, Distributions, Random
    using Test
    Random.seed!(514074)
    n_obj = 2
    n_nodes = 10
    cost = [rand(Uniform(0, 50), n_nodes, n_nodes) for _ in 1:n_obj] 
    iterations = 10_000
    method = RandomSearch(n_nodes=n_nodes)
    options = (trace = true, parallel = true, progress = false)
    result = optimize!(method, cost, iterations; options...)
    @test true
end

@safetestset "Nearest Neighbor Run" begin
    using AntColonyOptimization, Distributions, Random
    using Test
    Random.seed!(514074)
    n_obj = 2
    n_nodes = 10
    cost = [rand(Uniform(0, 50), n_nodes, n_nodes) for _ in 1:n_obj] 
    iterations = 10_000
    method = NearestNeighbor(n_nodes=n_nodes)
    options = (trace = true, parallel = true, progress = false)
    result = optimize!(method, cost, iterations; options...)
    @test true
end