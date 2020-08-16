function optimize!(method::AntColony, cost_matrix, iterations; trace=false)
    state = initialize(cost_matrix, method)
    ants = [Ant(method.n_nodes) for _ in 1:method.n_ants]
    for i in 1:iterations
        for ant in ants 
            find_path!(ant, method, state)
        end
        store_solutions!(method, state, ants)
        best_ants = select_best_ants(ants, method)
        set_best_path!(state, best_ants, trace)
        set_pheremones!(method, state, best_ants)
        compute_probabilities!(method, state)
    end
    return state
end

function compute_probabilities!(method, state)
    @unpack θ,cost,τ = state
    @unpack α,β = method
    @. θ = (τ^α) * (cost^β)
    θ ./= sum(θ, dims=2)
end

function find_path!(ant, method, state)
    @unpack start_node, end_node, n_nodes = method
    @unpack path = ant
    @unpack θ,cost = state
    path[1],path[end] = start_node,end_node
    visited = fill(false, n_nodes)
    visited[[1,end]] .= true
    n0 = start_node
    fitness = 0.0
    for n in 2:(n_nodes - 1)
        w = θ[n0,:]
        w[visited] .= 0.0
        n1 = sample(1:n_nodes, Weights(w))
        visited[n1] = true
        path[n] = n1
        n0 = n1
        fitness += cost[n0,n1] 
    end
    fitness += cost[n0,end_node]
    ant.fitness = fitness
    return nothing
end

function set_best_path!(state, ants, trace)
    best_ant = ants[1]
    if best_ant.fitness < state.best_fitness
        println("best fitness: ", best_ant.fitness)
        state.best_fitness = best_ant.fitness
        state.best_path = best_ant.path
    end
    return nothing
end

function set_pheremones!(method, state, ants)
    @unpack τ = state
    @unpack ρ = method
    for ant in ants
        set_pheremones!(τ, ρ, ant)
    end
    return nothing
end

function set_pheremones!(τ, ρ, ant::Ant)
    @unpack path,fitness = ant
    for i in 1:(length(path) - 1)
        n0,n1 = path[i],path[i + 1]
        τ[n0,n1] += (1 - ρ) * τ[n0,n1] +  (1 / fitness)
    end
    return nothing
end

function select_best_ants(ants, method)
    sort!(ants, by=x -> x.fitness)
    return ants[1:method.n_best]
end

function initialize(cost, method)
    state = ColonyState(fill(1.0, size(cost)), cost, zero(cost), Inf, Float64[],
    Dict{Array{Int64,1},Float64}())
    compute_probabilities!(method, state)
    return state
end

function store_solutions!(method, state, ants)
    if method.retain_solutions
        for ant in ants
            if !haskey(state.all_solutions, ant.path)
                state.all_solutions[ant.path] = ant.fitness
            end
        end
    end
    return nothing
end

is_dominated(x, y) = all(x .> y)                        