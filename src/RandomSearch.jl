struct RandomSearch <: PathFinder
    n_nodes::Int
    start_node::Int
    end_node::Int
end

mutable struct RandomState{T} <: State
    n_obj::Int
    cost::Array{Array{Float64,2},1}
    frontier::T
    fitness::Array{Float64,1}
    path::Array{Float64,1}
end

function find_path(method::RandomSearch, state::RandomState, cost)
    fitness = fill(0.0, lenth(cost)) 
    n_nodes = size(cost[1])[1]
    path[1],path[end] = start_node,end_node
    visited = fill(false, n_nodes)
    visited[[1,end]] .= true
    n0 = start_node
    for n in 2:(n_nodes - 1)
        obj_idx = rand(1:n_obj)
        w = θ[obj_idx][n0,:]
        w[visited] .= 0.0
        n1 = sample(rng, 1:n_nodes, Weights(w))
        visited[n1] = true
        path[n] = n1
        map!(i -> fitness[i] += cost[i][n0,n1], fitness, 1:n_obj) 
        n0 = n1
    end
    map!(i -> fitness[i] += cost[i][n0,end_node], fitness, 1:n_obj) 
    return ant.fitness
end

function initialize(method::RandomSearch, cost::Array{Float64,2})
    return initialize(method, [cost])
end

function initialize(method::RandomSearch, cost)
    n_obj = length(costs) 
    scheme = Scheme{n_obj}(0.1, is_minimizing=true)
    a = EpsBoxArchive(scheme)
    state = RandomState(n_obj, τ, η, costs, θ, a)
    compute_probabilities!(method, state)
    return state
end

function pfind_paths!(method::RandomSearch, state, rngs)
    @threads for ant in method.ants 
        rng = rngs[threadid()]
        find_path!(ant, method, state, rng)
    end
end

function find_paths!(method::RandomSearch, state, args...)
    for ant in method.ants
        find_path!(ant, method, state)
    end
end

find_path!(ant, method, state) = find_path!(ant, method, state, Random.GLOBAL_RNG)

function find_path!(method::RandomSearch, state, rng)
    @unpack start_node, end_node, n_nodes = method
    @unpack cost,n_obj,fitness,path = state
    fitness .= 0.0
    path[1],path[end] = start_node,end_node
    visited = fill(false, n_nodes)
    visited[[1,end]] .= true
    n0 = start_node
    for n in 2:(n_nodes - 1)
        obj_idx = rand(1:n_obj)
        w = θ[obj_idx][n0,:]
        w[visited] .= 0.0
        n1 = sample(rng, 1:n_nodes, Weights(w))
        visited[n1] = true
        path[n] = n1
        map!(i -> fitness[i] += cost[i][n0,n1], fitness, 1:n_obj) 
        n0 = n1
    end
    map!(i -> fitness[i] += cost[i][n0,end_node], fitness, 1:n_obj) 
    return nothing
end

function update!(method::RandomSearch, state::ColonyState)
    store_solutions!(method, state)
    reset_state!(state)
end

function store_solutions!(method::RandomSearch, state)
    add_candidate!(state.frontier, state.current_fitness, state.current_path, 2)
    return nothing
end

function reset_state!(state::RandomState)

end