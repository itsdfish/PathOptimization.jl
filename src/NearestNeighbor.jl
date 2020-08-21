struct NearestNeighbor <: PathFinder
    n_nodes::Int
    start_node::Int
    end_node::Int
end

NearestNeighbor(;n_nodes, start_node=1, end_node=n_nodes) = NearestNeighbor(n_nodes, start_node, end_node)

mutable struct NearestState{T} <: State
    n_obj::Int
    cost::Array{Array{Float64,2},1}
    frontier::T
    fitness::Array{Float64,1}
    path::Array{Int,1}
end

function initialize(method::NearestNeighbor, cost::Array{Float64,2})
    return initialize(method, [cost])
end

function initialize(method::NearestNeighbor, cost)
    n_obj = length(cost)
    scheme = Scheme{n_obj}(0.1, is_minimizing=true)
    a = EpsBoxArchive(scheme)
    fitness = fill(0.0, n_obj)
    path = fill(0, method.n_nodes)
    state = NearestState(n_obj, cost, a, fitness, path)
    return state
end

function pfind_paths!(method::NearestNeighbor, state, rngs)
    find_path!(method, state)
end

function find_paths!(method::NearestNeighbor, state, args...)
    find_path!(method, state)
end

find_path!(method::NearestNeighbor, state::NearestState) = find_path!(method, state, Random.GLOBAL_RNG)

function find_path!(method::NearestNeighbor, state::NearestState, rng)
    @unpack n_obj,fitness,path,cost = state
    @unpack n_nodes,start_node,end_node = method
    path[1],path[end] = start_node,end_node
    not_visited = [1:n_nodes;]
    deleteat!(not_visited, [start_node,end_node])
    n0 = start_node
    for n in 2:(n_nodes - 1)
        obj_idx = rand(1:n_obj)
        v = @view cost[obj_idx][n0,not_visited]
        _,min_idx = findmin(v)
        n1 = not_visited[min_idx]
        deleteat!(not_visited, min_idx)
        path[n] = n1
        map!(i -> fitness[i] += cost[i][n0,n1], fitness, 1:n_obj) 
        n0 = n1
    end
    map!(i -> fitness[i] += cost[i][n0,end_node], fitness, 1:n_obj) 
    return nothing
end

function update!(method::NearestNeighbor, state)
    store_solutions!(method, state)
    reset_state!(state)
end

function store_solutions!(method::NearestNeighbor, state)
    add_candidate!(state.frontier, Tuple(state.fitness), state.path, 2)
    return nothing
end

function reset_state!(state::NearestState)
    state.fitness .= 0.0
    state.path .= 0
end