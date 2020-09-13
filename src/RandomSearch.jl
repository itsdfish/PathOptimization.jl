struct RandomSearch <: PathFinder
    n_nodes::Int
    start_node::Int
    end_node::Int
end

RandomSearch(;n_nodes, start_node=1, end_node=n_nodes) = RandomSearch(n_nodes, start_node, end_node)

mutable struct RandomState{T} <: State
    n_obj::Int
    cost::Array{Array{Float64,2},1}
    frontier::T
    fitness::Array{Float64,1}
    path::Array{Int,1}
end

function initialize(method::RandomSearch, cost::Array{Float64,2})
    return initialize(method, [cost])
end

function initialize(method::RandomSearch, cost)
    # number of objective functions
    n_obj = length(cost)
    scheme = Scheme{n_obj}(0.1, is_minimizing=true)
    # object for managing Pareto frontier
    a = EpsBoxArchive(scheme)
    # initialize fitness 
    fitness = fill(0.0, n_obj)
    # initialize path
    path = fill(0, method.n_nodes)
    # initialize state object
    state = RandomState(n_obj, cost, a, fitness, path)
    return state
end

function pfind_path!(method::RandomSearch, state, rngs)
    find_path!(method, state)
end

function find_path!(method::RandomSearch, state, args...)
    find_path!(method, state)
end

find_path!(method::RandomSearch, state::RandomState) = find_path!(method, state, Random.GLOBAL_RNG)

function find_path!(method::RandomSearch, state::RandomState, rng)
    @unpack n_obj,fitness,path,cost = state
    @unpack n_nodes,start_node,end_node = method
    path[1],path[end] = start_node,end_node
    w = fill(1 / n_nodes, n_nodes)
    w[[start_node,end_node]] .= 0.0
    n0 = start_node
    for n in 2:(n_nodes - 1)
        obj_idx = rand(rng, 1:n_obj)
        n1 = sample(rng, 1:n_nodes, Weights(w))
        w[n1] = 0.0
        path[n] = n1
        map!(i -> fitness[i] += cost[i][n0,n1], fitness, 1:n_obj) 
        n0 = n1
    end
    map!(i -> fitness[i] += cost[i][n0,end_node], fitness, 1:n_obj) 
    return nothing
end

function update!(method::RandomSearch, state)
    store_solutions!(method, state)
    reset_state!(state)
end

function reset_state!(state::RandomState)
    state.fitness .= 0.0
    state.path .= 0
end