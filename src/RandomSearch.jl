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
    @unpack n_nodes,start_node,end_node = method
    # number of objective functions
    n_obj = length(cost)
    scheme = Scheme{n_obj}(0.1, is_minimizing=true)
    # object for managing Pareto frontier
    a = EpsBoxArchive(scheme)
    # initialize fitness 
    fitness = fill(0.0, n_obj)
    # initialize path
    d = setdiff(1:n_nodes, [start_node,end_node])
    path = [start_node;d;end_node]
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
    @unpack path,fitness,cost = state
    shuffle!(rng, @view path[2:end-1])
    fitness .= compute_path_cost(path, cost)
    return nothing
end

function update!(method::RandomSearch, state)
    store_solutions!(method, state)
end