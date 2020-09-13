"""
`NearestNeighbor` is an object that holds the parameters of the nearest neighbor optimization algorithm.

* `n_nodes::Int`: the number of nodes in the path
* `start_node::Int`: (1) the starting node of a path
* `end_node::Int`: (n_nodes) the ending node of a path
* `use2opt`: use 2-opt algorithm if true

Example: 
```@example
method = NearestNeighbor(n_nodes=20)
```
"""
struct NearestNeighbor <: PathFinder
    n_nodes::Int
    start_node::Int
    end_node::Int
    use2opt::Bool
end

function NearestNeighbor(;n_nodes, start_node=1, end_node=n_nodes, use2opt=false) 
    return NearestNeighbor(n_nodes, start_node, end_node, use2opt)
end
"""
`AntColony!` is an object that holds the parameters of the ant colony optimization algorithm.
* `n_obj::Int`: the number of objective
* `cost::Array{Array{Float64,2},1}`: an array of cost matrices
* `frontier::T`: the Pareto frontier
* `path::Array{Int,1}`: current best path
"""
mutable struct NearestState{T} <: State
    n_obj::Int
    cost::Array{Array{Float64,2},1}
    frontier::T
    fitness::Array{Float64,1}
    path::Array{Int,1}
end

"""
initialize the nearest neighbor algorithm and return state object
* `method`: nearest neighbor object
* `cost`: array of cost matrices
"""
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

"""
Parallel search not supported for nearest neighbor
* `method`: nearest neighbor object
* `state`: nearest neighbor object
*  `rngs`: array of random number generators, one for each thread.
"""
function pfind_path!(method::NearestNeighbor, state, rngs)
    find_path!(method, state)
end

function find_path!(method::NearestNeighbor, state, args...)
    find_path!(method, state)
    method.use2opt ? two_opt(state) : nothing
end

find_path!(method::NearestNeighbor, state::NearestState) = find_path!(method, state, Random.GLOBAL_RNG)

"""
Find path for nearest neigbhor
* `method`: nearest neigbhor object
* `state`: algorithm state object
*  `rng`: a random number generator object.
"""
function find_path!(method::NearestNeighbor, state::NearestState, rng)
    @unpack n_obj,fitness,path,cost = state
    @unpack n_nodes,start_node,end_node = method
    path[1],path[end] = start_node,end_node
    not_visited = [1:n_nodes;]
    deleteat!(not_visited, [start_node,end_node])
    n0 = start_node
    for n in 2:(n_nodes - 1)
        obj_idx = rand(rng, 1:n_obj)
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

function reset_state!(state::NearestState)
    state.fitness .= 0.0
    state.path .= 0
end