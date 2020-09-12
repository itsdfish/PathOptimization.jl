"""
Ant object for ant colony optimization
*  `fitness`: array of fitness values where each element corresponds to fitness of
each objective
* `path`: an ordered array of nodes representing a path.
https://abc.erciyes.edu.tr/software.htm
"""
mutable struct Bee
    fitness::Array{Float64,1}
    path::Array{Int,1}
end

Bee() = Bee(Float64[], Int[])


"""
`BeeColony!` is an object that holds the parameters of the bee colony optimization algorithm.
* `n_bees::Int`: the number of bees in the colony
* `bees::Vector{Bee}`: a vector of bees
* `n_nodes::Int`: the number of nodes in the path
* `start_node::Int`: (1) the starting node of a path
* `end_node::Int`: (n_nodes) the ending node of a path

Example: 
```@example
method = BeeColony(n_bees=100, n_nodes=20)
```
"""
struct BeeColony <: PathFinder
    n_bees::Int
    bees::Vector{Bee}
    n_nodes::Int
    start_node::Int
    end_node::Int
end

function BeeColony(;n_bees=20, n_nodes=10,
    start_node=1, end_node=n_nodes)
    ants = [Bee() for _ in 1:n_bees]
    return BeeColony(n_bees, bees, n_nodes, start_node, end_node)
end

"""
`BeeColonyState!` is an object that holds the state of the ant colony algorithm.
* `n_obj::Int`: the number of objectives
* `cost::Array{Array{Float64,2},1}`: an array of cost matrices
* `frontier::T`: the Pareto frontier
"""

mutable struct BeeColonyState{T} <: State
    n_obj::Int
    cost::Array{Array{Float64,2},1}
    frontier::T
end

"""
initialize ant colony algorithm and return ant colony state object
* `method`: ant colony object
* `cost`: array of cost matrices
"""

function initialize(method::BeeColony, cost::Array{Float64,2})
    return initialize(method, [cost])
end

function initialize(method::BeeColony, cost)
    n_obj = length(cost)
    n_nodes,_ = size(cost[1])
    map(a -> a.fitness = fill(0.0, n_obj), method.ants)
    map(a -> a.path = fill(0, n_nodes), method.ants) 
    τ = map(x -> zero(x) .+ method.τmax, cost)
    η = map(x -> median(x, dims=2) ./ x, cost)
    θ = zero.(cost)
    scheme = Scheme{n_obj}(0.1, is_minimizing=true)
    a = EpsBoxArchive(scheme)
    state = ColonyState(n_obj, τ, η, cost, θ, a)
    compute_probabilities!(method, state)
    return state
end

"""
Find path for each ant in parallel.
* `method`: ant colony object
* `state`: colony state object
*  `rngs`: array of random number generators, one for each thread.
"""

function pfind_paths!(method::BeeColony, state, rngs)
    @threads for ant in method.ants 
        rng = rngs[threadid()]
        find_path!(method, state, ant, rng)
    end
end

"""
Find path for each ant.
* `method`: ant colony object
* `state`: colony state object
"""

function find_paths!(method::BeeColony, state, args...)
    for ant in method.ants
        find_path!(method, state, ant)
    end
end

find_path!(method::BeeColony, state, ant) = find_path!(method, state, ant, Random.GLOBAL_RNG)

"""
Find best ant, set pheremones and compute new transition probabilities
* `method`: ant colony object
* `state`: colony state object
"""
function update!(method::BeeColony, state::ColonyState)
    store_solutions!(method, state)
    best_ants = get_best_ants(method, state)
    set_pheremones!(method, state, best_ants)
    compute_probabilities!(method, state)
end

"""
Find path for an individual ant.
* `method`: ant colony object
* `state`: colony state object
* `ant`: a single ant
*  `rng`: a random number generator object.
"""
function find_path!(method::BeeColony, state, ant, rng)
    @unpack start_node, end_node, n_nodes = method
    @unpack path,fitness = ant
    @unpack θ,cost,n_obj = state
    fitness .= 0.0
    path[1],path[end] = start_node,end_node
    visited = fill(false, n_nodes)
    visited[[start_node,end_node]] .= true
    n0 = start_node
    for n in 2:(n_nodes - 1)
        obj_idx = rand(rng, 1:n_obj)
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

function employed_bee!(method::BeeColony, state)

end

function onlooker_bee!(method::BeeColony, state)

end

function scout_bee!(method::BeeColony, state)

end

"""
Find best ants for current iteration and all iterations.
* `method`: ant colony object
* `state`: colony state object
"""
function get_best_ants(method, state)
    ants = method.ants
    arr = Array{Ant,1}
    best_ants = (current = arr(), all = arr())
    all_paths = get_best_paths(state.frontier)
    frontier = get_best_cost(state.frontier)
    for obj in 1:state.n_obj
        min_ant,_ = get_min(x -> x.fitness[obj], ants)
        push!(best_ants.current, min_ant)
        mn_obj,idx = get_min(x -> x[obj], frontier)
        push!(best_ants.all, Ant([mn_obj...], Int.(all_paths[idx])))
    end
    return best_ants
end

function get_min(f::Function, array)
    mv = Inf
    mo = array[1]
    idx = 1
    for (i,a) in enumerate(array) 
        x = f(a)
        if x < mv 
            mv = x
            mo = a
            idx = i
        end 
    end 
    return mo,idx
end

"""
Add non-dominated solutions to Pareto frontier
* `method`: ant colony object
* `state`: colony state object
"""
function store_solutions!(method, state)
    # T = NTuple{state.n_obj,Float64}
    for ant in method.ants
        # fitness::T = Tuple(get_fitness(ant)) 
        add_candidate!(state.frontier, Tuple(ant.fitness), ant.path, 2)
    end
    return nothing
end

get_fitness(ant) = ant.fitness  