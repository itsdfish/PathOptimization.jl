"""
Ant object for ant colony optimization
*  `fitness`: array of fitness values where each element corresponds to fitness of
each objective
* `path`: an ordered array of nodes representing a path.
"""
mutable struct Ant
    fitness::Array{Float64,1}
    path::Array{Int,1}
end

Ant() = Ant(Float64[], Int[])

import Base: ==

function ==(a1::Ant, a2::Ant)
    for f in fieldnames(Ant)
        if getfield(a1, f) != getfield(a2, f)
            return false
        end
    end
    return true
end

"""
`AntColony!` is an object that holds the parameters of the ant colony optimization algorithm.
* `n_ants::Int`: the number of ants in the colony
* `ants::Vector{Ant}`: a vector of ants
* `τmin::Float64`: the minimum path pheremone value
* `τmax::Float64`: the maximum path pheremone value
* `α::Float64`: (1) the influence of the pheremone trail on path finding
* `β::Float64`: (1) the influence of local heuristics on path finding
* `ρ::Float64`: (.1) decay rate of the pheremone trial
* `n_nodes::Int`: the number of nodes in the path
* `start_node::Int`: (1) the starting node of a path
* `end_node::Int`: (n_nodes) the ending node of a path

Example: 
```@example
method = AntColony(n_ants=100, n_nodes=20, β=4.0, ρ=.10,
    τmin=0.0, τmax=10.0)
```
"""
struct AntColony <: PathFinder
    n_ants::Int
    ants::Vector{Ant}
    τmin::Float64
    τmax::Float64
    α::Float64
    β::Float64
    ρ::Float64
    max_no_change::Int
    n_nodes::Int
    start_node::Int
    end_node::Int
    use2opt::Bool
end

function AntColony(;n_ants=20, τmin=1.0, τmax=10.0, α=1.0, β=1.0, ρ=0.1, max_no_change=50, n_nodes=10,
    start_node=1, end_node=n_nodes, use2opt=false)
    ants = [Ant() for _ in 1:n_ants]
    return AntColony(n_ants, ants, τmin, τmax, α, β, ρ, max_no_change, n_nodes, start_node, end_node,
        use2opt)
end

"""
`ColonyState!` is an object that holds the state of the ant colony algorithm.
* `n_obj::Int`: the number of objectives
* `τ::Array{Array{Float64,2},1}`: array of pheremone matrices corresponding to each objective
* `η::Array{Array{Float64,2},1}`: array of local heuristics
* `cost::Array{Array{Float64,2},1}`: an array of cost matrices
* `θ::Array{Array{Float64,2},1}`: array of probability matrices, one matrix for each objective 
* `frontier::T`: the Pareto frontier
"""

mutable struct ColonyState{T} <: State
    n_obj::Int
    τ::Array{Array{Float64,2},1}
    η::Array{Array{Float64,2},1}
    cost::Array{Array{Float64,2},1}
    θ::Array{Array{Float64,2},1}
    frontier::T
end

"""
initialize ant colony algorithm and return ant colony state object
* `method`: ant colony object
* `cost`: array of cost matrices
"""

function initialize(method::AntColony, cost::Array{Float64,2})
    return initialize(method, [cost])
end

function initialize(method::AntColony, cost)
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

function pfind_paths!(method::AntColony, state, rngs)
    @threads for ant in method.ants 
        rng = rngs[threadid()]
        find_path!(method, state, ant, rng)
    end
    method.use2opt ? two_opt(method, state) : nothing
end

"""
Find path for each ant.
* `method`: ant colony object
* `state`: colony state object
"""

function find_paths!(method::AntColony, state, args...)
    for ant in method.ants
        find_path!(method, state, ant)
    end
    method.use2opt ? two_opt(method, state) : nothing
end

find_path!(method::AntColony, state, ant) = find_path!(method, state, ant, Random.GLOBAL_RNG)

function two_opt(method::AntColony, state)
    for obj in 1:state.n_obj
        ant,_ = findmin(x->x.fitness[obj], method.ants)
        ant.path = two_opt(ant.path, state.cost)
        ant.fitness = compute_path_cost(ant.path, state.cost)
    end
end

"""
Find best ant, set pheremones and compute new transition probabilities
* `method`: ant colony object
* `state`: colony state object
"""
function update!(method::AntColony, state::ColonyState)
    store_solutions!(method, state)
    best_ants = get_best_ants(method, state)
    eval_progress!(method, state)
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
function find_path!(method::AntColony, state, ant, rng)
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

"""
Find path for an individual ant.
* `method`: ant colony object
* `state`: colony state object
* `best_ants`: best ants for current iteration and all iterations
"""
function set_pheremones!(method, state, best_ants)
    @unpack τ,n_obj = state
    @unpack ρ,τmin,τmax = method
    for obj in 1:n_obj
        current_fit = best_ants.current[obj].fitness[obj]
        best_fit = best_ants.all[obj].fitness[obj]
        path = best_ants.current[obj].path
        set_pheremones!(τ[obj], ρ, τmin, τmax, current_fit, best_fit, path)
    end
    return nothing
end

function set_pheremones!(τ, ρ, τmin, τmax, current_fit, best_fit, path)
    τ .*= (1 - ρ)
    τΔ = 1 / (1 + current_fit - best_fit)
    for i in 1:(length(path) - 1)
        n0,n1 = path[i],path[i + 1]
        τ[n0,n1] += τΔ
    end
    τ .= min.(τ, τmax)
    τ .= max.(τ, τmin)
    return nothing
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
Compute transition matrices using pheremone and local heuristic matrices
* `method`: ant colony object
* `state`: colony state object
"""
function compute_probabilities!(method, state)
    @unpack θ,τ,η = state
    @unpack α,β = method
    for (θ′, τ′, η′) in zip(θ, τ, η)
        compute_probabilities!(θ′, τ′, η′, α, β)
    end
end

function compute_probabilities!(θ, τ, η, α, β)
    @. θ = (τ^α) * (η^β)
    θ ./= sum(θ, dims=2)
end

"""
Add non-dominated solutions to Pareto frontier
* `method`: ant colony object
* `state`: colony state object
"""
function store_solutions!(method::AntColony, state)
    # T = NTuple{state.n_obj,Float64}
    for ant in method.ants
        # fitness::T = Tuple(get_fitness(ant)) 
        add_candidate!(state.frontier, Tuple(ant.fitness), ant.path, 2)
    end
    return nothing
end

get_fitness(ant) = ant.fitness

function eval_progress!(method::AntColony, state)
    frontier = state.frontier
    streak = noprogress_streak(state.frontier, since_restart=true)
    if streak > method.max_no_change * method.n_ants 
        notify!(frontier, :restart)
        reset_pheremones!(method, state)
    end
    return nothing
end

function reset_pheremones!(method, state)
    map!(x -> x .= method.τmax, state.τ, state.τ)
end