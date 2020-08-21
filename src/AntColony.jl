mutable struct Ant
    fitness::Array{Float64,1}
    path::Array{Int,1}
end

Ant() = Ant(Float64[], Int[])

struct AntColony <: PathFinder
    n_ants::Int
    ants::Vector{Ant}
    τmin::Float64
    τmax::Float64
    α::Float64
    β::Float64
    ρ::Float64
    n_nodes::Int
    start_node::Int
    end_node::Int
end

function AntColony(;n_ants=20, τmin=1.0, τmax=5.0, α=1.0, β=1.0, ρ=0.1, n_nodes=10,
    start_node=1, end_node=n_nodes, retain_solutions=false)
    ants = [Ant() for _ in 1:n_ants]
    return AntColony(n_ants, ants, τmin, τmax, α, β, ρ, n_nodes, start_node, end_node)
end

mutable struct ColonyState{T} <: State
    n_obj::Int
    τ::Array{Array{Float64,2},1}
    η::Array{Array{Float64,2},1}
    cost::Array{Array{Float64,2},1}
    θ::Array{Array{Float64,2},1}
    frontier::T
end

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

function pfind_paths!(method::AntColony, state, rngs)
    @threads for ant in method.ants 
        rng = rngs[threadid()]
        find_path!(method, state, ant, rng)
    end
end

function find_paths!(method::AntColony, state, args...)
    for ant in method.ants
        find_path!(method, state, ant)
    end
end

find_path!(method::AntColony, state, ant) = find_path!(method, state, ant, Random.GLOBAL_RNG)

function update!(method::AntColony, state::ColonyState)
    store_solutions!(method, state)
    best_ants = get_best_ants(method, state)
    set_pheremones!(method, state, best_ants)
    compute_probabilities!(method, state)
end

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

function get_best_ants(method, state)
    ants = method.ants
    ant = ants[1]
    n_obj = length(ant.fitness)
    arr = Array{typeof(ant),1}()
    best_ants = (current = arr, all = arr)
    all_paths = get_best_paths(state.frontier)
    frontier = get_best_cost(state.frontier)
    for obj in 1:n_obj
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

function store_solutions!(method, state)
    T = NTuple{state.n_obj,Float64}
    for ant in method.ants
        fitness::T = Tuple(get_fitness(ant)) 
        add_candidate!(state.frontier, fitness, ant.path, 2)
    end
    return nothing
end

get_fitness(ant) = ant.fitness