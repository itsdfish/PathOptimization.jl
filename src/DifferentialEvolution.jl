"""
* `θ`: a vector of parameters
* `samples`: a 2-dimensional array containing all acccepted proposals
* `accept`: proposal acceptance. 1: accept, 0: reject
* `weight`: particle weight based on model fit (currently posterior log likelihood)
* `lp`: a vector of log posterior probabilities associated with each accepted proposal
"""
mutable struct Particle
    obj_idx::Int
    path::Array{Int,1}
    Θ::Array{Float64,1}
    fitness::Array{Float64,1}
    n_same::Int
end

Base.broadcastable(x::Particle) = Ref(x)

function Particle(n_nodes::Int, n_obj, start_node, end_node)
    d = setdiff(1:n_nodes, [start_node,end_node])
    path = [start_node;d;end_node]
    shuffle!(@view path[2:(end-1)])
    Θ = path*1.0
    fitness = fill(0.0, n_obj)
    obj_idx = rand(1:n_obj)
    Particle(obj_idx, path, Θ, fitness, 0)
end

function Particle(;Θ=Θ)
    n_nodes = length(Θ)
    path = [1:n_nodes;]
    fitness = Float64[]
    Particle(0,path, Θ, fitness, 0)
end

"""
* `n_particles`: number of particles (default=50)
* `n_nodes`: number of nodes in graph
* `start_node`: the starting node of the route (default = 1)
* `end_node`: the ending node of the route (default = n_nodes)
* `κ`: mutation probability (default = .50)
* `Θneighbor`: probability of initializing particle with nearest neighbor (default = .10)
* `θbest`: probability that a particle will swap path elements with best particle
* `θswap`: probablity that a selected particle will swap a path element with the path element of the best particle
* `γmin`: γ min for differential (default = .60)
* `γmax`: γ max for differential (default = .80)
* `recombination!`: a recombination function (default `exponential`!)
* `max_evals`: maximum number of evaluations a particle can have without improving fitness 
"""
struct DE{F<:Function} <: PathFinder
    n_particles::Int
    n_nodes::Int
    start_node::Int
    end_node::Int
    κ::Float64
    Θneighbor::Float64
    θbest::Float64
    θswap::Float64
    γmin::Float64
    γmax::Float64
    recombination!::F
    max_evals::Int
    max_no_change::Int
end

function DE(;n_particles=50, n_nodes, start_node=1, end_node=n_nodes,κ=.50, Θneighbor=.10, 
    θbest=.9, θswap=.8, γmin=.6, γmax=.8, recombination! = exponential!, max_evals=10, max_no_change=100)
    return DE(n_particles, n_nodes, start_node, end_node, κ, Θneighbor,
        θbest, θswap, γmin, γmax, recombination!, max_evals, max_no_change)
end

"""
* `mid_nodes`: all nodes excluding end points, which can be reordered
"""
mutable struct DEState{T} <: State
    n_evals::Int
    n_obj::Int
    cost::Array{Array{Float64,2},1}
    frontier::T
    particles::Array{Particle,1}
    mid_nodes::Array{Int,1}
end

function initialize(method::DE, cost::Array{Float64,2})
    return initialize(method, [cost])
end

function initialize(method::DE, cost)
    @unpack start_node,end_node,n_nodes,n_particles = method
    n_obj = length(cost)
    scheme = Scheme{n_obj}(0.1, is_minimizing=true)
    a = EpsBoxArchive(scheme)
    fitness = fill(0.0, n_obj)
    mid_nodes = setdiff([1:n_nodes;], [start_node,end_node])
    particles = [Particle(n_nodes, n_obj, start_node,end_node) for _ in 1:n_particles]
    state = DEState(0, n_obj, cost, a, particles, mid_nodes)
    nearest_neighbor!(method, state)
    map(x->compute_path_cost!(x, cost), particles)
    return state
end

function pfind_path!(method::DE, state, rngs)
    @threads for particle in method.particles 
        rng = rngs[threadid()]
        find_path!(method, state, particle, rng)
    end
end

function find_path!(method::DE, state, args...)
    for particle in state.particles
        find_path!(method, state, particle)
    end
end

find_path!(method::DE, state::DEState, particle::Particle) = find_path!(method, state, particle, Random.GLOBAL_RNG)

function find_path!(method::DE, state::DEState, particle::Particle, rng)
    @unpack n_obj,fitness,path,cost = state
    @unpack n_nodes,start_node,end_node = method
    path[1],path[end] = start_node,end_node
    w = fill(1 / n_nodes, n_nodes)
    w[[start_node,end_node]] .= 0.0
    n0 = start_node
end

function update!(method::DE, state)
    store_solutions!(method, state)
    set_best_path(method, state)
    adapt_gamma(method, state)
end

function store_solutions!(method, state)
    add_candidate!(state.frontier, Tuple(state.fitness), state.path, 2)
    return nothing
end

function cross_over(de, Pt, particles)
    idxs = findall(x -> x != Pt, particles)
    others = @view particles[idxs]
    Pm,Pn = sample(others, 2, replace=false)
    γ = 2.38
    # compute proposal value
    proposal = Pt + γ * (Pm - Pn)
    proposal.obj_idx = Pt.obj_idx
    return proposal
end

"""
Update particle based on Greedy Rule.
* `current`: current particle
* `proposal`: proposal particle
"""
function update_particle!(current, proposal)
    if proposal.fitness < current.fitness
        current.path = proposal.path
        current.fitness = proposal.fitness
    end
    return nothing
end

function nearest_neighbor!(method::DE, state)
    for particle in state.particles
        if rand() ≤ method.Θneighbor
            nearest_neighbor!(method, state, particle)
            reverse_map!(method, state, particle)
        end
    end
    return nothing
end

function nearest_neighbor!(method::PathFinder, state, particle)
    @unpack fitness,path = particle
    nearest_neighbor!(method::PathFinder, state, fitness, path)
end

function adapt_gamma(method, state)
    @unpack γmin, γmax = method
    @unpack max_evals, n_evals = state
    return (γmin - γmax) / max_evals * n_evals + γmax
end

function rank_order!(method, state, particle)
    particle.path[1] = method.start_node
    particle.path[end] = method.end_node
    cpath = @view particle.Θ[2:end-1]
    path = @view particle.path[2:end-1]
    idx = sortperm(cpath)
    reorder!(path, idx, state.mid_nodes)
    return nothing
end

"""
"""
function reverse_map!(method, state, particle)
    cpath = @view particle.Θ[2:end-1]
    path = @view particle.path[2:end-1]
    idx = sortperm(path)
    reorder!(cpath, idx, state.mid_nodes)
    return nothing
end

function reorder!(path, idx, nodes)
    for (i,r) in zip(idx,nodes)
        path[i] = r
    end
    return nothing
end

function best_match_rank!(method, state, particle)
    @unpack n_nodes,θbest,θswap = method
    @unpack particles = state
    path = particle.path
    rank_order!(method, state, particle)
    best_particle,_ = findmin(x->x.fitness[1], particles)
    best_path = best_particle.path
    if rand() ≤ θbest
        for n in 2:(n_nodes-1)
            if rand() ≤ θswap
                swap_nodes!(path, best_path, n)
            end
        end
    end
end

function swap_nodes!(path, best_path, n)
    old_node = path[n]
    idx = findfirst(x->x == best_path[n], path)
    path[n] = best_path[n]
    path[idx] = old_node
    return nothing
end

function eval_progress!(method::DE, state)
    if exceed_max_no_change(method, state, method.particles)
        #reset_pheremones!(method, state)
    end
    return nothing
end

function compute_path_cost!(particle, cost)
    particle.fitness .= compute_path_cost(particle.path, cost)
end

# Type-stable arithmatic operations for Union{Array{T,1},T} types (which return Any otherwise)
import Base: +, - ,*

function +(x::Particle, y::Particle)
    N = length(x.Θ)
    z = similar(x.Θ)
    for i in 1:N
        z[i] = x.Θ[i] + y.Θ[i]
    end
    return Particle(Θ=z)
end

+(x::Float64, y::Particle) = +(y, x)

function +(x::Particle, y::Float64)
    N = length(x.Θ)
    z = similar(x.Θ)
    for i in 1:N
        z[i] = x.Θ[i] .+ y
    end
    return Particle(Θ=z)
end

function *(x::Particle, y::Particle)
    N = length(x.Θ)
    z = similar(x.Θ)
    for i in 1:N
        z[i] = x.Θ[i] .* y.Θ[i]
    end
    return Particle(Θ=z)
end

*(x::Float64, y::Particle) = *(y, x)

function *(x::Particle, y::Float64)
    N = length(x.Θ)
    z = similar(x.Θ)
    for i in 1:N
        z[i] = x.Θ[i] .* y
    end
    return Particle(Θ=z)
end

function -(x::Particle, y::Particle)
    N = length(x.Θ)
    z = similar(x.Θ)
    for i in 1:N
        z[i] = x.Θ[i] - y.Θ[i]
    end
    return Particle(Θ=z)
end

-(x::Float64, y::Particle) = -(y, x)

function -(x::Particle, y::Float64)
    N = length(x.Θ)
    z = similar(x.Θ)
    for i in 1:N
        z[i] = x.Θ[i] .- y
    end
    return Particle(Θ=z)
end
