"""
* `θ`: a vector of parameters
* `samples`: a 2-dimensional array containing all acccepted proposals
* `accept`: proposal acceptance. 1: accept, 0: reject
* `weight`: particle weight based on model fit (currently posterior log likelihood)
* `lp`: a vector of log posterior probabilities associated with each accepted proposal
"""
mutable struct Particle
    path::Array{Int,1}
    Θ::Array{Float64,1}
    fitness::Array{Float64,1}
    n_same::Int
end

Base.broadcastable(x::Particle) = Ref(x)

function Particle(n_nodes::Int, n_obj, start_node, end_node)
    path = [start_node;2:(n_nodes-1);end_node]
    Θ = path*1.0
    fitness = fill(0.0, n_obj)
    Particle(path, Θ, fitness, 0)
end

function Particle(;Θ=Θ)
    n_nodes = length(Θ)
    path = [1:n_nodes;]
    cpath = [1.0:n_nodes;]
    fitness = Float64[]
    Particle(path, Θ, fitness, 0)
end

"""
* `n_particles`: number of particles (default=50)
* `n_nodes`: number of nodes in graph
* `start_node`: the starting node of the route (default = 1)
* `end_node`: the ending node of the route (default = n_nodes)
* `κ`: mutation probability (default = .50)
* `θcluster`: cluster probability (default = .10)
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
    θcluster::Float64
    θbest::Float64
    θswap::Float64
    γmin::Float64
    γmax::Float64
    recombination!::F
    max_evals::Int
end

function DE(;n_particles=50, n_nodes, start_node=1, end_node=n_nodes,κ=.50, θcluster=.10, 
    θbest=.9, θswap=.8, γmin=.6, γmax=.8, recombination! = exponential!, max_evals=10)
    return DE(n_particles, n_nodes, start_node, end_node, κ, θcluster,
        θbest, θswap, γmin, γmax, recombination!, max_evals)
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
    return state
end

function pfind_paths!(method::DE, state, rngs)
    @threads for particle in method.particles 
        rng = rngs[threadid()]
        find_path!(method, state, particle, rng)
    end
end

function find_paths!(method::DE, state, args...)
    for particle in state.particles
        find_path!(method, state, particle)
    end
end

find_path!(method::DE, state::DEState, particle) = find_path!(method, state, particle, Random.GLOBAL_RNG)

function find_path!(method::DE, state::DEState, particle, rng)
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

function cross_over!(de, Pt, group)
    group_diff = setdiff(group, [Pt])
    # sample particles for θm and θn
    Pm,Pn = sample(group_diff, 2, replace=false)
    # sample gamma weights
    γ = 2.38
    # compute proposal value
    Θp = Pt + γ * (Pm - Pn)
    # reset each parameter to previous value with probability (1-κ)
    recombination!(de, Pt, Θp)
    return Θp
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

function k_means_cluster(method, state)

end

function adapt_gamma(method, state)
    @unpack γmin, γmax = method
    @unpack max_evals, n_evals = state
    return (γmin - γmax) / max_evals * n_evals + γmax
end

function rank_order!(state, particle)
    cpath = @view particle.Θ[2:end-1]
    path = @view particle.path[2:end-1]
    idx = sortperm(cpath)
    path .= state.mid_nodes[idx]
    # particle.path[1] = start_node
    # particle.path[end] = end_node
end

function best_match_rank(method, state, particle)
    @unpack particles,n_nodes,θbest,θswap = method
    path = particle.path
    rank_order!(particle)
    best_particle,_ = findmin(x->x.fitness[1], particles)
    best_path = best_particle.path
    if rand() ≤ θbest
        for n in 1:n_nodes
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
        z[i] = x.Θ[i] .*′ y.Θ[i]
    end
    return Particle(Θ=z)
end

*(x::Float64, y::Particle) = *(y, x)

function *(x::Particle, y::Float64)
    N = length(x.Θ)
    z = similar(x.Θ)
    for i in 1:N
        z[i] = x.Θ[i] .*′ y
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
