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
    fitness::Float64
    n_same::Int
end

Base.broadcastable(x::Particle) = Ref(x)

function Particle(;path=Int[],cpath=Float64[], fitness=Inf)
    Particle(path, cpath, fitness)
end

struct DifferentialEvolution{<:F} <: PathFinder
    n_particles::Int
    n_nodes::Int
    start_node::Int
    end_node::Int
    κ::Float64
    θcluster::Float64
    γmin::Float64
    γmax::Float64
    recombination!::F
end

function DifferentialEvolution(;n_particles=50, n_nodes, start_node=1, end_node=n_nodes,
    κ=.50, θcluster=.10, γmin=.6, γmax=.8, recombination!=exponential!)
    return DifferentialEvolution(n_particles, n_nodes, start_node, end_node, κ, θcluster,
    γmin, γmax)
end

mutable struct DEState{T} <: State
    n_evals::Int
    max_evals::Int
    n_obj::Int
    cost::Array{Array{Float64,2},1}
    frontier::T
    fitness::Array{Float64,1}
    particles::Array{Particle,1}
end

function initialize(method::DifferentialEvolution, cost::Array{Float64,2})
    return initialize(method, [cost])
end

function initialize(method::DifferentialEvolution, cost)
    n_obj = length(cost)
    scheme = Scheme{n_obj}(0.1, is_minimizing=true)
    a = EpsBoxArchive(scheme)
    fitness = fill(0.0, n_obj)
    path = fill(0, method.n_nodes)
    state = DEState(n_obj, cost, a, fitness, path)
    return state
end

function pfind_paths!(method::DifferentialEvolution, state, rngs)
    find_path!(method, state)
end

function find_paths!(method::DifferentialEvolution, state, args...)
    for particle in state.particles
        find_path!(method, state, particle)
    end
end

find_path!(method::DifferentialEvolution, state::DEState, particle) = find_path!(method, state, Random.GLOBAL_RNG)

function find_path!(method::DifferentialEvolution, state::DEState, rng)
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

function update!(method::DifferentialEvolution, state)
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

"""
* `method`: differential evolution  object
* `path`: current path
* `proposal`: proposal path
"""
function exponential!(method, path, proposal)
    N = length(path)
    κ = method.κ
    i,j = 1,1
    while (rand() ≤ κ) && (i ≤ N)
        proposal[j] = path[j] 
        i += 1
        j = mod(j + 1, N)
    end
    return nothing
end

function binomial!(method, path, proposal)
    N = length(path)
    κ = method.κ
    for i in 1:N
        proposal[i] = rand() <= κ ? path[i] : proposal[i]
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

function rank_order!(particle)
    sortperm!(particle.path, particle.cpath, rev=true)
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
