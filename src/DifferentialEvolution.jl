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
    cross_over_id::Int
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

function Particle(;Θ)
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
* `n_cycles`: number of calibration cycles for ensemble crossover method
"""
struct DE{F1,F2<:Function} <: PathFinder
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
    recombination!::F1
    max_evals::Int
    max_no_change::Int
    cross_over_fun::F2
    n_cycles::Int
end

function DE(;n_particles=50, n_nodes, start_node=1, end_node=n_nodes,κ=.50, Θneighbor=.10, 
    θbest=.9, θswap=.8, γmin=.6, γmax=.8, recombination! = exponential!, max_evals=10, max_no_change=100*n_particles,
    cross_over_fun=cross_over_ensemble, n_cycles=10)
    return DE(n_particles, n_nodes, start_node, end_node, κ, Θneighbor, θbest, θswap, γmin, γmax,
        recombination!, max_evals, max_no_change, cross_over_fun, n_cycles)
end

"""
* `mid_nodes`: all nodes excluding end points, which can be reordered
"""
mutable struct DEState{T1,T2} <: State
    n_evals::Int
    iter::Int
    n_obj::Int
    cost::Array{Array{Float64,2},1}
    frontier::T1
    particles::Array{Particle,1}
    mid_nodes::Array{Int,1}
    cross_over_funs::T2
    best_cross_over::Array{Int,1}
    exploring::Bool
    best_fun_id::Int
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
    set_group_id!(particles)
    cross_over_funs = (cross_over,cross_over_best,cross_over_trig)
    best_cross_over = fill(0,3)
    state = DEState(0, 0, n_obj, cost, a, particles, mid_nodes, cross_over_funs,
        best_cross_over, true, 0)
    nearest_neighbor!(method, state)
    map(x->compute_path_cost!(x, cost), particles)
    return state
end

function set_group_id!(particles)
    n_particles = length(particles)
    group_sizes = map(_->div(n_particles, 3), 1:3)
    remainder = mod(n_particles, 3)
    idx = sample(1:3, remainder, replace = false)
    group_sizes[idx] .+= 1
    ids = mapreduce((i,n)->fill(i, n), vcat, 1:3, group_sizes)
    for (id,p) in zip(ids,particles)
        p.cross_over_id = id
    end
    return nothing
end

function pfind_path!(method::DE, state, rngs)
    state.iter += 1
    switch_ensemble_state!(method, state)
    @threads for particle in method.particles 
        rng = rngs[threadid()]
        find_path!(method, state, particle, rng)
    end
end

function find_path!(method::DE, state, args...)
    state.iter += 1
    switch_ensemble_state!(method, state)
    for particle in state.particles
        find_path!(method, state, particle)
    end
end

find_path!(method::DE, state::DEState, particle::Particle) = find_path!(method, state, particle, Random.GLOBAL_RNG)

function find_path!(method::DE, state::DEState, particle::Particle, rng)
    state.n_evals += 1
    set_objective!(state, particle)
    proposal = method.cross_over_fun(method, state, particle, rng)
    exponential!(method, particle.path, proposal.path)
    compute_path_cost!(proposal, state.cost)
    update_particle!(particle, proposal)
    best_match_rank!(method, state, particle)
end

function set_objective!(state, particle)
    particle.obj_idx = sample_objective(state)
    return nothing
end

function sample_objective(state)
    return rand(1:state.n_obj)
end

function update!(method::DE, state)
    @unpack particles = state
    two_opt(method, state)
    eval_ensemble!(method, state)
    fix_stuck!(method, state)
    store_solutions!(method, state)
    reset_same_counter!(state)
end

function store_solutions!(method::DE, state)
    for particle in state.particles
        add_candidate!(state.frontier, Tuple(particle.fitness), particle.path, 2)
    end
    return nothing
end

cross_over(de, state, Pt) = cross_over(de, state, Pt, Random.GLOBAL_RNG)

function cross_over(de, state, Pt, rng)
    @unpack particles = state
    idxs = findall(x -> x != Pt, particles)
    others = @view particles[idxs]
    P₁,P₂,P₃ = sample(rng, others, 3, replace=false)
    γ = adapt_gamma(de, state)
    # compute proposal value
    proposal = P₁ + γ * (P₂ - P₃)
    proposal.obj_idx = Pt.obj_idx
    return proposal
end

function cross_over_best(de, state, particle, rng)
    @unpack particles, = state
    @unpack obj_idx = particle
    Pb,_ = findmin(x->x.fitness[obj_idx], particles) 
    P₁,P₂,P₃,P₄ = sample(rng, particles, 4, replace=false)
    γ = adapt_gamma(de, state)
    # compute proposal value
    proposal = Pb + γ * (P₁ - P₂) + γ * (P₃ - P₄)
    proposal.obj_idx = obj_idx
    return proposal
end

function cross_over_trig(de, state, Pt, rng)
    @unpack particles = state
    @unpack obj_idx = Pt
    idxs = findall(x -> x != Pt, particles)
    others = @view particles[idxs]
    P₁,P₂,P₃ = sample(rng, others, 3, replace=false)
    # compute proposal value
    W = map(x->x.fitness[obj_idx], [P₁,P₂,P₃])
    w₁,w₂,w₃ = W ./sum(W)
    proposal = (P₁ + P₂ + P₃)/3.0 + (w₂-w₁) * (P₁ - P₂) +
        (w₃-w₂) * (P₂ - P₃) + (w₁-w₃) * (P₃ - P₁) 
    proposal.obj_idx = Pt.obj_idx
    return proposal
end

function cross_over_ensemble(method, state, Pt, rng)
    id = state.exploring ? Pt.cross_over_id : state.best_fun_id
    proposal = state.cross_over_funs[id](method, state, Pt, rng)
    return proposal
end

function switch_ensemble_state!(method, state)
    if mod(state.iter, method.n_cycles) == 0
        state.exploring = !state.exploring
        _,idx = findmax(state.best_cross_over)
        state.best_fun_id = idx
        state.best_cross_over .= 0
    end
    return nothing
end

"""
Update particle based on Greedy Rule.
* `current`: current particle
* `proposal`: proposal particle
"""
function update_particle!(current, proposal)
    idx = current.obj_idx
    if all(proposal.fitness .< current.fitness)
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

function adapt_gamma(γmin, γmax, n_evals, max_evals)
    return (γmin - γmax) / max_evals * n_evals + γmax
end

function adapt_gamma(method, state)
    @unpack max_evals, γmin, γmax = method
    @unpack n_evals = state
    adapt_gamma(γmin, γmax, n_evals, max_evals)
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

function fix_stuck!(method::DE, state)
    @unpack cost,particles = state
    if exceed_max_no_change!(method, state)
        for particle in particles
            new_path = two_opt(particle.path, cost)
            particle.path = new_path
            compute_path_cost!(particle, cost)
        end
    end
    return nothing
end

function compute_path_cost!(particle, cost)
    particle.fitness = compute_path_cost(particle.path, cost)
end

function two_opt(method::DE, state)
    @unpack n_obj,particles, cost = state
    for obj in 1:n_obj
        particle,_ = findmin(x->x.fitness[obj], particles)
        new_path = two_opt(particle.path, cost)
        particle.path = new_path
        compute_path_cost!(particle, cost)
    end
    return nothing
end

function reset_same_counter!(state)
    #state.cross_count .= 0


end

function eval_ensemble!(method::DE{T,C}, state) where {T,C <: typeof(cross_over_ensemble)}
    obj = sample_objective(state)
    best_particle,_ = findmin(x->x.fitness[obj], state.particles)
    id = best_particle.cross_over_id
    state.best_cross_over[id] += 1
    return nothing
end

function eval_ensemble!(method, state)
    # blank by default
    return nothing
end

# Type-stable arithmatic operations for Union{Array{T,1},T} types (which return Any otherwise)
import Base: +, -, *, /

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

function /(x::Particle, y::Particle)
    N = length(x.Θ)
    z = similar(x.Θ)
    for i in 1:N
        z[i] = x.Θ[i] ./ y.Θ[i]
    end
    return Particle(Θ=z)
end

/(x::Float64, y::Particle) = /(y, x)

function /(x::Particle, y::Float64)
    N = length(x.Θ)
    z = similar(x.Θ)
    for i in 1:N
        z[i] = x.Θ[i] ./ y
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
