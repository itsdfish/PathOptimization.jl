function optimize!(method::AntColony, cost_matrix, iterations; trace=false, parallel=true,
    progress=false)
    meter = Progress(iterations)
    state = initialize(cost_matrix, method)
    ants = [Ant(method, state.n_obj) for _ in 1:method.n_ants]
    seeds = rand(UInt, nthreads())
    rngs = MersenneTwister.(seeds)
    _find_paths! = parallel ? pfind_paths! : find_paths!
    for i in 1:iterations
        _find_paths!(ants, method, state, rngs)
        store_solutions!(method, state, ants)
        best_ants = get_best_ants(ants, state)
        set_pheremones!(method, state, best_ants)
        compute_probabilities!(method, state)
        progress ? next!(meter) : nothing
    end
    return state
end

function initialize(cost::Array{Float64,2}, method)
    return initialize([cost], method)
end

function initialize(costs, method)
    n_obj = length(costs) 
    τ = map(x -> zero(x) .+ 1.0, costs)
    η = map(x -> median(x, dims=2) ./ x, costs)
    θ = zero.(costs)
    scheme = Scheme{n_obj}(0.1, is_minimizing=true)
    a = EpsBoxArchive(scheme)
    state = ColonyState(n_obj, τ, η, costs, θ, a)
    compute_probabilities!(method, state)
    return state
end

function pfind_paths!(ants, method, state, rngs)
    @threads for ant in ants 
        rng = rngs[threadid()]
        find_path!(ant, method, state, rng)
    end
end

function find_paths!(ants, method, state, args...)
    for ant in ants
        find_path!(ant, method, state)
    end
end

find_path!(ant, method, state) = find_path!(ant, method, state, Random.GLOBAL_RNG)

function find_path!(ant, method, state, rng)
    @unpack start_node, end_node, n_nodes = method
    @unpack path,fitness = ant
    @unpack θ,cost,n_obj = state
    fitness .= 0.0
    path[1],path[end] = start_node,end_node
    visited = fill(false, n_nodes)
    visited[[1,end]] .= true
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
    @unpack ρ = method
    for obj in 1:n_obj
        current_fit = best_ants.current[obj].fitness[obj]
        best_fit = best_ants.all[obj].fitness[obj]
        path = best_ants.current[obj].path
        set_pheremones!(τ[obj], ρ, current_fit, best_fit, path)
    end
    return nothing
end

function set_pheremones!(τ, ρ, current_fit, best_fit, path)
    τ .*= (1 - ρ)
    τΔ = 1 / (1 + current_fit - best_fit)
    for i in 1:(length(path) - 1)
        n0,n1 = path[i],path[i + 1]
        τ[n0,n1] += τΔ
    end
    return nothing
end

function get_best_ants(ants, state)
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

function store_solutions!(method, state, ants)
    T = NTuple{state.n_obj,Float64}
    for ant in ants
        fitness::T = Tuple(get_fitness(ant)) 
        add_candidate!(state.frontier, fitness, ant.path, 2)
    end
    return nothing
end

get_fitness(ant) = ant.fitness