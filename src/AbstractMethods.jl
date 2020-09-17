function store_solutions!(method::PathFinder, state)
    add_candidate!(state.frontier, Tuple(state.fitness), state.path, 2)
    return nothing
end

"""
Incrementally optimize solution by changing local neighborhoods of size two_opt
* `state`: algorithm State object
"""
function two_opt(state::State)
    state.path = two_opt(state.path, state.cost)
    state.fitness = compute_path_cost(state)
end

function two_opt(path, cost, max_evals=1000)
    n = length(path)
    temp_path = copy(path)
    searching = true
    n_evals = 0
    while searching && (n_evals ≤ max_evals)
        searching = false
        for r in 2:(n-1)
            for c in (r+1):(n-1)
                n_evals += 1
                δ = map(x->relative_cost(temp_path, x, r, c), cost)
                if all(x->x < 0.0 , δ)
                    searching = true
                    reverse!(temp_path, r, c)
                    break 
                end
            end
        end
    end
    return temp_path
end

function relative_cost(path, cost, r, c)
    δ = 0.0
    δ -= cost[path[r-1],path[r]]
    δ -= cost[path[c],path[c+1]]
    δ += cost[path[r-1],path[c]]
    δ += cost[path[r],path[c+1]]
    for i in r:(c-1)
        δ -= cost[path[i],path[i+1]]
        δ += cost[path[c+r-i],path[c+r-i-1]]
    end
    return δ
end

function compute_path_cost(state::State)
    return compute_path_cost(state.path, state.cost)
end

function compute_path_cost(path, cost)
    return map(x->compute_path_cost(path, x), cost)
end

function compute_path_cost(path, cost::Array{Float64,2})
    c = 0.0
    for i in 1:length(path)-1
        c += cost[path[i],path[i+1]]
    end
    return c
end

function findmin(fun::Function, X)
    min_val = fun(X[1])
    min_idx = 1
    for (i,x) in enumerate(X)
        val = fun(x)
        if val < min_val 
            min_val = val
            min_idx = i
        end
    end
    return X[min_idx],min_idx
end

"""
* `method`: differential evolution  object
* `path`: current path
* `proposal`: proposal path
"""
function exponential!(method, path, proposal)
    N = length(path)
    κ = method.κ
    i,j = 1,0
    while (rand() ≤ κ) && (i ≤ N)
        i += 1
        j = 1 + mod(j, N)
        proposal[j] = path[j] 
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

function exceed_max_no_change!(method::PathFinder, state, n=1)
    frontier = state.frontier
    streak = noprogress_streak(state.frontier, since_restart=true)
    if streak > method.max_no_change * n
        notify!(frontier, :restart)
        return true
    end
    return false
end

function nearest_neighbor!(method::PathFinder, state, fitness, path)
   return nearest_neighbor!(method::PathFinder, state, fitness, path, Random.GLOBAL_RNG)
end

function nearest_neighbor!(method::PathFinder, state, fitness, path, rng)
    @unpack n_obj,cost = state
    @unpack n_nodes,start_node,end_node = method
    path[1],path[end] = start_node,end_node
    not_visited = [1:n_nodes;]
    deleteat!(not_visited, [start_node,end_node])
    n0 = start_node
    for n in 2:(n_nodes - 1)
        obj_idx = rand(1:n_obj)
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
