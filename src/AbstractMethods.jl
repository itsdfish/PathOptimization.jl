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
                if any(x->x < 0.0 , δ)
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