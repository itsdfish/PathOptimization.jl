abstract type PathFinder end

abstract type State end

function optimize!(method::PathFinder, cost_matrix, iterations; trace=false, parallel=true,
    progress=false)
    meter = Progress(iterations)
    state = initialize(method, cost_matrix)
    seeds = rand(UInt, nthreads())
    rngs = MersenneTwister.(seeds)
    _find_paths! = parallel ? pfind_paths! : find_paths!
    for i in 1:iterations
        _find_paths!(method, state, rngs)
        update!(method, state)
        progress ? next!(meter) : nothing
    end
    return state
end