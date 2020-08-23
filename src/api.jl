"""
Subtypes of PathFinder are algorithms for path optimization.
Each path finding algorithm must impliment the following functions:
initialize(method::M, cost)
find_paths!(method::M, state::S, args...)
pfind_paths!(method::M, state::S, rngs)
pfind_path
update!(method::M, state::S)
where M <: PathFinder and S <: State
"""
abstract type PathFinder end
"""
Subtypes of State store state information for path optimization aogorithms.
"""
abstract type State end

"""
`optimize!` optimizes the path using any algorithm that is a subtype of PathFinder. 
* `method`: a path optimization object that is a subtype of op
* `cost`: a matrix or array of matrices representing the cost of traveling from node to node
* `iterations`: the number of repetitions of the algorithm 
* `parallel`: (true) use multi-threading if applicable
* `progress`: show (false) progress bar

Example: 
```@example
n_obj = 2
n_nodes = 20
cost = [rand(Uniform(0, 10), n_nodes, n_nodes) for _ in 1:n_obj]

iterations = 1000
method = AntColony(n_ants=100, n_nodes=n_nodes, β=4.0, ρ=.10,
    τmin=0.0, τmax=10.0)
options = (parallel = true, progress = false)
result = optimize!(method, cost, iterations; options...)
```
"""
function optimize(method::PathFinder, cost, iterations; parallel=true,
    progress=false)
    meter = Progress(iterations)
    state = initialize(method, cost)
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