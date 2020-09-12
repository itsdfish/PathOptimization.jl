# PathOptimization

A Julia package for path optimization.

# Example
In the example below, ant colony optimization is used to find the Pareto frontier of a path with two objective cost functions. 
```@julia
using Revise, PathOptimization, Distributions, Random, Plots
using Fitness
Random.seed!(5214)
n_obj = 2
n_nodes = 25
cost = [rand(Uniform(0, 50), n_nodes, n_nodes) for _ in 1:n_obj]
iterations = 2000
method = AntColony(n_ants=100, n_nodes=n_nodes, β=4.0)
options = (parallel = true, progress = false)
result = optimize(method, cost, iterations; options...)

frontier = get_best_cost(result.frontier)
pyplot()
scatter(frontier, grid=false, leg=false, ylims=(0,800), xlims=(0,800),
    size=(600,400), markersize=9, markerstrokewidth=2, color=:purple, xlabel="Cost 1",
     ylabel="Cost 2", xaxis=font(12), yaxis=font(12))
```

<img src="Examples/example.png" alt="" width="600" height="400">

# API

The package PathOptimizaiton.jl provides a lightweight API for adding new algorithms, which only requires two objects and four methods. Additional methods can be called from the four required methods. A minimal working example can be found in src/RandomSearch.jl and Examples/Random_Search_Example.jl.

## Objects

### PathFinder

Parameters of each algorithm are defined in a subtype of PathFinder. For example:

```@julia
struct RandomSearch <: PathFinder
    n_nodes::Int
    start_node::Int
    end_node::Int
end
```

### State

The state of each algorithm is tracked in a subtype of State. Here is a simple example:

```@julia
mutable struct RandomState{T} <: State
    n_obj::Int
    cost::Array{Array{Float64,2},1}
    frontier::T
    fitness::Array{Float64,1}
    path::Array{Int,1}
end
```

## Methods


### initialize

The initialize function sets up the Pareto fronier, and other algorithmic specific configurations, and returns the state object.

```@julia
initialize(method::M, cost)
```

### find_paths!

A single threaded method that finds a path each iteration.

```@julia
d_paths!(method::M, state, args...)
```

### pfind_paths!

An optional multithreaded method that finds a path on each iteration.

```@julia
pfind_paths!(method::RandomSearch, state, rngs)
```

### update!

A method for updating the algorithm and state

```@julia
update!(method::M, state)
```