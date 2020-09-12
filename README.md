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

The package PathOptimizaiton.jl has a lightweight API, requring only two objects and four methods.

## Objects

Each algorithm is defined by a subtype of PathFinder, which contains parameters of the algorithm, and a subytype of State, which contains state information such as Pareto frontier and cost matrices. 

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