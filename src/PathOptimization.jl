module PathOptimization
    using Parameters, StatsFuns, StatsBase, Statistics, Random, ProgressMeter
    using Fitness
    import Base.Threads: @threads, nthreads, threadid
    import Base: findmin
    export AntColony, Ant, ColonyState, optimize, find_path!, select_best_ants
    export initialize, set_pheremones!, compute_probabilities!, set_pheremones!
    export pfind_paths!, find_path!, get_best_ants, compute_probabilities!, store_solutions!
    export RandomSearch, NearestNeighbor, cross_over, cross_over_best, cross_over_trig
    export cross_over_ensemble
    export DE, DEState, Particle, rank_order, exponential!, binomial!

    include("api.jl")
    include("AntColony.jl")
    include("RandomSearch.jl")
    include("NearestNeighbor.jl")
    include("DifferentialEvolution.jl")
    include("AbstractMethods.jl")
end
