module AntColonyOptimization
    using Parameters, StatsFuns, StatsBase, Statistics, Random, ProgressMeter
    import Base.Threads: @threads, nthreads, threadid
    export AntColony, Ant, ColonyState, optimize!, find_path!, select_best_ants
    export initialize, set_pheremones!, compute_probabilities!, set_pheremones!
    include("structs.jl")
    include("main.jl")
end
