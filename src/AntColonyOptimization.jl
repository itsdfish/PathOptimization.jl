module AntColonyOptimization
    using Parameters, StatsFuns, StatsBase, Statistics, Random, ProgressMeter
    using Fitness
    import Base.Threads: @threads, nthreads, threadid
    export AntColony, Ant, ColonyState, optimize!, find_path!, select_best_ants
    export initialize, set_pheremones!, compute_probabilities!, set_pheremones!
    export pfind_paths!, find_path!,get_best_ants, compute_probabilities!, store_solutions!

    include("structs.jl")
    include("main.jl")
end
