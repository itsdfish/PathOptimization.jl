struct AntColony
    n_ants::Int
    α::Float64
    β::Float64
    ρ::Float64
    depost_factor::Float64
    n_best::Int
    n_nodes::Int
    start_node::Int
    end_node::Int
    retain_solutions::Bool
end

function AntColony(;n_ants=20, α=1.0, β=1.0, ρ=0.1, deposit_factor=1.0, n_best=1, n_nodes=10,
    start_node=1, end_node=n_nodes, retain_solutions=false)
    return AntColony(n_ants, α, β, ρ, deposit_factor, n_best, n_nodes, start_node, end_node, 
    retain_solutions)
end

mutable struct ColonyState
    τ::Array{Float64,2}
    cost::Array{Float64,2}
    θ::Array{Float64,2}
    best_fitness::Float64
    best_path::Array{Float64,1}
    all_solutions::Dict{Array{Int64,1},Float64}
end

mutable struct Ant
    fitness::Float64
    path::Array{Int,1}
end

Ant(n) = Ant(0.0, fill(0, n))