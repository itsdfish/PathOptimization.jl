struct AntColony
    n_ants::Int
    α::Float64
    β::Float64
    ρ::Float64
    n_nodes::Int
    start_node::Int
    end_node::Int
end

function AntColony(;n_ants=20, α=1.0, β=1.0, ρ=0.1, n_nodes=10,
    start_node=1, end_node=n_nodes, retain_solutions=false)
    return AntColony(n_ants, α, β, ρ, n_nodes, start_node, end_node)
end
mutable struct ColonyState{T}
    n_obj::Int
    τ::Array{Array{Float64,2},1}
    η::Array{Array{Float64,2},1}
    cost::Array{Array{Float64,2},1}
    θ::Array{Array{Float64,2},1}
    frontier::T
end

mutable struct Ant
    fitness::Array{Float64,1}
    path::Array{Int,1}
end

Ant(method::AntColony, n_obj) = Ant(fill(0.0, n_obj), fill(0, method.n_nodes))