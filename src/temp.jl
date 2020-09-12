
using Optim, Distributions, CSV, DataFrames

logistic(β0, β1, w) = θ = 1/(1 + exp(-(β0 + β1*w)))

function sim(β0, β1, w, n)
    θ = logistic(β0, β1, w)
    return rand(Bernoulli(θ), n)
end
n = 1390
β0 = 1.88
β1 = -.039
data = sim(β0, β1, 1.0, n)
push!(data, sim(β0, β1, 3.0, n)...)
w = repeat([1,4], inner=n)
g = rand(2*n) .< .15

csv_data = CSV.read("/home/dfish/Projects/ACTR_HMT/performanceData.csv") |> DataFrame

function parse_array(a, type=Float64)
    a = split(split(a, "[")[2], "]")[1]
    a = split(a, ",")
    return parse.(type, a)
end

w = parse_array(csv_data.workloadPVT[1])
push!(w, parse_array(csv_data.workloadVisualSearch[1])...)
push!(w, parse_array(csv_data.workloadAuditorySearch[1])...)
push!(w, parse_array(csv_data.workloadDecisionTask[1])...)

data = parse_array(csv_data.pvtAccuracy[1])
push!(data, parse_array(csv_data.visualSearchAccuracy[1])...)
push!(data, parse_array(csv_data.auditorySearchAccuracy[1])...)
push!(data, parse_array(csv_data.decisionAccuracy[1])...)

g = fill(false, length(parse_array(csv_data.pvtAccuracy[1])))
push!(g, parse_array(csv_data.guessIndicatorVisualSearch[1], Bool)...)
push!(g, parse_array(csv_data.guessIndicatorAuditorySearch[1], Bool)...)
push!(g, fill(false, length(parse_array(csv_data.decisionAccuracy[1])))...)

function loglike(parms, data, w, g)
    LL = 0.0
    for i in 1:length(data)
        if g[i] 
            LL += log(.5)
        else
            θ = logistic(parms[1], parms[2], w[i])
            LL += logpdf(Bernoulli(θ), data[i])
        end
    end
    return -LL
end

x0 = [0.0,0.0]

results = optimize(x->loglike(x, data, w, g), x0, NelderMead())
Optim.minimizer(results)
#Optim.minimum(results)