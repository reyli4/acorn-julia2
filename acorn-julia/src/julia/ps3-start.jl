

using JuMP
using HiGHS
using CSV
using DataFrames
using Statistics
using Printf
import MathOptInterface as MOI

# --------------------------
# Scalar parameters 
# --------------------------
const POWER_VIOL_PENALTY = 10_000.0  

# --------------------------
# Load data
# --------------------------
const TS_FILE   = "ps3-data.csv"
const GEN_FILE  = "ps3-generators.csv"
const STOR_FILE = "ps3-storage.csv"

const FALLBACK_DIR = "/home/fs01/jl2966"

function resolve_path(fname::AbstractString)::String
    if isfile(fname)
        return fname
    end
    alt = joinpath(FALLBACK_DIR, fname)
    @assert isfile(alt) "Missing required input file: $(fname)"
    return alt
end

ts_df   = CSV.read(resolve_path(TS_FILE), DataFrame)
gen_df  = CSV.read(resolve_path(GEN_FILE), DataFrame)
stor_df = CSV.read(resolve_path(STOR_FILE), DataFrame)

@assert :demand in propertynames(ts_df) "ps3-data.csv must contain a 'demand' column."
@assert :generator in propertynames(gen_df) "ps3-generators.csv must contain a 'generator' column."
@assert :storage in propertynames(stor_df) "ps3-storage.csv must contain a 'storage' column."

T = nrow(ts_df)
TIME = 1:T

demand = Vector{Float64}(ts_df[!, :demand])

# Sets
gen_names = String.(gen_df[!, :generator])
stor_names = String.(stor_df[!, :storage])

const G = Symbol.(gen_names)
const S = Symbol.(stor_names)

# Helper
function find_exact_column(df::DataFrame, name::AbstractString)::Symbol
    target = lowercase(String(name))
    cols = Symbol[]
    for c in propertynames(df)
        if lowercase(String(c)) == target
            push!(cols, c)
        end
    end
    @assert length(cols) == 1 "ps3-data.csv must contain exactly one column named '$(name)' (case-insensitive exact match). Found $(length(cols))."
    return cols[1]
end

# Indexed parameters
cap_cost = Dict{Symbol, Float64}()
op_cost  = Dict{Symbol, Float64}()

for r in eachrow(gen_df)
    g = Symbol(String(r.generator))
    cap_cost[g] = Float64(r.capacity_cost_gen)
    op_cost[g]  = Float64(r.operating_cost)
end

stor_cap_cost      = Dict{Symbol, Float64}()
stor_duration_hr   = Dict{Symbol, Float64}()
stor_discharge_eff = Dict{Symbol, Float64}()
stor_charge_eff    = Dict{Symbol, Float64}()

for r in eachrow(stor_df)
    s = Symbol(String(r.storage))
    stor_cap_cost[s]      = Float64(r.capacity_cost_stor)
    stor_duration_hr[s]   = Float64(r.storage_duration_hr)
    stor_discharge_eff[s] = Float64(r.discharge_eff)
    stor_charge_eff[s]    = Float64(r.charge_eff)
end

# Hourly availability for each generator
availability = Dict{Symbol, Vector{Float64}}()
for gstr in gen_names
    g = Symbol(gstr)
    maybe_col = filter(c -> lowercase(String(c)) == lowercase(gstr), propertynames(ts_df))
    if length(maybe_col) == 1
        availability[g] = Vector{Float64}(ts_df[!, maybe_col[1]])
    else
        availability[g] = ones(T)
    end
end

# --------------------------
# Build model
# --------------------------
model = Model(HiGHS.Optimizer)

# Investment variables
@variable(model, x_gen[g in G] >= 0)   
@variable(model, x_stor[s in S] >= 0)  

# PS4(a): existing nuclear can only be retained/retired from 600 MW; no additional build.
if :NUC_EXIST in G
    @constraint(model, x_gen[:NUC_EXIST] <= 600.0)
end

# Operational variables
@variable(model, y[g in G, t in TIME] >= 0)          
@variable(model, p_ch[s in S, t in TIME] >= 0)       
@variable(model, p_dis[s in S, t in TIME] >= 0)    
@variable(model, e[s in S, t in TIME] >= 0)       
@variable(model, load_shed[t in TIME] >= 0)        

# Objective: investment + operations + load-shed penalty
@objective(model, Min,
    sum(cap_cost[g] * x_gen[g] for g in G) +
    sum(stor_cap_cost[s] * x_stor[s] for s in S) +
    sum(op_cost[g] * y[g, t] for g in G, t in TIME) +
    sum(POWER_VIOL_PENALTY * load_shed[t] for t in TIME)
)

# Generator availability constraints
@constraint(model, [g in G, t in TIME], y[g, t] <= availability[g][t] * x_gen[g])

# Storage power limits
@constraint(model, [s in S, t in TIME], p_ch[s, t] <= x_stor[s])
@constraint(model, [s in S, t in TIME], p_dis[s, t] <= x_stor[s])

# Storage energy capacity 
@constraint(model, [s in S, t in TIME], e[s, t] <= stor_duration_hr[s] * x_stor[s])

# Storage state of charge 
@constraint(model, [s in S, t in TIME],
    e[s, t] == e[s, t == 1 ? T : t - 1] +
        stor_charge_eff[s] * p_ch[s, t] -
        (1.0 / stor_discharge_eff[s]) * p_dis[s, t]
)

# Power balance with load-shedding penalty
@constraint(model, [t in TIME],
    sum(y[g, t] for g in G) +
    sum(p_dis[s, t] for s in S) -
    sum(p_ch[s, t] for s in S) +
    load_shed[t] == demand[t]
)

optimize!(model)

# --------------------------
# Reporting
# --------------------------
const OUT_FILE = joinpath(@__DIR__, "ps4-solution.txt")

open(OUT_FILE, "w") do io
    println(io, "PS4 Capacity Expansion with Storage (Wind/Solar + 4-hr Battery)")
    println(io, "Status: ", termination_status(model))
    println(io, "Objective value: ", objective_value(model))
    println(io)
    println(io, "Installed generation capacities (MW):")
    for g in G
        @printf(io, "  %s: %.4f\n", String(g), value(x_gen[g]))
    end
    println(io)
    println(io, "Installed storage power capacities (MW):")
    for s in S
        @printf(io, "  %s: %.4f\n", String(s), value(x_stor[s]))
    end
    println(io)
    println(io, "Installed storage energy capacities (MWh):")
    for s in S
        @printf(io, "  %s: %.4f\n", String(s), value(x_stor[s]) * stor_duration_hr[s])
    end
    println(io)
    total_shed = sum(value(load_shed[t]) for t in TIME)
    @printf(io, "Total load shed (MWh): %.4f\n", total_shed)
end

println("Wrote solution to ", OUT_FILE)
