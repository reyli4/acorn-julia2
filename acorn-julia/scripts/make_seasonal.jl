#!/usr/bin/env julia
using CSV, DataFrames, Random, Statistics

# --- inputs you’ll edit per scenario ---
const PROJECT_DIR = "/home/fs01/jl2966/acorn-julia2/acorn-julia"
const RUN_NAME    = "low_RE_mod_elec_iter0"
const RUN_DIR     = "$(PROJECT_DIR)/runs/$(RUN_NAME)"
const INPUTS      = "$(RUN_DIR)/inputs"

const TOTAL_CHARGE_MW = 6000.0
const TOTAL_ENERGY_MWH = 13360

# Seasonal behavior defaults (set only duration; keep other defaults reasonable)
const DURATION_HOURS = 24.0 * 90.0  # 3 months
const RATE_CH  = 1.0 / DURATION_HOURS
const RATE_DIS = 1.0 / DURATION_HOURS
const EFF      = 0.75
const INIT_SOC = 0.30
const END_SOC  = 0.0
const DISC_COST= 0.0

# ---------- helper: choose target bus list ----------
function load_busprop()
    p1 = "$(PROJECT_DIR)/data/grid/bus_prop_boyuan.csv"
    p2 = "$(PROJECT_DIR)/data/grid/bus_prop_liu_etal_2024.csv"
    path = isfile(p1) ? p1 : (isfile(p2) ? p2 : error("No bus_prop file found"))
    df = CSV.read(path, DataFrame)
    if !("BUS_I" in names(df)); error("bus_prop missing BUS_I"); end
    return df
end

function select_zone_buses(df; zone="A")
    col = "BUS_ZONE" in names(df) ? :BUS_ZONE :
          ("ZONE" in names(df) ? :ZONE : error("No zone column"))
    df[df[!, col] .== zone, :BUS_I]
end

function select_top_load_buses(df; zone_filter=nothing, k=5)
    d = copy(df)
    if zone_filter !== nothing
        col = "BUS_ZONE" in names(d) ? :BUS_ZONE : (:ZONE in names(d) ? :ZONE : error("No zone column"))
        # Build a boolean mask: true if this bus's zone is in zone_filter
        mask = [z in zone_filter for z in d[!, col]]
        d = d[mask, :]
    end
    if !("PD" in names(d))
        error("bus_prop missing PD (load)")
    end
    perm = sortperm(d.PD, rev=true)
    return d[perm[1:min(k, nrow(d))], :BUS_I]
end


"""
    select_gen_buses(; zone_filter=nothing)

Select buses with renewable generators (wind / PV) based on gen_prop_boyuan.csv.
Optionally filter by GEN_ZONE using `zone_filter = ["A","B",...]`.
"""
function select_gen_buses(; zone_filter=nothing)
    # Try a couple of plausible locations for the file; adjust if needed.
    p1 = "$(PROJECT_DIR)/data/grid/gen_prop_boyuan.csv"
    p2 = "$(PROJECT_DIR)/data/gen_prop_boyuan.csv"
    gen_path = isfile(p1) ? p1 : (isfile(p2) ? p2 : error("gen_prop_boyuan.csv not found"))

    gen = CSV.read(gen_path, DataFrame)

    if !("UNIT_TYPE" in names(gen)) || !("GEN_BUS" in names(gen))
        error("gen_prop_boyuan.csv missing UNIT_TYPE or GEN_BUS columns")
    end

    # Treat WT as wind and PV as solar; tweak if your codes differ
    mask = (gen.UNIT_TYPE .== "WT") .| (gen.UNIT_TYPE .== "PV")

    # Optional zone filter (if GEN_ZONE exists)
    if zone_filter !== nothing
        if "GEN_ZONE" in names(gen)
            mask .&= gen.GEN_ZONE .∈ zone_filter
        else
            @warn "GEN_ZONE column not found in gen_prop_boyuan.csv; ignoring zone_filter"
        end
    end

    buses = unique(gen[mask, :GEN_BUS])
    if isempty(buses)
        error("No renewable buses found in gen_prop_boyuan.csv (check UNIT_TYPE/GEN_ZONE/paths)")
    end

    return Vector{Int}(buses)
end

# ---------- strategy builders (pick one per run) ----------
function strat_zoneA_distributed_equal()
    dfb = load_busprop()
    buses = Vector{Int}(select_zone_buses(dfb; zone="A"))
    return buses, fill(1.0/length(buses), length(buses))
end

function strat_zoneA_concentrated(bus::Int)
    return [bus], [1.0]
end

function strat_near_renewables()
    # If you want to restrict to internal zones, you can pass zone_filter here
    # e.g. zone_filter=["A","B","C","D","E","F","G","H","I","J","K"]
    buses = select_gen_buses(; zone_filter=nothing)
    if isempty(buses); error("No renewable buses found"); end
    w = fill(1.0/length(buses), length(buses))
    return Vector{Int}(buses), w
end

function strat_topload_NYC(k::Int=5)
    dfb = load_busprop()
    buses = Vector{Int}(select_top_load_buses(dfb; zone_filter=["J","K"], k=k))
    w = fill(1.0/length(buses), length(buses))
    return buses, w
end

# ---------- builder for the seasonal CSV ----------
function make_seasonal_csv(outpath::String; template::DataFrame)
    n = nrow(template)
    # Keep placement and power capacity identical to the base template,
    # but scale energy to enforce the target duration.
    storage_capacity_mwh = template.charge_capacity_MW .* DURATION_HOURS
    df = DataFrame(
        bus_id = template.bus_id,
        charge_capacity_MW   = template.charge_capacity_MW,
        storage_capacity_mwh = storage_capacity_mwh,
        is_seasonal = ones(Int, n),
        max_charge_rate_frac    = fill(RATE_CH,  n),
        max_discharge_rate_frac = fill(RATE_DIS, n),
        eff = fill(EFF, n),
        init_soc = fill(INIT_SOC, n),
        end_soc_min = fill(END_SOC, n),
        discharge_cost_per_mwh = fill(DISC_COST, n),
    )
    CSV.write(outpath, df)
    println("Wrote seasonal file: $outpath")
end

# ---------- pick ONE strategy per invocation ----------
# Example 1: Zone A, distributed equally
#b, w = strat_zoneA_distributed_equal()

# Example 2: Zone A, concentrated at bus 55
# b, w = strat_zoneA_concentrated(55)

# Example 3: Near renewables (WT/PV buses from gen_prop_boyuan.csv)
#b, w = strat_near_renewables()

mkpath(INPUTS)
out = "$(INPUTS)/seasonal_zoneA_test_stage1_3month.csv"   # change per run to keep files distinct
base_template = CSV.read("$(INPUTS)/storage_assignment.csv", DataFrame)
make_seasonal_csv(out; template=base_template)
