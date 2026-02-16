using CSV
using DataFrames
using Dates
using JuMP
using Gurobi
using Statistics
include("./utils.jl")

function run_acorn(
    run_name,
    climate_scenario,
    sim_year,
    branchprop_name,
    busprop_name,
    if_lim_name,
    save_name;
    exclude_external_zones=true,
    include_new_hvdc=false,
    storage_eff=0.75,                     
    storage_filename::String="storage_assignment.csv",  # NEW
    seasonal_on::Bool=false,  # NEW
    seasonal_filename::String="seasonal_storage_assignment.csv",    # NEW
    seasonal_rate_caps_on::Bool=true,   # toggle extra seasonal constraints
    )


    ############################
    # Read all data
    ############################
    data_directory = "$(project_path)/data"
    run_directory = "$(project_path)/runs/$(run_name)"
    out_path = "$(run_directory)/outputs/$(climate_scenario)/$(save_name)"
    
    if !isdir(out_path)
        mkpath(out_path)
    end

    # Read the load
    load = CSV.read("$(run_directory)/inputs/load_$(climate_scenario).csv", DataFrame)

    # Read generation
    solar_upv = CSV.read("$(run_directory)/inputs/solar_upv_$(climate_scenario).csv", DataFrame)
    wind = CSV.read("$(run_directory)/inputs/wind_$(climate_scenario).csv", DataFrame)
    solar_dpv = CSV.read("$(run_directory)/inputs/solar_dpv_$(climate_scenario).csv", DataFrame)

    # Read hydro
    hydro_scenario = split(climate_scenario, "_")[1]
    large_hydro = CSV.read("$(run_directory)/inputs/large_hydro_$(hydro_scenario).csv", DataFrame)
    small_hydro = CSV.read("$(run_directory)/inputs/small_hydro_$(hydro_scenario).csv", DataFrame)



    # --- STORAGE (baseline + optional seasonal, simple) ------------------

    # always load baseline fleet
    base_storage = CSV.read("$(run_directory)/inputs/$(storage_filename)", DataFrame)
    base_storage.is_seasonal = fill(0, nrow(base_storage))   # <— tag baseline

    if seasonal_on
        seasonal_storage = CSV.read("$(run_directory)/inputs/$(seasonal_filename)", DataFrame)
        seasonal_storage.is_seasonal = fill(1, nrow(seasonal_storage))  # <— tag seasonal
        # If seasonal has extra columns, :union prevents column-mismatch errors
        storage = vcat(base_storage, seasonal_storage; cols = :union)
    else
        storage = base_storage
    end

    # Ensure column names are Symbols for consistent access (e.g., :is_seasonal)
    if eltype(names(storage)) == String
        rename!(storage, Symbol.(names(storage)))
    end

    println("DEBUG seasonal_on = ", seasonal_on)
    println("DEBUG seasonal_filename = ", seasonal_filename)
    println("DEBUG base_storage rows = ", nrow(base_storage))
    if seasonal_on
        println("DEBUG seasonal_storage rows = ", nrow(seasonal_storage))
    end
    println("DEBUG storage total rows = ", nrow(storage))
    println("DEBUG storage columns = ", names(storage))
    println("DEBUG storage column type = ", eltype(names(storage)))
    println("DEBUG seasonal_rate_caps_on = ", seasonal_rate_caps_on)

    if :is_seasonal ∈ names(storage) || "is_seasonal" ∈ names(storage)

        # handle either symbol or string column
        is_seasonal_col = :is_seasonal ∈ names(storage) ? :is_seasonal : "is_seasonal"
        println("DEBUG unique(is_seasonal) = ", unique(skipmissing(storage[!, is_seasonal_col])))
        println("DEBUG counts by tag:")
        println(combine(groupby(storage, is_seasonal_col), nrow => :count))
    else
        println("DEBUG: no :is_seasonal column on storage!")
    end

    # write a small snapshot to the outputs folder you’re reading from in Python
    mkpath(out_path)
    CSV.write("$(out_path)/_storage_debug.csv", storage)





    # --- Build per-asset parameter vectors robustly (no hascol, no missings) ---
    n_s = nrow(storage)

    # Helper to pull/convert a numeric column with per-row defaults
    # Handles both Symbol and String column names
    getvec = function(sym::Symbol, default)
        col = if sym ∈ names(storage)
            sym
        elseif String(sym) ∈ names(storage)
            String(sym)
        else
            nothing
        end
        if col === nothing
            return fill(default, n_s)
        end
        return [ismissing(v) ? default : Float64(v) for v in storage[!, col]]
    end

    # Ensure is_seasonal exists as 0/1 ints (harmless if unused elsewhere)
    if :is_seasonal ∉ names(storage) && "is_seasonal" ∉ names(storage)
        storage.is_seasonal = zeros(Int, n_s)
    else
        is_seasonal_col = :is_seasonal ∈ names(storage) ? :is_seasonal : "is_seasonal"
        storage.is_seasonal = [ismissing(v) ? 0 : Int(v) for v in storage[!, is_seasonal_col]]
    end

    eff_vec         = getvec(:eff,                     storage_eff)
    init_soc_vec    = getvec(:init_soc,                0.30)
    end_soc_min_vec = getvec(:end_soc_min,             0.0)
    # Optional C-rate caps (fractions of energy capacity per hour).
    # Defaults impose no extra rate limit if columns are absent.
    max_charge_rate_frac    = getvec(:max_charge_rate_frac,    Inf)
    max_discharge_rate_frac = getvec(:max_discharge_rate_frac, Inf)





    # Read generators
    genprop_nuclear = CSV.read("$(run_directory)/inputs/genprop_nuclear_matched.csv", DataFrame, stringtype=String)
    genprop_ng = CSV.read("$(run_directory)/inputs/genprop_ng_matched.csv", DataFrame, stringtype=String)
    genprop_hydro = CSV.read("$(run_directory)/inputs/genprop_hydro.csv", DataFrame, stringtype=String)
    genprop = vcat(genprop_nuclear, genprop_ng, genprop_hydro)

    # Read bus data
    busprop = CSV.read("$(data_directory)/grid/bus_prop_$(busprop_name).csv", DataFrame)
    bus_ids = busprop[:, "BUS_I"]

    # Read branch data
    branchprop = CSV.read("$(data_directory)/grid/branch_prop_$(branchprop_name).csv", DataFrame)

    ##### Remove external zones if specified
    if exclude_external_zones
        external_buses = [21, 29, 35, 100, 102, 103, 124, 125, 132, 134, 138]

        # Buses
        busprop = busprop[findall(.!in(external_buses), busprop.BUS_I), :]
        bus_ids = busprop[:, "BUS_I"]

        # Branches
        branchprop = branchprop[findall(.!in(external_buses), branchprop.F_BUS), :]
        branchprop = branchprop[findall(.!in(external_buses), branchprop.T_BUS), :]
    end

    #############################
    # Load adjustments 
    #############################
    # Subtract small solar 
    load = subtract_solar_dpv(load, solar_dpv, sim_year)

    # Subtract small hydro
    load = subtract_small_hydro(load, small_hydro, sim_year)

    if !exclude_external_zones
        # Fill the missing load buses with zero (external ones)
        println("NOTE: External buses all have zero load")
        load = leftjoin(DataFrame(bus_id=bus_ids), load, on=:bus_id)
        load = coalesce.(load, 0.0)
        load = sort(load, [:bus_id])
        load = max.(load, 0)
    end

    # Get final load data
    sim_dates = names(load)[2:end]
    nt = length(sim_dates)
    load_data = Matrix(load[:, sim_dates])
    #new
    # Storage arrays that depend on nt
    storage_bus_ids    = storage[:, "bus_id"]
    storage_charge_cap = repeat(storage[:, "charge_capacity_MW"], 1, nt)
    storage_energy_cap = repeat(storage[:, "storage_capacity_mwh"], 1, nt + 1)


    #############################
    # Add generators
    #############################
    # Add solar and wind generators
    wind_bus_ids = wind[:, "bus_id"]
    genprop = add_wind_generators(genprop, wind_bus_ids)

    solar_upv_bus_ids = solar_upv[:, "bus_id"]
    genprop = add_solar_generators(genprop, solar_upv_bus_ids)

    if !exclude_external_zones
        if include_new_hvdc
            genprop = add_hvdc_generators(genprop, true)
        else
            genprop = add_hvdc_generators(genprop, false)
        end
    end

    # Get generator limits
    g_max = repeat(genprop[:, "PMAX"], 1, nt) # Maximum real power output (MW)
    g_min = repeat(genprop[:, "PMIN"], 1, nt) # Minimum real power output (MW)

    # Update for renewables
    wind_idx = findall(x -> x == "Wind", genprop[:, "UNIT_TYPE"])
    g_max[wind_idx, :] .= wind[:, sim_dates]

    solar_upv_idx = findall(x -> x == "SolarUPV", genprop[:, "UNIT_TYPE"])
    g_max[solar_upv_idx, :] .= solar_upv[:, sim_dates]

    # Get generator ramp rates
    ramp_down = max.(repeat(genprop[:, "RAMP_30"], 1, nt) .* 2, repeat(genprop[:, "PMAX"], 1, nt)) # max of 2*RAMP_30, PMAX
    ramp_up = copy(ramp_down)

    # Generator cost
    gencost = repeat(genprop[:, "COST_1"], 1, nt) # Cost per unit power generated

    #########################
    # Interface limits
    #########################
    # Read interface limits
    if_lims = CSV.read("$(data_directory)/nyiso/interface_limits/if_lim_$(if_lim_name).csv", DataFrame)

    # Remove eternal zones if specified
    if exclude_external_zones
        external_zones = ["NE", "IESO", "PJM"]
        if_lims = if_lims[findall(.!in(external_zones), if_lims.FROM_ZONE), :]
        if_lims = if_lims[findall(.!in(external_zones), if_lims.TO_ZONE), :]
    end

    if_lim_up = repeat(if_lims[:, "IF_MAX"], 1, nt)
    if_lim_down = repeat(if_lims[:, "IF_MIN"], 1, nt)

    # Get IF lim map and update branchprop
    if_lim_map, branchprop = create_interface_map(if_lims, branchprop)

    # Branch limits
    branch_lims = repeat(Float64.(branchprop[:, "RATE_A"]), 1, nt)
    branch_lims[branch_lims.==0] .= 99999.0
    #new
    
    ########################
    # Optimization 
    ########################
    n_gen = size(genprop, 1)
    n_bus = size(busprop, 1)
    n_branch = size(branchprop, 1)

    model = Model(Gurobi.Optimizer)

    # Indices of storage rows (requires storage.is_seasonal already set earlier)
    seasonal_storage_idx = findall(==(1), storage.is_seasonal)
    base_storage_idx = findall(==(0), storage.is_seasonal)

    base_storage_bus_ids = storage_bus_ids[base_storage_idx]
    seasonal_storage_bus_ids = storage_bus_ids[seasonal_storage_idx]

    base_charge_cap = storage_charge_cap[base_storage_idx, 1:nt]
    seasonal_charge_cap = storage_charge_cap[seasonal_storage_idx, 1:nt]
    base_energy_cap = storage_energy_cap[base_storage_idx, 1:nt+1]
    seasonal_energy_cap = storage_energy_cap[seasonal_storage_idx, 1:nt+1]

    base_eff_vec = eff_vec[base_storage_idx]
    seasonal_eff_vec = eff_vec[seasonal_storage_idx]
    base_init_soc_vec = init_soc_vec[base_storage_idx]
    seasonal_init_soc_vec = init_soc_vec[seasonal_storage_idx]
    base_end_soc_min_vec = end_soc_min_vec[base_storage_idx]
    seasonal_end_soc_min_vec = end_soc_min_vec[seasonal_storage_idx]

    seasonal_max_charge_rate_frac = max_charge_rate_frac[seasonal_storage_idx]
    seasonal_max_discharge_rate_frac = max_discharge_rate_frac[seasonal_storage_idx]

    ## Define variables
    @variable(model, pg[1:n_gen, 1:nt])
    @variable(model, flow[1:n_branch, 1:nt])
    @variable(model, bus_angle[1:n_bus, 1:nt])
    @variable(model, charge_base[1:length(base_storage_idx), 1:nt])
    @variable(model, discharge_base[1:length(base_storage_idx), 1:nt])
    @variable(model, storage_state_base[1:length(base_storage_idx), 1:nt+1])

    @variable(model, charge_seasonal[1:length(seasonal_storage_idx), 1:nt])
    @variable(model, discharge_seasonal[1:length(seasonal_storage_idx), 1:nt])
    @variable(model, storage_state_seasonal[1:length(seasonal_storage_idx), 1:nt+1])
    @variable(model, load_shedding[1:n_bus, 1:nt])


    # Per-hour C-rate caps for seasonal assets (MW <= rate_frac * MWh)
    if seasonal_on && seasonal_rate_caps_on && !isempty(seasonal_storage_idx)
        @constraint(model,
            charge_seasonal[:, 1:nt] .<=
                seasonal_max_charge_rate_frac .* seasonal_energy_cap[:, 1:nt]
        )
        @constraint(model,
            discharge_seasonal[:, 1:nt] .<=
                seasonal_max_discharge_rate_frac .* seasonal_energy_cap[:, 1:nt]
        )
    end

    # Seasonal throughput cap (cycles per year)
    # Limits total seasonal discharge to cycles_per_year * total seasonal energy capacity
    # NOTE: currently disabled per request (no cycle limit)
    # seasonal_cycles_per_year = 1.0
    # if seasonal_on && !isempty(seasonal_storage_idx)
    #     @constraint(model,
    #         sum(discharge_seasonal) <= seasonal_cycles_per_year * sum(seasonal_energy_cap[:, 1])
    #     )
    # end

    
    # Removed seasonal SOC floor so Stage 0 runs treat seasonal storage like batteries
    
    ## Constraints 
    # Branch flow limits and power flow equations
    for l in 1:n_branch
        idx_from_bus = findfirst(x -> x == branchprop[l, "F_BUS"], busprop[:, "BUS_I"])
        idx_to_bus = findfirst(x -> x == branchprop[l, "T_BUS"], busprop[:, "BUS_I"])
        # Branch flow limits
        @constraint(model, -branch_lims[l, :] .<= flow[l, :] .<= branch_lims[l, :])
        # DC power flow equations
        @constraint(model, flow[l, :] .== (100 / branchprop[l, "BR_X"]) .*
                                          (bus_angle[idx_from_bus, :] .- bus_angle[idx_to_bus, :]))
    end

   # NEW Node balance and phase angle constraints
    for idx in 1:n_bus
        bus_id = bus_ids[idx]
        base_rows = findall(==(bus_id), base_storage_bus_ids)
        seasonal_rows = findall(==(bus_id), seasonal_storage_bus_ids)

        # build vector-length nt so dimensions match load_data[idx, 1:nt]
        dis_sum_base = isempty(base_rows) ? zeros(nt) : vec(sum(discharge_base[base_rows, 1:nt]; dims=1))
        ch_sum_base  = isempty(base_rows) ? zeros(nt) : vec(sum(charge_base[base_rows,    1:nt]; dims=1))
        dis_sum_seasonal = isempty(seasonal_rows) ? zeros(nt) : vec(sum(discharge_seasonal[seasonal_rows, 1:nt]; dims=1))
        ch_sum_seasonal  = isempty(seasonal_rows) ? zeros(nt) : vec(sum(charge_seasonal[seasonal_rows,    1:nt]; dims=1))

        dis_sum = dis_sum_base .+ dis_sum_seasonal
        ch_sum  = ch_sum_base  .+ ch_sum_seasonal

        if busprop[idx, "BUS_TYPE"] != 3  # not slack
            @constraint(model,
                load_data[idx, 1:nt] .==
                  -sum(flow[l, 1:nt] for l in findall(x -> x == bus_id, branchprop[:, "F_BUS"])) .+
                   sum(flow[l, 1:nt] for l in findall(x -> x == bus_id, branchprop[:, "T_BUS"])) .+
                   sum(pg[l, 1:nt]   for l in findall(x -> x == bus_id, genprop[:, "GEN_BUS"])) .+
                   dis_sum .- ch_sum .+
                   load_shedding[idx, 1:nt]
            )
            @constraint(model, -2π .<= bus_angle[idx, 1:nt] .<= 2π)
        else
            @constraint(model,
                load_data[idx, 1:nt] .==
                  -sum(flow[l, 1:nt] for l in findall(x -> x == bus_id, branchprop[:, "F_BUS"])) .+
                   sum(flow[l, 1:nt] for l in findall(x -> x == bus_id, branchprop[:, "T_BUS"])) .+
                   sum(pg[l, 1:nt]   for l in findall(x -> x == bus_id, genprop[:, "GEN_BUS"])) .+
                   load_shedding[idx, 1:nt]
            )
            @constraint(model, bus_angle[idx, 1:nt] .== 0.2979/180*π)
        end
    end


    # Storage constraints
    @constraint(model, 0 .<= charge_base .<= base_charge_cap)
    @constraint(model, 0 .<= discharge_base .<= base_charge_cap)
    @constraint(model, 0 .<= charge_seasonal .<= seasonal_charge_cap)
    @constraint(model, 0 .<= discharge_seasonal .<= seasonal_charge_cap)

    # Storage state dynamics (per-row efficiency)
    for t in 1:nt
        @constraint(model,
            storage_state_base[:, t+1] .== storage_state_base[:, t] .+
            (sqrt.(base_eff_vec) .* charge_base[:, t]) .- ((1.0 ./ sqrt.(base_eff_vec)) .* discharge_base[:, t])
        )
        @constraint(model,
            storage_state_seasonal[:, t+1] .== storage_state_seasonal[:, t] .+
            (sqrt.(seasonal_eff_vec) .* charge_seasonal[:, t]) .- ((1.0 ./ sqrt.(seasonal_eff_vec)) .* discharge_seasonal[:, t])
        )
    end

    # Capacity bounds + SOC boundary conditions
    @constraint(model, 0.0 .* base_energy_cap .<= storage_state_base .<= base_energy_cap)
    @constraint(model, storage_state_base[:, 1]    .== base_init_soc_vec    .* base_energy_cap[:, 1])
    @constraint(model, storage_state_base[:, nt+1] .>= base_end_soc_min_vec .* base_energy_cap[:, nt+1])

    @constraint(model, 0.0 .* seasonal_energy_cap .<= storage_state_seasonal .<= seasonal_energy_cap)
    @constraint(model, storage_state_seasonal[:, 1]    .== seasonal_init_soc_vec    .* seasonal_energy_cap[:, 1])
    @constraint(model, storage_state_seasonal[:, nt+1] .>= seasonal_end_soc_min_vec .* seasonal_energy_cap[:, nt+1])

    # Impose interface limits
    n_if_lims = size(if_lim_up)[1]

    for i in 1:n_if_lims
        # Sum flow across the interfaces
        branch_idx = if_lim_map[findall(==(i), if_lim_map[:, "IF_ID"]), "BR_IDX"]
        idx_signs = sign.(branch_idx)
        idx_abs = abs.(branch_idx)

        flow_sum = [sum(idx_signs .* flow[Int.(idx_abs), t]) for t in 1:nt]
        # Constraint
        @constraint(model, if_lim_down[i, 1:nt] .<= flow_sum .<= if_lim_up[i, 1:nt])
    end

    # Nuclear generators always fully dispatch
    nuclear_idx = findall(x -> x == "Nuclear", genprop[!, "UNIT_TYPE"])
    for idx in nuclear_idx
        @constraint(model, pg[idx, :] .== g_max[idx, :])
    end

    # Hydro generators always fully dispatch at a weekly timescale
    niagara_idx = findfirst(x -> x == "Moses Niagara (Fleet)", genprop[!, "GEN_NAME"])
    moses_saund_idx = findfirst(x -> x == "St Lawrence - FDR (Fleet)", genprop[!, "GEN_NAME"])

    moses_saund_hydro = Matrix(filter_by_year(large_hydro, sim_year))[1, 2:end]
    niagara_hydro = Matrix(filter_by_year(large_hydro, sim_year))[2, 2:end]

    # Calculate the capacity rate of Moses Saunders
    hydro_pmax = genprop[moses_saund_idx, "PMAX"]
    hours_in_week = 24 * 7
    cap_rate = maximum(moses_saund_hydro ./ hours_in_week / hydro_pmax)
    if cap_rate > 1
        g_max[moses_saund_idx, :] .= g_max[moses_saund_idx, :] .* cap_rate
    end

    # Calculate the capacity rate of Niagara
    hydro_pmax = genprop[niagara_idx, "PMAX"]
    hours_in_week = 24 * 7
    cap_rate = maximum(niagara_hydro ./ hours_in_week / hydro_pmax)
    if cap_rate > 1
        g_max[niagara_idx, :] .= g_max[niagara_idx, :] .* cap_rate
    end

    # Do manually for now, update later
    last_hydro_day = split(names(filter_by_year(large_hydro, sim_year))[end], "-")[end]
    if last_hydro_day == "30"
        weekly_hours = vcat(fill(7 * 24, 52), [14])
    elseif last_hydro_day == "31"
        weekly_hours = vcat(fill(7 * 24, 52), [7])
    else
        throw(DomainError(last_hydro_error, "Error with hydro"))
    end

    # Cumulative time counter
    ct = 0
    for t in 1:48
        # Add constraints for generator power sum
        @constraint(model, sum(pg[niagara_idx, ct+1:ct+weekly_hours[t]]) == niagara_hydro[t])
        @constraint(model, sum(pg[moses_saund_idx, ct+1:ct+weekly_hours[t]]) == moses_saund_hydro[t])
        ct += weekly_hours[t]
    end
    # Generator capacity constraints
    @constraint(model, g_min .<= pg .<= g_max)

    if !exclude_external_zones
        # HVDC constraints (modelled as two dummy generators on each side of the lines)
        csc_idx = findall(x -> x == "HVDC_CSC", genprop[!, "GEN_NAME"])
        @constraint(model, pg[csc_idx[1], :] .== -pg[csc_idx[2], :]) # SC+NPX1385

        neptune_idx = findall(x -> x == "HVDC_Neptune", genprop[!, "GEN_NAME"])
        @constraint(model, pg[neptune_idx[1], :] .== -pg[neptune_idx[2], :]) # Neptune

        vft_idx = findall(x -> x == "HVDC_VFT", genprop[!, "GEN_NAME"])
        @constraint(model, pg[vft_idx[1], :] .== -pg[vft_idx[2], :]) # VFT

        htp_idx = findall(x -> x == "HVDC_HTP", genprop[!, "GEN_NAME"])
        @constraint(model, pg[htp_idx[1], :] .== -pg[htp_idx[2], :]) # HTP
    end

    if include_new_hvdc
        clean_path_idx = findall(x -> x == "HVDC_NYCleanPath", genprop[!, "GEN_NAME"])
        @constraint(model, pg[clean_path_idx[1], :] .== -pg[clean_path_idx[2], :]) # CleanPath

        chp_express_idx = findall(x -> x == "HVDC_CHPexpress", genprop[!, "GEN_NAME"])
        @constraint(model, pg[chp_express_idx[1], :] .== -pg[chp_express_idx[2], :]) # CHP Express
    end

    # Generator ramping constraints
    @constraint(model, -ramp_down[:, 2:nt] .<= pg[:, 2:nt] .- pg[:, 1:nt-1] .<= ramp_up[:, 2:nt])

    # Load shedding constraints
    @constraint(model, 0.0 .<= load_shedding .<= max.(load_data, 0))

    # Extract generation for wind and calculate curtailment
    wind_gen = pg[wind_idx, :]
    wind_curt = Matrix(wind[:, sim_dates]) .- wind_gen

    # Extract generation for utility-scale solar (UPV) and calculate curtailment
    solar_gen = pg[solar_upv_idx, :]
    solar_curt = Matrix(solar_upv[:, sim_dates]) .- solar_gen

    # Objective function: Minimize load shedding and storage operation costs
    λ = 0.01  # Appendix D anti-simultaneous charge/discharge penalty
    seasonal_penalty_frac = 15  # fraction of λ applied to seasonal storage usage

    base_usage = sum(charge_base) + sum(discharge_base)
    seasonal_usage = sum(charge_seasonal) + sum(discharge_seasonal)
    seasonal_penalty = seasonal_on ? seasonal_penalty_frac : 1.0
    seasonal_term = λ * seasonal_penalty * seasonal_usage

    @objective(model, Min,
     10000 * sum(load_shedding) +
     sum(gencost .* pg) +
     λ * base_usage +
     seasonal_term)


    # RUN IT
    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        # Extract results
        pg_result = value.(pg);
        flow_result = value.(flow);
        charge_base_result = value.(charge_base);
        discharge_base_result = value.(discharge_base);
        charge_seasonal_result = value.(charge_seasonal);
        discharge_seasonal_result = value.(discharge_seasonal);
        storage_state_base_result = value.(storage_state_base);
        storage_state_seasonal_result = value.(storage_state_seasonal);

        n_s = nrow(storage)
        charge_result = zeros(n_s, nt)
        discharge_result = zeros(n_s, nt)
        storage_state_result = zeros(n_s, nt + 1)
        if !isempty(base_storage_idx)
            charge_result[base_storage_idx, :] = charge_base_result
            discharge_result[base_storage_idx, :] = discharge_base_result
            storage_state_result[base_storage_idx, :] = storage_state_base_result
        end
        if !isempty(seasonal_storage_idx)
            charge_result[seasonal_storage_idx, :] = charge_seasonal_result
            discharge_result[seasonal_storage_idx, :] = discharge_seasonal_result
            storage_state_result[seasonal_storage_idx, :] = storage_state_seasonal_result
        end
        # --- tag storage rows as base vs seasonal ---
        asset_type = "is_seasonal" ∈ names(storage) ?
          [storage.is_seasonal[i] == 1 ? "seasonal" : "base" for i in 1:n_s] :
            fill("base", n_s)
        load_shedding_result = value.(load_shedding);
        wind_curtail_result = value.(wind_curt);
        solar_curtail_result = value.(solar_curt);
    else
        println("Error with optimization")
    end

    ### Save results to files
    # Add bus/branch IDs and datetime to output files
    flow_result = hcat([branchprop[:, "F_BUS"] branchprop[:, "FROM_ZONE"]], flow_result)
    flow_result = hcat([branchprop[:, "T_BUS"] branchprop[:, "TO_ZONE"]], flow_result)
    flow_result = vcat(hcat(["from_bus" "from_bus_zone" "to_bus" "to_bus_zone"], reshape(sim_dates, 1, :)), flow_result)

    pg_result = hcat([genprop[:, "GEN_BUS"] map(x -> bus_to_zone[x], genprop[:, "GEN_BUS"])], pg_result)
    pg_result = vcat(hcat(["bus_id" "zone"], reshape(sim_dates, 1, :)), pg_result)

    # New charge_result with asset_type
    charge_result = hcat([storage_bus_ids asset_type map(x -> bus_to_zone[x], storage_bus_ids)], charge_result)
    charge_result = vcat(hcat(["bus_id" "asset_type" "zone"], reshape(sim_dates, 1, :)), charge_result)

    #NEW discharge_result with asset_type
    discharge_result = hcat([storage_bus_ids asset_type map(x -> bus_to_zone[x], storage_bus_ids)], discharge_result)
    discharge_result = vcat(hcat(["bus_id" "asset_type" "zone"], reshape(sim_dates, 1, :)), discharge_result)

    #new storage_state_result with asset_type (note final column is "end")
    storage_state_result = hcat([storage_bus_ids asset_type map(x -> bus_to_zone[x], storage_bus_ids)], storage_state_result)
    storage_state_result = vcat(hcat(["bus_id" "asset_type" "zone"], reshape(vcat(sim_dates, "end"), 1, :)), storage_state_result)

    # Base-only outputs
    charge_base_out = hcat([base_storage_bus_ids map(x -> bus_to_zone[x], base_storage_bus_ids)], charge_base_result)
    charge_base_out = vcat(hcat(["bus_id" "zone"], reshape(sim_dates, 1, :)), charge_base_out)
    discharge_base_out = hcat([base_storage_bus_ids map(x -> bus_to_zone[x], base_storage_bus_ids)], discharge_base_result)
    discharge_base_out = vcat(hcat(["bus_id" "zone"], reshape(sim_dates, 1, :)), discharge_base_out)
    storage_state_base_out = hcat([base_storage_bus_ids map(x -> bus_to_zone[x], base_storage_bus_ids)], storage_state_base_result)
    storage_state_base_out = vcat(hcat(["bus_id" "zone"], reshape(vcat(sim_dates, "end"), 1, :)), storage_state_base_out)

    # Seasonal-only outputs
    charge_seasonal_out = hcat([seasonal_storage_bus_ids map(x -> bus_to_zone[x], seasonal_storage_bus_ids)], charge_seasonal_result)
    charge_seasonal_out = vcat(hcat(["bus_id" "zone"], reshape(sim_dates, 1, :)), charge_seasonal_out)
    discharge_seasonal_out = hcat([seasonal_storage_bus_ids map(x -> bus_to_zone[x], seasonal_storage_bus_ids)], discharge_seasonal_result)
    discharge_seasonal_out = vcat(hcat(["bus_id" "zone"], reshape(sim_dates, 1, :)), discharge_seasonal_out)
    storage_state_seasonal_out = hcat([seasonal_storage_bus_ids map(x -> bus_to_zone[x], seasonal_storage_bus_ids)], storage_state_seasonal_result)
    storage_state_seasonal_out = vcat(hcat(["bus_id" "zone"], reshape(vcat(sim_dates, "end"), 1, :)), storage_state_seasonal_out)

    load_shedding_result = hcat([bus_ids map(x -> bus_to_zone[x], bus_ids)], load_shedding_result)
    load_shedding_result = vcat(hcat(["bus_id" "zone"], reshape(sim_dates, 1, :)), load_shedding_result)

    wind_curtail_result = hcat([wind_bus_ids map(x -> bus_to_zone[x], wind_bus_ids)], wind_curtail_result)
    wind_curtail_result = vcat(hcat(["bus_id" "zone"], reshape(sim_dates, 1, :)), wind_curtail_result)

    solar_curtail_result = hcat([solar_upv_bus_ids map(x -> bus_to_zone[x], solar_upv_bus_ids)], solar_curtail_result)
    solar_curtail_result = vcat(hcat(["bus_id" "zone"], reshape(sim_dates, 1, :)), solar_curtail_result)

    # Also for load
    load_data_out = hcat([bus_ids map(x -> bus_to_zone[x], bus_ids)], load_data)
    load_data_out = vcat(hcat(["bus_id" "zone"], reshape(sim_dates, 1, :)), load_data_out)

    # Save results as CSV files
    CSV.write("$(out_path)/gen_$(sim_year).csv", DataFrame(pg_result, :auto), header=false)
    CSV.write("$(out_path)/flow_$(sim_year).csv", DataFrame(flow_result, :auto), header=false)
    CSV.write("$(out_path)/charge_$(sim_year).csv", DataFrame(charge_result, :auto), header=false)
    CSV.write("$(out_path)/discharge_$(sim_year).csv", DataFrame(discharge_result, :auto), header=false)
    CSV.write("$(out_path)/wind_curtailment_$(sim_year).csv", DataFrame(wind_curtail_result, :auto), header=false)
    CSV.write("$(out_path)/solar_curtailment_$(sim_year).csv", DataFrame(solar_curtail_result, :auto), header=false)
    CSV.write("$(out_path)/storage_state_$(sim_year).csv", DataFrame(storage_state_result, :auto), header=false)
    CSV.write("$(out_path)/charge_base_$(sim_year).csv", DataFrame(charge_base_out, :auto), header=false)
    CSV.write("$(out_path)/discharge_base_$(sim_year).csv", DataFrame(discharge_base_out, :auto), header=false)
    CSV.write("$(out_path)/storage_state_base_$(sim_year).csv", DataFrame(storage_state_base_out, :auto), header=false)
    CSV.write("$(out_path)/charge_seasonal_$(sim_year).csv", DataFrame(charge_seasonal_out, :auto), header=false)
    CSV.write("$(out_path)/discharge_seasonal_$(sim_year).csv", DataFrame(discharge_seasonal_out, :auto), header=false)
    CSV.write("$(out_path)/storage_state_seasonal_$(sim_year).csv", DataFrame(storage_state_seasonal_out, :auto), header=false)
    CSV.write("$(out_path)/load_shedding_$(sim_year).csv", DataFrame(load_shedding_result, :auto), header=false)
    CSV.write("$(out_path)/residual_load_$(sim_year).csv", DataFrame(load_data_out, :auto), header=false)
end
