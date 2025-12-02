using JuMP
using Gurobi
using CSV
using DataFrames
using MAT
using LinearAlgebra
using MathOptInterface
using Dates

include("./old_acorn_utils.jl")

##############################
# Set data directories
##############################
data_dir = joinpath(dirname(@__DIR__), "data")
tmp_data_dir = joinpath(dirname(@__DIR__), "data_tmp")

##############################
# MAIN
##############################
function run_model(scenario, year, gen_prop_name, branch_prop_name, bus_prop_name, out_path)
    # Define constants
    batt_duration = 8
    storage_eff = 0.85 # Efficiency for general storage
    gilboa_eff = 0.75 # Efficiency for specific storage (e.g., Gilboa)
    # Get number of hours in the year
    # nt = Dates.daysinyear(year) * 24
    nt = 365 * 24

    n_if_lims = 15

    # Set flags
    newHVDC = true
    HydroCon = true
    tranRating = true
    networkcon = true

    # Read grid data
    gen_prop = CSV.read("$(data_dir)/grid/gen_prop_$(gen_prop_name).csv", DataFrame, header=true, stringtype=String)
    bus_prop = CSV.read("$(data_dir)/grid/bus_prop_$(bus_prop_name).csv", DataFrame, header=true, stringtype=String)
    branch_prop = CSV.read("$(data_dir)/grid/branch_prop_$(branch_prop_name).csv", DataFrame, header=true, stringtype=String)

    bus_ids = bus_prop[:, 1]

    # Get scaling factors
    cc_scenario, bd_rate, ev_rate, wind_scalar, solar_scalar, batt_scalar = read_scaling_factors(scenario)

    ############## Load ####################
    load = get_load(cc_scenario, year, ev_rate, bd_rate, bus_ids, nt)
    load = subtract_small_hydro(load, bus_ids, nt)
    load = subtract_solar_dpv(load, bus_ids, cc_scenario, year, solar_scalar, nt)

    ############## Supply ##############
    # Read hydro
    niagra_hydro, moses_saund_hydro = get_hydro(cc_scenario, year)

    # Add wind generators to the model
    wind_gen, wind_bus_ids = get_wind(year, wind_scalar, nt)
    gen_prop = add_wind_generators(gen_prop, wind_bus_ids)

    # Add utility solar generators
    solar_upv_gen, solar_upv_bus_ids = get_solar_upv(cc_scenario, year, solar_scalar, nt)
    gen_prop = add_upv_generators(gen_prop, solar_upv_bus_ids)

    # HVDC generators
    gen_prop = add_hvdc(gen_prop)

    # Get generator limits
    g_max = repeat(gen_prop[:, "PMAX"], 1, nt) # Maximum real power output (MW)
    g_min = repeat(gen_prop[:, "PMIN"], 1, nt) # Minimum real power output (MW)

    # Update for renewables
    wind_idx = findall(x -> x == "Wind", gen_prop[:, "UNIT_TYPE"])
    g_max[wind_idx, :] = wind_gen[:, 1:nt]

    solar_upv_idx = findall(x -> x == "SolarUPV", gen_prop[:, "UNIT_TYPE"])
    g_max[solar_upv_idx, :] = solar_upv_gen[:, 1:nt]

    # Get generator ramp rates
    ramp_down = max.(repeat(gen_prop[:, "RAMP_30"], 1, nt) .* 2, repeat(gen_prop[:, "PMAX"], 1, nt)) # max of RAMP_30, PMAX??
    ramp_up = copy(ramp_down)

    # Note in the original model there is cost info: skipping for now
    # since it's not used in the 2040 analysis

    ############## Grid ##############
    # Transmission interface limits
    if_lim_up, if_lim_dn, if_lim_map = get_if_lims(year, n_if_lims, nt)

    # Branch limits
    branch_lims = repeat(Float64.(branch_prop[:, "RATE_A"]), 1, nt)
    branch_lims[branch_lims.==0] .= 99999.0

    # Storage
    charge_cap, storage_cap, storage_bus_ids = get_storage(batt_scalar, batt_duration, nt)

    ########## Optimization ##############
    n_gen = size(gen_prop, 1)
    n_bus = size(bus_prop, 1)
    n_branch = size(branch_prop, 1)

    model = Model(Gurobi.Optimizer)

    ## Define variables
    @variable(model, pg[1:n_gen, 1:nt])
    @variable(model, flow[1:n_branch, 1:nt])
    @variable(model, bus_angle[1:n_bus, 1:nt])
    @variable(model, charge[1:length(storage_bus_ids), 1:nt])
    @variable(model, discharge[1:length(storage_bus_ids), 1:nt])
    @variable(model, batt_state[1:length(storage_bus_ids), 1:nt+1])
    @variable(model, load_shedding[1:n_bus, 1:nt])

    ## Constraints 
    # Branch flow limits and power flow equations
    for l in 1:n_branch
        idx_from_bus = findfirst(x -> x == branch_prop[l, "F_BUS"], bus_prop[:, "BUS_I"])
        idx_to_bus = findfirst(x -> x == branch_prop[l, "T_BUS"], bus_prop[:, "BUS_I"])
        @constraint(model, -branch_lims[l, :] .<= flow[l, :] .<= branch_lims[l, :])  # Branch flow limits
        @constraint(model, flow[l, :] .== (100 / branch_prop[l, "BR_X"]) .*  # DC power flow equations
                                          (bus_angle[idx_from_bus, :] .-
                                           bus_angle[idx_to_bus, :]))
    end

    # Node balance and phase angle constraints
    for idx in 1:n_bus
        bus_id = bus_ids[idx]
        if bus_prop[idx, "BUS_TYPE"] != 3  # Not the slack bus
            if bus_id in storage_bus_ids
                # Node balance with storage devices
                storage_idx = findfirst(==(bus_id), storage_bus_ids)
                @constraint(model, load[idx, 1:nt] .==
                                   -sum(flow[l, 1:nt] for l in findall(x -> x == bus_id, branch_prop[:, "F_BUS"])) .+
                                   sum(flow[l, 1:nt] for l in findall(x -> x == bus_id, branch_prop[:, "T_BUS"])) .+
                                   sum(pg[l, 1:nt] for l in findall(x -> x == bus_id, gen_prop[:, "GEN_BUS"])) .+
                                   discharge[storage_idx, 1:nt] .-
                                   charge[storage_idx, 1:nt] .+
                                   load_shedding[idx, 1:nt])
            else
                # Node balance without storage devices
                @constraint(model, load[idx, 1:nt] .==
                                   -sum(flow[l, 1:nt] for l in findall(x -> x == bus_id, branch_prop[:, "F_BUS"])) .+
                                   sum(flow[l, 1:nt] for l in findall(x -> x == bus_id, branch_prop[:, "T_BUS"])) .+
                                   sum(pg[l, 1:nt] for l in findall(x -> x == bus_id, gen_prop[:, "GEN_BUS"])) .+
                                   load_shedding[idx, 1:nt])
            end
            @constraint(model, -2 * pi .<= bus_angle[idx, 1:nt] .<= 2 * pi)  # Voltage angle limits
        else  # Slack bus
            # Node balance for slack bus
            @constraint(model, load[idx, 1:nt] .==
                               -sum(flow[l, 1:nt] for l in findall(x -> x == bus_id, branch_prop[:, "F_BUS"])) .+
                               sum(flow[l, 1:nt] for l in findall(x -> x == bus_id, branch_prop[:, "T_BUS"])) .+
                               sum(pg[l, 1:nt] for l in findall(x -> x == bus_id, gen_prop[:, "GEN_BUS"])) .+
                               load_shedding[idx, 1:nt])
            @constraint(model, bus_angle[idx, 1:nt] .== 0.2979 / 180 * pi)  # Fix voltage angle at slack bus
        end
    end

    # Storage constraints
    @constraint(model, 0 .<= charge .<= charge_cap)         # Charging limits
    @constraint(model, 0 .<= discharge .<= charge_cap)      # Discharging limits

    # Battery state dynamics for all time steps
    for t in 1:nt
        # Battery state dynamics for all but the last storage bus
        @constraint(model, batt_state[1:end-1, t+1] .== batt_state[1:end-1, t] .+ sqrt(storage_eff) .* charge[1:end-1, t] .- (1 / sqrt(storage_eff)) .* discharge[1:end-1, t])

        # Battery state dynamics Gilboa
        @constraint(model, batt_state[end, t+1] .== batt_state[end, t] .+ sqrt(gilboa_eff) .* charge[end, t] .- (1 / sqrt(gilboa_eff)) .* discharge[end, t])
    end

    # Battery capacity constraints
    @constraint(model, 0.0 .* storage_cap .<= batt_state .<= storage_cap)

    # Initial battery state (assuming 30% of capacity)
    @constraint(model, batt_state[:, 1] .== 0.3 .* storage_cap[:, 1])

    # Interface flow constraints
    # Internal limits set to infinity if desired
    if !networkcon
        if_lim_dn[1:12, :] .= -Inf
        if_lim_up[1:12, :] .= Inf
    end

    # Impose interface limits
    for i in 1:n_if_lims
        # Sum flow across the interfaces
        idx = if_lim_map[findall(==(i), if_lim_map[:, "IF_ID"]), "BUS_ID"]
        idx_signs = sign.(idx)
        idx_abs = abs.(idx)
        flow_sum = [sum(idx_signs .* flow[Int.(idx_abs), t]) for t in 1:nt]
        # Constraint
        @constraint(model, if_lim_dn[i, 1:nt] .<= flow_sum .<= if_lim_up[i, 1:nt])
    end

    # Nuclear generators always fully dispatch
    nuclear_idx = findall(x -> x == "Nuclear", gen_prop[!, "UNIT_TYPE"])
    for idx in nuclear_idx
        @constraint(model, pg[idx, :] .== g_max[idx, :])
    end

    # Hydro generators always fully dispatch
    niagra_idx = findfirst(x -> x == "Niagra", gen_prop[!, "UNIT_TYPE"])
    moses_saund_idx = findfirst(x -> x == "MosesSaunders", gen_prop[!, "UNIT_TYPE"])
    if HydroCon
        # Load the 'qm_to_numdays.csv' file into a DataFrame
        dayofqm = CSV.read("$(tmp_data_dir)/qm_to_numdays.csv", DataFrame)
        nhours = dayofqm.Days .* 24  # Convert days to hours

        # Calculate the capacity rate of Moses Saunders
        hydro_pmax = gen_prop[moses_saund_idx, "PMAX"]
        cap_rate = maximum(moses_saund_hydro ./ nhours / hydro_pmax)
        if cap_rate > 1
            g_max[moses_saund_idx, :] .= g_max[moses_saund_idx, :] .* cap_rate
        end

        # Cumulative time counter
        ct = 0
        for t in 1:48
            # Add constraints for generator power sum based on nyhy and moses_saund_hydro
            @constraint(model, sum(pg[niagra_idx, ct+1:ct+nhours[t]]) == niagra_hydro[t])
            @constraint(model, sum(pg[moses_saund_idx, ct+1:ct+nhours[t]]) == moses_saund_hydro[t])
            ct += nhours[t]
        end
    end

    # Generator capacity constraints
    @constraint(model, g_min .<= pg .<= g_max)

    # HVDC constraints (modelled as two dummy generators on each side of the lines)
    csc_idx = findall(x -> x == "HVDC_CSC", gen_prop[!, "UNIT_TYPE"])
    @constraint(model, pg[csc_idx[1], :] .== -pg[csc_idx[2], :]) # SC+NPX1385

    neptune_idx = findall(x -> x == "HVDC_Neptune", gen_prop[!, "UNIT_TYPE"])
    @constraint(model, pg[neptune_idx[1], :] .== -pg[neptune_idx[2], :]) # Neptune

    vft_idx = findall(x -> x == "HVDC_VFT", gen_prop[!, "UNIT_TYPE"])
    @constraint(model, pg[vft_idx[1], :] .== -pg[vft_idx[2], :]) # VFT

    htp_idx = findall(x -> x == "HVDC_HTP", gen_prop[!, "UNIT_TYPE"])
    @constraint(model, pg[htp_idx[1], :] .== -pg[htp_idx[2], :]) # HTP

    clean_path_idx = findall(x -> x == "HVDC_NYCleanPath", gen_prop[!, "UNIT_TYPE"])
    @constraint(model, pg[clean_path_idx[1], :] .== -pg[clean_path_idx[2], :]) # CleanPath

    chp_express_idx = findall(x -> x == "HVDC_CHPexpress", gen_prop[!, "UNIT_TYPE"])
    @constraint(model, pg[chp_express_idx[1], :] .== -pg[chp_express_idx[2], :]) # CHP Express

    # @constraint(model, pg[end-4, :] .== -pg[end-1, :]) # ?????

    # Generator ramping constraints
    @constraint(model, -ramp_down[:, 2:nt] .<= pg[:, 2:nt] .- pg[:, 1:nt-1] .<= ramp_up[:, 2:nt])

    # Load shedding constraints
    @constraint(model, 0.0 .<= load_shedding .<= max.(load, 0))

    # Extract generation for wind and calculate curtailment
    wg = pg[wind_idx, :]
    wc = wind_gen .- wg

    # Extract generation for utility-scale solar (UPV) and calculate curtailment
    sg = pg[solar_upv_idx, :]
    sc = solar_upv_gen .- sg

    # Objective function: Minimize load shedding and storage operation costs
    @objective(model, Min, sum(load_shedding) + 0.05 * (sum(charge) + sum(discharge)))

    # SOLVE
    optimize!(model)

    # Check if the solver found an optimal solution
    if termination_status(model) == MOI.OPTIMAL
        # Extract results
        pg_result = value.(pg)
        flow_result = value.(flow)
        charge_result = value.(charge)
        discharge_result = value.(discharge)
        batt_state_result = value.(batt_state)
        load_shedding_result = value.(load_shedding)
        wind_curtail_result = value.(wc)
        solar_curtail_result = value.(sc)
    else
        println("Optimization did not find an optimal solution for year $(year).")
    end

    # Save results to files
    if !isdir(out_path)
        mkdir(out_path)
    end

    # Save results as CSV files
    CSV.write("$(out_path)/gen_$(year).csv", DataFrame(pg_result, :auto), header=false)
    CSV.write("$(out_path)/flow_$(year).csv", DataFrame(flow_result, :auto), header=false)
    CSV.write("$(out_path)/charge_$(year).csv", DataFrame(charge_result, :auto), header=false)
    CSV.write("$(out_path)/disch_$(year).csv", DataFrame(discharge_result, :auto), header=false)
    CSV.write("$(out_path)/wind_curtailment_$(year).csv", DataFrame(wind_curtail_result, :auto), header=false)
    CSV.write("$(out_path)/solar_curtailment_$(year).csv", DataFrame(solar_curtail_result, :auto), header=false)
    CSV.write("$(out_path)/batt_state_$(year).csv", DataFrame(batt_state_result, :auto), header=false)
    CSV.write("$(out_path)/load_shedding_$(year).csv", DataFrame(load_shedding_result, :auto), header=false)
end
