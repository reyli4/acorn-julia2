using CSV
using DataFrames
using Dates

##############################################
# Paths
##############################################
project_path = "/home/fs01/jl2966/acorn-julia2/acorn-julia"

##############################################
# General utils
##############################################

function filter_by_year(df, target_year)
    date_cols = names(df)[2:end]  # Assuming first column is always bus_id
    year_cols = filter(col -> year(DateTime(col, "yyyy-mm-dd HH:MM:SS+00:00")) == target_year, date_cols)
    return select(df, [:bus_id; Symbol.(year_cols)])
end

function add_hvdc_generators(gen_prop, new_hvdc=true)
    """
    Adds in HVDC lines as dummy generators.

    NOTE: The bus indices here are different from the original model becuase we use
    the original indexing (we do not apply MATPOWER's ex2int).

    NOTE: We assume HVDC has zero cost! Is this valid? 
    """
    # Convert genprop number cols in Float64, otherwise runinto issues
    for col in names(gen_prop)
        if eltype(gen_prop[!, col]) <: Number
            gen_prop[!, col] = Float64.(gen_prop[!, col])
        end
    end

    # Existing HVDC lines (from the 2019 paper)
    csc1 = similar(gen_prop, 1)
    csc1[1, :] = reshape(["HVDC_CSC", 2, 0.0, 0, 2, 0.0, 0.0, 21, -409.833333333333, 0.0, 100.0, -100.0, 1.01, 100.0, 1.0, 530.0, -530.0, zeros(11)..., "NA", "HVDC", "HVDC", 2, 0, 0], 1, :) # CSC+NPX1358

    csc2 = similar(gen_prop, 1)
    csc2[1, :] = reshape(["HVDC_CSC", 2, 0.0, 0, 2, 0.0, 0.0, 80, 409.833333333333, 0.0, 100.0, -100.0, 1.0, 100.0, 1.0, 530.0, -530.0, zeros(11)..., "NA", "HVDC", "HVDC", 2, 0, 0], 1, :) # CSC+NPX1358

    neptune1 = similar(gen_prop, 1)
    neptune1[1, :] = reshape(["HVDC_Neptune", 2, 0.0, 0, 2, 0.0, 0.0, 124, -660.0, 0.0, 100.0, -100.0, 1.01, 100.0, 1.0, 660.0, -660.0, zeros(11)..., "NA", "HVDC", "HVDC", 2, 0, 0], 1, :) # Neptune

    neptune2 = similar(gen_prop, 1)
    neptune2[1, :] = reshape(["HVDC_Neptune", 2, 0.0, 0, 2, 0.0, 0.0, 79, 660.0, 0.0, 100.0, -100.0, 1.0, 100.0, 1.0, 660.0, -660.0, zeros(11)..., "NA", "HVDC", "HVDC", 2, 0, 0], 1, :) # Neptune

    vft1 = similar(gen_prop, 1)
    vft1[1, :] = reshape(["HVDC_VFT", 2, 0.0, 0, 2, 0.0, 0.0, 125, -560.833333333333, 0.0, 100.0, -100.0, 1.01, 100.0, 1.0, 660.0, -660.0, zeros(11)..., "NA", "HVDC", "HVDC", 2, 0, 0], 1, :) # VFT

    vft2 = similar(gen_prop, 1)
    vft2[1, :] = reshape(["HVDC_VFT", 2, 0.0, 0, 2, 0.0, 0.0, 81, 560.833333333333, 0.0, 100.0, -100.0, 1.0, 100.0, 1.0, 660.0, -660.0, zeros(11)..., "NA", "HVDC", "HVDC", 2, 0, 0], 1, :) # VFT

    htp1 = similar(gen_prop, 1)
    htp1[1, :] = reshape(["HVDC_HTP", 2, 0.0, 0, 2, 0.0, 0.0, 125, -315.0, 0.0, 100.0, -100.0, 1.01, 100.0, 1.0, 660.0, -660.0, zeros(11)..., "NA", "HVDC", "HVDC", 2, 0, 0], 1, :) # HTP

    htp2 = similar(gen_prop, 1)
    htp2[1, :] = reshape(["HVDC_HTP", 2, 0.0, 0, 2, 0.0, 0.0, 81, 315.0, 0.0, 100.0, -100.0, 1.0, 100.0, 1.0, 660.0, -660.0, zeros(11)..., "NA", "HVDC", "HVDC", 2, 0, 0], 1, :) # HTP

    gen_prop = vcat(gen_prop, csc1, neptune1, vft1, htp1, csc2, neptune2, vft2, htp2)

    # Proposed new HVDC lines
    if new_hvdc
        cleanpath1 = similar(gen_prop, 1)
        cleanpath1[1, :] = reshape(["HVDC_NYCleanPath", 2, 0.0, 0, 2, 0.0, 0.0, 69, 0, 0, 100, -100, 1, 100, 1, 1300, -1300, zeros(11)..., "NA", "HVDC", "HVDC", 2, 0, 0], 1, :) # NY Clean Path

        cleanpath2 = similar(gen_prop, 1)
        cleanpath2[1, :] = reshape(["HVDC_NYCleanPath", 2, 0.0, 0, 2, 0.0, 0.0, 81, 0, 0, 100, -100, 1, 100, 1, 1300, -1300, zeros(11)..., "NA", "HVDC", "HVDC", 2, 0, 0], 1, :) # NY Clean Path

        CHPexpress1 = similar(gen_prop, 1)
        CHPexpress1[1, :] = reshape(["HVDC_CHPexpress", 2, 0.0, 0, 2, 0.0, 0.0, 48, 0, 0, 100, -100, 1, 100, 1, 1250, -1250, zeros(11)..., "NA", "HVDC", "HVDC", 2, 0, 0], 1, :) # Champlain Hudson Power Express

        CHPexpress2 = similar(gen_prop, 1)
        CHPexpress2[1, :] = reshape(["HVDC_CHPexpress", 2, 0.0, 0, 2, 0.0, 0.0, 81, 0, 0, 100, -100, 1, 100, 1, 1250, -1250, zeros(11)..., "NA", "HVDC", "HVDC", 2, 0, 0], 1, :) # Champlain Hudson Power Express

        HQgen = similar(gen_prop, 1)
        HQgen[1, :] = reshape(["HVDC_HQ", 2, 0.0, 0, 2, 0.0, 0.0, 48, 0, 0, 100, -100, 1, 100, 1, 1250, -1250, zeros(11)..., "NA", "HVDC", "HVDC", 2, 0, 0], 1, :) # HydroQuebec

        gen_prop = vcat(gen_prop, HQgen, cleanpath1, cleanpath2, CHPexpress1, CHPexpress2)
    end

    return gen_prop
end

function create_interface_map(interfaces_df, branches_df)
    # Add branch IDs (1-based indexing)
    branches_with_id = copy(branches_df)
    branches_with_id.BR_ID = 1:nrow(branches_with_id)
    
    # Initialize result vectors
    interface_ids = Int[]
    branch_ids = Int[]
    
    # Loop through each interface
    for interface_row in eachrow(interfaces_df)
        if_id = interface_row.IF_ID
        from_zone = interface_row.FROM_ZONE
        to_zone = interface_row.TO_ZONE
        
        # Find branches that cross this interface
        for branch_row in eachrow(branches_with_id)
            branch_from_zone = branch_row.FROM_ZONE
            branch_to_zone = branch_row.TO_ZONE
            branch_id = branch_row.BR_ID
            
            # Check if branch crosses the interface
            if (branch_from_zone == from_zone && branch_to_zone == to_zone)
                # Branch goes in same direction as interface (positive)
                push!(interface_ids, if_id)
                push!(branch_ids, branch_id)
            elseif (branch_from_zone == to_zone && branch_to_zone == from_zone)
                # Branch goes in opposite direction to interface (negative)
                push!(interface_ids, if_id)
                push!(branch_ids, -branch_id)
            end
        end
    end
    
    # Create result dataframe
    interface_map = DataFrame(
        IF_ID = interface_ids,
        BR_IDX = branch_ids
    )
    
    return interface_map, branches_with_id
end

##############################################
# Generation utils
##############################################
function add_solar_generators(genprop, solar_bus_ids)
    # Solar generator info
    solar = similar(genprop, length(solar_bus_ids))

    solar[:, 1] .= "Solar farm" # Generator name
    solar[:, 2] .= 2 # Model (not important)
    solar[:, 3] .= 0.0 # Startup
    solar[:, 4] .= 0 # Shutdown
    solar[:, 5] .= 2 # NCOST
    solar[:, 6] .= 0.0 # COST_1
    solar[:, 7] .= 0.0 # COST_0
    solar[:, 8] .= solar_bus_ids # Bus number
    solar[:, 9] .= 0 # Pg
    solar[:, 10] .= 0 # Qg
    solar[:, 11] .= 9999 # Qmax
    solar[:, 12] .= -9999 # Qmin
    solar[:, 13] .= 1 # Vg
    solar[:, 14] .= 100 # mBase
    solar[:, 15] .= 1 # status
    solar[:, 16] .= 0 # Pmax
    solar[:, 17] .= 0 # Pmin
    solar[:, 18] .= 0 # Pc1
    solar[:, 19] .= 0 # Pc2
    solar[:, 20] .= 0 # Qc1min
    solar[:, 21] .= 0 # Qc1max
    solar[:, 22] .= 0 # Qc2min
    solar[:, 23] .= 0 # Qc2max
    solar[:, 24] .= 9999 # ramp rate for load following/AGC
    solar[:, 25] .= 9999 # ramp rate for 10 minute reserves
    solar[:, 26] .= 9999 # ramp rate for 30 minute reserves
    solar[:, 27] .= 0 # ramp rate for reactive power
    solar[:, 28] .= 0 # area participation factor
    solar[:, 29] .= "NA" # zone
    solar[:, 30] .= "SolarUPV" # generation type
    solar[:, 31] .= "SolarUPV" # fuel type
    solar[:, 32] .= 2 # CMT_KEY
    solar[:, 33] .= 0 # MIN_UP_TIME
    solar[:, 34] .= 0 # MIN_DOWN_TIME

    # Append to genprop
    return vcat(genprop, solar)
end

function add_wind_generators(genprop, wind_bus_ids)
    # Wind generator info
    wind = similar(genprop, length(wind_bus_ids))

    wind[:, 1] .= "Wind farm" # Generator name
    wind[:, 2] .= 2 # Model (not important)
    wind[:, 3] .= 0.0 # Startup
    wind[:, 4] .= 0 # Shutdown
    wind[:, 5] .= 2 # NCOST
    wind[:, 6] .= 0.0 # COST_1
    wind[:, 7] .= 0.0 # COST_0
    wind[:, 8] .= wind_bus_ids # Bus number
    wind[:, 9] .= 0 # Pg
    wind[:, 10] .= 0 # Qg
    wind[:, 11] .= 9999 # Qmax
    wind[:, 12] .= -9999 # Qmin
    wind[:, 13] .= 1 # Vg
    wind[:, 14] .= 100 # mBase
    wind[:, 15] .= 1 # status
    wind[:, 16] .= 0 # Pmax
    wind[:, 17] .= 0 # Pmin
    wind[:, 18] .= 0 # Pc1
    wind[:, 19] .= 0 # Pc2
    wind[:, 20] .= 0 # Qc1min
    wind[:, 21] .= 0 # Qc1max
    wind[:, 22] .= 0 # Qc2min
    wind[:, 23] .= 0 # Qc2max
    wind[:, 24] .= 9999 # ramp rate for load following/AGC
    wind[:, 25] .= 9999 # ramp rate for 10 minute reserves
    wind[:, 26] .= 9999 # ramp rate for 30 minute reserves
    wind[:, 27] .= 0 # ramp rate for reactive power
    wind[:, 28] .= 0 # area participation factor
    wind[:, 29] .= "NA" # zone
    wind[:, 30] .= "Wind" # generation type
    wind[:, 31] .= "Wind" # fuel type
    wind[:, 32] .= 2 # CMT_KEY
    wind[:, 33] .= 0 # MIN_UP_TIME
    wind[:, 34] .= 0 # MIN_DOWN_TIME

    # Append to genprop
    return vcat(genprop, wind)
end

##############################################
# Load utils
##############################################

function subtract_small_hydro(load, small_hydro, sim_year)
    """
    Adjusts the load data by subtracting small hydro
    """
    # Make copy
    load = copy(load)
    small_hydro = copy(small_hydro)

    # Filter to correct year
    load = filter_by_year(load, sim_year)
    small_hydro = filter_by_year(small_hydro, sim_year)

    # Filter both DataFrames to only include common columns
    common_cols = intersect(names(load), names(small_hydro))
    load = select(load, common_cols)
    small_hydro = select(small_hydro, common_cols)

    # Get small hydro bus ids
    small_hydro_bus_ids = small_hydro."bus_id"

    # Adjust loads with small hydro
    for i in eachindex(small_hydro_bus_ids)
        bus_idx = findfirst(==(small_hydro_bus_ids[i]), load."bus_id")
        load[bus_idx, 2:end] = Vector(load[bus_idx, 2:end]) .- Vector(small_hydro[i, 2:end])
    end

    return load
end

function subtract_solar_dpv(load, solar_dpv, sim_year)
    """
    Adjusts the load data by subtracting behind-the-meter solar (Solar DPV)
    """
    # Make copy
    load = copy(load)
    solar_dpv = copy(solar_dpv)

    # Filter to correct year
    load = filter_by_year(load, sim_year)
    solar_dpv = filter_by_year(solar_dpv, sim_year)

    # Filter both DataFrames to only include common columns
    common_cols = intersect(names(load), names(solar_dpv))
    load = select(load, common_cols)
    solar_dpv = select(solar_dpv, common_cols)

    # Get solar DPV bus ids
    solar_dpv_bus_ids = solar_dpv."bus_id"

    # Adjust loads with behind-the-meter solar
    for i in eachindex(solar_dpv_bus_ids)
        bus_idx = findfirst(==(solar_dpv_bus_ids[i]), load."bus_id")
        load[bus_idx, 2:end] = Vector(load[bus_idx, 2:end]) .- Vector(solar_dpv[i, 2:end])
    end

    return load
end

#######################
# Bus to zone mapping
#######################
bus_to_zone = Dict(
    21 => "NE",
    29 => "NE",
    35 => "NE",
    37=>"F",
    38=>"E",
    39=>"G",
    40=>"F",
    41=>"F",
    42=>"F",
    43=>"E",
    44=>"E",
    45=>"E",
    46=>"E",
    47=>"E",
    48=>"D",
    49=>"D",
    50=>"C",
    51=>"C",
    52=>"B",
    53=>"B",
    54=>"A",
    55=>"A",
    56=>"A",
    57=>"A",
    58=>"A",
    59=>"A",
    60=>"A",
    61=>"A",
    62=>"B",
    63=>"C",
    64=>"C",
    65=>"C",
    66=>"C",
    67=>"C",
    68=>"C",
    69=>"E",
    70=>"C",
    71=>"C",
    72=>"C",
    73=>"G",
    74=>"H",
    75=>"G",
    76=>"G",
    77=>"G",
    78=>"I",
    79=>"K",
    80=>"K",
    81=>"J",
    82=>"J",
    100=>"IESO",
    102=>"IESO",
    103=>"IESO",
    124=>"PJM",
    125=>"PJM",
    132=>"PJM",
    134=>"PJM",
    138=>"PJM",
)