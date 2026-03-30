# instantiate and precompile environment
using Pkg; Pkg.activate(dirname(@__DIR__))
Pkg.instantiate(); Pkg.precompile()

# load dependencies
using YAML
using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--project-dir"
            help = "Project directory"
            arg_type = String
            required = true
        "--run-dir"
            help = "Run directory"
            arg_type = String
            required = true
        "--if_lim_name"
            help = "Interface limit file"
            arg_type = String
            required = true
        "--exclude_external_zones"
            help = "Exclude external zones or not"
            arg_type = Int
            required = true
        "--include_new_hvdc"
            help = "Include new HVDC or not"
            arg_type = Int
            required = true
        "--save_name"
            help = "Save subdirectory name"
            arg_type = String
            required = true
        "--storage_filename"
            help = "File in run_dir/inputs to use for storage assignment (default: storage_assignment.csv)"
            arg_type = String
            default = "storage_assignment.csv"
        "--seasonal_on"
            help = "Include seasonal storage layer? 0=no,1=yes (default 0)"
            arg_type = Int
            default = 0
        "--seasonal_filename"
            help = "File in run_dir/inputs for seasonal storage (default: seasonal_storage_assignment.csv)"
            arg_type = String
            default = "seasonal_storage_assignment.csv"
        "--sim-sequence"
            help = "Optional comma-separated sequence of years (e.g., 1985,1986,1985,1986). Overrides config."
            arg_type = String
            default = ""
        "--wrap-boundary-years"
            help = "Repeat the first and last years this many extra times around a contiguous range from config sim_years."
            arg_type = Int
            default = 0
    end
    return parse_args(s)
end

to_int_year(v) = Int(round(parse(Float64, string(v))))

function parse_sequence_arg(seq_str::String)
    yrs = Int[]
    for token in split(seq_str, ",")
        t = strip(token)
        isempty(t) && continue
        push!(yrs, parse(Int, t))
    end
    return yrs
end

function build_wrapped_sequence(base_years::Vector{Int}, wrap_count::Int)
    isempty(base_years) && return Int[], String[], Int[], Int[]

    seq = Int[]
    roles = String[]
    keep = Int[]
    analysis_idx = Int[]

    for _ in 1:wrap_count
        push!(seq, first(base_years))
        push!(roles, "burn_in")
        push!(keep, 0)
        push!(analysis_idx, 0)
    end

    for (i, year) in enumerate(base_years)
        push!(seq, year)
        push!(roles, "analysis")
        push!(keep, 1)
        push!(analysis_idx, i)
    end

    for _ in 1:wrap_count
        push!(seq, last(base_years))
        push!(roles, "burn_out")
        push!(keep, 0)
        push!(analysis_idx, 0)
    end

    return seq, roles, keep, analysis_idx
end

# Main script
# -----------
args = parse_commandline()
project_dir = args["project-dir"]
run_dir = args["run-dir"]
if_lim_name = args["if_lim_name"]
exclude_external_zones = args["exclude_external_zones"]
include_new_hvdc = args["include_new_hvdc"]
save_name = args["save_name"]
storage_filename = args["storage_filename"]
seasonal_on = convert(Bool, args["seasonal_on"])
seasonal_filename = args["seasonal_filename"]
sim_sequence_arg = args["sim-sequence"]
wrap_boundary_years = haskey(args, "wrap-boundary-years") ? args["wrap-boundary-years"] : get(args, "wrap_boundary_years", 0)

println("Run parameters:")
println("  Save name: $(save_name)")
println("  Storage file: $(storage_filename)")
println("  Seasonal on?: $(seasonal_on)")
println("  Seasonal file: $(seasonal_filename)")
println("  sim-sequence arg: $(sim_sequence_arg)")
println("  wrap-boundary-years: $(wrap_boundary_years)")

# Load custom functions
include("$(project_dir)/src/julia/utils.jl")
include("$(project_dir)/src/julia/acorn.jl")

# Read run parameters
config = YAML.load_file("$(run_dir)/config.yml")

run_name = config["run_name"]
climate_scenario_years = config["climate_scenario_years"]
sim_years = config["sim_years"]
exclude_external_zones_bool = convert(Bool, exclude_external_zones)
include_new_hvdc_bool = convert(Bool, include_new_hvdc)

# Determine sequence
sim_sequence = Int[]
sequence_roles = String[]
keep_for_analysis = Int[]
analysis_year_idx = Int[]
if !isempty(strip(sim_sequence_arg))
    if wrap_boundary_years > 0
        println("NOTE: --wrap-boundary-years is ignored because --sim-sequence was provided explicitly.")
    end
    sim_sequence = parse_sequence_arg(sim_sequence_arg)
    sequence_roles = fill("analysis", length(sim_sequence))
    keep_for_analysis = fill(1, length(sim_sequence))
    analysis_year_idx = collect(1:length(sim_sequence))
elseif haskey(config, "sim_year_sequence")
    if wrap_boundary_years > 0
        println("NOTE: --wrap-boundary-years is ignored because config sim_year_sequence was provided explicitly.")
    end
    sim_sequence = [to_int_year(y) for y in config["sim_year_sequence"]]
    sequence_roles = fill("analysis", length(sim_sequence))
    keep_for_analysis = fill(1, length(sim_sequence))
    analysis_year_idx = collect(1:length(sim_sequence))
else
    base_years = collect(to_int_year(sim_years[1]):to_int_year(sim_years[2]))
    sim_sequence, sequence_roles, keep_for_analysis, analysis_year_idx = build_wrapped_sequence(base_years, wrap_boundary_years)
end

isempty(sim_sequence) && error("No simulation years found. Provide --sim-sequence or set sim_year_sequence/sim_years in config.yml.")

println("Sequence run parameters:")
println("  run_name: $(run_name)")
println("  climate scenario years: $(climate_scenario_years)")
println("  simulation sequence: $(sim_sequence)")
println("  sequence length: $(length(sim_sequence))")
if any(sequence_roles .!= "analysis")
    println("  sequence roles: $(sequence_roles)")
    kept_years = [sim_sequence[i] for i in eachindex(sim_sequence) if keep_for_analysis[i] == 1]
    println("  analysis years retained after trimming boundary repeats: $(kept_years)")
end
flush(stdout)

out_dir = "$(run_dir)/outputs/$(climate_scenario_years)/$(save_name)"
mkpath(out_dir)

manifest_path = "$(out_dir)/_sequence_manifest.csv"
open(manifest_path, "w") do io
    write(io, "seq_idx,sim_year,token,prev_token,sequence_role,keep_for_analysis,analysis_year_idx\n")

    for (i, sim_year) in enumerate(sim_sequence)
        token = "seq$(lpad(string(i), 3, '0'))_y$(sim_year)"
        prev_token = i > 1 ? "seq$(lpad(string(i - 1), 3, '0'))_y$(sim_sequence[i - 1])" : ""
        role = sequence_roles[i]
        keep = keep_for_analysis[i]
        analysis_idx = analysis_year_idx[i]

        println("Now running sequence step $(i)/$(length(sim_sequence)): year $(sim_year), token=$(token), role=$(role), keep=$(keep)")
        flush(stdout)

        # Safe resume behavior for sequence runs
        if isfile("$(out_dir)/load_shedding_$(token).csv")
            println("Step output exists, skipping token=$(token)...")
            write(io, "$(i),$(sim_year),$(token),$(prev_token),$(role),$(keep),$(analysis_idx)\n")
            continue
        end

        carry_base = i > 1 ? "$(out_dir)/storage_state_base_$(prev_token).csv" : nothing
        carry_seasonal = i > 1 ? "$(out_dir)/storage_state_seasonal_$(prev_token).csv" : nothing

        run_acorn(
            run_name,
            climate_scenario_years,
            sim_year,
            config["branchprop_name"],
            config["busprop_name"],
            if_lim_name,
            save_name;
            exclude_external_zones=exclude_external_zones_bool,
            include_new_hvdc=include_new_hvdc_bool,
            storage_filename=storage_filename,
            seasonal_on=seasonal_on,
            seasonal_filename=seasonal_filename,
            first_sim_year=sim_sequence[1],
            carryover_base_file=carry_base,
            carryover_seasonal_file=carry_seasonal,
            output_suffix=token,
        )

        write(io, "$(i),$(sim_year),$(token),$(prev_token),$(role),$(keep),$(analysis_idx)\n")
    end
end

println("Wrote sequence manifest: $(manifest_path)")
