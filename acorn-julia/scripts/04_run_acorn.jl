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
            help = "Include new HVD or not"
            arg_type = Int
            required = true
        "--save_name"
            help = "Save subdirectory name"
            arg_type = String
            required = true
        #NEW
        "--storage_filename"
            help = "File in run_dir/inputs to use for storage assignment (default: storage_assignment.csv)"
            arg_type = String
            default = "storage_assignment.csv"
        # NEW:
        "--seasonal_on"
            help = "Include seasonal storage layer? 0=no,1=yes (default 0)"
            arg_type = Int
            default = 0
        "--seasonal_filename"
            help = "File in run_dir/inputs for SEASONAL storage (default: seasonal_storage_assignment.csv)"
            arg_type = String
            default = "seasonal_storage_assignment.csv"
        "--seasonal_rate_caps_on"
            help = "Apply seasonal C-rate constraints? 0=no,1=yes (default 1)"
            arg_type = Int
            default = 1

    end
    return parse_args(s)
end

# Main script
# -----------
args = parse_commandline()
storage_filename = args["storage_filename"]   # <-- add this line
project_dir = args["project-dir"]
run_dir = args["run-dir"]
if_lim_name = args["if_lim_name"]
exclude_external_zones = args["exclude_external_zones"]
include_new_hvdc = args["include_new_hvdc"]
save_name = args["save_name"]
# NEW
storage_filename=args["storage_filename"]
seasonal_on_int = args["seasonal_on"]
seasonal_filename = args["seasonal_filename"]
seasonal_rate_caps_on = convert(Bool, args["seasonal_rate_caps_on"])
seasonal_on = convert(Bool, seasonal_on_int)

println("Run parameters:")
println("  Save name: $(save_name)")
println("  Storage file: $(storage_filename)")
println("  Seasonal on?: $(seasonal_on)")
println("  Seasonal file: $(seasonal_filename)")
println("  Seasonal rate caps on?: $(seasonal_rate_caps_on)")
println("  Seasonal SOC floor on?: removed (always off in Stage 0 build)")

# Load custom functions
include("$(project_dir)/src/julia/utils.jl")
include("$(project_dir)/src/julia/acorn.jl")

# Read run parameters
config = YAML.load_file("$(run_dir)/config.yml")

genX_file_name = config["genX_file_name"]
genX_downscaled_file_name = config["genX_downscaled_file_name"]
genX_iter = config["genX_iter"]
busprop_name = config["busprop_name"]
genprop_name = config["genprop_name"]
branchprop_name = config["branchprop_name"]
climate_scenario_years = config["climate_scenario_years"]
sim_years = config["sim_years"]
resstock_upgrade = config["resstock_upgrade"]
comstock_upgrade = config["comstock_upgrade"]
run_name = config["run_name"]

# Print run parameters
println("Run parameters:")
println("  run_name: $(run_name)")
println("  GenX file: $(genX_file_name)")
println("  GenX downscaled file: $(genX_downscaled_file_name)")
println("  GenX iter: $(genX_iter)")
println("  Busprop file: $(busprop_name)")
println("  Genprop file: $(genprop_name)")
println("  Branchprop file: $(branchprop_name)")
println("  Climate scenario years: $(climate_scenario_years)")
println("  Sim years: $(sim_years)")
println("  Exclude external zones: $(exclude_external_zones)")
println("  Include new HVDC: $(include_new_hvdc)")
println("  Save name: $(save_name)")
flush(stdout)  # Force output to appear immediately

exclude_external_zones_bool = convert(Bool, exclude_external_zones)
include_new_hvdc_bool = convert(Bool, include_new_hvdc)

# Loop through years and run model
for sim_year in sim_years[1]:sim_years[2]
    println("Now running year $(sim_year)...")
    flush(stdout)
    # Check if run already exists
    if isfile("$(run_dir)/outputs/$(climate_scenario_years)/$(save_name)/load_shedding_$(sim_year).csv")
        println("Run already exists, skipping...")
        continue
    end

    # Run ACORN
    run_acorn(
        run_name,
        climate_scenario_years,
        sim_year,
        branchprop_name,
        busprop_name,
        if_lim_name,
        save_name;
        exclude_external_zones=exclude_external_zones_bool,
        include_new_hvdc=include_new_hvdc_bool,
        storage_filename=storage_filename,   #NEW
        seasonal_on=seasonal_on,          # NEW
        seasonal_filename=seasonal_filename,   # NEW
        seasonal_rate_caps_on=seasonal_rate_caps_on,
    )
end
