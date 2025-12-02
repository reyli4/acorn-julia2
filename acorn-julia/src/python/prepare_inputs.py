import pandas as pd
import random
from typing import Dict
import geopandas as gpd

from python.utils import project_path


def resource_mapping(resource):
    if "battery" in resource:
        return "battery"
    elif "biomass" in resource:
        return "biomass"
    elif "hydroelectric" in resource and "storage" in resource:
        return "hydroelectric_storage"
    elif "hydroelectric" in resource and "storage" not in resource:
        return "hydroelectric"
    elif "distributed_generation" in resource:
        return "distributed_generation"
    elif "landbasedwind" in resource:
        return "onshore_wind_new"
    elif "onshore_wind" in resource:
        return "onshore_wind_existing"
    elif "natural_gas" in resource or "naturalgas" in resource:
        return "natural_gas"
    elif "nuclear" in resource:
        return "nuclear"
    elif "offshorewind" in resource:
        return "offshore_wind"
    elif "res_water_heat" in resource:
        return "water_heating"
    elif "solar" in resource:
        return "solar_existing"
    elif "utilitypv" in resource:
        return "solar_new"
    elif "trans_light_duty" in resource:
        return "transportation"
    else:
        return "other"  # fallback


def tidy_genX(df_genX: pd.DataFrame) -> pd.DataFrame:
    """
    Tidy the GenX data
    """
    # Filter to NYS
    df_genX = df_genX[df_genX["Zone"].isin([2, 3, 4, 5, 6, 7, 8, 9])].copy()
    # Add resource info
    df_genX["Zone"] = df_genX["Resource"].apply(lambda x: x.split("_")[2])
    df_genX["Resource"] = df_genX["Resource"].apply(
        lambda x: "_".join(x.split("_")[3:])
    )
    df_genX["Resource"] = df_genX["Resource"].apply(lambda x: resource_mapping(x))

    return df_genX


def generate_random_sites(
    df_genX,
    sites_per_zone=1,
    columns_to_scale=["EndCap"],
):
    """
    Generate random sites by creating random points in each target zone.
    We need this for GenX existing resources since it's now downscaled
    to specific lat/lons.
    """
    # Read NYISO GDF
    nyiso_gdf = gpd.read_file(
        f"{project_path}/data/nyiso/gis/NYISO_Load_Zone_Dissolved.shp"
    )

    # Merge dataframes
    gdf = pd.merge(
        nyiso_gdf,
        df_genX,
        left_on="zone",
        right_on="genX_zone",
    )

    # If only one point per zone, use original approach
    if sites_per_zone == 1:
        gdf["geometry"] = gdf.geometry.sample_points(1)
        gdf["latitude"] = gdf["geometry"].apply(lambda p: p.y)
        gdf["longitude"] = gdf["geometry"].apply(lambda p: p.x)
        return gdf

    # For multiple points, repeat rows and sample points
    # Repeat each row sites_per_zone times
    repeated_gdf = gdf.loc[gdf.index.repeat(sites_per_zone)].reset_index(drop=True)

    # Sample one point per row since we've already duplicated the rows
    repeated_gdf["geometry"] = repeated_gdf.geometry.sample_points(1)
    repeated_gdf["latitude"] = repeated_gdf["geometry"].apply(lambda p: p.y)
    repeated_gdf["longitude"] = repeated_gdf["geometry"].apply(lambda p: p.x)

    # Split EndCap
    if sites_per_zone > 1.0:
        for col in columns_to_scale:
            repeated_gdf[col] = repeated_gdf[col] / sites_per_zone

    return repeated_gdf


def split_combined_zones(
    df_genx: pd.DataFrame, df_genprop: pd.DataFrame
) -> pd.DataFrame:
    """
    Split combined zones (C&E and G-I) based on existing capacity distribution in genprop
    """
    df_genx_split = df_genx.copy()

    # Handle C&E zone split
    if "C&E" in df_genx_split["Zone"].values:
        # Get existing NG capacity in zones C and E
        ng_in_c = df_genprop[
            (df_genprop["GEN_ZONE"] == "C") & (df_genprop["FUEL_TYPE"] == "NG")
        ]["PMAX"].sum()

        ng_in_e = df_genprop[
            (df_genprop["GEN_ZONE"] == "E") & (df_genprop["FUEL_TYPE"] == "NG")
        ]["PMAX"].sum()

        total_ce = ng_in_c + ng_in_e

        # Get the combined capacity from GenX
        ce_row = df_genx_split[df_genx_split["Zone"] == "C&E"]
        if not ce_row.empty:
            ce_capacity = ce_row["EndCap"].iloc[0]

            # Split proportionally, but handle zero case
            if total_ce > 0:
                c_ratio = ng_in_c / total_ce
                e_ratio = ng_in_e / total_ce
            else:
                c_ratio = 0.5
                e_ratio = 0.5

            # Remove C&E row and add C and E rows
            df_genx_split = df_genx_split[df_genx_split["Zone"] != "C&E"]

            # Add zone C
            c_row = ce_row.copy()
            c_row["Zone"] = "C"
            c_row["EndCap"] = ce_capacity * c_ratio
            df_genx_split = pd.concat([df_genx_split, c_row], ignore_index=True)

            # Add zone E
            e_row = ce_row.copy()
            e_row["Zone"] = "E"
            e_row["EndCap"] = ce_capacity * e_ratio
            df_genx_split = pd.concat([df_genx_split, e_row], ignore_index=True)

    # Handle G-I zone split (assuming this might exist)
    if "G-I" in df_genx_split["Zone"].values:
        zones_gi = ["G", "H", "I"]
        existing_capacities = {}

        for zone in zones_gi:
            existing_capacities[zone] = df_genprop[
                (df_genprop["GEN_ZONE"] == zone) & (df_genprop["FUEL_TYPE"] == "NG")
            ]["PMAX"].sum()

        total_gi = sum(existing_capacities.values())

        # Get the combined capacity from GenX
        gi_row = df_genx_split[df_genx_split["Zone"] == "G-I"]
        if not gi_row.empty:
            gi_capacity = gi_row["EndCap"].iloc[0]

            # Remove G-I row
            df_genx_split = df_genx_split[df_genx_split["Zone"] != "G-I"]

            # Add individual zone rows
            for zone in zones_gi:
                if total_gi > 0:
                    zone_ratio = existing_capacities[zone] / total_gi
                else:
                    zone_ratio = 1.0 / len(zones_gi)

                zone_row = gi_row.copy()
                zone_row["Zone"] = zone
                zone_row["EndCap"] = gi_capacity * zone_ratio
                df_genx_split = pd.concat([df_genx_split, zone_row], ignore_index=True)

    return df_genx_split


def get_ng_capacity_by_zone(df_genx: pd.DataFrame) -> Dict[str, float]:
    """
    Extract natural gas capacity by zone from GenX results
    """
    ng_data = df_genx[df_genx["Resource"] == "natural_gas"].copy()
    return dict(zip(ng_data["Zone"], ng_data["EndCap"]))


def get_current_ng_capacity_by_zone(df_genprop: pd.DataFrame) -> Dict[str, float]:
    """
    Get current natural gas capacity by zone from genprop (only active generators)
    """
    ng_gens = df_genprop[
        (df_genprop["FUEL_TYPE"] == "NG")
        & (df_genprop["GEN_STATUS"] == 1)  # Only count active generators
    ].copy()
    return ng_gens.groupby("GEN_ZONE")["PMAX"].sum().to_dict()


def retire_generators(
    df_genprop: pd.DataFrame,
    zone: str,
    target_reduction: float,
    retirement_method: str = "random",
    store_path: str = None,
) -> pd.DataFrame:
    """
    Retire generators to reduce capacity in a zone
    """
    df_modified = df_genprop.copy()
    zone_ng_gens = df_modified[
        (df_modified["GEN_ZONE"] == zone)
        & (df_modified["FUEL_TYPE"] == "NG")
        & (df_modified["GEN_STATUS"] == 1)  # Only retire active generators
    ].copy()

    if zone_ng_gens.empty:
        msg = f"Warning: No active NG generators found in zone {zone}"
        print(msg)
        if store_path:
            with open(f"{store_path}/NG_matching.txt", "a") as f:
                print(msg, file=f)
        return df_modified

    retired_capacity = 0.0
    generators_to_retire = []

    if retirement_method == "random":
        # Randomly select generators to retire
        available_gens = zone_ng_gens.index.tolist()
        random.shuffle(available_gens)

        for gen_idx in available_gens:
            if retired_capacity >= target_reduction:
                break
            gen_capacity = df_modified.loc[gen_idx, "PMAX"]
            generators_to_retire.append(gen_idx)
            retired_capacity += gen_capacity

    elif retirement_method == "smallest_first":
        # Retire smallest generators first
        sorted_gens = zone_ng_gens.sort_values("PMAX")

        for gen_idx, gen_row in sorted_gens.iterrows():
            if retired_capacity >= target_reduction:
                break
            generators_to_retire.append(gen_idx)
            retired_capacity += gen_row["PMAX"]

    elif retirement_method == "largest_first":
        # Retire largest generators first
        sorted_gens = zone_ng_gens.sort_values("PMAX", ascending=False)

        for gen_idx, gen_row in sorted_gens.iterrows():
            if retired_capacity >= target_reduction:
                break
            generators_to_retire.append(gen_idx)
            retired_capacity += gen_row["PMAX"]

    elif retirement_method == "highest_cost_first":
        # Retire highest cost generators first (most expensive to operate)
        sorted_gens = zone_ng_gens.sort_values("COST_1", ascending=False)

        for gen_idx, gen_row in sorted_gens.iterrows():
            if retired_capacity >= target_reduction:
                break
            generators_to_retire.append(gen_idx)
            retired_capacity += gen_row["PMAX"]

    elif retirement_method == "lowest_cost_first":
        # Retire lowest cost generators first (cheapest to operate)
        sorted_gens = zone_ng_gens.sort_values("COST_1", ascending=True)

        for gen_idx, gen_row in sorted_gens.iterrows():
            if retired_capacity >= target_reduction:
                break
            generators_to_retire.append(gen_idx)
            retired_capacity += gen_row["PMAX"]

    # Retire selected generators
    for gen_idx in generators_to_retire:
        df_modified.loc[gen_idx, "GEN_STATUS"] = 0  # Set to retired

    msg = (
        f"Zone {zone}: Retired {len(generators_to_retire)} generators, "
        f"reducing capacity by {retired_capacity:.1f} MW (target: {target_reduction:.1f} MW)"
    )
    print(msg)
    if store_path:
        with open(f"{store_path}/NG_matching.txt", "a") as f:
            print(msg, file=f)

    return df_modified


def duplicate_generators(
    df_genprop: pd.DataFrame,
    zone: str,
    target_increase: float,
    store_path: str,
) -> pd.DataFrame:
    """
    Duplicate generators to increase capacity in a zone
    """
    df_modified = df_genprop.copy()
    zone_ng_gens = df_modified[
        (df_modified["GEN_ZONE"] == zone)
        & (df_modified["FUEL_TYPE"] == "NG")
        & (df_modified["GEN_STATUS"] == 1)
    ].copy()

    if zone_ng_gens.empty:
        msg = f"Warning: No active NG generators found in zone {zone} for duplication"
        print(msg)
        with open(f"{store_path}/NG_matching.txt", "a") as f:
            print(msg, file=f)
        return df_modified

    added_capacity = 0.0
    new_generators = []

    # Randomly select generators to duplicate until we reach target
    available_gens = zone_ng_gens.index.tolist()

    while added_capacity < target_increase and available_gens:
        # Select a random generator to duplicate
        gen_idx = random.choice(available_gens)
        gen_row = df_modified.loc[gen_idx].copy()

        # Modify the generator name to indicate it's a duplicate
        original_name = gen_row["GEN_NAME"]
        duplicate_count = (
            len([g for g in new_generators if original_name in g["GEN_NAME"]]) + 1
        )
        gen_row["GEN_NAME"] = f"{original_name}_DUP_{duplicate_count}"

        new_generators.append(gen_row)
        added_capacity += gen_row["PMAX"]

    # Add new generators to dataframe
    if new_generators:
        new_df = pd.DataFrame(new_generators)
        df_modified = pd.concat([df_modified, new_df], ignore_index=True)

    msg = (
        f"Zone {zone}: Added {len(new_generators)} duplicate generators, "
        f"increasing capacity by {added_capacity:.1f} MW (target: {target_increase:.1f} MW)"
    )
    print(msg)
    with open(f"{store_path}/NG_matching.txt", "a") as f:
        print(msg, file=f)

    return df_modified


def validate_results(
    df_original: pd.DataFrame,
    df_modified: pd.DataFrame,
    target_capacities: Dict[str, float],
    store_path: str,
) -> None:
    """
    Validate that the modifications achieved the target capacities
    """
    header = "\n" + "=" * 60 + "\nVALIDATION RESULTS\n" + "=" * 60
    print(header)
    with open(f"{store_path}/NG_matching.txt", "a") as f:
        print(header, file=f)

    # Get final capacities
    final_capacities = get_current_ng_capacity_by_zone(df_modified)

    table_header = (
        "Zone | Original (MW) | Target (MW) | Final (MW) | Error (MW)\n" + "-" * 65
    )
    print(table_header)
    with open(f"{store_path}/NG_matching.txt", "a") as f:
        print(table_header, file=f)

    for zone in sorted(target_capacities.keys()):
        original_cap = get_current_ng_capacity_by_zone(df_original).get(zone, 0.0)
        target_cap = target_capacities[zone]
        final_cap = final_capacities.get(zone, 0.0)
        error = final_cap - target_cap

        row = f"{zone:4s} | {original_cap:12.1f} | {target_cap:10.1f} | {final_cap:9.1f} | {error:9.1f}"
        print(row)
        with open(f"{store_path}/NG_matching.txt", "a") as f:
            print(row, file=f)


def match_ng_capacity(
    df_genx: pd.DataFrame,
    df_genprop: pd.DataFrame,
    store_path: str,
    retirement_method: str = "random",
    tolerance: float = 0.01,
) -> pd.DataFrame:
    """
    Main function to match natural gas capacity between GenX and genprop
    """
    with open(f"{store_path}/NG_matching.txt", "a") as f:
        print("Starting capacity matching process...", file=f)
    print("Starting capacity matching process...")

    # Natural gas only
    df_genx = df_genx[df_genx["Resource"] == "natural_gas"].copy()

    # Aggregate to zone level
    df_genx = df_genx.groupby("Zone")[["EndCap"]].sum().reset_index()
    df_genx["Resource"] = "natural_gas"

    # Split combined zones in GenX data
    df_genx_split = split_combined_zones(df_genx, df_genprop)

    # Get target capacities from GenX
    target_capacities = get_ng_capacity_by_zone(df_genx_split)

    # Get current capacities from genprop
    current_capacities = get_current_ng_capacity_by_zone(df_genprop)

    header = (
        "\nCapacity comparison:\nZone | Current (MW) | Target (MW) | Difference (MW)\n"
        + "-" * 55
    )
    print(header)
    with open(f"{store_path}/NG_matching.txt", "a") as f:
        print(header, file=f)

    df_modified = df_genprop[df_genprop["FUEL_TYPE"] == "NG"].copy()

    for zone in sorted(target_capacities.keys()):
        current_cap = current_capacities.get(zone, 0.0)
        target_cap = target_capacities[zone]
        difference = target_cap - current_cap

        row = (
            f"{zone:4s} | {current_cap:11.1f} | {target_cap:10.1f} | {difference:13.1f}"
        )
        print(row)
        with open(f"{store_path}/NG_matching.txt", "a") as f:
            print(row, file=f)

        # Only modify if difference is significant
        if abs(difference) > tolerance:
            if difference < 0:  # Need to retire capacity
                df_modified = retire_generators(
                    df_modified, zone, abs(difference), retirement_method, store_path
                )
            else:  # Need to add capacity
                df_modified = duplicate_generators(
                    df_modified, zone, difference, store_path
                )

    # Validate results
    validate_results(df_genprop, df_modified, target_capacities, store_path)

    # Store
    df_modified[df_modified["GEN_STATUS"] == 1].to_csv(
        f"{store_path}/inputs/genprop_NG_matched.csv", index=False
    )

    return df_modified
