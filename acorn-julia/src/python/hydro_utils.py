import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from python.utils import project_path, merge_to_zones, nearest_neighbor_lat_lon


def disaggregate_weekly_to_hourly(
    df, method="average", morning_peak_hour=8, evening_peak_hour=18
):
    """
    Disaggregate weekly hydropower data to hourly data.

    Parameters:
    -----------
    df : pandas.DataFrame
        Weekly data with columns: datetime, eia_id, power_predicted_mwh, p_max, p_min, p_avg
    method : str
        'average' for constant hourly values or 'diurnal' for daily cycle
    morning_peak_hour : int
        Hour of day for morning peak (0-23), used only for diurnal method
    evening_peak_hour : int
        Hour of day for evening peak (0-23), used only for diurnal method

    Returns:
    --------
    pandas.DataFrame
        Hourly data with datetime and power_mw columns, plus original metadata
    """

    def create_diurnal_pattern(morning_peak=8, evening_peak=18):
        """Create a 24-hour normalized diurnal pattern with two peaks."""
        hours = np.arange(24)

        # Create two gaussian-like peaks
        morning_component = np.exp(-0.5 * ((hours - morning_peak) / 2.5) ** 2)
        evening_component = np.exp(-0.5 * ((hours - evening_peak) / 3.0) ** 2)

        # Combine peaks with evening peak being stronger
        pattern = 0.6 * morning_component + 1.0 * evening_component

        # Add baseline to avoid zeros
        pattern += 0.3

        # Normalize so mean = 1 (preserves total energy)
        pattern = pattern / pattern.mean()

        return pattern

    def create_pattern_for_period(daily_pattern, n_hours):
        """Create pattern for given number of hours."""
        n_full_days = n_hours // 24
        remaining_hours = n_hours % 24

        # Create pattern for full days
        if n_full_days > 0:
            full_days_pattern = np.tile(daily_pattern, n_full_days)
        else:
            full_days_pattern = np.array([])

        # Add partial day if needed
        if remaining_hours > 0:
            partial_day_pattern = daily_pattern[:remaining_hours]
            pattern = np.concatenate([full_days_pattern, partial_day_pattern])
        else:
            pattern = full_days_pattern

        return pattern

    def scale_pattern_to_constraints(pattern, p_avg, p_max, p_min, total_mwh):
        """Scale pattern to match average, and respect max/min constraints."""
        # Start with pattern scaled to match average
        scaled_pattern = pattern * p_avg

        # Check if we violate constraints
        pattern_max = scaled_pattern.max()
        pattern_min = scaled_pattern.min()

        # If we exceed p_max, compress the pattern
        if pattern_max > p_max:
            # Compress pattern to fit within [p_min, p_max]
            pattern_range = pattern.max() - pattern.min()
            available_range = p_max - p_min

            if pattern_range > 0:
                compression_factor = available_range / pattern_range
                scaled_pattern = p_min + (pattern - pattern.min()) * compression_factor

        # Final adjustment to ensure total energy conservation
        current_total = scaled_pattern.sum()
        target_total = total_mwh
        adjustment_factor = target_total / current_total
        scaled_pattern *= adjustment_factor

        return scaled_pattern

    if method == "average":
        # Average and ffill
        final_df = (
            df.set_index("datetime")
            .groupby("eia_id")
            .resample("h")[["p_avg"]]
            .ffill()
            .reset_index()
            .rename(columns={"p_avg": "power_MW"})
        )

    else:  # diurnal method
        # Process each plant separately for diurnal method (needs the complex logic)
        result_dfs = []

        for eia_id in df["eia_id"].unique():
            plant_df = df[df["eia_id"] == eia_id].copy()

            hourly_data = []

            for _, row in plant_df.iterrows():
                period_start = pd.to_datetime(row["datetime"])
                n_hours = int(row["n_hours"])

                # Create diurnal cycle
                daily_pattern = create_diurnal_pattern(
                    morning_peak_hour, evening_peak_hour
                )
                period_pattern = create_pattern_for_period(daily_pattern, n_hours)

                # Scale pattern to match constraints and conserve energy
                hourly_power = scale_pattern_to_constraints(
                    period_pattern,
                    row["p_avg"],
                    row["p_max"],
                    row["p_min"],
                    row["power_predicted_mwh"],
                )

                # Create hourly timestamps
                hourly_timestamps = pd.date_range(
                    start=period_start, periods=n_hours, freq="H"
                )

                # Create hourly records
                for i, (timestamp, power_mw) in enumerate(
                    zip(hourly_timestamps, hourly_power)
                ):
                    hourly_data.append(
                        {
                            "datetime": timestamp,
                            "eia_id": row["eia_id"],
                            "power_MW": power_mw,
                            # "period_start": period_start,
                            # "hour_of_period": i,
                            # "hour_of_day": timestamp.hour,
                            # "day_of_week": timestamp.dayofweek,
                            # "scenario": row["scenario"],
                            # "original_n_hours": n_hours,
                            # "original_p_avg": row["p_avg"],
                            # "original_p_max": row["p_max"],
                            # "original_p_min": row["p_min"],
                            # "original_total_mwh": row["power_predicted_mwh"],
                        }
                    )

            plant_hourly_df = pd.DataFrame(hourly_data)
            result_dfs.append(plant_hourly_df)

        # Combine all plants
        final_df = pd.concat(result_dfs, ignore_index=True)

    # Convert to UTC datetime
    final_df["datetime"] = final_df["datetime"].dt.tz_localize("UTC")

    # Sort by plant and datetime
    final_df = final_df.set_index(["eia_id", "datetime"])

    return final_df


def assign_hydro_GD_to_buses(
    hydro_scenario,
    downsample_method="average",
):
    """
    Assign hydro GDs to buses.
    """
    # Read hydro production and plants
    df_hydro = pd.read_csv(
        f"{project_path}/data/hydro/godeeep-hydro/godeeep-hydro-{hydro_scenario}-weekly.csv"
    )
    df_hydro_plants = pd.read_csv(
        f"{project_path}/data/hydro/godeeep-hydro/godeeep-hydro-plants.csv"
    )
    # Merge plant info to zones
    df_hydro_plants = merge_to_zones(df_hydro_plants, join="left")

    # Subset to NYISO plants
    df_hydro_plants = df_hydro_plants[df_hydro_plants["ba"] == "NYIS"]
    df_hydro = df_hydro[df_hydro["eia_id"].isin(df_hydro_plants["eia_id"])].copy()
    df_hydro["datetime"] = pd.to_datetime(df_hydro["datetime"])
    df_hydro_plants["geometry"] = [
        Point(x, y) for x, y in zip(df_hydro_plants["lon"], df_hydro_plants["lat"])
    ]

    # We model Robert-Moses Niagara and St Lawrence FDR separately
    large_hydro_plants = ["Robert Moses Niagara", "Robert Moses - St. Lawrence"]
    df_large_hydro = df_hydro[df_hydro["plant"].isin(large_hydro_plants)].copy()
    # Assign buses manually
    df_large_hydro.loc[df_large_hydro["plant"] == "Robert Moses Niagara", "bus_id"] = 55
    df_large_hydro.loc[
        df_large_hydro["plant"] == "Robert Moses - St. Lawrence", "bus_id"
    ] = 48

    # For the rest, we use the average pattern
    df_small_hydro = disaggregate_weekly_to_hourly(
        df=df_hydro[~df_hydro["plant"].isin(large_hydro_plants)].copy(),
        method=downsample_method,
    )

    # There are some without zonal overlap -- assign manually
    df_hydro_plants.loc[df_hydro_plants["zone"].isna(), "zone"] = (
        "C"  # C for first two based on coordinates
    )

    # Merge to assign to buses
    df_small_hydro = pd.merge(
        df_small_hydro.reset_index(),
        df_hydro_plants[["eia_id", "zone", "lat", "lon", "geometry"]],
        on="eia_id",
        how="left",
    )

    # Get unique locations for quicker assignment
    df_small_hydro_unique_locs = df_small_hydro.drop_duplicates("eia_id")
    df_small_hydro_unique_locs = nearest_neighbor_lat_lon(
        gpd.GeoDataFrame(df_small_hydro_unique_locs), match_zones=True
    )

    df_small_hydro = (
        pd.merge(
            df_small_hydro,
            df_small_hydro_unique_locs[["eia_id", "bus_id"]],
            on="eia_id",
            how="outer",
        )
        .groupby(["bus_id", "datetime"])[["power_MW"]]
        .sum()
    )

    # Tidy large hydro
    df_large_hydro = df_large_hydro.reset_index().set_index(["bus_id", "datetime"])[
        ["power_predicted_mwh"]
    ]

    return df_small_hydro, df_large_hydro
