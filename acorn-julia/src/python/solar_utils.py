from glob import glob
from multiprocessing import Pool, cpu_count

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import salem
import xarray as xr
from scipy.optimize import minimize_scalar

from python.utils import (
    project_path,
    nearest_neighbor_lat_lon,
    zone_names,
    month_keys,
    merge_to_zones,
)


def read_all_sind():
    """
    Read all SIND data from NREL SIND.
    """
    sind_files = glob(f"{project_path}/data/nrel/sind/ny-pv-2006/Actual_*.csv")

    df_all = []

    # Loop through all
    for file in sind_files:
        # Read
        df = pd.read_csv(file)

        # Convert to UTC
        df["datetime"] = pd.to_datetime(df["LocalTime"], format="%m/%d/%y %H:%M")
        df["datetime"] = df["datetime"].dt.tz_localize(
            "America/New_York", ambiguous="NaT", nonexistent="NaT"
        )
        df["datetime"] = df["datetime"].dt.tz_convert("UTC")

        # Resample to hourly
        df = df.set_index("datetime")
        df = df.resample("h").mean(numeric_only=True)

        # Add lat/lon
        lat, lon = file.split("_")[1], file.split("_")[2]
        df["sind_lat"] = float(lat)
        df["sind_lon"] = float(lon)

        # Add system type
        df["solar_type"] = file.split("_")[4]

        # Fix naming
        df = df.rename(columns={"Power(MW)": "actual_power_MW"})

        # Normalize
        power_rating = float(file.split("_")[5].replace("MW", ""))
        df["actual_power_norm"] = df["actual_power_MW"] / power_rating

        # Append to all
        df_all.append(df)

    return pd.concat(df_all)


def calculate_solar_power(
    Geff,  # Incident solar radiation (W/m²)
    Ta,  # Ambient air temperature (°C)
    Pg_star=1.0,  # Rated capacity (MW) - set to 1 MW in the paper
    G_star=1000.0,  # Reference solar radiation (W/m²) - standard value
    Tc_star=25.0,  # Reference cell temperature (°C) - standard value
    beta=0.45,  # Temperature loss coefficient (%/°C) - from the paper
    NOCT=46.0,  # Nominal operating cell temperature (°C) - from the paper
):
    """
    Calculate solar power generation based on Perpiñan et al. model.
    Reference: https://onlinelibrary.wiley.com/doi/abs/10.1002/pip.728
    See also:

    Parameters:
    -----------
    Geff : float or array
        Incident solar radiation (W/m²)
    Ta : float or array
        Ambient air temperature (°C)
    Pg_star : float, optional
        Rated capacity (MW), default is 1.0 MW
    G_star : float, optional
        Reference solar radiation (W/m²), default is 1000 W/m²
    Tc_star : float, optional
        Reference cell temperature (°C), default is 25°C
    beta : float, optional
        Temperature loss coefficient (%/°C), default is 0.45%/°C
    NOCT : float, optional
        Nominal operating cell temperature (°C), default is 46°C

    Returns:
    --------
    P_DC : float or array
        Generated solar power (MW)
    """
    # Calculate the conversion parameter CT (Equation 13)
    CT = (NOCT - 20) / (0.8 * G_star)

    # Calculate cell temperature (Equation 12)
    Tc = Ta + CT * Geff

    # Calculate efficiency ratio (Equation 11)
    efficiency_ratio = 1 - (beta / 100) * (Tc - Tc_star)

    # Calculate DC power output (Equation 10)
    P_DC = Pg_star * (Geff / G_star) * efficiency_ratio

    return P_DC


def _read_solar_climate_data(args):
    """
    Function to help read and subset solar climate data in parallel.

    Parameters:
    -----------
    args : tuple
        Arguments
    file : str
        Path to the climate data file
    lat_name : str
        Name of the latitude variable
    lon_name : str
        Name of the longitude variable
    x_min : float
        Minimum longitude
    x_max : float
        Maximum longitude
    y_min : float
        Minimum latitude
    y_max : float
        Maximum latitude
    solar_vars : list
        Variables to keep
    use_salem : bool
        Whether to use Salem to read the data (useful for WRF data like TGW)

    Returns:
    --------
    ds : xr.Dataset
        Dataset containing the climate data
    """
    # Unpack
    file, lat_name, lon_name, x_min, x_max, y_min, y_max, solar_vars, use_salem = args

    # Open
    if use_salem:
        ds = salem.open_wrf_dataset(file)
    else:
        ds = xr.open_dataset(file)

    # Subset
    ds = ds[solar_vars].sel(
        {lat_name: slice(y_min, y_max), lon_name: slice(x_min, x_max)}
    )

    # Return
    return ds.load()


def _select_solar_climate_data_point(args):
    """
    Function to help select solar climate data points in parallel.

    Parameters:
    -----------
    args : tuple
        Arguments
    ds : xr.Dataset
        Dataset containing the climate data
    lat_name : str
        Name of the latitude variable
    lon_name : str
        Name of the longitude variable
    time_name : str
        Name of the time variable
    lat : float
        Latitude of the desired point
    lon : float
        Longitude of the desired point
    curvilinear : bool
        Whether the climate data is on a curvilinear grid

    Returns:
    --------
    df : pd.DataFrame
        DataFrame containing the climate data
    """
    # Unpack
    ds, lat_name, lon_name, time_name, lat, lon, curvilinear = args

    # Select climate data point
    if curvilinear:
        ds_crs = ccrs.Projection(ds.pyproj_srs)
        x, y = ds_crs.transform_point(
            np.round(lon, 2), np.round(lat, 2), src_crs=ccrs.PlateCarree()
        )
    else:
        x, y = lon, lat

    ds_sel = ds.sel({lon_name: x, lat_name: y}, method="nearest")

    # Take only the solar data
    df = ds_sel.to_dataframe().reset_index()

    # Add info
    df["desired_lat"] = lat
    df["desired_lon"] = lon
    df["datetime"] = pd.to_datetime(df[time_name])
    df["datetime"] = df["datetime"].dt.tz_localize("UTC")
    df = df.drop(columns=time_name)

    # Return
    return df


def prepare_solar_data(
    climate_paths,
    solar_vars,
    lat_name="lat",
    lon_name="lon",
    time_name="time",
    curvilinear=False,
    use_salem=True,
    parallel=False,
    sites="sind",
    sind_site_type=None,
    sind_keep_every=1,
    min_lat=39,  # approx NYS
    max_lat=45,  # approx NYS
    min_lon=-80,  # approx NYS
    max_lon=-71,  # approx NYS
):
    """
    Gather input data for solar power generation.

    Parameters:
    -----------
    climate_paths : list
        List of climate data file paths
    solar_vars : list
        List of solar variables to extract (e.g., temperature, shortwave radiation)
    lat_name : str
        Name of the latitude variable in climate data
    lon_name : str
        Name of the longitude variable in climate data
    time_name : str
        Name of the time variable in climate data
    curvilinear : bool
        Whether the climate data is on a curvilinear grid
    use_salem : bool
        Whether to use Salem to read the data (useful for WRF data like TGW)
    parallel : bool
        Whether to run in parallel using multiprocessing
    sites : str or list
        Whether to use SIND sites ("sind") or a custom set of lat/lon points
    sind_site_type : str, optional
        Type of SIND site to subset to (UPV or DPV)
    sind_keep_every : int
        Every nth SIND site to keep
    min_lat : float
    max_lat : float
    min_lon : float
    max_lon : float
        Subset climate data based on these bounds

    Returns:
    --------
    df : pd.DataFrame
        DataFrame containing climate data
    """
    # Get bounds for NYS
    if curvilinear:
        # Get CRS
        ds_tmp = salem.open_wrf_dataset(climate_paths[0])
        ds_crs = ccrs.Projection(ds_tmp.pyproj_srs)
        # Get bounds
        x_min, y_min = ds_crs.transform_point(
            min_lon, min_lat, src_crs=ccrs.PlateCarree()
        )
        x_max, y_max = ds_crs.transform_point(
            max_lon, max_lat, src_crs=ccrs.PlateCarree()
        )
    else:
        x_min, y_min = min_lon, min_lat
        x_max, y_max = max_lon, max_lat

    # Read in parallel
    if parallel:
        n_cores = cpu_count() - 1
        # Prepare args for each file
        args = [
            (
                file,
                lat_name,
                lon_name,
                x_min,
                x_max,
                y_min,
                y_max,
                solar_vars,
                use_salem,
            )
            for file in np.sort(climate_paths)
        ]
        with Pool(processes=n_cores - 1) as pool:
            # Read climate data
            ds_all = pool.map(_read_solar_climate_data, args)
        ds = xr.concat(ds_all, dim="time")
    else:
        ds = []
        for file in np.sort(climate_paths):
            # Read climate data
            ds_tmp = _read_solar_climate_data(
                (
                    file,
                    lat_name,
                    lon_name,
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    solar_vars,
                    use_salem,
                )
            )

            # Append
            ds.append(ds_tmp)
        ds = xr.concat(ds, dim="time")

    # Subset to desired time range and sort
    ds = ds.sortby("time")
    assert len(ds.time) > 0, "No data found"

    # Get lat/lons from solar sites
    if isinstance(sites, str) and sites == "sind":
        df_sind = read_all_sind()
        # Apply keep_every sampling
        if sind_keep_every > 1:
            df_sind_sampled = df_sind.iloc[::sind_keep_every]
        else:
            df_sind_sampled = df_sind
        latlons = (
            df_sind_sampled[["sind_lat", "sind_lon"]].value_counts().index.to_numpy()
        )
    elif isinstance(sites, (list, np.ndarray)):
        latlons = sites
    else:
        raise ValueError(f"Invalid sites: {sites}")

    # Loop through lat/lons
    if parallel:
        n_cores = cpu_count() - 1
        with Pool(processes=n_cores - 1) as pool:
            # Prepare args
            args = [
                (
                    ds,
                    lat_name,
                    lon_name,
                    time_name,
                    lat,
                    lon,
                    curvilinear,
                )
                for lat, lon in latlons
            ]
            # Select climate data point
            df_all = pool.map(_select_solar_climate_data_point, args)
    else:
        df_all = []
        for lat, lon in latlons:
            # Select climate data point
            df = _select_solar_climate_data_point(
                (
                    ds,
                    lat_name,
                    lon_name,
                    time_name,
                    lat,
                    lon,
                    curvilinear,
                )
            )
            # Append
            df_all.append(df)

    # Combine all
    df_all = pd.concat(df_all, ignore_index=True)

    # Merge
    if isinstance(sites, str) and sites == "sind":
        df_all = pd.merge(
            df_all,
            df_sind.reset_index(),
            right_on=["datetime", "sind_lat", "sind_lon"],
            left_on=["datetime", "desired_lat", "desired_lon"],
        )

        # Subset to site type if specified
        if sind_site_type is not None:
            df_all = df_all[df_all["solar_type"] == sind_site_type]

    elif isinstance(sites, (list, np.ndarray)):
        df_all = pd.merge(
            df_all,
            pd.DataFrame(latlons, columns=["desired_lat", "desired_lon"]),
            on=["desired_lat", "desired_lon"],
        )

    # Drop duplicates
    df_all = (
        df_all.set_index(["desired_lat", "desired_lon", "datetime"])
        .sort_index()
        .reset_index()
    ).drop_duplicates()

    # Add datetime info
    df_all["month"] = df_all["datetime"].dt.month
    df_all["dayofyear"] = df_all["datetime"].dt.dayofyear
    df_all["hour"] = df_all["datetime"].dt.hour

    # Return
    return df_all


def get_solar_correction_factors(
    df,
    temperature_var,
    shortwave_var,
    beta,
    lookup_cols=["month", "hour"],
    apply_correction=True,
):
    """
    Calculates solar power correction factors from climate data.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing climate data
    temperature_var : str
        Name of the temperature variable [C]
    shortwave_var : str
        Name of the shortwave radiation variable [W/m2]
    beta : float
        Temperature loss coefficient [%]
    lookup_cols : list
        Columns to use for the lookup table

    Returns:
    --------
    df : pd.DataFrame
        DataFrame containing the solar power data
    correction_lookup : pd.DataFrame
        DataFrame containing the correction factors
    """

    # Calculate solar power
    df["sim_power_norm"] = calculate_solar_power(
        df[shortwave_var], df[temperature_var], beta=beta
    )

    # Create lookup table: average bias by doy and hour
    df["bias"] = df["actual_power_norm"] - df["sim_power_norm"]
    correction_lookup = (
        df.groupby(lookup_cols)["bias"].mean().to_frame(name="bias_correction")
    )

    # Apply correction if specified
    if apply_correction:
        df = pd.merge(df, correction_lookup.reset_index(), on=lookup_cols)
        df["sim_power_norm_corrected"] = df["sim_power_norm"] + df["bias_correction"]
    # Set negative values to zero
    df["sim_power_norm_corrected"] = df["sim_power_norm_corrected"].clip(lower=0.0)

    # Return
    return df, correction_lookup


def optimize_beta(df, temperature_var, shortwave_var, lookup_cols=["month", "hour"]):
    """
    Optimizes the beta parameter for the solar power correction. Note that the
    optimization is done alongside the bias correction.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing climate data
    temperature_var : str
        Name of the temperature variable [C]
    shortwave_var : str
        Name of the shortwave radiation variable [W/m2]
    lookup_cols : list, optional
        Columns to use for the lookup table

    Returns:
    --------
    beta : float
        Optimized beta parameter
    """

    # Objective function
    def _objective(beta, df, temperature_var, shortwave_var, lookup_cols):
        # Calculate solar power (with bias correction if specified)
        if lookup_cols is not None:
            df, df_correction = get_solar_correction_factors(
                df, temperature_var, shortwave_var, beta, lookup_cols=lookup_cols
            )
        else:
            df["sim_power_norm_corrected"] = calculate_solar_power(
                df[shortwave_var], df[temperature_var], beta=beta
            )

        # Calculate RMSE
        rmse = np.sqrt(
            np.mean((df["sim_power_norm_corrected"] - df["actual_power_norm"]) ** 2)
        )

        # Return
        return rmse

    # Optimize
    res = minimize_scalar(
        _objective,
        bounds=(0.01, 5.0),
        args=(df, temperature_var, shortwave_var, lookup_cols),
        method="bounded",
    )

    # Return
    return res.x


def plot_solar_correction_fit(
    df, x_col, y_col, x_name=None, y_name=None, daily=False, zonal=False, save_path=None
):
    """
    Plot the solar correction fit.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing climate data
    x_col : str
        Name of the x-axis variable
    y_col : str
        Name of the y-axis variable
    x_name : str, optional
        Name of the x-axis variable
    y_name : str, optional
        Name of the y-axis variable
    daily : bool, optional
        Whether to plot daily averages
    zonal : bool, optional
        Whether to plot zonal averages
    save_path : str, optional
        Path to save the plot
    """
    # Daily averages if selected
    if daily:
        df = (
            df.groupby([df["datetime"].dt.date, "sind_lat", "sind_lon"])
            .mean(numeric_only=True)
            .reset_index()
        )

    # Zonal averages if selected
    if zonal:
        df = merge_to_zones(df, lat_name="sind_lat", lon_name="sind_lon")

    # Loop through counter variables
    if zonal:
        counter_var = "ZONE"
    else:
        counter_var = "month"

    # Plot
    if zonal:
        n_zones = len(df[counter_var].unique())
        n_cols = 3
        n_rows = int(np.ceil(n_zones / n_cols))
    else:
        n_cols = 3
        n_rows = 4
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, n_rows * 2.5))
    axs = axs.flatten()

    # Loop through counter variables
    for idc, counter in enumerate(df[counter_var].unique()):
        df_counter = df[df[counter_var] == counter]
        df_counter.plot(
            y=y_col,
            x=x_col,
            kind="scatter",
            s=3,
            ax=axs[idc],
            alpha=0.5,
        )
        # Add fit info
        r2 = (
            np.corrcoef(df_counter.dropna()[x_col], df_counter.dropna()[y_col])[0, 1]
            ** 2
        )
        rmse = np.sqrt(
            np.mean((df_counter.dropna()[x_col] - df_counter.dropna()[y_col]) ** 2)
        )
        # Add 1:1 line
        axs[idc].plot(
            [0, 1], [0, 1], transform=axs[idc].transAxes, ls="--", color="black"
        )
        # Tidy
        if zonal:
            counter_names = zone_names
        else:
            counter_names = month_keys

        axs[idc].set_title(
            f"{counter_names[counter]} (R$^2$: {r2:.2f}, RMSE: {rmse:.2f})"
        )
        axs[idc].grid(alpha=0.5)
        axs[idc].set_xlabel("")
        axs[idc].set_ylabel("")

    fig.supxlabel(x_name if x_name is not None else x_col)
    fig.supylabel(y_name if y_name is not None else y_col)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def apply_solar_correction_factors(
    df,
    df_correction,
    temperature_var,
    shortwave_var,
    beta,
    lookup_cols=["month", "hour"],
):
    """
    Calculates solar power correction factors from climate data.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing climate data
    df_correction : pd.DataFrame
        DataFrame containing correction factors
    lookup_cols : list
        Columns to use for the lookup table
    """
    # Make sure datetime is present
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["month"] = df["datetime"].dt.month
    df["dayofyear"] = df["datetime"].dt.dayofyear
    df["hour"] = df["datetime"].dt.hour

    # Calculate solar power
    df["sim_power_norm"] = calculate_solar_power(
        df[shortwave_var], df[temperature_var], beta=beta
    )

    # Apply correction
    df = pd.merge(df, df_correction, on=lookup_cols)
    df["sim_power_norm_corrected"] = df["sim_power_norm"] + df["bias_correction"]
    # Set negative values to zero
    df["sim_power_norm_corrected"] = df["sim_power_norm_corrected"].clip(lower=0.0)

    # Return
    return df


def calculate_solar_timeseries_from_genX(
    df_genX,
    climate_paths,
    correction_file,
    match_zones=True,
    PV_bus_only=True,
    solar_vars=["T2C", "SWDOWN"],
    correction_cols=["month", "hour"],
    lat_name="south_north",
    lon_name="west_east",
    curvilinear=True,
    parallel=True,
):
    """
    Calculate solar power generation timeseries at bus level using genX outputs and climate data.

    This function combines solar resource data from climate model outputs with genX capacity
    information to calculate hourly solar power generation at each bus in the grid. It handles
    the conversion of solar resource data to power using a temperature-dependent correction factor
    and aggregates generation to the bus level.

    Parameters
    ----------
    df_genX : pd.DataFrame
        DataFrame containing genX outputs with columns ['latitude', 'longitude', 'EndCap', 'genX_zone']
    climate_paths : list
        List of paths to climate data files
    correction_file : str
        Path to correction factors file
    match_zones : bool, optional
        Whether to match zones when assigning to buses, by default True
    solar_vars : list, optional
        Solar variables to extract from climate data, expected temperature in C and shortwave radiation in W/m2, by default ["T2C", "SWDOWN"]
    correction_cols : list, optional
        Columns to use for correction factors, by default ["month", "hour"]
    lat_name : str, optional
        Name of latitude variable in climate data, by default "south_north"
    lon_name : str, optional
        Name of longitude variable in climate data, by default "west_east"
    curvilinear : bool, optional
        Whether climate data is on curvilinear grid, by default True
    parallel : bool, optional
        Whether to process data in parallel, by default True

    Returns
    -------
    pd.DataFrame
        DataFrame with hourly solar power generation at each bus, indexed by ['BUS_I', 'datetime']
        with column 'power_mw' containing the generation in megawatts
    """
    # Get raw wind data from climate outputs
    sites = np.column_stack((df_genX["latitude"], df_genX["longitude"]))

    df_solar = prepare_solar_data(
        climate_paths=climate_paths,
        solar_vars=solar_vars,
        sites=sites,
        lat_name=lat_name,
        lon_name=lon_name,
        curvilinear=curvilinear,
        parallel=parallel,
    )

    # Merge with genX outputs
    df = pd.merge(
        df_solar[["desired_lat", "desired_lon", "datetime"] + solar_vars],
        df_genX[["latitude", "longitude", "EndCap", "genX_zone"]],
        how="outer",
        left_on=["desired_lat", "desired_lon"],
        right_on=["latitude", "longitude"],
    ).drop(columns=["desired_lat", "desired_lon"])

    # Calculate power
    df_correction = pd.read_csv(correction_file)
    beta = df_correction["optimized_beta"].values[0]
    df = apply_solar_correction_factors(
        df,
        df_correction,
        solar_vars[0],
        solar_vars[1],
        beta,
        lookup_cols=correction_cols,
    )
    df["power_MW"] = df["sim_power_norm_corrected"] * df["EndCap"]

    # Get unique genX locations (easier to assign to buses)
    gdf_genX_unique_locs = gpd.GeoDataFrame(
        df_genX[["latitude", "longitude", "genX_zone"]],
        geometry=gpd.points_from_xy(df_genX["longitude"], df_genX["latitude"]),
        crs="EPSG:4326",
    )

    # Assign to buses
    gdf_genX_unique_locs = nearest_neighbor_lat_lon(
        gdf_genX_unique_locs.rename(columns={"genX_zone": "zone"}),
        match_zones=match_zones,
        PV_bus_only=PV_bus_only,
    )

    # Merge with timeseries and sum by bus
    df_out = (
        pd.merge(df, gdf_genX_unique_locs, on=["latitude", "longitude"], how="outer")
        .groupby(["bus_id", "datetime"])[["power_MW"]]
        .sum()
    )

    # Return
    return df_out
