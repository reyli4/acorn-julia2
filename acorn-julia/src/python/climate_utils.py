import geopandas as gpd
import pandas as pd
import salem
import numpy as np
from glob import glob

from python.utils import project_path, tgw_path


def generate_tgw_filelist(climate_scenario_years: str, years="all"):
    """
    Get list of TGW climate data
    """
    # Get info
    climate_scenario, year_start, year_end = climate_scenario_years.split("_")

    if years == "all":
        climate_paths = np.sort(
            glob(
                f"{tgw_path}/{climate_scenario_years}/hourly/tgw_wrf_{climate_scenario}_hourly_*.nc"
            )
        )
    else:
        climate_paths = np.sort(
            [
                glob(
                    f"{tgw_path}/{climate_scenario_years}/hourly/tgw_wrf_{climate_scenario}_hourly_{year}*.nc"
                )
                for year in range(int(years[0]), 1 + int(years[1]))
            ]
        ).flatten()

    return climate_paths


def tgw_to_zones(
    tgw_file_path: str,
    tgw_vars: list[str],
    nyiso_zone_shp_path: str = f"{project_path}/data/nyiso/gis/NYISO_Load_Zone_Dissolved.shp",
) -> pd.DataFrame:
    """
    Converts TGW output variables to NYISO load zone (area weighted) averages.

    Code based on: https://github.com/IMMM-SFA/im3components/blob/main/im3components/wrf_to_tell/wrf_tell_counties.py
    """

    # Read TGW
    tgw = salem.open_wrf_dataset(tgw_file_path)[tgw_vars]
    tgw = tgw.where((tgw.lat > 40) & (tgw.lon > -80), drop=True)  # subset NYS

    tgw_crs = tgw.pyproj_srs
    geometry = tgw.salem.grid.to_geometry().geometry

    # Read NYISO zones
    gdf = gpd.read_file(nyiso_zone_shp_path).to_crs(tgw_crs)
    # Normalize zone column name to 'ZONE'
    zone_candidates = ["ZONE", "zone", "zone_name"]
    zcol = next((c for c in zone_candidates if c in gdf.columns), None)
    if zcol is None:
        raise KeyError(
            f"Expected one of {zone_candidates} in {nyiso_zone_shp_path}, found {list(gdf.columns)}"
        )
    gdf = gdf.rename(columns={zcol: "ZONE"})

    # Get the intersection df
    tgw_df_single = tgw.isel(time=0).to_dataframe().reset_index(drop=True)
    tgw_df_single = gpd.GeoDataFrame(tgw_df_single, geometry=geometry).set_crs(tgw_crs)
    intersection_single = gpd.overlay(gdf, tgw_df_single, how="intersection")

    # Convert TGW to dataframe
    tgw_df = tgw.to_dataframe().reset_index()

    # Merge
    intersection = pd.merge(
        tgw_df[["time", "lat", "lon"] + tgw_vars],
        intersection_single[["ZONE", "lat", "lon", "geometry"]],
        how="inner",
        on=["lat", "lon"],
    )
    intersection = gpd.GeoDataFrame(intersection, geometry=intersection["geometry"])

    # Area weighting
    intersection["area"] = intersection.area
    intersection["weight"] = intersection["area"] / intersection[
        ["ZONE", "time", "area"]
    ].groupby(["ZONE", "time"]).area.transform("sum")

    # Compute area-weighted average
    out = (
        intersection[tgw_vars]
        .multiply(intersection["weight"], axis="index")
        .join(intersection[["ZONE", "time"]])
        .groupby(["ZONE", "time"])
        .sum()
    )

    # 🔧 make zone/time real columns with the names the model expects
    out.reset_index().rename(columns={"ZONE": "zone"})
    return out
