#!/usr/bin/env python3
from glob import glob
import random

import numpy as np
import xarray as xr
import salem
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from python import wind_utils as wu
from python import climate_utils as cu
from python import utils as pu

# Get TGW input wind data
start_year = 2007
end_year = 2013
wtk_keep_every = 5

climate_paths = cu.generate_tgw_filelist('historical_1980_2019', [start_year, end_year])

df = wu.prepare_wind_data(
    climate_paths = climate_paths,
    wind_vars = ['U10', 'V10'],
    lat_name="south_north",
    lon_name="west_east",
    curvilinear=True,
    parallel=True,
    wtk_keep_every=wtk_keep_every,
)

# Compute and store the stability coefficients
res = wu.get_stability_coefficients(df, 'ws', 'ws_wtk').reset_index()
res.to_csv(f"{pu.project_path}/data/wind/models/tgw_stability_coefficients_{start_year}-{end_year}_every{wtk_keep_every}.csv", index=False)
print(f"Wrote: {pu.project_path}/data/wind/models/tgw_stability_coefficients_{start_year}-{end_year}_every{wtk_keep_every}.csv")
