# Generation modeling

We generally follow [Doering et al. (2023)](https://doi.org/10.1093/ooenergy/oiad003) for modeling and bias-correcting the solar and wind generation.

## Solar

We generate bias-correction factors based on correcting the simulated solar generation agsinst the [NREL SIND data](https://www.nrel.gov/grid/solar-power-data). We follow [Doering et al. (2023)](https://doi.org/10.1093/ooenergy/oiad003) by simulating solar production as a function of incoming shortwave radiation and temperature, based on [Perpiñan et al. (2007)](https://onlinelibrary.wiley.com/doi/10.1002/pip.728). The bias-correction optimizes the `beta` parameter (controls the temperature sensitivity of solar production) alongside additive factors based on hour of the day and month of the year. 

### Wind

We generate stability coefficients for interpolating the climate data wind speeds (typically at 10m) to wind turbine hub heights (assumed to be 100m). Hub height wind speeds are taken from (NREL Wind Toolkit)[https://www.nrel.gov/grid/wind-toolkit] (specifically the Techno-Economic subset). We follow the scaling expression (Eq. 9) in [Doering et al. (2023)](https://doi.org/10.1093/ooenergy/oiad003) to perform the interpolation, allowing stability coefficents to vary by month of the year, hour of the day, and zone. Note that for computational efficiency, you can choose to look at every n-th WTK site. Currently the default value is to consider every 10th site. When translating wind speed into normalized power production, we need to assume the IEC power curve, taken from Table 2 in [NREL Technical Report: Validation of Power Output for the WIND Toolkit (2014)](https://docs.nrel.gov/docs/fy14osti/61714.pdf). By default, we use `iec` for onshore wind and `offshore` for offshore locations.
