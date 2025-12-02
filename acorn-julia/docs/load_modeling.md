# Load modeling

We model three distinct load sources -- the "baseline" load from historical NYISO data, plus potential future load increases due to electrification of residential and commerical buildings, from NREL ResStock and ComStock data, respectively.

## Baseline model

Historical load from 2020-2023 is provided by NYISO at a zonal, hourly scale. We train an multi-layer perceptron neural network (NN) to predict this load as a function of hourly zonal temperature, hour of the day, day of the year, and the previous day's average load. The NN predicts all zones simultaneously in order to capture important zonal correlations. Out-of-sample R-squared is generally above 90% across zones.

The baseline load is disaggregated to the bus level using the historical load ratios from the original 140-bus NPCC system. Note that only buses with non-zero load ratio based on the NPCC system are considered, following [Liu et al. (2024)](https://arxiv.org/abs/2307.15079).

## Electrification upgrades

We rely on NREL ResStock and Comstock data to model the temperature dependence of the New York state (NYS) building stock under possible future electrification scenarios. Specifically, we use the [ResStock 2024.2 release](https://resstock.nrel.gov/datasets) and the [ComStock 2024_1 release](https://comstock.nrel.gov/page/datasets), both of which rely on the *Actual Meteorological Year 2018*. The NREL data provides energy savings (i.e. increased electricity demand) of different building types under various upgrades. We again use a multi-layer perceptron to model the temperature dependence of this electricity load, fitting a separate model per building type and upgrade. Predictor variables include the hourly temperature, hour of the day, and the previous day's average temperature. Note that we fit this model to the ResStock/ComStock data aggregated across all of NYS and assume the inferred temperature relationships hold at finer scales. 

Once the model has been constructed, we can use it to predict hourly total NYS electricity load due to building electrification. We need to disaggregate this to the bus level for ACORN -- we first disaggregate to the county level using NREL building weight files, which provide a normalized weight of the number of building types in each county. Then to disaggregate to the bus level, the load is distributed evenly for counties with one or more buses, and the loads for counties without a bus is distributed to the nearest single bus. 

 Note also that NREL provides many possible electrification upgrades for both residential and commerical buildings, so we need to choose one of each for the ACORN runs — the defaults are upgrade 1 for ResStock (ENERGY STAR heat pump with elec backup) and upgrade 31 (Package 3 - Wall and Roof Insulation, New Windows, LED Lighting and ASHP-Boiler) for ComStock. The ResStock load is typically much larger than ComStock.