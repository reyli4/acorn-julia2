# Notes on coupling GenX to ACORN

ACORN can accept the outputs from a GenX capacity expansion run to inform which generation resources it uses. More specifically, GenX will specify the zonal distribution of natural gas, on/off-shore wind, solar UPV and DPV, nuclear, hydro, battery storage, as well as the peak load over the capacity expansion period. We need to translate these generation capacities into specific generators assigned to specific buses, and scale ACORN's modeled peak future load to be in line with what GenX has optimized against.

## Load
Load in ACORN is modeled as a combination of baseline load (from NYISO historical data) plus expected future electrification (from NREL's ResStock and ComStock data) -- see the load modeling documentation for more details. GenX provides load input timeseries for representative periods throughout the year. In general the naive load increases from ResStock and ComStock (but mostly ResStock) are much larger than what GenX assumes, so we scale down the ResStock peak load over the entire time period be half of the GenX peak load. When added to the baseline and ComStock load, this usually results in the ACORN average load being comfortably below the GenX maximum, but the ACORN peak load is larger than the GenX peak. Since GenX only includes representative load periods and ACORN has 40 years of hourly data (using the TGW data), we would expect ACORN to exhibit wider extrema. However, this procedure is somewhat ad-hoc currently as we can scale as much or as little as we like. It woud be good to align both GenX and ACORN with (e.g.) the NYISO Golden Book forecasts or some other source. 

Also note we do not include future load from EVs, as was done in [Lui et al. (2024)](https://arxiv.org/abs/2307.15079). This could be added in future.  

## Thermal
Natural gas zonal capacities from GenX and mapped to specific thermal generators in ACORN. The baseline capacities (i.e. in the ACORN `gen_prop` file and GenX's assumed existing capacity) do not match, so we retire plants when necessary following a highest-cost-first protocol, and duplicate plants when ACORN needs more capacity to match GenX. We are generally able to match the zonal capacities to within a few hundred MW. Note that ACORN also has other fossil fuel generators (e.g. using kerosene or fuel oil) that have no equivalent in GenX, so they are all retired.

## Nuclear
GenX shows around 2700 MW of nuclear capacity in zone C — this corresponds to three specific plants in the ACORN generator list which have a capacity of 2736 MW. We retire the other plants in the ACORN generator list.

## Hydro
GenX includes `hydroelectric` and `hydroelectric_storage` resources. Using all hydro plants in the GO-DEEEP hydro TGW dataset, we match the `hydroelectric` category faily closely. ACORN does not have an explicit representation of hydro storage, so this is a mismatch (of around 2000 MW). 

## Storage
Storage capacity from GenX is mapped to assigned to random buses in ACORN, where the user can determine how many sites per zone are generated. The storage and charging limits are taken from `EndEnergyCap` and `EndCap` respectively in the GenX output files. Both the storage and charging limits are scaled down if more than one site per zone is assumed — this keeps the implied battery duration constant.

## Solar
The downscaled *new* solar UPV is passed seamlessly from GenX to ACORN. The downscaled GenX outputs provide lat/lon coordinates and nameplate capacities of specific sites. 

The *existing* solar UPV and DPV in GenX is only given at the zonal level, so we generate random points within the correct zone and then assign the generation timeseries to the nearest bus. For DPV, we generate 10 ``sites" per zone since we would expect DPV to be more diffuse. For UPV we generate one site per zone.

## Wind
For existing onshore wind, GenX again only provides the zonal capacity. We disaggregate this by generating one site per zone and assigning the wind generation timeseries to the nearest bus.

For existing *offshore* wind, GenX provided many "sites" per zone (usually zone J — NYC). We generate quasi-randomly distributed offshore sites in the southeast direction from zone J, one for each "site" in the GenX output file. These are then allocated to the nearest bus, which will be one of the two in NYC.

For calculating the power generation in ACORN, we need to assume a specific power curve — we use NREL data for this, specifically the `offshore`" category for offshore sites and `iec1` for onshore (although they are fairly similar).

## Other
The `biomass`, `water_heating`, and `transportation` categories in GenX have no equivalents (currently) in ACORN, so these are neglected. In the end, the total NYISO capacity in ACORN is usually a few GW less than GenX. 