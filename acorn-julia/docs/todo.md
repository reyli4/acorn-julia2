1. *Representation of external zones in ACORN*
    - Currently, no external loads or generators are modeled. There is the option to remove all external buses and model a "NYISO-only" system, or to keep the external buses for the added network connections, which can be helpful for moving power around.
    - I'm not sure how external loads are modeled in previous works -- in [Liu et al. (2024)](https://arxiv.org/abs/2307.15079) it seems to vary across the year and by climate change scenario, but not year-to-year.
    - I'm also unsure where parameters of the external or "import" generators come from.
2. *Additional hydro scenarios*
3. *Future electric vehicle loads*
4. *Additional climate scenarios/data*
    - Current experiments are all based on the [IM3/HyperFACETS TGW data](https://tgw-data.msdlive.org/).
    - Other interesting-looking datasets include NREL's [NCDB](https://climate.nrel.gov/about/what-is-the-ncdb) and [Sup3rCC](https://data.openei.org/submissions/5839).
    - I tried to make the climate processing scripts somewhat generalized but additional work will likely be required here.