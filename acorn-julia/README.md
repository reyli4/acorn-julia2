# ACORN: A Climate-informed Operational Reliability Network

This repo contains a Julia implementation of the New York State power system model developed by the [Anderson Research Group](https://andersonenergylab-cornell.github.io/). For past work using ACORN, see [Liu et al. (2024)](https://arxiv.org/abs/2307.15079) and [Kabir et al. (2024)](https://doi.org/10.1016/j.renene.2024.120013). ACORN was validated using NYISO data from 2019, outlined in [Liu et al. (2023)](https://ieeexplore.ieee.org/document/9866561).

## Dependencies
This repo uses Python (mainly for data processing) and Julia (for running ACORN). Python dependencies are given in `pyproject.toml` and can be installed into a virtual environment using a tool like [uv](https://docs.astral.sh/uv/). Julia dependencies are given in `Project.toml`.

## Docs
See the following files for more information on the model construction and data processing: 
- [ACORN](docs/acorn.md)
- [Load modeling](docs/load_modeling.md)
- [Solar + wind modeling](docs/gen_modeling.md)
- [Coupling with GenX](docs/coupling_to_GenX.md) (including hydro, natural gas, nuclear matching)
- [Future work](docs/todo.md)

## Reproducing ACORN runs

To re-download and re-process all necessary input datasets, run the scripts and notebooks in the `scripts` directory in order. The paths to data/code are somewhat self-contained but you will need to update the paths in `src/python/utils.py` and `src/julia/utils.jl`, as well as the paths in all the bash scipts in `scripts/01_data_download`. The IM3 TGW data can be downloaded from Globus, which you will need to do manually (see `04_im3_tgw.md`). 

For constructing the input datasets for a specific ACORN run where the parameters are informed by GenX outputs, you need to then run through the `construct_inputs.ipynb` notebook in the corresponding run folder. User-defined parameters for each run are set in the `config.yml` file. The only parameter that might need updating in the  `construct_inputs.ipynb` notebook is the `genX_max_load` -- you will need access to the GenX input files in order to set this correctly. Otherwise, `construct_inputs.ipynb` performs each matching step as documented in the [Coupling with GenX](docs/coupling_to_GenX.md) file. These steps could be automated into a script but there is some randomness around generating wind/solar sites so it's good to view the location plots as they are produced. You can then run the `run_acorn.sh` script which will submit a SLURM script to run ACORN. Make sure Gurobi is available (on Hopper, `module load gurobi/11.0.3`) and you have the correct [license](https://www.gurobi.com/features/academic-wls-license/).

The initial experiments were performed on the Cornell Hopper cluster --- I tried to make all paths general but some code may end up being specific to that system. On Hopper, a version of this repository is available in the `vs498_0001` shared directory with all the input and output files that were too large for GitHub. A reduced form version with some data removed is available [at this link](https://drive.google.com/file/d/1kD-hCTRH4RHN8uew-LTRdBV2TIucItWY/view?usp=sharing) (around 16GB uncompressed).