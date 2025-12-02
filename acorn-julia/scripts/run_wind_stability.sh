#!/bin/bash
#SBATCH --job-name=wind_stability_run
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --time=04:00:00
#SBATCH --mem=20GB
#SBATCH --output=logs/out-%A.txt
#SBATCH --error=logs/err-%A.txt

# activate your env
source ~/opt/miniconda3/etc/profile.d/conda.sh
conda activate acorn-py312

cd /home/fs01/jl2966/acorn-julia2

# Make sure logs and output dirs exist
mkdir -p logs data/wind/models

# Let Python find acorn-julia/src/python as the package "python"
export PYTHONPATH="${PWD}/acorn-julia/src:${PYTHONPATH}"


python acorn-julia/scripts/compute_wind_stability.py
