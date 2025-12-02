#!/bin/bash
#SBATCH --job-name=acorn_run
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --time=06:00:00
#SBATCH --mem=20GB
#SBATCH --output=logs/out-%A.txt
#SBATCH --error=logs/err-%A.txt

# Start time
echo "Starting job at $(date)"

# Load Gurobi
module load gurobi/11.0.3
export GRB_THREADS=10

# Constants
PROJECT_DIR="/home/fs01/dcl257/projects/acorn-julia"
RUN_DIR=$PWD

echo "Project directory: $PROJECT_DIR"
echo "Run directory: $RUN_DIR"

# Run script
# ------------
# NYISO ONLY
julia $PROJECT_DIR/scripts/04_run_acorn.jl \
--project-dir $PROJECT_DIR \
--run-dir $RUN_DIR \
--if_lim_name vivienne_2023_paper \
--exclude_external_zones 1 \
--include_new_hvdc 0 \
--save_name nyiso_only

# NYISO + EXTERNAL ZONES
julia $PROJECT_DIR/scripts/04_run_acorn.jl \
--project-dir $PROJECT_DIR \
--run-dir $RUN_DIR \
--if_lim_name vivienne_2023_paper \
--exclude_external_zones 0 \
--include_new_hvdc 0 \
--save_name external_zones

# ------------

# End time
echo "Ending job at $(date)"

exit 0