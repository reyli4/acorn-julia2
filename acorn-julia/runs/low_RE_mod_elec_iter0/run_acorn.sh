#!/bin/bash
#SBATCH --job-name=acorn_run
#SBATCH --nodes=10
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=20
#SBATCH --time=06:00:00
#SBATCH --mem=20GB
#SBATCH --output=logs/out-%A.txt
#SBATCH --error=logs/err-%A.txt


# Start time
echo "Starting job at $(date)"

# Load Gurobi
module load gurobi/13.0.0
export GRB_THREADS=20

# Constants
PROJECT_DIR="/home/fs01/jl2966/acorn-julia2/acorn-julia"
RUN_DIR="$PROJECT_DIR/runs/low_RE_mod_elec_iter0"

echo "Project directory: $PROJECT_DIR"
echo "Run directory: $RUN_DIR"

# Run script
# ------------
# NYISO ONLY
CPUS=${SLURM_CPUS_PER_TASK:-20}
srun -c ${CPUS} julia $PROJECT_DIR/scripts/04_run_acorn.jl \
--project-dir $PROJECT_DIR \
--run-dir $RUN_DIR \
--if_lim_name vivienne_2023_paper \
--exclude_external_zones 1 \
--include_new_hvdc 0 \
--save_name nyiso_only

# NYISO + EXTERNAL ZONES
srun -c ${CPUS} julia $PROJECT_DIR/scripts/04_run_acorn.jl \
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








    






