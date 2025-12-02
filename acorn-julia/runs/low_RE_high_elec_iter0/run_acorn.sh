#!/bin/bash
#SBATCH --job-name=acorn_run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=04:00:00
#SBATCH --mem=20G
#SBATCH --output=logs/out-%A.txt
#SBATCH --error=logs/err-%A.txt

echo "Starting job at $(date)"
module load gurobi/11.0.3

# paths
PROJECT_DIR="/home/fs01/jl2966/acorn-julia2/acorn-julia"                     # repo root (data/config live here)
JULIA_PROJ="/home/fs01/jl2966/acorn-julia2/acorn-julia"          # Julia env (Project.toml)
SCRIPT="${JULIA_PROJ}/scripts/04_run_acorn.jl"
RUN_DIR="$PWD"                                                    # scenario folder

echo "Project directory: $PROJECT_DIR"
echo "Run directory: $RUN_DIR"

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-10}
export OMP_NUM_THREADS=$JULIA_NUM_THREADS
export MKL_NUM_THREADS=$JULIA_NUM_THREADS

julia --project="$JULIA_PROJ" "$SCRIPT" \
  --project-dir "$PROJECT_DIR" \
  --run-dir "$RUN_DIR" \
  --if_lim_name vivienne_2023_paper \
  --exclude_external_zones 1 \
  --include_new_hvdc 0 \
  --save_name nyiso_only

julia --project="$JULIA_PROJ" "$SCRIPT" \
  --project-dir "$PROJECT_DIR" \
  --run-dir "$RUN_DIR" \
  --if_lim_name vivienne_2023_paper \
  --exclude_external_zones 0 \
  --include_new_hvdc 0 \
  --save_name external_zones

echo "Ending job at $(date)"
