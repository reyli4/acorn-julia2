#!/bin/bash
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=1            # Number of tasks per node
#SBATCH --mem=3GB                      # Memory per node
#SBATCH --time=02:00:00                # Maximum run time


###################################
# Download NREL WTK data
# We use the techno-economic release
# More info: https://www.nrel.gov/grid/wind-toolkit
###################################

echo "Job started on `hostname` at `date`"

############################
# Speicify save path
OUT_DIR="/home/fs01/jl2966/acorn-julia2/data/nrel/wtk"
############################
cd $OUT_DIR

PREFIX="https://nrel-pds-wtk.s3.amazonaws.com/wtk-techno-economic/pywtk-data/met_data"
SAVE_DIR="met_data"
CSV_FILE="wtk_site_metadata.csv"

# Process each line of the CSV
while IFS="," read -r id lon lat state county col6 col7 col8 col9 col10 col11 filepath; do
    # Check if state is New York
    if [ "$state" = "New York" ]; then
        echo "Found New York entry: $id"

        # Extract just the filename portion (after the last slash)
        filepath=$(echo "$filepath" | awk -F, '{print $NF}' | tr -d '\r\n')
        filename=$(basename "$filepath")

        # Download the file
        wget -nc "${PREFIX}/$filepath" -O "${SAVE_DIR}/$filename"
    fi
done < "$CSV_FILE"

echo "Job Ended at `date`"