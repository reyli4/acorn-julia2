#!/bin/bash

# NREL Residential Stock Data Download Script
# Downloads timeseries aggregates for NYISO region
# More info: https://resstock.nrel.gov/datasets
# We are using the 2024.2 release

# Base URL
BASE_URL="https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2024/resstock_amy2018_release_2/timeseries_aggregates/by_iso_rto_region"

# Array of unit types
UNIT_TYPES=(
    "mobile_home"
    "multi-family_with_2_-_4_units"
    "multi-family_with_5plus_units"
    "single-family_attached"
    "single-family_detached"
)

# Create output directory
OUTPUT_DIR="/home/fs01/jl2966/acorn-julia2/data/nrel/resstock"
mkdir -p "$OUTPUT_DIR"

echo "Starting download of NREL building stock data..."
echo "Output directory: $OUTPUT_DIR"
echo ""

# First download the metadata
wget -q --show-progress -P "$OUTPUT_DIR" "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2024/resstock_amy2018_release_2/metadata/baseline.parquet"

# Loop through upgrade numbers (0 to 16)
for upgrade in {0..16}; do
    echo "Processing upgrade $upgrade..."
    
    # Loop through unit types
    for unit_type in "${UNIT_TYPES[@]}"; do
        # Construct filename and URL
        filename="up$(printf "%02d" $upgrade)-nyiso-${unit_type}.csv"
        url="${BASE_URL}/upgrade%3D${upgrade}/iso_rto_region%3DNYISO/${filename}"
        
        echo "  Downloading: $filename"
        
        # Download with wget
        wget -q --show-progress -P "$OUTPUT_DIR" "$url"
        
        # Check if download was successful
        if [ $? -eq 0 ]; then
            echo "    ✓ Success"
        else
            echo "    ✗ Failed to download $filename"
        fi
    done
    echo ""
done

echo "Download complete!"
echo "Files saved in: $OUTPUT_DIR"

# Show summary
total_files=$(ls -1 "$OUTPUT_DIR"/*.csv 2>/dev/null | wc -l)
echo "Total files downloaded: $total_files"