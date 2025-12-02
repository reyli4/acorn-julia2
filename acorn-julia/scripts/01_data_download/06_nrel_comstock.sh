#!/bin/bash

# NREL Commercial Building Stock Data Download Script
# Downloads timeseries aggregates for NY state
# We are using the 2024_1 release
# More info: https://comstock.nrel.gov/page/datasets

# Base URL
BASE_URL="https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2024/comstock_amy2018_release_1/timeseries_aggregates/by_iso_rto_region"

# Array of commercial building types
BUILDING_TYPES=(
    "fullservicerestaurant"
    "hospital"
    "largehotel"
    "largeoffice"
    "mediumoffice"
    "outpatient"
    "primaryschool"
    "quickservicerestaurant"
    "retailstandalone"
    "retailstripmall"
    "secondaryschool"
    "smallhotel"
    "smalloffice"
    "warehouse"
)

# Create output directory
OUTPUT_DIR="/home/fs01/jl2966/acorn-julia2/data/nrel/comstock"
mkdir -p "$OUTPUT_DIR"

echo "Starting download of NREL commercial building stock data..."
echo "Output directory: $OUTPUT_DIR"
echo ""

# First download the metadata
wget -q --show-progress -P "$OUTPUT_DIR" "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2024/comstock_amy2018_release_1/metadata/baseline.parquet"

# Also need the measure descriptions
wget -q --show-progress -P "$OUTPUT_DIR" "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2024/comstock_amy2018_release_1/measure_name_crosswalk.csv"

# Loop through upgrade numbers (0 to 39)
for upgrade in {0..39}; do
    echo "Processing upgrade $upgrade..."
    
    # Loop through building types
    for building_type in "${BUILDING_TYPES[@]}"; do
        # Construct filename and URL
        filename="up$(printf "%02d" $upgrade)-nyiso-${building_type}.csv"
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