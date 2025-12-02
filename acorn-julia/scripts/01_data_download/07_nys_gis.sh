#!/bin/bash

#####################################
# Download NYS county shapefile
# Taken from census.gov
# URL: https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html
#####################################
# Base URL
BASE_URL="https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_5m.zip"

# Output directory
OUTPUT_DIR="/home/fs01/jl2966/acorn-julia2/data/nys/"
mkdir -p "$OUTPUT_DIR"

# Download
wget -q --show-progress -P "$OUTPUT_DIR" "$BASE_URL"

# Unzip
unzip "$OUTPUT_DIR/cb_2018_us_county_5m.zip" -d "$OUTPUT_DIR/gis"

# Remove zip file
rm "$OUTPUT_DIR/cb_2018_us_county_5m.zip"