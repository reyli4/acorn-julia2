#!/bin/bash

######################################################################
# NREL SIND data from https://www.nrel.gov/grid/solar-power-data.html
# NOTE: Actual is real power output, DA is day ahead forecast, 
# HA4 is 4 hour ahead forecast
######################################################################

# Specify output dir
OUT_DIR="/home/fs01/jl2966/acorn-julia2/data/nrel/sind"

# Download
wget https://www.nrel.gov/grid/assets/downloads/ny-pv-2006.zip -P $OUT_DIR

# Unzip
unzip $OUT_DIR/ny-pv-2006.zip -d $OUT_DIR/ny-pv-2006