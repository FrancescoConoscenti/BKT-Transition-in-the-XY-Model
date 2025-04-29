#!/bin/bash

# === Simulation Parameters ===
Lx=32          # System size (linear dimension)
Tmin=0.5       # Minimum temperature
Tmax=1.5       # Maximum temperature
NumberTemp=20  # Number of temperatures between Tmin and Tmax
TempRangeType=1  # How to space temperatures: 0=geometric, 1=linear
NumCores=4     # Number of cores for parallelization (adjust based on your machine)
PreTherm=1000  # Number of pre-thermalization steps (initial discard)
Therm=5000     # Number of thermalization steps (discard at each temp)
Measure=10000  # Number of measurement steps

# === Create data directory structure ===
# The expected data directory that will be created by mcpt-xy.py
DATA_DIR="./testL=${Lx}"

# === Run Simulation (Only if data doesn't exist) ===
if [ ! -d "$DATA_DIR" ] || [ ! -f "${DATA_DIR}/variables.data" ]; then
    echo "Data not found. Starting XY model simulation..."
    # Run the Monte Carlo simulation (output redirected to log.txt)
    python3 ./mcpt-xy.py ${Lx} ${Tmin} ${Tmax} ${NumberTemp} ${TempRangeType} ${NumCores} ${PreTherm} ${Therm} ${Measure} >& log.txt
    
    # Check if simulation created the data directory
    if [ ! -d "$DATA_DIR" ] || [ ! -f "${DATA_DIR}/variables.data" ]; then
        echo "ERROR: Simulation failed to create expected data in ${DATA_DIR}"
        echo "Check log.txt for details."
        exit 1
    fi
else
    echo "Data directory ${DATA_DIR} already exists. Skipping simulation."
fi

# === Run Data Analysis ===
echo "Starting data analysis..."
# Run the data processing script (output redirected to log_analysis.txt)
python3 ./all_data_process.py ${Lx} >& log_analysis.txt

# Check analysis exit status
if [ $? -ne 0 ]; then
    echo "ERROR: Data analysis failed. Check log_analysis.txt for details."
    exit 1
fi

echo "Analysis finished. Check log.txt and log_analysis.txt for details."
echo "Results should be saved in ${DATA_DIR}finalData/ directory."
echo "You can visualize results using simple_plot.ipynb." 