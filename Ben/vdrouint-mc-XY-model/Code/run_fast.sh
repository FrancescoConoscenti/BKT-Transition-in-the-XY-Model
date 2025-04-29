#!/bin/bash

# === MINIMAL SIMULATION PARAMETERS FOR QUICK RUN ===
Lx=8            # Reduced system size (8x8 instead of 32x32)
Tmin=0.8        # Focus near the BKT transition (expected ~0.89)
Tmax=1.0        # Smaller temperature range
NumberTemp=3    # Just 3 temperature points
TempRangeType=1 # Linear temperature spacing
NumCores=4      # Use same number of cores
PreTherm=100    # 10x fewer pre-thermalization steps
Therm=500       # 10x fewer thermalization steps
Measure=1000    # 10x fewer measurement steps

# === Create data directory structure ===
# Important: The simulation hardcodes the data directory name as "testL=${Lx}" (not "testL=${Lx}_fast")
DATA_DIR="./testL=${Lx}"

echo "Starting a FAST minimal XY model simulation..."
echo "System size: ${Lx}x${Lx}, Temperatures: ${Tmin}-${Tmax} (${NumberTemp} points)"
echo "Steps: PreTherm=${PreTherm}, Therm=${Therm}, Measure=${Measure}"

# Check if data already exists to avoid running simulation again
if [ -d "$DATA_DIR" ] && [ -f "${DATA_DIR}/variables.data" ]; then
    echo "Data directory already exists. Skipping simulation."
else
    # === Run the minimal simulation ===
    echo "Running minimal simulation..."
    # The simulation will create the data directory
    python3 ./mcpt-xy.py ${Lx} ${Tmin} ${Tmax} ${NumberTemp} ${TempRangeType} ${NumCores} ${PreTherm} ${Therm} ${Measure} >& log_fast.txt
    
    # Check if simulation completed successfully
    if [ ! -d "$DATA_DIR" ] && [ ! -f "${DATA_DIR}/variables.data" ]; then
        # Check if files were created in current directory (this is likely)
        if [ -f "./variables.data" ] && [ -f "./configatT=08000.data" ]; then
            echo "Data files created in current directory. Moving to ${DATA_DIR}..."
            mkdir -p "${DATA_DIR}"
            mv variables.data configatT=*.data outputatT=*.data "${DATA_DIR}/"
        else
            echo "ERROR: Simulation failed to create expected data"
            echo "Check log_fast.txt for details."
            exit 1
        fi
    fi
fi

echo "Simulation data available. Running quick analysis..."

# Run the data processing script
python3 ./all_data_process.py ${Lx} >& log_analysis_fast.txt

# Check if analysis was successful
if [ ! -d "${DATA_DIR}finalData" ]; then
    echo "ERROR: Data analysis failed. Check log_analysis_fast.txt for details."
    exit 1
fi

echo "Fast run complete! Check results in ${DATA_DIR}finalData/"
echo "You can visualize results using simple_plot.ipynb." 