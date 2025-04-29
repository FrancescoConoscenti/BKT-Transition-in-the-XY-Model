#!/bin/bash

# === Simulation Parameters (Double Layer/Modified Model) ===
N=32           # System size (linear dimension)
J2=0.0         # Parameter J2 (adjust as needed)
Lambda=1.0     # Parameter Lambda (adjust as needed)
NumberTemp=20  # Number of temperatures
NumCores=4     # Number of cores (adjust based on your machine)
Tmin=0.5       # Minimum temperature
Tmax=1.5       # Maximum temperature
# Note: The original script didn't specify Therm/Measure steps for mcptdoublel.py
# You might need to check mcptdoublel.py or add them if required.

# === Execution ===

echo "Starting Double Layer XY model simulation..."
# Run the Monte Carlo simulation (output redirected to log_doublel.txt)
python3 ./mcptdoublel.py ${N} ${J2} ${Lambda} ${NumberTemp} ${NumCores} ${Tmin} ${Tmax} >& log_doublel.txt

echo "Simulation finished. Starting data analysis..."
# Run the data processing script (output redirected to log_analysis_doublel.txt)
python3 ./all_data_process.py ${N} ${J2} ${Lambda} >& log_analysis_doublel.txt

echo "Analysis finished. Check log_doublel.txt and log_analysis_doublel.txt for details."
echo "Results should be saved in data files." 