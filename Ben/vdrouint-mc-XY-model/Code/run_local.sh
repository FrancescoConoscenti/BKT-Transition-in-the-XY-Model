#!/bin/bash

# === Simulation Parameters ===
Lx=32          # System size (linear dimension)
Tmin=0.5       # Minimum temperature
Tmax=1.5       # Maximum temperature
NumberTemp=20  # Number of temperatures between Tmin and Tmax
Type="WOLFF"   # Algorithm type ("WOLFF" or "METRO")
NumCores=4     # Number of cores for parallelization (adjust based on your machine)
PreTherm=1000  # Number of pre-thermalization steps (initial discard)
Therm=5000     # Number of thermalization steps (discard at each temp)
Measure=10000  # Number of measurement steps

# === Execution ===

echo "Starting XY model simulation..."
# Run the Monte Carlo simulation (output redirected to log.txt)
python3 ./mcpt-xy.py ${Lx} ${Tmin} ${Tmax} ${NumberTemp} ${Type} ${NumCores} ${PreTherm} ${Therm} ${Measure} >& log.txt

echo "Simulation finished. Starting data analysis..."
# Run the data processing script (output redirected to log_analysis.txt)
python3 ./all_data_process.py ${Lx} >& log_analysis.txt

echo "Analysis finished. Check log.txt and log_analysis.txt for details."
echo "Results should be saved in data files (check mcpt-xy.py and all_data_process.py for naming conventions)."
echo "You can visualize results using simple_plot.ipynb." 