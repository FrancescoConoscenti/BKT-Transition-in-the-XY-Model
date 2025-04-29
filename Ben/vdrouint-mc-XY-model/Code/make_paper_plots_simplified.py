#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import subprocess
import argparse
import time
from scipy.optimize import curve_fit
from datetime import datetime

# Set matplotlib parameters for paper-quality plots
rc('font', **{'family': 'sans-serif', 'size': 10})
rc('text', usetex=True)
plt.rcParams['figure.figsize'] = (12, 4)

# Define colors for plots
colors = {
    'L6': 'red',
    'L8': 'orange',
    'L10': 'green',
    'L12': 'blue'
}

# System sizes to simulate - using very small sizes for quick results
system_sizes = [6, 8, 10, 12]

# Temperature range for simulation - fewer points focused around transition
temp_min = 0.7  # Starting closer to transition
temp_max = 1.1  # Ending just after transition
num_temps = 10  # Very few temperature points

# Ultra minimal simulation steps
simulation_steps = {
    'pretherm': 100,    # Minimal pre-thermalization
    'therm': 500,       # Minimal thermalization
    'measure': 1000     # Minimal measurement
}

def ensure_dir(dir_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")
    return dir_path

def run_simulation(L, temp_min, temp_max, num_temps, temp_type, 
                   pretherm_steps, therm_steps, measure_steps, num_cores, force=False):
    """Run the Monte Carlo simulation for a specific system size"""
    data_dir = f"./testL={L}"
    log_file = f"log_L{L}_quick.txt"
    
    # Skip if data already exists and force=False
    if os.path.exists(data_dir) and os.path.exists(f"{data_dir}/variables.data") and not force:
        print(f"Data for L={L} already exists. Skipping simulation.")
        return True
    
    print(f"Starting quick simulation for L={L}...")
    command = [
        "python3", "./mcpt-xy.py", 
        str(L), str(temp_min), str(temp_max), str(num_temps), str(temp_type),
        str(num_cores), str(pretherm_steps), str(therm_steps), str(measure_steps)
    ]
    
    start_time = time.time()
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(command, stdout=f, stderr=f)
            
        if result.returncode != 0:
            print(f"Error running simulation for L={L}. Check {log_file}")
            return False
        
        elapsed_time = time.time() - start_time
        print(f"Simulation for L={L} completed in {elapsed_time:.2f} seconds")
        return True
    except Exception as e:
        print(f"Error running simulation for L={L}: {str(e)}")
        return False

def run_analysis(L, force=False):
    """Run the data analysis for a specific system size"""
    data_dir = f"./testL={L}finalData"
    log_file = f"log_analysis_L{L}_quick.txt"
    
    # Skip if data already exists and force=False
    if os.path.exists(data_dir) and not force:
        print(f"Analysis data for L={L} already exists. Skipping analysis.")
        return True
    
    print(f"Starting data analysis for L={L}...")
    command = ["python3", "./all_data_process.py", str(L)]
    
    start_time = time.time()
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(command, stdout=f, stderr=f)
            
        if result.returncode != 0:
            print(f"Error running analysis for L={L}. Check {log_file}")
            return False
        
        elapsed_time = time.time() - start_time
        print(f"Analysis for L={L} completed in {elapsed_time:.2f} seconds")
        return True
    except Exception as e:
        print(f"Error running analysis for L={L}: {str(e)}")
        return False

def load_data(L):
    """Load the processed data for a specific system size"""
    base_path = f"./testL={L}finalData"
    
    if not os.path.exists(f"{base_path}/variables.data"):
        raise FileNotFoundError(f"Data for L={L} not found. Run simulation and analysis first.")
    
    # Load temperature and thermodynamic data
    variables = np.loadtxt(f"{base_path}/variables.data")
    temps = variables[3:]
    thermo_data = np.loadtxt(f"{base_path}/thermo_output.data")
    
    # Get number of temperature points
    nt = len(temps)
    
    return temps, nt, thermo_data

def extract_T_KT(temps, spin_stiffness, errors):
    """Extract T_KT by finding intersection with the universal jump line"""
    # Calculate the universal jump line: rho_s = 2T/pi
    jump_line = 2.0 * temps / np.pi
    
    # Calculate the intersection by finding where the difference changes sign
    diff = spin_stiffness - jump_line
    
    # Look for sign change in the difference
    sign_changes = np.where(np.diff(np.signbit(diff)))[0]
    
    if len(sign_changes) == 0:
        # No intersection found
        print("Warning: No intersection found with universal jump line")
        return None, None
    
    # Get the temperatures and values around the intersection
    idx = sign_changes[0]
    t1, t2 = temps[idx], temps[idx+1]
    v1, v2 = spin_stiffness[idx], spin_stiffness[idx+1]
    e1, e2 = errors[idx], errors[idx+1]
    
    # Linear interpolation to find intersection
    slope = (v2 - v1) / (t2 - t1)
    jump_slope = 2.0 / np.pi
    
    # Solve for intersection: v1 + slope*(t-t1) = 2t/pi
    # Rearranging: t = (v1 - slope*t1) / (2/pi - slope)
    if abs(slope - jump_slope) < 1e-6:
        # Lines are almost parallel, use midpoint
        T_KT = (t1 + t2) / 2.0
    else:
        T_KT = (v1 - slope*t1) / (2.0/np.pi - slope)
    
    # Estimate error based on the original errors
    error = max(e1, e2)
    
    return T_KT, error

def create_combined_figure(output_dir):
    """Create a combined figure with the most important plots from Figure 7 and 8"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Data for the T_KT scaling plot
    T_KT_values = []
    T_KT_errors = []
    L_values = []
    
    # Loop through each system size and add to plots
    for L in system_sizes:
        try:
            temps, nt, thermo_data = load_data(L)
            
            # Specific Heat (top left)
            specific_heat = thermo_data[0:nt, 1]
            specific_heat_err = thermo_data[nt:(2*nt), 1]
            axes[0, 0].errorbar(temps, specific_heat, yerr=specific_heat_err, 
                               label=f"$L = {L}$", marker='o', markersize=4, 
                               color=colors[f'L{L}'], alpha=0.8)
            
            # Spin Stiffness (top right)
            spin_stiffness = thermo_data[0:nt, 9]
            spin_stiffness_err = thermo_data[nt:(2*nt), 9]
            axes[0, 1].errorbar(temps, spin_stiffness, yerr=spin_stiffness_err, 
                               label=f"$L = {L}$", marker='o', markersize=4, 
                               color=colors[f'L{L}'], alpha=0.8)
            
            # Plot the universal prediction line
            if L == system_sizes[0]:  # Only do this once
                axes[0, 1].plot(temps, 2*temps/np.pi, 'k--', label=r'$\rho_s = \frac{2T}{\pi}$')
            
            # Vortex Density (bottom left)
            vortex_density = thermo_data[0:nt, 11]
            vortex_density_err = thermo_data[nt:(2*nt), 11]
            axes[1, 0].errorbar(temps, vortex_density, yerr=vortex_density_err, 
                               label=f"$L = {L}$", marker='o', markersize=4, 
                               color=colors[f'L{L}'], alpha=0.8)
            
            # Susceptibility (bottom right)
            susc = thermo_data[0:nt, 5]
            susc_err = thermo_data[nt:(2*nt), 5]
            axes[1, 1].errorbar(temps, susc, yerr=susc_err, 
                               label=f"$L = {L}$", marker='o', markersize=4, 
                               color=colors[f'L{L}'], alpha=0.8)
            
            # Extract T_KT for this system size
            T_KT, T_KT_err = extract_T_KT(temps, spin_stiffness, spin_stiffness_err)
            if T_KT is not None:
                T_KT_values.append(T_KT)
                T_KT_errors.append(T_KT_err)
                L_values.append(L)
                print(f"System size L={L}: T_KT = {T_KT:.4f} ± {T_KT_err:.4f}")
            
        except FileNotFoundError:
            print(f"Data for L={L} not found, skipping in plots")
    
    # Customize the plots
    axes[0, 0].set_title("Specific Heat", fontsize=12)
    axes[0, 0].set_xlabel("$T/J$")
    axes[0, 0].set_ylabel("Specific Heat per Site $c_v$")
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title("Spin Stiffness", fontsize=12)
    axes[0, 1].set_xlabel("$T/J$")
    axes[0, 1].set_ylabel("Spin stiffness $\\rho_s/J$")
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title("Vortex Density", fontsize=12)
    axes[1, 0].set_xlabel("$T/J$")
    axes[1, 0].set_ylabel("Density of vortices $\\omega_v$")
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title("Susceptibility", fontsize=12)
    axes[1, 1].set_xlabel("$T/J$")
    axes[1, 1].set_ylabel("Spin susceptibility $\chi$")
    axes[1, 1].set_yscale('log')  # Use log scale for susceptibility
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add a legend to the top right plot
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), 
              ncol=len(system_sizes), frameon=False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for the legend
    
    # Add a text annotation with observed T_KT values
    if T_KT_values:
        info_text = "Estimated $T_{KT}$ values:\n"
        for i, L in enumerate(L_values):
            info_text += f"$L={L}$: ${T_KT_values[i]:.4f} \\pm {T_KT_errors[i]:.4f}$\n"
        axes[1, 1].text(0.5, 0.1, info_text, transform=axes[1, 1].transAxes, 
                       bbox=dict(facecolor='white', alpha=0.7))
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, "bkt_results_simplified.png"), dpi=300, bbox_inches='tight')
    print(f"Combined figure saved to {os.path.join(output_dir, 'bkt_results_simplified.png')}")
    plt.close()

def main():
    # Declare global variables
    global system_sizes
    
    parser = argparse.ArgumentParser(description="Simplified script to reproduce BKT transition plots")
    
    # General options
    parser.add_argument('--force', action='store_true', help="Force rerun of simulations even if data exists")
    parser.add_argument('--skip-sim', action='store_true', help="Skip simulations and only make plots from existing data")
    parser.add_argument('--output-dir', type=str, default='./quick_plots', help="Output directory for plots")
    parser.add_argument('--cores', type=int, default=4, help="Number of CPU cores to use")
    
    # Simulation parameters (simplified options)
    parser.add_argument('--sizes', type=int, nargs='+', default=system_sizes, help="System sizes to simulate")
    
    args = parser.parse_args()
    
    # Override system_sizes if custom sizes provided
    if args.sizes:
        system_sizes = args.sizes
    
    # Create output directory
    output_dir = ensure_dir(args.output_dir)
    
    # Record start time
    start_time = time.time()
    print(f"Starting simplified BKT analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"System sizes: {system_sizes}")
    print(f"Temperature range: {temp_min} to {temp_max} ({num_temps} points)")
    print(f"Simulation steps: PreTherm={simulation_steps['pretherm']}, " + 
          f"Therm={simulation_steps['therm']}, Measure={simulation_steps['measure']}")
    
    # Run simulations for each system size sequentially
    if not args.skip_sim:
        for L in system_sizes:
            success = run_simulation(
                L, temp_min, temp_max, num_temps, 1,  # Type 1 = linear spacing
                simulation_steps['pretherm'], 
                simulation_steps['therm'], 
                simulation_steps['measure'],
                args.cores,
                args.force
            )
            
            if success:
                # Run analysis immediately after each simulation
                run_analysis(L, args.force)
            else:
                print(f"Skipping analysis for L={L} due to simulation failure")
    else:
        print("Skipping simulations as requested")
    
    # Create plot
    create_combined_figure(output_dir)
    
    # Print total elapsed time
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Total runtime: {int(minutes)}m {seconds:.2f}s")
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    main() 