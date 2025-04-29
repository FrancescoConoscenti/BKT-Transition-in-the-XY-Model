#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import subprocess
import argparse
import time
from scipy.optimize import curve_fit
import multiprocessing
from datetime import datetime

# Set matplotlib parameters for paper-quality plots
rc('font', **{'family': 'sans-serif', 'size': 10})
rc('text', usetex=True)
plt.rcParams['figure.figsize'] = (12, 4)

# Define some nice colors for plots (same as in the paper)
colors = {
    'L10': 'red',
    'L12': 'orange',
    'L16': 'green',
    'L20': 'blue'
}

# System sizes to simulate
system_sizes = [10, 12, 16, 20]  # Smaller system sizes for reasonable runtime

# Temperature range for simulation
temp_min = 0.5
temp_max = 1.5
num_temps = 20  # Reduced number of temperature points

# Control variables for how much computation to do
simulation_steps = {
    'pretherm': 1000,   # Reduced pre-thermalization steps
    'therm': 5000,      # Reduced thermalization steps 
    'measure': 10000    # Reduced measurement steps
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
    log_file = f"log_L{L}.txt"
    
    # Skip if data already exists and force=False
    if os.path.exists(data_dir) and os.path.exists(f"{data_dir}/variables.data") and not force:
        print(f"Data for L={L} already exists. Skipping simulation.")
        return True
    
    print(f"Starting simulation for L={L}...")
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
    log_file = f"log_analysis_L{L}.txt"
    
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

def finite_size_scaling(T_KT_values, L_values):
    """Perform finite-size scaling fit to extract T_KT(L=∞)"""
    # Function to fit: T_KT(L) = T_KT(∞) + a/(ln(L))²
    def fit_func(x, T_KT_inf, a):
        return T_KT_inf + a*x
    
    # Convert L to 1/(ln(L))²
    x_data = 1.0 / (np.log(np.array(L_values)))**2
    y_data = np.array(T_KT_values)
    
    # Perform the fit
    popt, pcov = curve_fit(fit_func, x_data, y_data)
    T_KT_inf, a = popt
    
    # Calculate error bars from the covariance matrix
    perr = np.sqrt(np.diag(pcov))
    
    print(f"Extrapolated T_KT(L=∞) = {T_KT_inf:.4f} ± {perr[0]:.4f}")
    print(f"Coefficient a = {a:.4f} ± {perr[1]:.4f}")
    
    # Generate fit line for plotting
    x_fit = np.linspace(0, max(x_data)*1.1, 100)
    y_fit = fit_func(x_fit, T_KT_inf, a)
    
    return T_KT_inf, a, x_fit, y_fit

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

def run_simulations_parallel(args):
    """Run simulations for all system sizes in parallel"""
    num_cores_per_sim = max(1, multiprocessing.cpu_count() // len(system_sizes))
    print(f"Running simulations using {num_cores_per_sim} cores per simulation")
    
    processes = []
    for L in system_sizes:
        # Create and start process for each system size
        process = multiprocessing.Process(
            target=run_simulation,
            args=(L, args.temp_min, args.temp_max, args.num_temps, args.temp_type, 
                  args.pretherm, args.therm, args.measure, num_cores_per_sim, args.force)
        )
        processes.append(process)
        process.start()
        
        # Stagger starts to avoid I/O conflicts
        time.sleep(1)
    
    # Wait for all simulations to complete
    for process in processes:
        process.join()
    
    # Now run analysis sequentially (analysis is typically fast)
    for L in system_sizes:
        run_analysis(L, args.force)

def create_figure7(output_dir):
    """Create Figure 7: Specific Heat, Magnetization, and Susceptibility"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Loop through each system size and add to plots
    for L in system_sizes:
        try:
            temps, nt, thermo_data = load_data(L)
            
            # Panel (a): Specific Heat
            specific_heat = thermo_data[0:nt, 1]
            specific_heat_err = thermo_data[nt:(2*nt), 1]
            axes[0].errorbar(temps, specific_heat, yerr=specific_heat_err, 
                            label=f"$L = {L}$", marker='o', markersize=4, 
                            color=colors[f'L{L}'], alpha=0.8)
            
            # Panel (b): Magnetization
            mag = thermo_data[0:nt, 3]
            mag_err = thermo_data[nt:(2*nt), 3]
            axes[1].errorbar(temps, mag, yerr=mag_err, 
                            label=f"$L = {L}$", marker='o', markersize=4, 
                            color=colors[f'L{L}'], alpha=0.8)
            
            # Panel (c): Susceptibility
            susc = thermo_data[0:nt, 5]
            susc_err = thermo_data[nt:(2*nt), 5]
            # Use log scale for susceptibility
            axes[2].errorbar(temps, susc, yerr=susc_err, 
                            label=f"$L = {L}$", marker='o', markersize=4, 
                            color=colors[f'L{L}'], alpha=0.8)
            
        except FileNotFoundError:
            print(f"Data for L={L} not found, skipping in Figure 7")
    
    # Customize the plots
    axes[0].set_title("a)", loc='left')
    axes[0].set_xlabel("$T/J$")
    axes[0].set_ylabel("Specific Heat per Site $c_v$")
    axes[0].set_xlim(temp_min, temp_max)
    axes[0].set_ylim(0, 2.5)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title("b)", loc='left')
    axes[1].set_xlabel("$T/J$")
    axes[1].set_ylabel("Magnetization per site $m$")
    axes[1].set_xlim(temp_min, temp_max)
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_title("c)", loc='left')
    axes[2].set_xlabel("$T/J$")
    axes[2].set_ylabel("Spin susceptibility $\chi$")
    axes[2].set_xlim(temp_min, temp_max)
    axes[2].set_yscale('log')  # Use log scale for susceptibility
    axes[2].grid(True, alpha=0.3)
    
    # Add a legend to the last subplot
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
              ncol=len(system_sizes), frameon=False)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, "figure7.png"), dpi=300, bbox_inches='tight')
    print(f"Figure 7 saved to {os.path.join(output_dir, 'figure7.png')}")
    plt.close()

def create_figure8(output_dir):
    """Create Figure 8: Vortex Density, Spin Stiffness, and T_KT scaling"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Data for the T_KT scaling plot
    T_KT_values = []
    T_KT_errors = []
    L_values = []
    
    # Loop through each system size and add to plots
    for L in system_sizes:
        try:
            temps, nt, thermo_data = load_data(L)
            
            # Panel (a): Vortex Density
            vortex_density = thermo_data[0:nt, 11]
            vortex_density_err = thermo_data[nt:(2*nt), 11]
            axes[0].errorbar(temps, vortex_density, yerr=vortex_density_err, 
                            label=f"$L = {L}$", marker='o', markersize=4, 
                            color=colors[f'L{L}'], alpha=0.8)
            
            # Panel (b): Spin Stiffness
            spin_stiffness = thermo_data[0:nt, 9]
            spin_stiffness_err = thermo_data[nt:(2*nt), 9]
            axes[1].errorbar(temps, spin_stiffness, yerr=spin_stiffness_err, 
                            label=f"$L = {L}$", marker='o', markersize=4, 
                            color=colors[f'L{L}'], alpha=0.8)
            
            # Plot the universal prediction line
            if L == system_sizes[0]:  # Only do this once
                axes[1].plot(temps, 2*temps/np.pi, 'k--', label=r'$\rho_s = \frac{2T}{\pi}$')
            
            # Extract T_KT for this system size
            T_KT, T_KT_err = extract_T_KT(temps, spin_stiffness, spin_stiffness_err)
            if T_KT is not None:
                T_KT_values.append(T_KT)
                T_KT_errors.append(T_KT_err)
                L_values.append(L)
                print(f"System size L={L}: T_KT = {T_KT:.4f} ± {T_KT_err:.4f}")
            
        except FileNotFoundError:
            print(f"Data for L={L} not found, skipping in Figure 8")
    
    # Panel (c): Finite Size Scaling of T_KT
    if len(L_values) >= 3:  # Need at least 3 points for a meaningful fit
        T_KT_inf, a, x_fit, y_fit = finite_size_scaling(T_KT_values, L_values)
        
        # Plot the data points with error bars
        x_data = 1.0 / (np.log(np.array(L_values)))**2
        axes[2].errorbar(x_data, T_KT_values, yerr=T_KT_errors, fmt='o', 
                        color='blue', label='Data')
        
        # Plot the fit line
        axes[2].plot(x_fit, y_fit, 'r-', label=f'$T_{{KT}}(\\infty) = {T_KT_inf:.3f}$')
        
        # Highlight the extrapolated T_KT(∞) at x=0
        axes[2].plot(0, T_KT_inf, 'ro', markersize=8)
    else:
        print("Not enough data points for T_KT scaling analysis")
    
    # Customize the plots
    axes[0].set_title("a)", loc='left')
    axes[0].set_xlabel("$T/J$")
    axes[0].set_ylabel("Density of vortices $\\omega_v$")
    axes[0].set_xlim(temp_min, temp_max)
    axes[0].set_ylim(0, 0.3)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title("b)", loc='left')
    axes[1].set_xlabel("$T/J$")
    axes[1].set_ylabel("Spin stiffness $\\rho_s/J$")
    axes[1].set_xlim(temp_min, temp_max)
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_title("c)", loc='left')
    axes[2].set_xlabel("$1/(\\ln L)^2$")
    axes[2].set_ylabel("$T_{KT}(L)$")
    axes[2].set_xlim(-0.01, 0.17)
    axes[2].set_ylim(0.85, 0.95)
    axes[2].grid(True, alpha=0.3)
    
    # Add a legend to the first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
              ncol=len(system_sizes), frameon=False)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, "figure8.png"), dpi=300, bbox_inches='tight')
    print(f"Figure 8 saved to {os.path.join(output_dir, 'figure8.png')}")
    plt.close()

def main():
    # Declare global variables
    global system_sizes
    
    parser = argparse.ArgumentParser(description="Reproduce figures from BKT paper using Monte Carlo simulations")
    
    # General options
    parser.add_argument('--force', action='store_true', help="Force rerun of simulations even if data exists")
    parser.add_argument('--skip-sim', action='store_true', help="Skip simulations and only make plots from existing data")
    parser.add_argument('--output-dir', type=str, default='./paper_plots', help="Output directory for plots")
    
    # Simulation parameters
    parser.add_argument('--temp-min', type=float, default=temp_min, help="Minimum temperature")
    parser.add_argument('--temp-max', type=float, default=temp_max, help="Maximum temperature")
    parser.add_argument('--num-temps', type=int, default=num_temps, help="Number of temperature points")
    parser.add_argument('--temp-type', type=int, default=1, help="Temperature range type (0=geometric, 1=linear)")
    
    # Simulation steps
    parser.add_argument('--pretherm', type=int, default=simulation_steps['pretherm'], help="Pre-thermalization steps")
    parser.add_argument('--therm', type=int, default=simulation_steps['therm'], help="Thermalization steps")
    parser.add_argument('--measure', type=int, default=simulation_steps['measure'], help="Measurement steps")
    
    # Sizes to simulate
    parser.add_argument('--sizes', type=int, nargs='+', default=system_sizes, help="System sizes to simulate")
    
    args = parser.parse_args()
    
    # Override system_sizes if custom sizes provided
    if args.sizes:
        system_sizes = args.sizes
    
    # Create output directory
    output_dir = ensure_dir(args.output_dir)
    
    # Record start time
    start_time = time.time()
    print(f"Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"System sizes: {system_sizes}")
    print(f"Temperature range: {args.temp_min} to {args.temp_max} ({args.num_temps} points)")
    print(f"Simulation steps: PreTherm={args.pretherm}, Therm={args.therm}, Measure={args.measure}")
    
    # Run simulations if not skipped
    if not args.skip_sim:
        run_simulations_parallel(args)
    else:
        print("Skipping simulations as requested")
    
    # Create plots
    create_figure7(output_dir)
    create_figure8(output_dir)
    
    # Print total elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    main() 