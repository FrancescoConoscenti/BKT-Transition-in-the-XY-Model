#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import os

# Function to ensure the plots directory exists
def ensure_plots_dir(dir_path="./plots"):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created plots directory at {dir_path}")
    return dir_path

# Set matplotlib parameters for nicer plots
from matplotlib import rc
rc('font',**{'family':'sans-serif', 'size': 10})
rc('text', usetex=True)

# Define some nice colors for plots
color_red = (0.73, 0.13869999999999993, 0.)
color_orange = (1., 0.6699999999999999, 0.)
color_green = (0.14959999999999996, 0.43999999999999995, 0.12759999999999994)
color_blue = (0.06673600000000002, 0.164512, 0.776)
color_purple = (0.25091600000000003, 0.137378, 0.29800000000000004)
color_ocker = (0.6631400000000001, 0.71, 0.1491)
color_pink = (0.71, 0.1491, 0.44730000000000003)
color_brown = (0.651, 0.33331200000000005, 0.054683999999999955)

def load_data(N):
    """Load simulation data for the given system size N"""
    # Construct the paths where data should be found
    # First check if data is in the local directory structure
    name_dir = f'testL={N}finalData'
    local_base_path = f'./{name_dir}'
    results_base_path = f'../Results/{name_dir}'
    
    # Try local path first, then fallback to original path
    if os.path.exists(f"{local_base_path}/variables.data"):
        base_path = local_base_path
        print(f"Using local data path: {base_path}")
    elif os.path.exists(f"{results_base_path}/variables.data"):
        base_path = results_base_path
        print(f"Using Results directory path: {base_path}")
    else:
        raise FileNotFoundError(f"Could not find data in {local_base_path} or {results_base_path}")
    
    # Load variables and temperature data
    saving_variables = np.loadtxt(f'{base_path}/variables.data')
    range_temp = saving_variables[3:]
    nt = len(range_temp)
    
    # Load thermodynamic data
    data_thermo = np.loadtxt(f'{base_path}/thermo_output.data')
    
    print(f"Successfully loaded data for system size N={N}")
    return range_temp, nt, data_thermo

def plot_spin_stiffness(N, range_temp, nt, data_thermo, plots_dir):
    """Plot the spin stiffness and the predicted Nelson-Kosterlitz jump"""
    plt.figure(figsize=(8, 6))
    
    # Plot the spin stiffness with error bars
    plt.errorbar(range_temp, data_thermo[0:nt, 9], data_thermo[(nt):(2*nt), 9], 
                 marker='o', linestyle='-', label=f'Spin Stiffness (L={N})')
    
    # Plot the Nelson-Kosterlitz universal prediction (rho_s = 2T/pi)
    plt.plot(range_temp, 2*range_temp/np.pi, 'r--', label=r'$\rho_s = \frac{2T}{\pi}$')
    
    # Add labels and legend
    plt.xlabel('Temperature $T/J$', fontsize=12)
    plt.ylabel('Spin Stiffness $\\rho_s/J$', fontsize=12)
    plt.title(f'Spin Stiffness vs Temperature (L={N})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plot_file = os.path.join(plots_dir, f'spin_stiffness_L{N}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_file}")
    plt.close()

def plot_specific_heat(N, range_temp, nt, data_thermo, plots_dir):
    """Plot the specific heat vs temperature"""
    plt.figure(figsize=(8, 6))
    
    # Plot the specific heat with error bars
    plt.errorbar(range_temp, data_thermo[0:nt, 1]*(N**2), data_thermo[(nt):(2*nt), 1], 
                 marker='o', linestyle='-', label=f'Specific Heat (L={N})')
    
    # Add labels and legend
    plt.xlabel('Temperature $T/J$', fontsize=12)
    plt.ylabel('Specific Heat $C_V$', fontsize=12)
    plt.title(f'Specific Heat vs Temperature (L={N})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plot_file = os.path.join(plots_dir, f'specific_heat_L{N}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_file}")
    plt.close()

def plot_susceptibility(N, range_temp, nt, data_thermo, plots_dir):
    """Plot the magnetic susceptibility vs temperature"""
    plt.figure(figsize=(8, 6))
    
    # Plot the susceptibility with error bars
    plt.errorbar(range_temp, data_thermo[0:nt, 5]*(N**2), data_thermo[(nt):(2*nt), 5], 
                 marker='o', linestyle='-', label=f'Susceptibility (L={N})')
    
    # Add labels and legend
    plt.xlabel('Temperature $T/J$', fontsize=12)
    plt.ylabel('Susceptibility $\\chi$', fontsize=12)
    plt.title(f'Magnetic Susceptibility vs Temperature (L={N})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plot_file = os.path.join(plots_dir, f'susceptibility_L{N}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_file}")
    plt.close()

def main():
    # System size to analyze
    N = 8  # Use the same size as in our fast simulation
    
    # Create plots directory
    plots_dir = ensure_plots_dir()
    
    try:
        # Load data
        range_temp, nt, data_thermo = load_data(N)
        
        # Create and save plots
        plot_spin_stiffness(N, range_temp, nt, data_thermo, plots_dir)
        plot_specific_heat(N, range_temp, nt, data_thermo, plots_dir)
        plot_susceptibility(N, range_temp, nt, data_thermo, plots_dir)
        
        print(f"Analysis complete! Plots saved to '{plots_dir}' directory.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure to run the simulation and analysis first!")

if __name__ == "__main__":
    main() 