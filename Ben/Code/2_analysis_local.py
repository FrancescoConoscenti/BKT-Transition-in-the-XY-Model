#!/usr/bin/env python3

"""
Analysis Script for BKT Transition Simulation Data

This script analyzes the output data from the BKT Transition simulations.
It can be run locally after downloading the simulation data from Kaggle.

Usage:
    1. Run kaggle_clean.py on Kaggle to generate simulation data
    2. Download the output folders (testL=X and testL=XfinalData)
    3. Run this script to analyze the data and create plots

Example:
    python analyze_bkt.py --sizes 10 15 20 25 30
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os
import time
import sys
from datetime import datetime
from scipy.optimize import curve_fit
import math
from numba import jit
import shutil

try:
    from tqdm.auto import tqdm
    tqdm_available = True
except ImportError:
    tqdm_available = False
    def tqdm(iterable, **kwargs):
        if 'desc' in kwargs:
            print(f"Starting: {kwargs['desc']}")
        return iterable

# LaTeX setup
latex_available = shutil.which('latex') is not None
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 4)

if latex_available:
    try:
        rc('font', **{'family': 'sans-serif', 'size': 10})
        rc('text', usetex=True)
        print("Using LaTeX for high-quality rendering")
    except:
        print("Error setting up LaTeX rendering, falling back to standard text")
else:
    print("LaTeX rendering not available, using standard text rendering.")

# Helper functions
def ensure_dir(dir_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")
    return dir_path

def extract_T_KT(temps, spin_stiffness, errors):
    """Extract T_KT by finding intersection with universal jump line"""
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
    
    # Solve for intersection point
    T_KT = (t1 + t2) / 2.0 if abs(slope - jump_slope) < 1e-6 else (v1 - slope*t1) / (2.0/np.pi - slope)
    error = max(e1, e2)
    
    return T_KT, error

# Jackknife analysis functions
@jit(nopython=True)
def jackknife_analysis(data, num_blocks, block_length=None):
    """Combined jackknife analysis that handles both block creation and error estimation"""
    if block_length is None:
        blocks = data  # If data is already blocked
    else:
        # Create blocks
        block_length = int(block_length)
        blocks = np.zeros(num_blocks)
        for i in range(num_blocks):
            blocks[i] = np.mean(data[i*block_length:(i+1)*block_length])
    
    # Calculate jackknife error
    N_B = len(blocks)
    avg = np.sum(blocks)/N_B
    data_size = N_B * (block_length if block_length else 1)
    
    # Jackknife estimate for each "leave one out" sample
    jack_estimates = np.zeros(N_B)
    for i in range(N_B):
        jack_estimates[i] = (data_size*avg - (block_length if block_length else 1)*blocks[i])/(data_size - (block_length if block_length else 1))
    
    # Calculate error
    jack_mean = np.mean(jack_estimates)
    error_sq = ((N_B - 1)/N_B) * np.sum((jack_estimates - jack_mean)**2)
    
    return avg, np.sqrt(error_sq)

@jit(nopython=True)
def JackknifeErrorFromFullList(data, num_blocks, block_length):
    """Wrapper for backward compatibility"""
    return jackknife_analysis(data, num_blocks, block_length)

# Analysis functions
def run_analysis(L, force=False):
    """Run data analysis for a specific system size"""
    data_dir = f"testL={L}"
    folder_data_final = f"testL={L}finalData"
    ensure_dir(folder_data_final)
    
    print(f"Starting data analysis for L={L}...")
    
    # Check if required files exist
    if not os.path.exists(f"{data_dir}/variables.data"):
        print(f"Error: variables.data not found in {data_dir}")
        return False
    
    # Load parameters
    saving_variables = np.loadtxt(f"{data_dir}/variables.data")
    length_box = int(saving_variables[0])
    number_box = int(saving_variables[1])
    range_temp = saving_variables[3:]
    np.savetxt(f"{folder_data_final}/variables.data", saving_variables)
    
    nt = len(range_temp)
    factor_print = 10000
    
    print(f"Analyzing {nt} temperature points...")
    
    # Arrays for storing results (combined values and errors)
    results = np.zeros((2*nt, 12))
    
    # Process each temperature point
    for m in tqdm(range(nt), desc="Analyzing temperatures", unit="temp"):
        data_file = f"{data_dir}/outputatT={int(range_temp[m]*factor_print):05d}.data"
        if not os.path.exists(data_file):
            print(f"Error: {data_file} not found")
            continue
        
        # Load raw data for this temperature
        data = np.loadtxt(data_file)
        
        # Extract measurements and normalize
        E1_init = data[:,0] / L**2
        M1_re = data[:,1] / L**2
        M1_im = data[:,2] / L**2
        M1_init = M1_re + 1j*M1_im
        ordU_xi = data[:,3] / L**4
        ordU_xi_0 = (M1_re**2 + M1_im**2) / L**4
        
        # Calculate derived quantities
        E2 = E1_init**2
        E4 = E2**2
        
        M1_tot = np.abs(M1_init)
        M2 = M1_tot**2
        M4 = M2**2
        
        # Jackknife analysis
        E1_avg, E1_error = JackknifeErrorFromFullList(E1_init, number_box, length_box)
        E2_avg, E2_error = JackknifeErrorFromFullList(E2, number_box, length_box)
        E4_avg, E4_error = JackknifeErrorFromFullList(E4, number_box, length_box)
        
        M1_real_avg, M1_real_error = JackknifeErrorFromFullList(M1_re, number_box, length_box)
        M1_imag_avg, M1_imag_error = JackknifeErrorFromFullList(M1_im, number_box, length_box)
        M1_avg, M1_error = JackknifeErrorFromFullList(M1_tot, number_box, length_box)
        M2_avg, M2_error = JackknifeErrorFromFullList(M2, number_box, length_box)
        M4_avg, M4_error = JackknifeErrorFromFullList(M4, number_box, length_box)
        
        ordU_xi_avg, ordU_xi_err = JackknifeErrorFromFullList(ordU_xi, number_box, length_box)
        ordU_xi_0_avg, ordU_xi_0_err = JackknifeErrorFromFullList(ordU_xi_0, number_box, length_box)
        
        # Energy and related
        results[m, 0] = E1_avg
        results[nt+m, 0] = E1_error
        
        # Specific heat
        T = range_temp[m]
        div_sp = T**2
        results[m, 1] = (E2_avg - E1_avg**2)/div_sp
        results[nt+m, 1] = np.sqrt(E2_error**2 + (2*E1_error*E1_avg)**2)/div_sp
        
        # Order cumulant
        ord_cum = (E4_avg)/(E2_avg**2) - 1
        results[m, 2] = ord_cum
        results[nt+m, 2] = np.abs(ord_cum) * np.sqrt((E4_error/E4_avg)**2 + (2*E2_error/E2_avg)**2)
        
        # Order parameter
        u_op = M1_real_avg**2 + M1_imag_avg**2
        u_op_err = np.sqrt((2*M1_real_error*M1_real_avg)**2 + (2*M1_imag_error*M1_imag_avg)**2)
        
        results[m, 3] = np.sqrt(u_op)
        results[nt+m, 3] = 0.5*u_op_err/np.sqrt(u_op)
        
        results[m, 4] = M1_avg
        results[nt+m, 4] = M1_error
        
        # Susceptibility
        results[m, 5] = (M2_avg - M1_avg**2)/T
        results[nt+m, 5] = np.sqrt(M2_error**2 + (2*M1_avg*M1_error)**2)/T
        
        results[m, 6] = M2_avg/T
        results[nt+m, 6] = M2_error/T
        
        # Binder cumulant
        bind_cum = M4_avg/(M2_avg**2)
        results[m, 7] = 1 - bind_cum/3
        results[nt+m, 7] = (1/3)*np.abs(bind_cum)*np.sqrt((M4_error/M4_avg)**2 + (2*M2_error/M2_avg)**2)
        
        # Correlation length
        fact_sin = (1/(2*np.sin(np.pi/L))**2)
        val_U_c = (ordU_xi_0_avg/ordU_xi_avg) - 1
        results[m, 8] = fact_sin*val_U_c
        results[nt+m, 8] = fact_sin*np.sqrt((ordU_xi_0_err/ordU_xi_0_avg)**2 + (ordU_xi_err/ordU_xi_avg)**2)
        
        # Stiffness
        Stiff_tot_H = data[:,4]/L**2
        Stiff_tot_I = data[:,5]/L**2
        Stiff_tot_I2 = Stiff_tot_I**2
        Stiff_tot_I4 = Stiff_tot_I2**2
        
        Stiff_tot_H_avg, Stiff_tot_H_error = JackknifeErrorFromFullList(Stiff_tot_H, number_box, length_box)
        Stiff_tot_I_avg, Stiff_tot_I_error = JackknifeErrorFromFullList(Stiff_tot_I, number_box, length_box)
        Stiff_tot_I2_avg, Stiff_tot_I2_error = JackknifeErrorFromFullList(Stiff_tot_I2, number_box, length_box)
        Stiff_tot_I4_avg, Stiff_tot_I4_error = JackknifeErrorFromFullList(Stiff_tot_I4, number_box, length_box)
        
        results[m, 9] = Stiff_tot_H_avg - (L**2/T)*(Stiff_tot_I2_avg - Stiff_tot_I_avg**2)
        results[nt+m, 9] = np.sqrt(
            Stiff_tot_H_error**2 + 
            ((L**2)*Stiff_tot_I2_error/T)**2 + 
            ((L**2)*2*Stiff_tot_I_error*Stiff_tot_I_avg/T)**2
        )
        
        # Fourth order
        list_Ysq_tot = (Stiff_tot_H - (L**2/T)*(Stiff_tot_I2 - Stiff_tot_I**2))**2
        list_Ysq_tot_avg, list_Ysq_tot_error = JackknifeErrorFromFullList(list_Ysq_tot, number_box, length_box)
        
        results[m, 10] = (-4)*results[m, 9] + 3*(Stiff_tot_H_avg - (L**2/T)*(list_Ysq_tot_avg - results[m, 9]**2)) + 2*(L**6/(T**3))*Stiff_tot_I4_avg
        
        # Vorticity
        Vort = data[:,6]
        Vort_avg, Vort_error = JackknifeErrorFromFullList(Vort, number_box, length_box)
        results[m, 11] = Vort_avg/L**2
        results[nt+m, 11] = Vort_error/L**2
    
    # Save all results to a single file
    np.savetxt(f"{folder_data_final}/thermo_output.data", results)
    print(f"Analysis for L={L} completed successfully")
    return True

def load_data(L):
    """Load processed data for a specific system size"""
    base_path = f"testL={L}finalData"
    
    if not os.path.exists(f"{base_path}/variables.data"):
        raise FileNotFoundError(f"Data for L={L} not found. Run analysis first.")
    
    variables = np.loadtxt(f"{base_path}/variables.data")
    temps = variables[3:]
    thermo_data = np.loadtxt(f"{base_path}/thermo_output.data")
    
    nt = len(temps)
    return temps, nt, thermo_data

def finite_size_scaling(T_KT_values, L_values):
    """Perform finite-size scaling to extract T_KT in thermodynamic limit"""
    # T_KT(L) = T_KT(∞) + a/(ln L)²
    def fit_func(x, T_KT_inf, a):
        return T_KT_inf + a*x
    
    # Convert L to 1/(ln(L))²
    x_data = 1.0 / (np.log(np.array(L_values)))**2
    y_data = np.array(T_KT_values)
    
    # Perform the fit
    popt, pcov = curve_fit(fit_func, x_data, y_data)
    T_KT_inf, a = popt
    perr = np.sqrt(np.diag(pcov))
    
    print(f"Extrapolated T_KT(L=∞) = {T_KT_inf:.4f} ± {perr[0]:.4f}")
    print(f"Coefficient a = {a:.4f} ± {perr[1]:.4f}")
    
    # Generate fit line for plotting
    x_fit = np.linspace(0, max(x_data)*1.1, 100)
    y_fit = fit_func(x_fit, T_KT_inf, a)
    
    return T_KT_inf, a, x_fit, y_fit

def setup_subplot(ax, title, xlabel, ylabel, use_latex=True, yscale=None):
    """Helper function to set up plot styling consistently"""
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel if not use_latex else f"${xlabel}$")
    ax.set_ylabel(ylabel if not use_latex else f"${ylabel}$")
    if yscale:
        ax.set_yscale(yscale)
    ax.grid(True, alpha=0.3)
    return ax

def create_combined_figure(system_sizes, output_dir="./plots", latex_available=True):
    """Create combined figure with key observables"""
    ensure_dir(output_dir)
    
    # Setup figure and plotting style
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Color mapping for different system sizes
    colors = {f'L{L}': plt.cm.viridis(i/len(system_sizes)) 
              for i, L in enumerate(system_sizes)}
    
    # Data for finite-size scaling
    T_KT_values, T_KT_errors, L_values = [], [], []
    
    # Process each system size
    for L in system_sizes:
        try:
            temps, nt, thermo_data = load_data(L)
            
            # Plot data for this system size on all subplots
            for idx, (row, col, data_idx, title, ylabel, yscale) in enumerate([
                (0, 0, 1, "Specific Heat", "c_v", None),
                (0, 1, 9, "Spin Stiffness", "\\rho_s/J", None),
                (1, 0, 11, "Vortex Density", "\\omega_v", None),
                (1, 1, 5, "Susceptibility", "\\chi", "log")
            ]):
                data = thermo_data[0:nt, data_idx]
                errors = thermo_data[nt:(2*nt), data_idx]
                
                label = f"L = {L}" if not latex_available else f"$L = {L}$"
                axes[row, col].errorbar(temps, data, yerr=errors, 
                                      label=label, 
                                      marker='o', markersize=4, 
                                      color=colors[f'L{L}'], alpha=0.8)
                
                # Set up subplot style
                setup_subplot(axes[row, col], title, "T/J", ylabel, latex_available, yscale)
                
                # Add universal jump line to spin stiffness plot
                if idx == 1 and L == system_sizes[0]:
                    jump_line = 2*temps/np.pi
                    label = 'rho_s = 2T/π' if not latex_available else r'$\rho_s = \frac{2T}{\pi}$'
                    axes[row, col].plot(temps, jump_line, 'k--', label=label)
            
            # Extract T_KT for this system size
            spin_stiffness = thermo_data[0:nt, 9]
            spin_stiffness_err = thermo_data[nt:(2*nt), 9]
            T_KT, T_KT_err = extract_T_KT(temps, spin_stiffness, spin_stiffness_err)
            
            if T_KT is not None:
                T_KT_values.append(T_KT)
                T_KT_errors.append(T_KT_err)
                L_values.append(L)
                print(f"System size L={L}: T_KT = {T_KT:.4f} ± {T_KT_err:.4f}")
            
        except FileNotFoundError:
            print(f"Data for L={L} not found, skipping in plots")
    
    # Add legend at the top
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), 
              ncol=min(5, len(system_sizes)), frameon=False)
    
    # Adjust layout
    try:
        plt.tight_layout()
    except:
        print("Warning: tight_layout failed, using basic layout")
        plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1, wspace=0.3, hspace=0.3)
    
    plt.subplots_adjust(top=0.9)  # Make room for the legend
    
    # Add annotation with T_KT values
    if T_KT_values:
        if latex_available:
            info_text = "Estimated $T_{KT}$ values:\n"
            for i, L in enumerate(L_values):
                info_text += f"$L={L}$: ${T_KT_values[i]:.4f} \\pm {T_KT_errors[i]:.4f}$\n"
        else:
            info_text = "Estimated T_KT values:\n"
            for i, L in enumerate(L_values):
                info_text += f"L={L}: {T_KT_values[i]:.4f} ± {T_KT_errors[i]:.4f}\n"
        
        axes[1, 1].text(0.5, 0.1, info_text, transform=axes[1, 1].transAxes, 
                       bbox=dict(facecolor='white', alpha=0.7))
    
    # Perform finite-size scaling if enough data
    if len(L_values) >= 3:
        T_KT_inf, a, x_fit, y_fit = finite_size_scaling(T_KT_values, L_values)
        
        # Create separate figure for finite-size scaling
        fig_scaling, ax_scaling = plt.subplots(figsize=(8, 6))
        
        x_data = 1.0 / (np.log(np.array(L_values)))**2
        ax_scaling.errorbar(x_data, T_KT_values, yerr=T_KT_errors, fmt='o', 
                           color='blue', label='Data')
        
        fit_label = f'T_KT(inf) = {T_KT_inf:.3f}' if not latex_available else f'$T_{{KT}}(\\infty) = {T_KT_inf:.3f}$'
        ax_scaling.plot(x_fit, y_fit, 'r-', label=fit_label)
        ax_scaling.plot(0, T_KT_inf, 'ro', markersize=8)
        
        ax_scaling.set_title("Finite Size Scaling Analysis", fontsize=14)
        if latex_available:
            ax_scaling.set_xlabel("$1/(\\ln L)^2$", fontsize=12)
            ax_scaling.set_ylabel("$T_{KT}(L)$", fontsize=12)
        else:
            ax_scaling.set_xlabel("1/(ln L)^2", fontsize=12)
            ax_scaling.set_ylabel("T_KT(L)", fontsize=12)
        ax_scaling.grid(True, alpha=0.3)
        ax_scaling.legend()
        
        plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        plt.savefig(os.path.join(output_dir, "finite_size_scaling.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_scaling)
    
    # Save main figure
    plt.savefig(os.path.join(output_dir, "bkt_results.png"), dpi=300, bbox_inches='tight')
    print(f"Combined figure saved to {os.path.join(output_dir, 'bkt_results.png')}")
    plt.close(fig)

def check_requirements():
    """Check if all required packages are available"""
    requirements = {
        "numpy": True,
        "matplotlib": True,
        "scipy": True,
        "numba": True,
        "tqdm": tqdm_available,
        "latex": latex_available
    }
    
    missing = [pkg for pkg, available in requirements.items() if not available]
    
    if missing:
        print("WARNING: The following packages or tools are missing or unavailable:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("The script will still run, but with reduced functionality.")
        
        if not tqdm_available:
            print("  Without tqdm, progress bars will not be displayed.")
            print("  You can install it with: pip install tqdm")
            
        if not latex_available:
            print("  Without LaTeX, plots will use standard text rendering (not as pretty).")
    
    return len(missing) == 0

def main():
    """Main function to run analysis and create plots"""
    all_requirements_met = check_requirements()
    if not all_requirements_met:
        print("Continuing with reduced functionality...")
        print()
    
    # Configuration as a dictionary (replacing command line args)
    config = {
        "sizes": [10, 15, 20, 25, 30],  # System sizes to analyze
        "output_dir": "./plots",         # Directory to save output plots
        "skip_analysis": False,          # Skip data analysis and only create plots
        "force": False                   # Force reanalysis of existing data
    }
    
    ensure_dir(config["output_dir"])
    
    # Initialize timing
    start_time = time.time()
    print(f"Starting BKT data analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"System sizes: {config['sizes']}")
    
    # Run analysis
    if not config["skip_analysis"]:
        for L in tqdm(config["sizes"], desc="Running analysis", unit="size"):
            run_analysis(L, config["force"])
    else:
        print("Skipping analysis as requested")
    
    # Create plots
    create_combined_figure(config["sizes"], config["output_dir"], latex_available=latex_available)
    
    # Print timing information
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Plots saved to {config['output_dir']}")

if __name__ == "__main__":
    main() 