#!/usr/bin/env python3

"""
Visualization script for the XY model showing spin configurations and 
highlighting regions of high local spin misalignment (indicative of vortices).
Creates three plots:
1. Low temperature (T≈0.7): Mostly aligned spins
2. Near T_KT (T≈0.88): Phase transition region with emerging disorder
3. Above T_KT (T≈0.95): Disordered phase
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import subprocess
import argparse
import time
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

# Set matplotlib parameters for paper-quality plots
rc('font', **{'family': 'sans-serif', 'size': 10})
rc('text', usetex=True)
plt.rcParams['figure.figsize'] = (15, 5)

# Default temperatures to visualize
DEFAULT_TEMPS = [0.7, 0.88, 0.95]

# Colormap for highlighting local misalignment (e.g., 'viridis', 'plasma', 'YlOrRd')
# 'viridis' is often good for perception
MISALIGNMENT_CMAP = plt.cm.viridis 

def ensure_dir(dir_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")
    return dir_path

def run_simulation(L, temp, pretherm_steps, therm_steps, num_cores, work_dir):
    """Run a simulation at a single temperature to get a thermalized configuration"""
    data_dir = f"{work_dir}/config_L{L}_T{temp:.2f}"
    ensure_dir(data_dir)
    log_file = f"{data_dir}/log.txt"
    config_file = f"configL{L}_T{int(temp*10000):05d}.data"
    
    print(f"Thermalizing L={L} at T={temp:.2f}...")
    
    command = [
        "python3", "./mcpt-thermalize.py", 
        str(L), str(temp), str(pretherm_steps), str(therm_steps), str(num_cores)
    ]
    
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(command, stdout=f, stderr=f)
            
        if result.returncode != 0:
            print(f"Error running thermalization. Check {log_file}")
            return None
        
        if os.path.exists(config_file):
            config = np.loadtxt(config_file)
            np.savetxt(f"{data_dir}/config.data", config)
            os.rename(config_file, f"{data_dir}/original_config.data")
            return config.reshape(L, L)
        else:
            print(f"Configuration file not found: {config_file}")
            return None
    except Exception as e:
        print(f"Error in simulation: {str(e)}")
        return None

def calculate_local_misalignment(config):
    """
    Calculate a measure of local spin misalignment for each site.
    High values indicate regions where a spin differs significantly from its neighbors,
    often characteristic of vortex cores or other defects.
    
    Args:
        config: 2D array of spin angles
        
    Returns:
        2D array of local misalignment values
    """
    L = config.shape[0]
    misalignment = np.zeros((L, L))
    
    for i in range(L):
        for j in range(L):
            angle_center = config[i, j]
            local_misalignment = 0.0
            neighbors = [
                config[(i+1)%L, j], config[(i-1)%L, j], 
                config[i, (j+1)%L], config[i, (j-1)%L]
            ]
            
            for angle_neighbor in neighbors:
                # Calculate angle difference, wrap to [-pi, pi]
                diff = angle_center - angle_neighbor
                diff = (diff + np.pi) % (2*np.pi) - np.pi
                # Contribution to misalignment: 1 - cos(diff) is 0 for aligned, 2 for anti-aligned
                local_misalignment += (1.0 - np.cos(diff))
                
            misalignment[i, j] = local_misalignment
    
    return misalignment

def calculate_spin_correlation(config):
    """Calculate average nearest-neighbor correlation to quantify order"""
    L = config.shape[0]
    correlation = 0.0
    count = 0
    
    for i in range(L):
        for j in range(L):
            angle = config[i, j]
            right_angle = config[i, (j+1) % L]
            down_angle = config[(i+1) % L, j]
            corr_right = np.cos(angle - right_angle)
            corr_down = np.cos(angle - down_angle)
            correlation += corr_right + corr_down
            count += 2
    
    return correlation / count

def plot_configuration(config, title, ax):
    """
    Plot a configuration highlighting local spin misalignment.
    
    Args:
        config: 2D array of spin angles
        title: Title for the plot
        ax: Matplotlib axis to plot on
        
    Returns:
        Matplotlib image object for colorbar reference
    """
    L = config.shape[0]
    
    # Calculate local misalignment
    misalignment = calculate_local_misalignment(config)
    
    # Calculate spin correlation for the title
    correlation = calculate_spin_correlation(config)
    
    # Determine normalization range for misalignment colormap
    # Max possible misalignment is 4 * (1 - cos(pi)) = 8
    norm = Normalize(vmin=0, vmax=np.max(misalignment) if np.max(misalignment) > 0 else 1)
    
    # Plot the misalignment map as the background
    im = ax.imshow(misalignment, cmap=MISALIGNMENT_CMAP, 
                  interpolation='nearest', alpha=0.8,
                  norm=norm)
    
    # Plot spin directions as arrows
    X, Y = np.meshgrid(range(L), range(L))
    stride = max(1, L // 16) # Slightly denser arrows may be good
    
    # Use a contrasting color for arrows (e.g., white or black depending on cmap)
    arrow_color = 'white' 
    ax.quiver(X[::stride, ::stride], Y[::stride, ::stride], 
              np.cos(config)[::stride, ::stride], 
              np.sin(config)[::stride, ::stride], 
              pivot='mid', color=arrow_color, scale=25, width=0.005, 
              alpha=0.9, headlength=4, headwidth=3)
    
    # Add correlation value to title
    full_title = f"{title}\nSpin correlation: {correlation:.3f}"
    ax.set_title(full_title, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    
    return im

def create_spin_visualizations(L=16, temps=None, output_dir='./spin_visualizations', 
                        pretherm_steps=500, therm_steps=1000, num_cores=8):
    """Generate and plot spin configurations showing local misalignment"""
    if temps is None:
        temps = DEFAULT_TEMPS
    
    ensure_dir(output_dir)
    work_dir = f"{output_dir}/temp_configs"
    ensure_dir(work_dir)
    
    all_configs = []
    titles = []
    correlations = []
    
    # Generate configurations
    for temp in temps:
        config = run_simulation(L, temp, pretherm_steps, therm_steps, num_cores, work_dir)
        if config is not None:
            all_configs.append(config)
            correlation = calculate_spin_correlation(config)
            correlations.append(correlation)
            
            # Construct appropriate title
            if temp < 0.8:
                regime = "Below $T_{KT}$"
                desc = "Ordered phase (low misalignment)"
            elif temp < 0.9:
                regime = "Near $T_{KT}$"
                desc = "Critical phase (emerging misalignment)"
            else:
                regime = "Above $T_{KT}$"
                desc = "Disordered phase (high misalignment)"
                
            title = f"{regime} (T={temp:.2f})\n{desc}"
            titles.append(title)
    
    # Create a combined figure
    if all_configs:
        fig = plt.figure(figsize=(18, 6))
        gs = gridspec.GridSpec(1, len(all_configs))
        
        for i, (config, title) in enumerate(zip(all_configs, titles)):
            ax = plt.subplot(gs[i])
            im = plot_configuration(config, title, ax)
        
        # Add a colorbar for local misalignment
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(r'Local Spin Misalignment', fontsize=12)
        
        plt.suptitle(f"XY Model Spin Misalignment Around BKT Transition ($L={L}$)", fontsize=16)
        plt.tight_layout(rect=[0, 0.05, 0.9, 0.95])
        
        # Save the figure
        plt.savefig(f"{output_dir}/misalignment_configurations_L{L}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save individual configurations
        for i, (config, temp) in enumerate(zip(all_configs, temps)):
            plt.figure(figsize=(8, 8))
            ax = plt.gca()
            im = plot_configuration(config, f"Temperature T={temp:.2f}", ax)
            plt.colorbar(im, label='Local Spin Misalignment')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/misalignment_config_L{L}_T{temp:.2f}.png", dpi=300)
            plt.close()
        
        print(f"Visualization saved to {output_dir}/misalignment_configurations_L{L}.png")
        print(f"Individual configurations saved to {output_dir}/")
        
        print("\nStatistics:")
        for temp, corr in zip(temps, correlations):
            print(f"T={temp:.2f}: Correlation={corr:.4f}")
    else:
        print("Failed to generate any valid configurations")
        
    return True

def main():
    parser = argparse.ArgumentParser(description="Visualize spin configurations and local misalignment in the XY model")
    
    parser.add_argument('--L', type=int, default=16, help="Linear system size")
    parser.add_argument('--temps', type=float, nargs='+', default=None, help="Temperatures to visualize")
    parser.add_argument('--output-dir', type=str, default='./spin_visualizations', help="Output directory for plots")
    parser.add_argument('--pretherm', type=int, default=500, help="Pre-thermalization steps")
    parser.add_argument('--therm', type=int, default=1000, help="Thermalization steps")
    parser.add_argument('--cores', type=int, default=8, help="Number of CPU cores to use")
    
    args = parser.parse_args()
    start_time = time.time()
    
    create_spin_visualizations(
        L=args.L, temps=args.temps, output_dir=args.output_dir,
        pretherm_steps=args.pretherm, therm_steps=args.therm, num_cores=args.cores
    )
    
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Total runtime: {int(minutes)}m {seconds:.2f}s")

if __name__ == "__main__":
    main() 