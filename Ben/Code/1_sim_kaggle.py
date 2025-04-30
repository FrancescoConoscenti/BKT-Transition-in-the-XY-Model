#!/usr/bin/env python3

"""
REFACTORING JUSTIFICATIONS:
1. Combined similar jackknife functions to reduce code duplication
2. Optimized neighbor list creation with NumPy vectorization
3. Simplified temperature range generation using NumPy directly
4. Improved data structure usage for configuration management
5. Reduced redundant calculations in measurement functions
6. Streamlined parallel tempering implementation
7. Improved memory efficiency by reusing arrays where possible
8. Enhanced error handling with early returns
9. Used NumPy functionality more efficiently throughout
10. Separated analysis into a separate script (analyze_bkt.py) for Kaggle workflow
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os
import time
import sys
from datetime import datetime
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

# Core Monte Carlo functions
@jit(nopython=True)
def WolffUpdate(config, temp, N, neighbors_list):
    beta = 1./temp
    numItTot = N*N
    size = N*N
    avg_size_clust = 0.
    
    cluster = np.zeros(size, dtype=np.int8)
    listQ = np.zeros(size + 1, dtype=np.int64)

    for nn in range(numItTot):
        init = np.random.randint(0, size)
        listQ[0] = init + 1
        random_angle = (np.pi)*np.random.rand()
        
        cluster[init] = 1
        sc_in = 0
        sc_to_add = 1

        while listQ[sc_in] != 0:
            site_studied = listQ[sc_in] - 1
            sc_in += 1
            avg_size_clust += 1
                
            site_angle = config[site_studied]
            config[site_studied] = 2*random_angle - site_angle
 
            for kk in range(4):
                site_nn = neighbors_list[4*site_studied + kk]
                near_angle = config[site_nn]
                if cluster[site_nn] == 0:
                    energy_difference = (-1)*(np.cos(site_angle - near_angle) - 
                                             np.cos(site_angle - (2*random_angle - near_angle)))
                    freezProb_next = 1. - np.exp(beta*energy_difference)
                    if (np.random.rand() < freezProb_next):
                        listQ[sc_to_add] = site_nn + 1                   
                        cluster[site_nn] = 1
                        sc_to_add += 1
        listQ[:] = 0
        cluster[:] = 0

    return avg_size_clust/numItTot

@jit(nopython=True)
def MetropolisUpdate(config, temp, N, neighbors_list):
    beta = 1./temp
    numItTot = N*N

    for _ in range(numItTot):
        site = np.random.randint(0, numItTot)
        s_angle = config[site]
        new_angle = (2*np.pi)*np.random.rand()

        energy_past = energy_future = 0.0
        for kk in range(4):
            site_nn = neighbors_list[4*site + kk]
            past_angle = config[site_nn]
            energy_past += np.cos(s_angle - past_angle)
            energy_future += np.cos(new_angle - past_angle)

        cost = energy_past - energy_future
        if cost <= 0 or np.random.rand() < np.exp(-beta*cost):
            config[site] = new_angle

@jit(nopython=True)
def EnergyCalc(config, N):
    energy = 0.
    for i in range(N):
        for j in range(N):
            latt1 = config[N*i + j]
            latt1shiftX = config[N*((i-1) % N) + j]
            latt1shiftY = config[N*i + (j-1) % N]
            energy += (-1.0)*(np.cos(latt1-latt1shiftX) + np.cos(latt1-latt1shiftY))
    return energy

@jit(nopython=True)
def MeasureConfigNumba(config, N):
    tpi = 2*np.pi
    config_re = config.reshape((N,N))
    
    energy = H_tot = I_tot = 0.
    ord1 = 0. + 0.j
    ordU_xi_x = 0. + 0.j
    ordU_xi_y = 0. + 0.j
    vort = 0.
    
    for i in range(N):
        for j in range(N):
            platt1 = config_re[i,j]
            platt1shiftX = config_re[i-1,j]
            platt1shiftY = config_re[i,j-1]
            platt1shiftXshiftY = config_re[i-1,j-1]
            
            vcos = np.cos(platt1-platt1shiftX)
            energy += (-1.0)*(vcos + np.cos(platt1-platt1shiftY))
            H_tot += vcos
            I_tot += np.sin(platt1-platt1shiftX)
            ord1 += np.exp(1j*platt1)
            
            ordU_xi_x += np.cos(platt1)*np.exp(1j*i*tpi/N)
            ordU_xi_y += np.sin(platt1)*np.exp(1j*i*tpi/N)
            
            # Vortex calculation
            platt1v = np.mod(platt1, tpi)
            platt1shiftXv = np.mod(platt1shiftX, tpi)
            platt1shiftXshiftYv = np.mod(platt1shiftXshiftY, tpi)
            platt1shiftYv = np.mod(platt1shiftY, tpi)
            diff_list1 = np.array([platt1v - platt1shiftXv, 
                                  platt1shiftXv - platt1shiftXshiftYv,
                                  platt1shiftXshiftYv - platt1shiftYv, 
                                  platt1shiftYv - platt1v])
            
            vort_here = 0.
            for ll_1 in diff_list1:
                if ll_1 > np.pi:
                    ll_1 -= tpi
                elif ll_1 < -np.pi:
                    ll_1 += tpi   
                vort_here += ll_1/tpi
                    
            vort += np.abs(vort_here)

    ordU_xi_m = (ordU_xi_x.real)**2 + (ordU_xi_x.imag)**2 + (ordU_xi_y.real)**2 + (ordU_xi_y.imag)**2
    
    return np.array([energy, ord1.real, ord1.imag, ordU_xi_m, H_tot, I_tot, vort/tpi])

# Unified jackknife analysis functions
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

# Helper functions
def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")
    return dir_path

# Monte Carlo simulation functions
def update_mc(config, temp, N, neighbors_list, iterations, measure_func=None):
    """Unified update function that can perform thermalization or measurements"""
    config = config.copy()
    results = []
    
    for _ in range(iterations):
        WolffUpdate(config, temp, N, neighbors_list)
        if measure_func:
            results.append(measure_func(config, N))
    
    if measure_func:
        # Check if the measurement function returns an array or a scalar
        if isinstance(results[-1], np.ndarray) or hasattr(results[-1], '__getitem__'):
            # MeasureConfigNumba returns an array where first element is energy
            return config, results[-1][0], np.array(results)
        else:
            # EnergyCalc returns a scalar energy value
            return config, results[-1], np.array(results)
    else:
        return config, EnergyCalc(config, N), None

def run_simulation(L, temp_min, temp_max, num_temps, temp_type, 
                  pretherm_steps, therm_steps, measure_steps, num_cores=1, force=False):
    """Run Monte Carlo simulation for a specific system size"""
    print(f"Starting simulation for L={L}...")
    N = L
    
    # Initialize directory
    name_dir = f"testL={L}"
    ensure_dir(name_dir)
    
    # Generate temperature range
    list_temps = (np.linspace(temp_min, temp_max, num_temps) if temp_type == 1 else 
                 temp_min * np.power(temp_max/temp_min, np.arange(num_temps)/(num_temps-1)))
    
    # Create neighbor list more efficiently
    site_indices = np.arange(N*N).reshape(N, N)
    neighbors_list = np.zeros(4*(N*N), dtype=np.int64)
    
    i_indices, j_indices = np.indices((N, N))
    for offset, (di, dj) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
        neighbors = site_indices[(i_indices + di) % N, (j_indices + dj) % N].flatten()
        neighbors_list[np.arange(0, 4*N*N, 4) + offset] = neighbors
    
    print(f'Linear size: L={N}, Temperature range: {temp_min} to {temp_max} ({num_temps} points)')
    print(f'Steps: PreTherm={pretherm_steps}, Therm={therm_steps}, Measure={measure_steps}')
    
    # Initialize configurations
    config_at_T = np.array([2*np.pi*np.random.rand(N*N) for _ in range(num_temps)])
    
    # Setup for parallel tempering
    list_energies = np.zeros(num_temps)
    pt_TtoE = np.arange(num_temps)  # Maps temperature index to configuration index
    pt_EtoT = np.arange(num_temps)  # Maps configuration index to temperature index
    
    # Pre-thermalization
    print('Starting pre-thermalization...')
    start_time = time.time()
    
    niters = np.ones(num_temps, dtype=np.int64)
    list_avg_clus_size = np.zeros(num_temps)
    
    for m in tqdm(range(num_temps), desc="Pre-thermalization", unit="temp"):
        config, _, _ = update_mc(
            config_at_T[m], list_temps[m], N, neighbors_list, 
            pretherm_steps*niters[m], None
        )
        
        # Estimate average cluster size
        avg_size_clust = 0.0
        for _ in range(10):  # Sample a few times for better estimate
            avg_size_clust += WolffUpdate(config.copy(), list_temps[m], N, neighbors_list)
        
        list_avg_clus_size[m] = avg_size_clust / 10
        config_at_T[m] = config
    
    # Calculate optimal iterations based on cluster size
    niters = np.ceil((N*N)/np.maximum(list_avg_clus_size, 1)).astype(np.int64)
    print(f'Optimal iterations per temperature: {niters}')
    
    # Save parameters
    saving_variables = np.concatenate(([measure_steps, measure_steps, therm_steps], list_temps))
    np.savetxt(f"{name_dir}/variables.data", saving_variables)
    
    print(f'Pre-thermalization completed in {time.time() - start_time:.2f} seconds')
    
    # Setup for parallel tempering
    indices_temp = np.arange(num_temps)
    even_pairs = [indices_temp[i:i+2] for i in range(0, len(indices_temp)-1, 2)]
    odd_pairs = [indices_temp[i:i+2] for i in range(1, len(indices_temp)-1, 2)]
    pair_sets = [even_pairs, odd_pairs]
    
    # Thermalization
    print('Starting thermalization...')
    start_time = time.time()
    all_energies_therm = np.zeros((num_temps, therm_steps*measure_steps))
    
    for il in tqdm(range(therm_steps), desc="Thermalization", unit="step"):
        # Update all configurations
        for m in range(num_temps):
            config, energy, energies = update_mc(
                config_at_T[m], list_temps[pt_EtoT[m]], N, neighbors_list,
                measure_steps*niters[pt_EtoT[m]], EnergyCalc
            )
            list_energies[m] = energy
            config_at_T[m] = config
            
            # Store energy values - ensure we have the right shape and length
            if energies is not None:
                # Ensure we only take the right number of energy values
                if len(energies) > measure_steps:
                    # Take measure_steps values evenly spaced from the energies array
                    indices = np.linspace(0, len(energies)-1, measure_steps, dtype=int)
                    energies_subset = energies[indices]
                else:
                    # If we have too few values, repeat the last one to fill
                    energies_subset = np.pad(energies, (0, measure_steps - len(energies)), 'edge')

                # For EnergyCalc, we get an array of scalar values
                if len(energies_subset.shape) == 1:
                    all_energies_therm[pt_TtoE[m], measure_steps*il:measure_steps*(il+1)] = energies_subset
                # For MeasureConfigNumba, we would need to extract the first column (energy)
                else:
                    all_energies_therm[pt_TtoE[m], measure_steps*il:measure_steps*(il+1)] = energies_subset[:, 0]
        
        # Parallel tempering exchanges
        pair_idx = il % 2
        for pair in pair_sets[pair_idx]:
            if len(pair) < 2:
                continue
                
            i, j = pair
            Ti, Tj = list_temps[i], list_temps[j]
            Ei, Ej = list_energies[pt_TtoE[i]], list_energies[pt_TtoE[j]]
            
            # Metropolis criterion for exchange
            delta = (Ei - Ej) * (1/Ti - 1/Tj)
            if delta > 0 or np.random.rand() < np.exp(delta):
                # Swap configurations
                pt_TtoE[i], pt_TtoE[j] = pt_TtoE[j], pt_TtoE[i]
                pt_EtoT[pt_TtoE[i]], pt_EtoT[pt_TtoE[j]] = i, j
    
    print(f'Thermalization completed in {time.time() - start_time:.2f} seconds')
    
    # Measurement phase
    print('Starting measurements...')
    start_time = time.time()
    all_data_thermo = np.zeros((num_temps, measure_steps*measure_steps, 7))
    
    for il in tqdm(range(measure_steps), desc="Measurement", unit="step"):
        # Update and measure
        for m in range(num_temps):
            config, energy, measurements = update_mc(
                config_at_T[m], list_temps[pt_EtoT[m]], N, neighbors_list,
                measure_steps*niters[pt_EtoT[m]], MeasureConfigNumba
            )
            list_energies[m] = energy
            config_at_T[m] = config
            
            # Store measurements for this temperature - ensure correct length
            if len(measurements) > measure_steps:
                # Take measure_steps values evenly spaced from the measurements array
                indices = np.linspace(0, len(measurements)-1, measure_steps, dtype=int)
                measurements_subset = measurements[indices]
            else:
                # If we have too few values, repeat the last one to fill
                measurements_subset = np.pad(measurements, [(0, measure_steps - len(measurements)), (0, 0)], 'edge')
                
            # Store all 7 values
            all_data_thermo[pt_TtoE[m], measure_steps*il:measure_steps*(il+1)] = measurements_subset
        
        # Parallel tempering exchanges (same as in thermalization)
        pair_idx = il % 2
        for pair in pair_sets[pair_idx]:
            if len(pair) < 2:
                continue
                
            i, j = pair
            Ti, Tj = list_temps[i], list_temps[j]
            Ei, Ej = list_energies[pt_TtoE[i]], list_energies[pt_TtoE[j]]
            
            delta = (Ei - Ej) * (1/Ti - 1/Tj)
            if delta > 0 or np.random.rand() < np.exp(delta):
                pt_TtoE[i], pt_TtoE[j] = pt_TtoE[j], pt_TtoE[i]
                pt_EtoT[pt_TtoE[i]], pt_EtoT[pt_TtoE[j]] = i, j
    
    print(f'Measurements completed in {time.time() - start_time:.2f} seconds')
    
    # Save final data
    factor_print = 10000
    for q in range(num_temps):
        temp_init = list_temps[q]
        np.savetxt(f"{name_dir}/configatT={int(temp_init*factor_print):05d}.data", config_at_T[pt_TtoE[q]])
        np.savetxt(f"{name_dir}/outputatT={int(temp_init*factor_print):05d}.data", all_data_thermo[q])
    
    print('Data export complete')
    return True

def check_requirements():
    """Check if all required packages are available"""
    requirements = {
        "numpy": True,
        "matplotlib": True,
        "numba": True,
        "tqdm": tqdm_available
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
            
    return len(missing) == 0

def main():
    """Main function to run simulations"""
    all_requirements_met = check_requirements()
    if not all_requirements_met:
        print("Continuing with reduced functionality...")
        print()
    
    # Configuration as a dictionary (replacing command line args)
    config = {
        "sizes": [10, 15, 20, 25, 30],  # System sizes to simulate
        "temp_min": 0.7,                # Minimum temperature
        "temp_max": 1.1,                # Maximum temperature
        "temps": 15,                    # Number of temperature points
        "temp_type": 1,                 # Temperature spacing type: 0=geometric, 1=linear
        "pretherm": 100,                # Pre-thermalization steps
        "therm": 200,                   # Thermalization steps
        "measure": 500,                 # Measurement steps
        "force": False                  # Force simulation even if output exists
    }
    
    # Initialize timing
    start_time = time.time()
    print(f"Starting BKT simulation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"System sizes: {config['sizes']}")
    print(f"Temperature range: {config['temp_min']} to {config['temp_max']} ({config['temps']} points)")
    print(f"Simulation steps: PreTherm={config['pretherm']}, Therm={config['therm']}, Measure={config['measure']}")
    
    # Estimate runtime
    total_steps = config['therm'] + config['measure']
    largest_size = max(config['sizes'])
    estimated_minutes = len(config['sizes']) * config['temps'] * total_steps * (largest_size/10)**2 / 1000
    print(f"Estimated runtime: {estimated_minutes/60:.1f} hours" if estimated_minutes >= 60 
          else f"Estimated runtime: {estimated_minutes:.1f} minutes")
    print("Note: This is a rough estimate, actual runtime may vary based on hardware")
    
    # Run simulations for each system size
    for L in tqdm(config["sizes"], desc="Running simulations", unit="size"):
        success = run_simulation(
            L=L,
            temp_min=config["temp_min"],
            temp_max=config["temp_max"],
            num_temps=config["temps"],
            temp_type=config["temp_type"],
            pretherm_steps=config["pretherm"],
            therm_steps=config["therm"],
            measure_steps=config["measure"],
            force=config["force"]
        )
        
        if not success:
            print(f"Simulation for L={L} failed")
    
    # Print timing information
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Data directory: testL=X")
    print("\nTo analyze the data:")
    print("1. Download all the testL=X directories")
    print("2. Run analyze_bkt.py to process the data and create plots")

if __name__ == "__main__":
    main() 