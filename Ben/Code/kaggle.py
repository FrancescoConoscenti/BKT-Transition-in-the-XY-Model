#!/usr/bin/env python3

"""
XY Model Kosterlitz-Thouless Transition Simulation

This script performs Monte Carlo simulations of the 2D XY model to study
the Berezinskii-Kosterlitz-Thouless (BKT) phase transition. It uses the Wolff
cluster algorithm to efficiently sample configurations, especially near the
critical point.

To use this script in Kaggle/Colab:
1. Upload this file
2. Run it with the command:
   !python kaggle.py
3. To modify parameters, edit the config dictionary in the main() function:
   e.g., To use smaller system sizes and skip simulation:
   config["sizes"] = [6, 8]
   config["skip_sim"] = True

Required packages:
- numpy
- matplotlib
- scipy
- numba
- tqdm (optional, for progress bars)

The script will:
1. Run Monte Carlo simulations for different system sizes
2. Calculate thermodynamic observables (energy, specific heat, etc.)
3. Measure the spin stiffness to identify the BKT transition
4. Create plots of the key observables
5. Perform finite-size scaling analysis if possible

Created by: Victor Drouin-Touchette
Adaptation for Kaggle: [Your Name]

MEMORY CONSIDERATIONS:
# Memory usage increases with:
# - System size L (scales as L²)
# - Number of temperature points
# - Number of measurement steps
# For Kaggle/Colab with limited memory (<16GB), use L ≤ 30
# For high-end systems (32GB+), you can use L up to 80
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

# Try to import tqdm, but provide a fallback if not available
try:
    from tqdm.auto import tqdm
    tqdm_available = True
except ImportError:
    # Fallback implementation if tqdm is not available
    tqdm_available = False
    def tqdm(iterable, **kwargs):
        # Simple fallback function that just returns the original iterable
        # Optionally print a message about the operation
        if 'desc' in kwargs:
            print(f"Starting: {kwargs['desc']}")
        return iterable

# Set matplotlib parameters for paper-quality plots
# First, check if LaTeX is actually available
latex_available = shutil.which('latex') is not None

# Configure matplotlib based on LaTeX availability
if latex_available:
    try:
        rc('font', **{'family': 'sans-serif', 'size': 10})
        rc('text', usetex=True)
        plt.rcParams['figure.figsize'] = (12, 4)
        print("Using LaTeX for high-quality rendering")
    except:
        latex_available = False
        print("Error setting up LaTeX rendering, falling back to standard text")

# If LaTeX is not available, use standard text rendering
if not latex_available:
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.figsize'] = (12, 4)
    print("LaTeX rendering not available, using standard text rendering.")

############################
# Core Monte Carlo functions
############################

#one step of the Wolff algorithm
@jit(nopython=True)
def WolffUpdate(config, temp, N, neighbors_list):
    beta=1./temp
    
    #do wolff steps as long as the total cluster size flipped
    # is smaller than the size of the system
    numItTot = N*N
    size = N*N
    #initialize outputs
    avg_size_clust = 0.

    cluster = np.zeros(size, dtype = np.int8)
    listQ = np.zeros(size + 1, dtype = np.int64)

    for nn in range(numItTot):
        init = np.random.randint(0, size)
        listQ[0] = init + 1
        theta_rand = (np.pi)*np.random.rand()   
        random_angle = theta_rand
        
        cluster[init] = 1 #this site is in the cluster now
        sc_in = 0
        sc_to_add = 1

        while listQ[sc_in] != 0:
            site_studied = listQ[sc_in] + (-1)
            sc_in += 1
            avg_size_clust += 1
                
            prime_layer_rand = random_angle
            site_angle = config[site_studied]  #find the current angle of the studied site
            config[site_studied] = (2*prime_layer_rand - site_angle) #site selected has been flipped
 
            for kk in range(4):
                site_nn = neighbors_list[4*site_studied + kk]
                near_angle = config[site_nn]
                if cluster[site_nn] == 0:
                    energy_difference = (-1)*(np.cos(site_angle - near_angle) - np.cos(site_angle - (2*prime_layer_rand - near_angle)))
                    freezProb_next = 1. - np.exp(beta*energy_difference)
                    if (np.random.rand() < freezProb_next):
                        listQ[sc_to_add] = site_nn + (1)                    
                        cluster[site_nn] = 1
                        sc_to_add += 1
        listQ[:] = 0
        cluster[:] = 0

    #average size cluster
    avg_size_clust = avg_size_clust/numItTot    

    return avg_size_clust

#one step of the Metropolis algorithm
@jit(nopython=True)
def MetropolisUpdate(config, temp, N, neighbors_list):
    beta=1./temp
    
    numItTot = N*N
    size = N*N

    for nn in range(numItTot):
        site = np.random.randint(0, size)
        s_angle = config[site]

        new_angle = (2*np.pi)*np.random.rand()

        energy_past = 0.0
        energy_future = 0.0

        for kk in range(4):
            site_nn = neighbors_list[4*site + kk]
            past_angle = config[site_nn]
            energy_past += np.cos(s_angle - past_angle)
            energy_future += np.cos(new_angle - past_angle)

        cost = energy_past - energy_future

        if cost <= 0:
            config[site]= new_angle
        else:
            if np.random.rand() < np.exp(- beta*cost):
                config[site]= new_angle

@jit(nopython=True)
def EnergyCalc(config, N):
    energy = 0.
    #calculate the energy
    for i in range(N):
        for j in range(N):
            latt1 = config[N*i + j]
            latt1shiftX = config[N*(i-1) + j]
            latt1shiftY = config[N*i + j-1]
            energy += (-1.0)*(np.cos(latt1+(-1)*latt1shiftX) + 
                             np.cos(latt1+(-1)*latt1shiftY))
    return energy

@jit(nopython=True)
def MeasureConfigNumba(config, N):
    #######
    #all calculations ----------
    #######  
    tpi = 2*np.pi

    config_re = config
    config_re = config_re.reshape((N,N))
    mod_latt = np.mod(config_re, tpi)
 
    energy = 0.
    H_tot = 0.
    I_tot = 0.

    #total
    total = 0.
    #bond average
    bond_avg = 0.
    #ord
    ord1 = 0. + 0.j

    #U(1) correlation length
    ordU_xi_x = 0. + 0.j
    ordU_xi_y = 0. + 0.j
    
    #some vortex in there
    vort = 0.
    
    for i in range(N):
        for j in range(N):
            platt1 = config_re[i,j]
            platt1shiftX = config_re[i-1,j]
            platt1shiftY = config_re[i,j-1]
            platt1shiftXshiftY = config_re[i-1,j-1]
            
            vcos = np.cos(platt1+(-1)*platt1shiftX)
            energy += (-1.0)*(vcos + 
                          np.cos(platt1+(-1)*platt1shiftY))
            H_tot += vcos
            I_tot += np.sin(platt1+(-1)*platt1shiftX)

            ord1 += np.exp(1j*platt1)
          
            #U(1) correlation length
            ordU_xi_x += np.cos(platt1)*np.exp(1j*i*tpi/N)
            ordU_xi_y += np.sin(platt1)*np.exp(1j*i*tpi/N)
            
            #vortex calc
            platt1v = np.mod(platt1, tpi)
            platt1shiftXv = np.mod(platt1shiftX, tpi)
            platt1shiftXshiftYv = np.mod(platt1shiftXshiftY, tpi)
            platt1shiftYv = np.mod(platt1shiftY,tpi)
            diff_list1 = np.array([platt1v - platt1shiftXv, platt1shiftXv - platt1shiftXshiftYv,
                                   platt1shiftXshiftYv - platt1shiftYv, platt1shiftYv - platt1v])
            
            vort_here = 0.
            for ll_1 in diff_list1:
                if ll_1 > np.pi:
                    ll_1 = ll_1 - tpi
                if ll_1 < -np.pi:
                    ll_1 = ll_1 + tpi   
                ll_1 = ll_1/tpi
                vort_here += ll_1
                    
            vort += np.absolute(vort_here)

    vort = vort/tpi
    
    #U(1) correlation length
    ordU_xi_m = (ordU_xi_x.real)**2 + (ordU_xi_x.imag)**2 + (ordU_xi_y.real)**2 + (ordU_xi_y.imag)**2
    
    all_dat = np.array([energy, ord1.real, ord1.imag, ordU_xi_m,
                       H_tot, I_tot, vort])

    return all_dat

############################
# Jackknife analysis functions  
############################

@jit(nopython=True)
def jackBlocks(original_list, num_of_blocks, length_of_blocks):
    block_list = np.zeros(num_of_blocks)
    length_of_blocks = int(length_of_blocks)
    for i in range(num_of_blocks):
        block_list[i] = (1/length_of_blocks)*np.sum(original_list[i*(length_of_blocks) : (i + 1)*(length_of_blocks)])
    return block_list

@jit(nopython=True)
def JackknifeError(blocks, length_of_blocks):
    blocks = np.array(blocks)
    N_B = len(blocks)
    avg = np.sum(blocks)/N_B
    #length_of_blocks is k
    N_J = N_B*length_of_blocks #is basically N
    jack_block = (1/(N_J - length_of_blocks))*(N_J*np.ones(N_B)*avg - length_of_blocks*blocks)
    bar_o_j = np.sum(jack_block)/N_B
    error_sq = ((N_B - 1)/N_B)*np.sum((jack_block - bar_o_j*np.ones(N_B))**2)

    return avg, np.sqrt(error_sq)

@jit(nopython=True)
def JackknifeErrorFromFullList(original_list, num_of_blocks, length_of_blocks):
    blocks = jackBlocks(original_list, num_of_blocks, length_of_blocks)

    N_B = len(blocks)
    avg = np.sum(blocks)/N_B
    N_J = N_B*length_of_blocks #is basically N
    jack_block = (1/(N_J - length_of_blocks))*(N_J*np.ones(N_B)*avg - length_of_blocks*blocks)
    bar_o_j = np.sum(jack_block)/N_B
    error_sq = ((N_B - 1)/N_B)*np.sum((jack_block - bar_o_j*np.ones(N_B))**2)

    return avg, np.sqrt(error_sq)

############################
# Helper functions
############################

def ensure_dir(dir_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")
    return dir_path

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

############################
# Monte Carlo simulation functions
############################

def UpdatePreTherm(config_init, temp, N, neighbors_list, factorIter):
    """Pre-thermalization to obtain optimal factorIter"""
    #the config
    config = config_init.copy()

    #mc step
    avg_size_clust = 0.0
    for st in range(factorIter):
        size_clust = WolffUpdate(config, temp, N, neighbors_list)
        avg_size_clust += size_clust

    final_avg_size_clust = avg_size_clust/factorIter
    return [config, final_avg_size_clust]

def PTTstepTherm(config_init, temp, N, neighbors_list, factorIter):
    """Thermalization step"""
    #the config
    config = config_init.copy()
    energies = np.zeros(factorIter)

    for st in range(factorIter):
        WolffUpdate(config, temp, N, neighbors_list)
        energies[st] = EnergyCalc(config, N)

    #final energy
    energy = energies[-1]
    return [config, energy, np.array(energies)]

def PTTstepMeasure(config_init, temp, N, neighbors_list, factorIter):
    """Measurement step"""
    #the config
    config = config_init.copy()

    data_thermo = []
    for st in range(factorIter):
        WolffUpdate(config, temp, N, neighbors_list)
        data_thermo.append(MeasureConfigNumba(config, N))

    #final energy
    energy = data_thermo[-1][0]
    return [config, energy, np.array(data_thermo)]

def run_simulation(L, temp_min, temp_max, num_temps, temp_type, 
                  pretherm_steps, therm_steps, measure_steps, num_cores=1, force=False):
    """Run the Monte Carlo simulation for a specific system size"""
    print(f"Starting simulation for L={L}...")
    N = L  # Linear system size
    
    # Initialize directory structure
    name_dir = f"testL={L}"
    if not os.path.exists(name_dir):
        os.makedirs(name_dir)
    
    # Temperature range
    if temp_type == 0:  # Geometric spacing
        ratio_T = (temp_max/temp_min)**(1/(num_temps-1))
        list_temps = np.zeros(num_temps)
        for i in range(num_temps):
            list_temps[i] = temp_min*((ratio_T)**(i))
    else:  # Linear spacing
        list_temps = np.linspace(temp_min, temp_max, num_temps)
    
    # Initialize neighbor list
    neighbors_list = np.zeros(4*(N**2), dtype=np.int64)
    for i in range(N**2):
        vec_nn_x = [-1, 1, 0, 0]  # Neighbor offsets in x direction
        vec_nn_y = [0, 0, -1, 1]  # Neighbor offsets in y direction
        
        site_studied_1 = i//N  # Row
        site_studied_2 = i%N   # Column
        
        for p in range(4):
            # Calculate neighbor index with periodic boundary conditions
            neighbors_list[4*i + p] = (N*np.mod((site_studied_1 + vec_nn_x[p]), N) + 
                                       np.mod((site_studied_2 + vec_nn_y[p]), N))
    
    # Print simulation parameters
    print(f'Linear size of the system L={N}')
    print(f'Temperature range: {temp_min} to {temp_max} in {num_temps} steps')
    print(f'Pre-thermalization steps: {pretherm_steps}')
    print(f'Thermalization steps: {therm_steps}')
    print(f'Measurement steps: {measure_steps}')
    
    # Initialize random configurations
    config_start = []
    for q in range(num_temps):
        config_start.append(2*np.pi*np.random.rand(N**2))
    config_start = np.array(config_start)
    config_at_T = config_start.copy()
    
    # Initialize arrays
    list_energies = np.zeros(num_temps)
    pt_TtoE = [i for i in range(num_temps)]  # Maps temperature index to configuration index
    pt_EtoT = [i for i in range(num_temps)]  # Maps configuration index to temperature index
    
    print('Starting pre-thermalization...')
    start_time = time.time()
    
    # Pre-thermalization to determine optimal number of iterations per temperature
    niters = np.ones(num_temps, dtype=np.int64)
    list_avg_clus_size = np.zeros(num_temps)
    
    # Sequential pre-thermalization
    resultsPreTherm = []
    for m in tqdm(range(num_temps), desc="Pre-thermalization", unit="temp"):
        # Run pre-thermalization with initial number of iterations
        result = UpdatePreTherm(config_init=config_at_T[m],
                               temp=list_temps[m], 
                               N=N, 
                               neighbors_list=neighbors_list, 
                               factorIter=pretherm_steps*niters[m])
        resultsPreTherm.append(result)
    
    # Process pre-thermalization results
    for q in range(num_temps):
        list_avg_clus_size[q] = resultsPreTherm[q][1]
        config_at_T[q] = resultsPreTherm[q][0]
    
    # Calculate optimal number of iterations based on average cluster size
    niters = np.ceil((N**2)/list_avg_clus_size)*niters
    niters = niters.astype(np.int64)
    print('Optimal number of iterations per temperature:')
    print(niters)
    
    # Save simulation parameters
    saving_variables = np.array([measure_steps, measure_steps, therm_steps])
    saving_variables = np.append(saving_variables, list_temps)
    np.savetxt(f"{name_dir}/variables.data", saving_variables)
    
    print(f'Pre-thermalization completed in {time.time() - start_time:.2f} seconds')
    
    # Thermalization setup
    # Tuples for parallel tempering
    indices_temp = [i for i in range(num_temps)]
    tuples_1 = [indices_temp[i:i+2] for i in range(0, len(indices_temp), 2)]  # odd switch
    tuples_2 = [indices_temp[i:i+2] for i in range(1, len(indices_temp)-1, 2)]  # even switch
    tuples_tot = [tuples_1, tuples_2]
    half_length = int(num_temps/2)
    len_tuples_tot = [half_length, half_length - 1]
    
    # Main thermalization loop
    print('Starting thermalization...')
    start_time = time.time()
    
    all_the_energies_thermalization = np.zeros((num_temps, therm_steps*measure_steps))
    
    # Thermalization loop
    swap_even_pairs = 0  # Controls which pairs to swap in parallel tempering
    
    for il in tqdm(range(therm_steps), desc="Thermalization", unit="step"):
        # Sequential thermalization
        resultsTherm = []
        for m in range(num_temps):
            result = PTTstepTherm(config_init=config_at_T[m],
                                 temp=list_temps[pt_EtoT[m]], 
                                 N=N, 
                                 neighbors_list=neighbors_list,
                                 factorIter=measure_steps*niters[pt_EtoT[m]])
            resultsTherm.append(result)
        
        # Process thermalization results
        for q in range(num_temps):
            list_energies[q] = resultsTherm[q][1]
            config_at_T[q] = resultsTherm[q][0]
            data_extract = resultsTherm[pt_TtoE[q]][2]
            for ws in range(measure_steps):
                all_the_energies_thermalization[q][measure_steps*il + ws] = data_extract[ws]
        
        # Parallel tempering step
        tuples_used = tuples_tot[swap_even_pairs]
        for sw in range(len_tuples_tot[swap_even_pairs]):
            index_i = tuples_used[sw][0]
            index_j = tuples_used[sw][1]
            initial_i_temp = list_temps[index_i]
            initial_j_temp = list_temps[index_j]
            index_energy_i = pt_TtoE[index_i]
            index_energy_j = pt_TtoE[index_j]
            
            Delta_ij = (list_energies[index_energy_i] - list_energies[index_energy_j])*(1/initial_i_temp - 1/initial_j_temp)
            if Delta_ij > 0:
                pt_TtoE[index_i] = index_energy_j
                pt_TtoE[index_j] = index_energy_i
                pt_EtoT[index_energy_i] = index_j
                pt_EtoT[index_energy_j] = index_i
            else:
                if np.random.rand() < np.exp(Delta_ij):
                    pt_TtoE[index_i] = index_energy_j
                    pt_TtoE[index_j] = index_energy_i
                    pt_EtoT[index_energy_i] = index_j
                    pt_EtoT[index_energy_j] = index_i
        
        # Change the pair swapper for next run
        swap_even_pairs = (1 - swap_even_pairs)
    
    print(f'Thermalization completed in {time.time() - start_time:.2f} seconds')
    
    # Measurement stage
    print('Starting measurements...')
    start_time = time.time()
    
    all_data_thermo = np.zeros((num_temps, measure_steps*measure_steps, 7))
    
    # Measurement loop
    swap_even_pairs = 0
    
    for il in tqdm(range(measure_steps), desc="Measurement", unit="step"):
        # Sequential measurements
        resultsMeasure = []
        for m in range(num_temps):
            result = PTTstepMeasure(config_init=config_at_T[m],
                                   temp=list_temps[pt_EtoT[m]], 
                                   N=N, 
                                   neighbors_list=neighbors_list,
                                   factorIter=measure_steps*niters[pt_EtoT[m]])
            resultsMeasure.append(result)
        
        # Process measurement results
        for q in range(num_temps):
            list_energies[q] = resultsMeasure[q][1]
            config_at_T[q] = resultsMeasure[q][0]
        
        # Parallel tempering step
        tuples_used = tuples_tot[swap_even_pairs]
        for sw in range(len_tuples_tot[swap_even_pairs]):
            index_i = tuples_used[sw][0]
            index_j = tuples_used[sw][1]
            initial_i_temp = list_temps[index_i]
            initial_j_temp = list_temps[index_j]
            index_energy_i = pt_TtoE[index_i]
            index_energy_j = pt_TtoE[index_j]
            
            Delta_ij = (list_energies[index_energy_i] - list_energies[index_energy_j])*(1/initial_i_temp - 1/initial_j_temp)
            if Delta_ij > 0:
                pt_TtoE[index_i] = index_energy_j
                pt_TtoE[index_j] = index_energy_i
                pt_EtoT[index_energy_i] = index_j
                pt_EtoT[index_energy_j] = index_i
            else:
                if np.random.rand() < np.exp(Delta_ij):
                    pt_TtoE[index_i] = index_energy_j
                    pt_TtoE[index_j] = index_energy_i
                    pt_EtoT[index_energy_i] = index_j
                    pt_EtoT[index_energy_j] = index_i
        
        # Change the pair swapper for next run
        swap_even_pairs = (1 - swap_even_pairs)
        
        # Read and save the measurement data
        for q in range(num_temps):
            data_extract = resultsMeasure[pt_TtoE[q]][2]
            for ws in range(measure_steps):
                all_data_thermo[q][measure_steps*il + ws] = data_extract[ws]
    
    print(f'Measurements completed in {time.time() - start_time:.2f} seconds')
    
    # Export data
    factor_print = 10000  # For temperature formatting in filenames
    for q in range(num_temps):
        temp_init = list_temps[q]
        np.savetxt(f"{name_dir}/configatT={int(temp_init*factor_print):05d}.data", config_at_T[pt_TtoE[q]])
        np.savetxt(f"{name_dir}/outputatT={int(temp_init*factor_print):05d}.data", all_data_thermo[q])
    
    print('Done with exporting data')
    return True 

############################
# Data Analysis Functions
############################

def run_analysis(L, force=False):
    """Run the data analysis for a specific system size"""
    # Define directory paths
    data_dir = f"testL={L}"
    folder_data_final = f"testL={L}finalData"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(folder_data_final):
        os.makedirs(folder_data_final)
    
    print(f"Starting data analysis for L={L}...")
    
    # Load simulation parameters
    if not os.path.exists(f"{data_dir}/variables.data"):
        print(f"Error: variables.data not found in {data_dir}")
        return False
    
    saving_variables = np.loadtxt(f"{data_dir}/variables.data")
    length_box = int(saving_variables[0])
    number_box = int(saving_variables[1])
    range_temp = saving_variables[3:]
    np.savetxt(f"{folder_data_final}/variables.data", saving_variables)
    
    # Number of temperature steps
    nt = len(range_temp)
    factor_print = 10000  # For temperature formatting in filenames
    
    print(f"Analyzing {nt} temperature points...")
    
    # Initialize arrays for measurements
    Energy = np.zeros(2*nt)
    SpecificHeat = np.zeros(2*nt)
    OrderCumulant = np.zeros(2*nt)
    OrderParam = np.zeros(2*nt)
    OrderParam_BIS = np.zeros(2*nt)
    Susceptibility1 = np.zeros(2*nt)
    Susceptibility2 = np.zeros(2*nt)
    BinderCumulant = np.zeros(2*nt)
    CorrLengthU = np.zeros(2*nt)
    RhoTot = np.zeros(2*nt)
    fourthOrderTot = np.zeros(2*nt)
    Vorticity = np.zeros(2*nt)
    
    # Process data for each temperature
    for m in tqdm(range(nt), desc="Analyzing temperatures", unit="temp"):
        data_file = f"{data_dir}/outputatT={int(range_temp[m]*factor_print):05d}.data"
        if not os.path.exists(data_file):
            print(f"Error: {data_file} not found")
            continue
        
        data = np.loadtxt(data_file)
        
        # Extract basic measurements
        E1_init = np.divide(np.array(data[:,0]), L**2)
        M1_init = np.divide(np.array(data[:,1]) + 1j*np.array(data[:,2]), L**2)
        ordU_xi = np.divide(np.array(data[:,3]), L**4) 
        ordU_xi_0 = np.divide(np.array(data[:,1])**2 + np.array(data[:,2])**2, L**4) 
        
        # Derived quantities
        E1 = E1_init
        E2 = E1*E1
        E4 = E2*E2
        
        M1_real = np.real(M1_init)
        M1_imag = np.imag(M1_init)
        M1_tot = np.absolute(M1_init)
        M2 = np.absolute(M1_init)**2
        M4 = M2*M2
        
        # Jackknife error analysis
        E1_avg, E1_error = JackknifeErrorFromFullList(E1, number_box, length_box)
        E2_avg, E2_error = JackknifeErrorFromFullList(E2, number_box, length_box)
        E4_avg, E4_error = JackknifeErrorFromFullList(E4, number_box, length_box)
        
        M1_real_avg, M1_real_error = JackknifeErrorFromFullList(M1_real, number_box, length_box)
        M1_imag_avg, M1_imag_error = JackknifeErrorFromFullList(M1_imag, number_box, length_box)
        M1_avg, M1_error = JackknifeErrorFromFullList(M1_tot, number_box, length_box)
        M2_avg, M2_error = JackknifeErrorFromFullList(M2, number_box, length_box)
        M4_avg, M4_error = JackknifeErrorFromFullList(M4, number_box, length_box)
        
        ordU_xi_avg, ordU_xi_err = JackknifeErrorFromFullList(ordU_xi, number_box, length_box)
        ordU_xi_0_avg, ordU_xi_0_err = JackknifeErrorFromFullList(ordU_xi_0, number_box, length_box)
        
        # Energy related observables
        Energy[m] = E1_avg
        Energy[nt + m] = E1_error
        
        div_sp = (range_temp[m]**2)
        SpecificHeat[m] = (E2_avg - E1_avg**2)/div_sp 
        SpecificHeat[nt + m] = (((E2_error)**2 + (2*E1_error*E1_avg)**2)**(0.5))/div_sp
        
        ord_cum = (E4_avg)/(E2_avg**2) - 1
        OrderCumulant[m] = ord_cum
        OrderCumulant[nt + m] = np.fabs(ord_cum)*np.sqrt((E4_error/E4_avg)**2 + (2*E2_error/E2_avg)**2)
        
        # Order parameter related observables
        u_op = M1_real_avg**2 + M1_imag_avg**2
        u_op_err = np.sqrt((2*M1_real_error*M1_real_avg)**2 + (2*M1_imag_error*M1_imag_avg)**2)
        
        OrderParam[m] = np.sqrt(u_op)
        OrderParam[nt + m] = 0.5*u_op_err/np.sqrt(u_op)
        
        OrderParam_BIS[m] = M1_avg
        OrderParam_BIS[nt + m] = M1_error
        
        Susceptibility1[m] = (M2_avg - M1_avg**2)/(range_temp[m])
        Susceptibility1[nt + m] = np.sqrt((M2_error)**2 + (2*M1_avg*M1_error)**2)/(range_temp[m])
        
        Susceptibility2[m] = (M2_avg)/(range_temp[m])
        Susceptibility2[nt + m] = (M2_error)/(range_temp[m])
        
        bind_cum = M4_avg/(M2_avg**2)
        BinderCumulant[m] = 1 - bind_cum/3
        BinderCumulant[nt + m] = (1/3)*np.fabs(bind_cum)*np.sqrt((M4_error/M4_avg)**2 + (2*M2_error/M2_avg)**2) 
        
        # Correlation length
        fact_sin = (1/(2*np.sin(np.pi/L))**2)
        val_U_c = ((ordU_xi_0_avg/ordU_xi_avg) - 1)
        CorrLengthU[m] = fact_sin*val_U_c
        CorrLengthU[nt + m] = fact_sin*(np.sqrt((ordU_xi_0_err/ordU_xi_0_avg)**2 + (ordU_xi_err/ordU_xi_avg)**2))
        
        # Stiffness
        Stiff_tot_H = data[:,4]/L**2
        Stiff_tot_I = data[:,5]/L**2
        Stiff_tot_I2 = Stiff_tot_I*Stiff_tot_I
        Stiff_tot_I4 = Stiff_tot_I2*Stiff_tot_I2
        
        Stiff_tot_H_avg, Stiff_tot_H_error = JackknifeErrorFromFullList(Stiff_tot_H, number_box, length_box)
        Stiff_tot_I_avg, Stiff_tot_I_error = JackknifeErrorFromFullList(Stiff_tot_I, number_box, length_box)
        Stiff_tot_I2_avg, Stiff_tot_I2_error = JackknifeErrorFromFullList(Stiff_tot_I2, number_box, length_box)
        Stiff_tot_I4_avg, Stiff_tot_I4_error = JackknifeErrorFromFullList(Stiff_tot_I4, number_box, length_box)
        
        T = range_temp[m]    
        RhoTot[m] = Stiff_tot_H_avg - (L**2/T)*(Stiff_tot_I2_avg - Stiff_tot_I_avg**2)
        RhoTot[nt + m] = np.sqrt(Stiff_tot_H_error**2 + ((L**2)*Stiff_tot_I2_error/T)**2 + ((L**2)*2*Stiff_tot_I_error*Stiff_tot_I_avg/T)**2)
        
        # Fourth order observables
        list_Ysq_tot = (Stiff_tot_H - (L**2/T)*(Stiff_tot_I2 - Stiff_tot_I**2))**2
        list_Ysq_tot_avg, list_Ysq_tot_error = JackknifeErrorFromFullList(list_Ysq_tot, number_box, length_box)
        
        fourthOrderTot[m] = (-1)*4*RhoTot[m] + 3*(Stiff_tot_H_avg - (L**2/T)*(list_Ysq_tot_avg - RhoTot[m]**2)) + 2*(L**6/(T**3))*Stiff_tot_I4_avg
        
        # Vorticity
        Vort = data[:,6]
        Vort_avg, Vort_error = JackknifeErrorFromFullList(Vort, number_box, length_box)
        Vorticity[m] = Vort_avg/L**2
        Vorticity[nt + m] = Vort_error/L**2
    
    # Save the processed data
    np.savetxt(f"{folder_data_final}/thermo_output.data",
              np.c_[Energy, SpecificHeat, OrderCumulant,
                   OrderParam, OrderParam_BIS, Susceptibility1, Susceptibility2, 
                   BinderCumulant, CorrLengthU, RhoTot, fourthOrderTot, Vorticity])
    
    print(f"Analysis for L={L} completed successfully")
    return True

def load_data(L):
    """Load the processed data for a specific system size"""
    base_path = f"testL={L}finalData"
    
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

############################
# Plotting Functions
############################

def create_combined_figure(system_sizes, output_dir="./plots", latex_available=True):
    """Create a combined figure with the key observables from the simulation"""
    # Create output directory
    ensure_dir(output_dir)
    
    # Extra safeguard: explicitly disable LaTeX if not available
    if not latex_available:
        plt.rcParams['text.usetex'] = False
        
    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Define colors for different system sizes
    colors = {}
    color_list = ['red', 'orange', 'green', 'blue', 'purple', 'brown']
    for i, L in enumerate(system_sizes):
        colors[f'L{L}'] = color_list[i % len(color_list)]
    
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
                               label=f"L = {L}" if not latex_available else f"$L = {L}$", 
                               marker='o', markersize=4, 
                               color=colors[f'L{L}'], alpha=0.8)
            
            # Spin Stiffness (top right)
            spin_stiffness = thermo_data[0:nt, 9]
            spin_stiffness_err = thermo_data[nt:(2*nt), 9]
            axes[0, 1].errorbar(temps, spin_stiffness, yerr=spin_stiffness_err, 
                               label=f"L = {L}" if not latex_available else f"$L = {L}$", 
                               marker='o', markersize=4, 
                               color=colors[f'L{L}'], alpha=0.8)
            
            # Plot the universal prediction line
            if L == system_sizes[0]:  # Only do this once
                if latex_available:
                    axes[0, 1].plot(temps, 2*temps/np.pi, 'k--', label=r'$\rho_s = \frac{2T}{\pi}$')
                else:
                    axes[0, 1].plot(temps, 2*temps/np.pi, 'k--', label='rho_s = 2T/π')
            
            # Vortex Density (bottom left)
            vortex_density = thermo_data[0:nt, 11]
            vortex_density_err = thermo_data[nt:(2*nt), 11]
            axes[1, 0].errorbar(temps, vortex_density, yerr=vortex_density_err, 
                               label=f"L = {L}" if not latex_available else f"$L = {L}$", 
                               marker='o', markersize=4, 
                               color=colors[f'L{L}'], alpha=0.8)
            
            # Susceptibility (bottom right)
            susc = thermo_data[0:nt, 5]
            susc_err = thermo_data[nt:(2*nt), 5]
            axes[1, 1].errorbar(temps, susc, yerr=susc_err, 
                               label=f"L = {L}" if not latex_available else f"$L = {L}$", 
                               marker='o', markersize=4, 
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
    axes[0, 0].set_xlabel("T/J" if not latex_available else "$T/J$")
    axes[0, 0].set_ylabel("Specific Heat per Site" if not latex_available else "Specific Heat per Site $c_v$")
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title("Spin Stiffness", fontsize=12)
    axes[0, 1].set_xlabel("T/J" if not latex_available else "$T/J$")
    axes[0, 1].set_ylabel("Spin stiffness" if not latex_available else "Spin stiffness $\\rho_s/J$")
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title("Vortex Density", fontsize=12)
    axes[1, 0].set_xlabel("T/J" if not latex_available else "$T/J$")
    axes[1, 0].set_ylabel("Density of vortices" if not latex_available else "Density of vortices $\\omega_v$")
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title("Susceptibility", fontsize=12)
    axes[1, 1].set_xlabel("T/J" if not latex_available else "$T/J$")
    axes[1, 1].set_ylabel("Spin susceptibility" if not latex_available else "Spin susceptibility $\chi$")
    axes[1, 1].set_yscale('log')  # Use log scale for susceptibility
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add a legend to the top right plot
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), 
              ncol=len(system_sizes), frameon=False)
    
    # Use tight_layout only if LaTeX is available, otherwise use a simpler layout
    if latex_available:
        try:
            plt.tight_layout()
        except:
            # If tight_layout fails, fall back to basic layout
            print("Warning: tight_layout failed, using basic layout")
            plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1, wspace=0.3, hspace=0.3)
    else:
        # Just use basic layout adjustment for non-LaTeX mode
        plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1, wspace=0.3, hspace=0.3)
        
    # Always adjust top margin for the legend
    plt.subplots_adjust(top=0.9)  # Make room for the legend
    
    # Add a text annotation with observed T_KT values
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
    
    # If we have enough data points, perform finite size scaling
    if len(L_values) >= 3:
        T_KT_inf, a, x_fit, y_fit = finite_size_scaling(T_KT_values, L_values)
        
        # Create a separate figure for finite size scaling
        fig_scaling, ax_scaling = plt.subplots(figsize=(8, 6))
        
        # Extra safeguard: explicitly disable LaTeX for this plot too if not available
        if not latex_available:
            plt.rcParams['text.usetex'] = False
            
        # Plot the data points with error bars
        x_data = 1.0 / (np.log(np.array(L_values)))**2
        ax_scaling.errorbar(x_data, T_KT_values, yerr=T_KT_errors, fmt='o', 
                           color='blue', label='Data')
        
        # Plot the fit line
        if latex_available:
            ax_scaling.plot(x_fit, y_fit, 'r-', label=f'$T_{{KT}}(\\infty) = {T_KT_inf:.3f}$')
        else:
            ax_scaling.plot(x_fit, y_fit, 'r-', label=f'T_KT(inf) = {T_KT_inf:.3f}')
        
        # Highlight the extrapolated T_KT(∞) at x=0
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
        
        # Use safe layout settings instead of tight_layout
        plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        
        # Save the scaling figure
        plt.savefig(os.path.join(output_dir, "finite_size_scaling.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_scaling)
    
    # Save the main figure with tight_bbox to avoid layout issues
    plt.savefig(os.path.join(output_dir, "bkt_results.png"), dpi=300, bbox_inches='tight')
    print(f"Combined figure saved to {os.path.join(output_dir, 'bkt_results.png')}")
    plt.close(fig) 

############################
# Main Function
############################

def check_requirements():
    """Check if all required packages are available and print information"""
    requirements = {
        "numpy": True,  # We've already imported it, so it's available
        "matplotlib": True,  # We've already imported it, so it's available
        "scipy": True,  # We've already imported it, so it's available
        "numba": True,  # We've already imported it, so it's available
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
            print("  This is normal in Kaggle/Colab environments.")
    
    return len(missing) == 0

def main():
    """Main function to run simulations, analysis, and create plots"""
    # Check requirements
    all_requirements_met = check_requirements()
    if not all_requirements_met:
        print("Continuing with reduced functionality...")
        print()
        
    # Configuration dictionary with default values
    config = {
        # General options
        "output_dir": "./plots",
        "force": False,
        "skip_sim": False,
        "skip_analysis": False,
        
        # Simulation parameters
        "temps": 15,             # Number of temperature points
        "temp_min": 0.7,         # Minimum temperature
        "temp_max": 1.1,         # Maximum temperature (covering the transition at ~0.89)
        "temp_type": 1,          # 0=geometric, 1=linear
        
        # Simulation steps
        "pretherm": 100,         # Pre-thermalization steps
        "therm": 200,            # Thermalization steps (reduced for efficiency)
        "measure": 500,          # Measurement steps (reduced for efficiency)
        
        # System sizes - focus on L=10 to L=30 range
        "sizes": [10, 15, 20, 25, 30]  # Good range of system sizes to see scaling
    }
    
    # MODIFY CONFIGURATION HERE IF NEEDED
    # For example:
    # config["sizes"] = [6, 8]  # Use smaller system sizes for quick testing
    # config["skip_sim"] = True  # Skip simulation and only do analysis/plotting
    
    # QUICK TESTING CONFIGURATIONS:
    # Fast test run (few minutes): config["sizes"] = [6, 8]; config["therm"] = 10; config["measure"] = 10
    # Medium test run (30min-1hr): config["sizes"] = [10, 15, 20]; config["therm"] = 100; config["measure"] = 200
    
    # PRODUCTION CONFIGURATIONS:
    # Better statistics: config["measure"] = 1000
    # Higher precision: config["temps"] = 20  # More temperature points
    # Publication quality: config["sizes"] = [20, 30, 40, 60, 80]; config["therm"] = 500; config["measure"] = 2000
    
    # Create output directory
    ensure_dir(config["output_dir"])
    
    # Record start time
    start_time = time.time()
    print(f"Starting BKT simulation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"System sizes: {config['sizes']}")
    print(f"Temperature range: {config['temp_min']} to {config['temp_max']} ({config['temps']} points)")
    print(f"Simulation steps: PreTherm={config['pretherm']}, Therm={config['therm']}, Measure={config['measure']}")
    
    # Estimate runtime based on system sizes and steps
    total_steps = config['therm'] + config['measure']
    largest_size = max(config['sizes'])
    # Very rough estimation - actual runtime depends on hardware, system sizes, etc.
    estimated_minutes = len(config['sizes']) * config['temps'] * total_steps * (largest_size/10)**2 / 1000
    if estimated_minutes < 60:
        print(f"Estimated runtime: {estimated_minutes:.1f} minutes")
    else:
        print(f"Estimated runtime: {estimated_minutes/60:.1f} hours")
    print("Note: This is a rough estimate, actual runtime may vary based on hardware")
    
    # Run simulations for each system size sequentially
    if not config["skip_sim"]:
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
    else:
        print("Skipping simulations as requested")
    
    # Run analysis
    if not config["skip_analysis"]:
        for L in tqdm(config["sizes"], desc="Running analysis", unit="size"):
            run_analysis(L, config["force"])
    else:
        print("Skipping analysis as requested")
    
    # Create plots
    create_combined_figure(config["sizes"], config["output_dir"], latex_available=latex_available)
    
    # Print total elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Plots saved to {config['output_dir']}")

if __name__ == "__main__":
    main()