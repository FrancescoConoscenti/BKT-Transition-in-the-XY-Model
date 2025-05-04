#!/usr/bin/env python3
"""
Monte Carlo Parallel Tempering implementation for the 2D XY model.
Implements the Berezinskii–Kosterlitz–Thouless (BKT) phase transition simulation.
"""
from __future__ import division, print_function
import numpy as np
import time
import sys
import os
from joblib import Parallel, delayed
from numpy import pi, cos, sin, exp, mod, absolute
from numpy.random import rand, randint

# Import MC update functions
from functions_mcstep import WolffUpdate as mcUpdate
from functions_mcstep import EnergyCalc
from functions_mcstep import MeasureConfigNumba


def update_pre_therm(config_init, temp, N, neighbors_list, factorIter):
    """
    Pre-thermalization step to determine optimal cluster size.
    
    Args:
        config_init: Initial configuration
        temp: Temperature
        N: System size (length)
        neighbors_list: List of neighboring sites
        factorIter: Number of MC steps to perform
        
    Returns:
        Updated configuration and average cluster size
    """
    config = config_init.copy()
    avg_size_clust = 0.0
    
    for _ in range(factorIter):
        size_clust = mcUpdate(config, temp, N, neighbors_list)
        avg_size_clust += size_clust

    return [config, avg_size_clust / factorIter]


def ptt_step_therm(config_init, temp, N, neighbors_list, factorIter):
    """
    Thermalization step for parallel tempering.
    
    Args:
        config_init: Initial configuration
        temp: Temperature
        N: System size (length)
        neighbors_list: List of neighboring sites
        factorIter: Number of MC steps to perform
        
    Returns:
        Updated configuration, final energy, and energy trajectory
    """
    config = config_init.copy()
    energies = np.zeros(factorIter)

    for st in range(factorIter):
        mcUpdate(config, temp, N, neighbors_list)
        energies[st] = EnergyCalc(config, N)

    return [config, energies[-1], np.array(energies)]


def ptt_step_measure(config_init, temp, N, neighbors_list, factorIter):
    """
    Measurement step for parallel tempering.
    
    Args:
        config_init: Initial configuration
        temp: Temperature
        N: System size (length)
        neighbors_list: List of neighboring sites
        factorIter: Number of MC steps to perform
        
    Returns:
        Updated configuration, final energy, and thermodynamic data
    """
    config = config_init.copy()
    data_thermo = []
    
    for _ in range(factorIter):
        mcUpdate(config, temp, N, neighbors_list)
        data_thermo.append(MeasureConfigNumba(config, N))

    return [config, data_thermo[-1][0], np.array(data_thermo)]


def main():
    """Main function that runs the Monte Carlo simulation with parallel tempering."""
    # Parse command line arguments
    N = int(sys.argv[1])          # Linear system size L
    Tmin = float(sys.argv[2])     # Minimum temperature
    Tmax = float(sys.argv[3])     # Maximum temperature
    nt = int(sys.argv[4])         # Number of temperature points
    type_of_temp_range = int(sys.argv[5])  # Temperature spacing: 0=geometric, 1=linear
    num_cores = int(sys.argv[6])  # Number of cores for parallel computation
    therm = int(sys.argv[7])      # Number of thermalization bins
    number_box = int(sys.argv[8]) # Number of measurement bins
    
    # Fixed parameters
    factor_print = 10000          # Scaling factor for filenames
    jint = 1.0                    # Interaction strength
    length_box = 100              # Number of MC steps in each bin
    pre_therm = 100               # Number of pre-thermalization steps
    
    # Setup temperature range
    if type_of_temp_range == 0:
        # Geometric temperature range
        ratio_T = (Tmax/Tmin)**(1/(nt-1))
        range_temp = np.zeros(nt)
        for i in range(nt):
            range_temp[i] = Tmin * (ratio_T**i)
    else:
        # Linear temperature range
        range_temp = np.linspace(Tmin, Tmax, nt)
    
    list_temps = range_temp
    list_energies = np.zeros(nt)

    # Initialize neighbor list
    neighbors_list = np.zeros(4*(N**2), dtype=int)
    for i in range(N**2):
        site_studied_1 = i // N
        site_studied_2 = i % N
        vec_nn_x = [-1, 1, 0, 0]
        vec_nn_y = [0, 0, -1, 1]
        
        for p in range(4):
            neighbors_list[4*i + p] = (N * mod((site_studied_1 + vec_nn_x[p]), N) + 
                                       mod((site_studied_2 + vec_nn_y[p]), N))

    # Setup output directory
    name_dir = f'testL={N}'
    where_to_save = './'
    
    if not os.path.exists(where_to_save + name_dir):
        os.mkdir(where_to_save + name_dir)

    # Print simulation parameters
    print()
    print(f'Linear size of the system L={N}')
    print('Interaction strength:')
    print(f'J = {jint}')
    print()
    print(f'From temperature Tmax={Tmax} to Tmin={Tmin}')
    print(f'In {nt} steps')
    print()
    print(f'Size of bins: {length_box}')
    print(f'Number of thermalization bins: {therm}')
    print(f'Number of measurement bins: {number_box}')
    print(f'Number of cores: {num_cores}')
    print()

    # Initialize configurations
    config_at_T = np.array([2*pi*rand(N**2) for _ in range(nt)])

    # Setup for parallel tempering
    pt_TtoE = list(range(nt))  # Maps temperature index to energy (config) index
    pt_EtoT = list(range(nt))  # Maps energy (config) index to temperature index

    print('Starting the initialization step')
    print()
    start = time.time()

    print('List of temperatures:')
    print(list_temps)

    print('Starting pre-thermalization')
    
    # Pre-thermalization to determine optimal number of iterations
    niters = np.ones(nt, dtype=int)
    list_avg_clus_size = np.zeros(nt)

    with Parallel(n_jobs=num_cores, max_nbytes='10M') as parallel:
        results_pre_therm = parallel(delayed(update_pre_therm)(
            config_init=config_at_T[m],
            temp=list_temps[m],
            N=N,
            neighbors_list=neighbors_list,
            factorIter=pre_therm*niters[m]
        ) for m in range(nt))

        for q in range(nt):
            list_avg_clus_size[q] = results_pre_therm[q][1]
            config_at_T[q] = results_pre_therm[q][0]

    # Adjust number of iterations based on average cluster size
    niters = np.ceil((N**2)/list_avg_clus_size).astype(int)
    print('New number of iterations per temperature:')
    print(niters)

    end = time.time()
    print()
    print(f'Done with initialization in {end - start:.2f} seconds')
    print()

    # Save simulation parameters
    saving_variables_pre = np.array([length_box, number_box, therm])
    saving_variables = np.append(saving_variables_pre, list_temps)
    np.savetxt(f'{where_to_save}{name_dir}/variables.data', saving_variables)

    # Setup for parallel tempering swaps
    tuples_1 = [pt_TtoE[i:i+2] for i in range(0, len(pt_TtoE), 2)]      # odd switches
    tuples_2 = [pt_TtoE[i:i+2] for i in range(1, len(pt_TtoE)-1, 2)]    # even switches
    tuples_tot = [tuples_1, tuples_2]
    half_length = int(nt/2)
    len_tuples_tot = [half_length, half_length - 1]

    # Start thermalization
    print()
    print('Starting the thermalization')
    print()
    start = time.time()

    swap_even_pairs = 0  # Flag to alternate between even/odd swaps
    all_the_energies_thermalization = np.zeros((nt, therm*length_box))

    with Parallel(n_jobs=num_cores, max_nbytes='5M') as parallel:
        for il in range(therm):
            # Run MC steps at each temperature
            results_therm = parallel(delayed(ptt_step_therm)(
                config_init=config_at_T[m],
                temp=list_temps[pt_EtoT[m]],
                N=N,
                neighbors_list=neighbors_list,
                factorIter=length_box*niters[pt_EtoT[m]]
            ) for m in range(nt))
            
            for q in range(nt):
                list_energies[q] = results_therm[q][1]
                config_at_T[q] = results_therm[q][0]
                data_extract = results_therm[pt_TtoE[q]][2]
                for ws in range(length_box):
                    all_the_energies_thermalization[q][length_box*il + ws] = data_extract[ws]

            # Parallel tempering swap step
            tuples_used = tuples_tot[swap_even_pairs]
            for sw in range(len_tuples_tot[swap_even_pairs]):
                index_i = tuples_used[sw][0]
                index_j = tuples_used[sw][1]
                initial_i_temp = list_temps[index_i]
                initial_j_temp = list_temps[index_j]
                index_energy_i = pt_TtoE[index_i]
                index_energy_j = pt_TtoE[index_j]

                # Calculate Metropolis acceptance criterion
                Delta_ij = (list_energies[index_energy_i] - list_energies[index_energy_j]) * \
                           (1/initial_i_temp - 1/initial_j_temp)
                
                if Delta_ij > 0 or rand() < exp(Delta_ij):
                    # Swap the configurations
                    pt_TtoE[index_i] = index_energy_j
                    pt_TtoE[index_j] = index_energy_i
                    pt_EtoT[index_energy_i] = index_j
                    pt_EtoT[index_energy_j] = index_i

            # Toggle between even and odd pairs for next swap attempt
            swap_even_pairs = 1 - swap_even_pairs
            print(f'Done with therm step {il} out of {therm}')

    end = time.time()
    print()
    print('Done with thermalization')
    print(f'in {end - start:.2f} seconds')
    print(f'Number of steps: {therm*length_box}')
    print(f'Time per step (full PT + MC): {(end - start)/(therm*length_box):.6f}')
    print(f'Time per step per temp (full PT + MC): {(end - start)/(therm*length_box*nt):.6f}')
    print()

    # Display thermalization progress
    print('Comparing energies for the first and last quarter of thermalization:')
    print()
    for q in range(nt):
        startv1 = int(length_box*therm/2)
        endv1 = int(3*length_box*therm/4)
        startv2 = endv1
        endv2 = int(length_box*therm)
        
        print(f'T = {range_temp[q]:.6f}')
        print('First average:')
        dat1 = all_the_energies_thermalization[q][startv1:endv1]/(N*N)
        print(f'{np.mean(dat1):.6f} ± {np.std(dat1):.6f}')
        
        print('Second average:')
        dat2 = all_the_energies_thermalization[q][startv2:endv2]/(N*N)
        print(f'{np.mean(dat2):.6f} ± {np.std(dat2):.6f}')
        
        print(f'Difference: {np.mean(dat1) - np.mean(dat2):.6f}')
        print()

    # Start measurements
    print()
    print('Starting the measurements')
    print()
    start = time.time()

    all_data_thermo = np.zeros((nt, number_box*length_box, 7))
    swap_even_pairs = 0

    with Parallel(n_jobs=num_cores, max_nbytes='5M') as parallel:
        for il in range(number_box):
            # Run MC steps at each temperature
            results_measure = parallel(delayed(ptt_step_measure)(
                config_init=config_at_T[m],
                temp=list_temps[pt_EtoT[m]],
                N=N,
                neighbors_list=neighbors_list,
                factorIter=length_box*niters[pt_EtoT[m]]
            ) for m in range(nt))
            
            for q in range(nt):
                list_energies[q] = results_measure[q][1]
                config_at_T[q] = results_measure[q][0]

            # Parallel tempering swap step
            tuples_used = tuples_tot[swap_even_pairs]
            for sw in range(len_tuples_tot[swap_even_pairs]):
                index_i = tuples_used[sw][0]
                index_j = tuples_used[sw][1]
                initial_i_temp = list_temps[index_i]
                initial_j_temp = list_temps[index_j]
                index_energy_i = pt_TtoE[index_i]
                index_energy_j = pt_TtoE[index_j]

                # Calculate Metropolis acceptance criterion
                Delta_ij = (list_energies[index_energy_i] - list_energies[index_energy_j]) * \
                           (1/initial_i_temp - 1/initial_j_temp)
                
                if Delta_ij > 0 or rand() < exp(Delta_ij):
                    # Swap the configurations
                    pt_TtoE[index_i] = index_energy_j
                    pt_TtoE[index_j] = index_energy_i
                    pt_EtoT[index_energy_i] = index_j
                    pt_EtoT[index_energy_j] = index_i

            # Toggle between even and odd pairs for next swap attempt
            swap_even_pairs = 1 - swap_even_pairs

            # Save measurement data
            for q in range(nt):
                data_extract = results_measure[pt_TtoE[q]][2]
                for ws in range(length_box):
                    all_data_thermo[q][length_box*il + ws] = data_extract[ws]

            print(f'Done with measure step {il} out of {number_box}')

    end = time.time()
    print()
    print('Done with measurements')
    print(f'in {end - start:.2f} seconds')
    print(f'Number of PT steps: {number_box*length_box}')
    print(f'Time per step (full PT + MC): {(end - start)/(number_box*length_box):.6f}')
    print(f'Time per step per temp (full PT + MC): {(end - start)/(number_box*length_box*nt):.6f}')
    print()

    # Export data
    for q in range(nt):
        temp_init = list_temps[q]
        np.savetxt(
            f'{where_to_save}{name_dir}/configatT={int(temp_init*factor_print):05d}.data',
            config_at_T[pt_TtoE[q]]
        )
        np.savetxt(
            f'{where_to_save}{name_dir}/outputatT={int(temp_init*factor_print):05d}.data',
            all_data_thermo[q]
        )

    print()
    print('Done with exporting data')


if __name__ == '__main__':
    main()
