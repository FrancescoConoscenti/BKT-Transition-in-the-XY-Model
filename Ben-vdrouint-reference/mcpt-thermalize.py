#!/usr/bin/env python3
"""
Simplified Monte Carlo thermalization for the 2D XY model.

This version only performs thermalization steps without measurements,
making it suitable for generating thermalized configurations for visualization.
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


def perform_thermalization(config_init, temp, N, neighbors_list, factorIter):
    """
    Thermalization step for a single configuration.
    
    Args:
        config_init: Initial configuration
        temp: Temperature
        N: System size (length)
        neighbors_list: List of neighboring sites
        factorIter: Number of MC steps to perform
        
    Returns:
        Updated configuration and final energy
    """
    config = config_init.copy()

    for _ in range(factorIter):
        mcUpdate(config, temp, N, neighbors_list)

    # Calculate final energy
    energy = EnergyCalc(config, N)
    return [config, energy]


def main():
    """Main function that runs the Monte Carlo thermalization."""
    # Parse command line arguments
    if len(sys.argv) < 5:
        print("Usage: python3 mcpt-thermalize.py L temp pre_therm therm num_cores")
        sys.exit(1)
        
    N = int(sys.argv[1])          # Linear system size L
    temp = float(sys.argv[2])     # Temperature
    pre_therm = int(sys.argv[3])  # Number of pre-thermalization steps
    therm = int(sys.argv[4])      # Number of thermalization steps
    num_cores = int(sys.argv[5]) if len(sys.argv) > 5 else 4  # Number of cores (default: 4)
    
    # Fixed parameters
    factor_print = 10000          # Scaling factor for filenames
    jint = 1.0                    # Interaction strength
    length_box = 100              # Number of MC steps in each bin
    
    # Setup output directory
    name_dir = f'thermalized_L{N}_T{int(temp*factor_print):05d}'
    where_to_save = './'
    
    if not os.path.exists(where_to_save + name_dir):
        os.mkdir(where_to_save + name_dir)

    # Print simulation parameters
    print()
    print(f'Thermalization for XY model')
    print(f'Linear size of the system L={N}')
    print(f'Temperature T={temp}')
    print(f'Interaction strength J={jint}')
    print(f'Pre-thermalization steps: {pre_therm}')
    print(f'Thermalization steps: {therm}')
    print(f'Number of cores: {num_cores}')
    print()

    # Initialize configuration
    config = 2*pi*rand(N**2)

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

    print('Starting pre-thermalization')
    start = time.time()
    
    # Pre-thermalization to determine optimal number of iterations
    niters = 1
    
    with Parallel(n_jobs=num_cores, max_nbytes='10M') as parallel:
        results_pre_therm = parallel(delayed(update_pre_therm)(
            config_init=config,
            temp=temp,
            N=N,
            neighbors_list=neighbors_list,
            factorIter=pre_therm
        ) for _ in range(1))  # Only need one process for pre-thermalization

        config = results_pre_therm[0][0]
        avg_clust_size = results_pre_therm[0][1]

    # Adjust number of iterations based on average cluster size
    niters = max(1, int(np.ceil((N**2)/avg_clust_size)))
    print(f'Optimal iterations per therm step: {niters}')

    # Save pre-thermalized configuration
    np.savetxt(f'{where_to_save}{name_dir}/pretherm_config.data', config)
    
    end = time.time()
    print(f'Done with pre-thermalization in {end - start:.2f} seconds')
    
    # Start main thermalization
    print()
    print('Starting main thermalization')
    start = time.time()

    with Parallel(n_jobs=num_cores, max_nbytes='5M') as parallel:
        results_therm = parallel(delayed(perform_thermalization)(
            config_init=config,
            temp=temp,
            N=N,
            neighbors_list=neighbors_list,
            factorIter=therm*niters
        ) for _ in range(1))  # Only need one process for thermalization
        
        config = results_therm[0][0]
        energy = results_therm[0][1]

    end = time.time()
    print()
    print('Done with thermalization')
    print(f'in {end - start:.2f} seconds')
    print(f'Final energy: {energy}')
    print(f'Energy per site: {energy/(N*N):.6f}')
    
    # Save thermalized configuration
    output_file = f'{where_to_save}configL{N}_T{int(temp*factor_print):05d}.data'
    np.savetxt(output_file, config)
    print(f'Thermalized configuration saved to {output_file}')


if __name__ == '__main__':
    main() 