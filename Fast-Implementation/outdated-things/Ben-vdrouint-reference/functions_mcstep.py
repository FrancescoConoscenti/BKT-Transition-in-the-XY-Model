#!/usr/bin/env python3
"""
Implementation of Monte Carlo update algorithms for the 2D XY model.
Contains Wolff cluster update, Metropolis single-spin update, energy calculation,
and measurement functions optimized with Numba.
"""
from __future__ import division
import numpy as np
from numpy import pi, cos, sin, exp, mod, absolute
from numba import jit


@jit(nopython=True)
def WolffUpdate(config, temp, N, neighbors_list):
    """
    Performs a Wolff cluster update on the XY model configuration.
    
    This algorithm identifies and flips a cluster of spins, where spins
    are added to the cluster with a probability based on their alignment.
    
    Args:
        config: Array of spin angles (shape N²)
        temp: Temperature
        N: System size (linear dimension)
        neighbors_list: Array of neighboring site indices
        
    Returns:
        Average cluster size
    """
    beta = 1.0 / temp
    
    # Total number of iterations equals system size
    num_sites = N * N
    
    # Initialize arrays for tracking cluster
    cluster = np.zeros(num_sites, dtype=np.int8)
    site_queue = np.zeros(num_sites + 1, dtype=np.int64)
    
    # Track average cluster size
    avg_size_clust = 0.0

    # Perform num_sites cluster updates
    for _ in range(num_sites):
        # Choose random site to start cluster
        init_site = np.random.randint(0, num_sites)
        site_queue[0] = init_site + 1  # +1 offset for 0-check termination
        
        # Choose random reflection angle
        random_angle = np.pi * np.random.rand()
        
        # Mark initial site as part of cluster
        cluster[init_site] = 1
        
        # Initialize queue indices
        current_idx = 0
        next_idx = 1

        # Process all sites in the cluster
        while site_queue[current_idx] != 0:
            # Get next site from queue (adjusting for the +1 offset)
            site = site_queue[current_idx] - 1
            current_idx += 1
            avg_size_clust += 1
            
            # Get current spin angle and reflect it
            site_angle = config[site]
            config[site] = 2 * random_angle - site_angle  # Reflect through random_angle
            
            # Check all neighbors
            for k in range(4):
                neighbor = neighbors_list[4*site + k]
                neighbor_angle = config[neighbor]
                
                # If neighbor not in cluster yet, try to add it
                if cluster[neighbor] == 0:
                    # Calculate energy difference from flipping this spin
                    energy_diff = -1 * (cos(site_angle - neighbor_angle) - 
                                        cos(site_angle - (2*random_angle - neighbor_angle)))
                    
                    # Calculate freezing probability
                    freeze_prob = 1.0 - exp(beta * energy_diff)
                    
                    # Add to cluster with calculated probability
                    if np.random.rand() < freeze_prob:
                        site_queue[next_idx] = neighbor + 1  # +1 offset
                        cluster[neighbor] = 1
                        next_idx += 1
        
        # Reset arrays for next iteration
        site_queue[:] = 0
        cluster[:] = 0

    # Calculate average cluster size
    return avg_size_clust / num_sites


@jit(nopython=True)
def MetropolisUpdate(config, temp, N, neighbors_list):
    """
    Performs a Metropolis single-spin update on the XY model configuration.
    
    This algorithm attempts to update each spin once by proposing a new random angle
    and accepting/rejecting based on the energy difference.
    
    Args:
        config: Array of spin angles (shape N²)
        temp: Temperature
        N: System size (linear dimension)
        neighbors_list: Array of neighboring site indices
    """
    beta = 1.0 / temp
    num_sites = N * N
    
    # Attempt to update each spin once
    for _ in range(num_sites):
        # Choose random site
        site = np.random.randint(0, num_sites)
        current_angle = config[site]
        
        # Propose new random angle
        new_angle = 2 * np.pi * np.random.rand()
        
        # Calculate energy difference
        energy_before = 0.0
        energy_after = 0.0
        
        # Sum over neighbors
        for k in range(4):
            neighbor = neighbors_list[4*site + k]
            neighbor_angle = config[neighbor]
            energy_before += cos(current_angle - neighbor_angle)
            energy_after += cos(new_angle - neighbor_angle)
        
        # Calculate energy cost (negative means favorable)
        energy_cost = energy_before - energy_after
        
        # Accept or reject based on Metropolis criterion
        if energy_cost <= 0 or np.random.rand() < exp(-beta * energy_cost):
            config[site] = new_angle


@jit(nopython=True)
def EnergyCalc(config, N):
    """
    Calculates the total energy of the XY model configuration.
    
    Args:
        config: Array of spin angles (shape N²)
        N: System size (linear dimension)
        
    Returns:
        Total energy of the configuration
    """
    energy = 0.0
    
    # Sum over all sites and their left & up neighbors
    for i in range(N):
        for j in range(N):
            site_angle = config[N*i + j]
            left_angle = config[N*((i-1) % N) + j]  # Periodic boundary
            up_angle = config[N*i + (j-1) % N]      # Periodic boundary
            
            # Add interaction energy (negative sign for ferromagnetic coupling)
            energy += -1.0 * (cos(site_angle - left_angle) + 
                              cos(site_angle - up_angle))
                              
    return energy


@jit(nopython=True)
def MeasureConfigNumba(config, N):
    """
    Measures various physical observables for the XY model configuration.
    
    Calculates energy, magnetization, correlation functions, and vortex density.
    
    Args:
        config: Array of spin angles (shape N²)
        N: System size (linear dimension)
        
    Returns:
        Array of measured quantities: 
        [energy, order_param_real, order_param_imag, correlation_length,
         stiffness_h, stiffness_i, vortex_density]
    """
    # Constant
    two_pi = 2 * pi
    
    # Reshape configuration for easier 2D access
    config_2d = config.reshape((N, N))
    
    # Initialize observables
    energy = 0.0
    h_tot = 0.0
    i_tot = 0.0
    ord1 = 0.0
    ordu_xi_x = 0.0
    ordu_xi_y = 0.0
    vort = 0.0
    
    # Loop over all lattice sites
    for i in range(N):
        for j in range(N):
            # Get angles at this site and neighbors (with periodic boundaries)
            angle = config_2d[i, j]
            angle_left = config_2d[i-1, j]
            angle_up = config_2d[i, j-1]
            angle_diag = config_2d[i-1, j-1]
            
            # Calculate cosine term for energy
            cos_term = cos(angle - angle_left)
            
            # Energy contribution
            energy += -1.0 * (cos_term + cos(angle - angle_up))
            
            # Stiffness terms
            h_tot += cos_term
            i_tot += sin(angle - angle_left)
            
            # Magnetization (complex order parameter)
            ord1 += exp(1j * angle)
            
            # Correlation length terms
            ordu_xi_x += cos(angle) * exp(1j * i * two_pi / N)
            ordu_xi_y += sin(angle) * exp(1j * i * two_pi / N)
            
            # Vortex calculation
            # First bring all angles to [0, 2π) range
            angle_mod = mod(angle, two_pi)
            angle_left_mod = mod(angle_left, two_pi)
            angle_diag_mod = mod(angle_diag, two_pi)
            angle_up_mod = mod(angle_up, two_pi)
            
            # Calculate angle differences around plaquette
            diff_list = np.array([angle_mod - angle_left_mod, 
                                  angle_left_mod - angle_diag_mod,
                                  angle_diag_mod - angle_up_mod, 
                                  angle_up_mod - angle_mod])
            
            # Calculate vorticity (normalize differences to [-π, π])
            vort_here = 0.0
            for diff in diff_list:
                if diff > pi:
                    diff = diff - two_pi
                elif diff < -pi:
                    diff = diff + two_pi
                vort_here += diff / two_pi
            
            # Add absolute vorticity
            vort += absolute(vort_here)
    
    # Normalize vorticity
    vort = vort / two_pi
    
    # Calculate squared magnitude of correlation functions
    ordu_xi_m = (ordu_xi_x.real)**2 + (ordu_xi_x.imag)**2 + (ordu_xi_y.real)**2 + (ordu_xi_y.imag)**2
    
    # Return array of measured quantities
    return np.array([energy, ord1.real, ord1.imag, ordu_xi_m, h_tot, i_tot, vort])