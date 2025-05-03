#!/usr/bin/env python3
"""
2D XY Model Simulation with Kosterlitz-Thouless Phase Transition Analysis

This module implements a high-performance Monte Carlo simulation of the classical
XY model on a 2D lattice, specifically designed to investigate the 
Kosterlitz-Thouless phase transition. The implementation uses Numba JIT compilation
for maximum computational efficiency.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import time
import os
import sys
import argparse
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any, Union
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numba
from numba import jit, njit, prange, float64, int64, boolean, types
from scipy.optimize import curve_fit # For finite-size scaling fit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class SimulationParameters:
    """Container for simulation parameters."""
    L: int  # Lattice size
    T_min: float  # Minimum temperature
    T_max: float  # Maximum temperature
    num_points: int  # Number of temperature points
    sweeps: int  # Monte Carlo sweeps per temperature
    thermalize_sweeps: int  # Thermalization sweeps
    J: float = 1.0  # Coupling constant
    temperatures: Optional[np.ndarray] = None # Field added for convenience


@dataclass
class SimulationResults:
    """Container for simulation results."""
    temperatures: np.ndarray
    energies: np.ndarray
    energy_errors: np.ndarray
    magnetizations: np.ndarray
    mag_errors: np.ndarray
    specific_heat: np.ndarray
    susceptibility: np.ndarray
    vortices: np.ndarray
    stiffnesses: np.ndarray
    stiffness_errors: np.ndarray
    T_BKT: float  # Estimated BKT transition temperature


# ============================================================================
# JIT-compiled computational kernels - MODIFIED SECTION
# ============================================================================

@njit(float64(float64[:,:], int64, float64))
def total_energy(spins, L, J):
    """
    Calculate the total energy of the system.
    
    Args:
        spins: 2D array of spin angles
        L: Linear size of the lattice
        J: Coupling constant
        
    Returns:
        float: Total energy of the system
    """
    energy = 0.0
    
    # Calculate with explicit loops for Numba optimization
    for i in range(L):
        for j in range(L):
            # Current spin
            theta = spins[i, j]
            sx, sy = np.cos(theta), np.sin(theta)
            
            # Right neighbor (with PBC)
            j_right = (j + 1) % L
            theta_right = spins[i, j_right]
            sx_right, sy_right = np.cos(theta_right), np.sin(theta_right)
            
            # Down neighbor (with PBC)
            i_down = (i + 1) % L
            theta_down = spins[i_down, j]
            sx_down, sy_down = np.cos(theta_down), np.sin(theta_down)
            
            # Add interaction energy (negative for ferromagnetic)
            energy -= J * (sx * sx_right + sy * sy_right)
            energy -= J * (sx * sx_down + sy * sy_down)
            
    return energy


@njit(float64(float64[:,:], int64, int64, int64, float64))
def site_energy(spins, i, j, L, J):
    """
    Calculate the energy contribution of a single site.
    
    Args:
        spins: 2D array of spin angles
        i, j: Site coordinates
        L: Linear size of the lattice
        J: Coupling constant
        
    Returns:
        float: Energy contribution of the site
    """
    # Get the spin angle
    theta = spins[i, j]
    sx, sy = np.cos(theta), np.sin(theta)
    
    # Get neighbor indices with periodic boundary conditions
    i_up = (i - 1) % L
    i_down = (i + 1) % L
    j_left = (j - 1) % L
    j_right = (j + 1) % L
    
    # Get neighbor spin components
    theta_up = spins[i_up, j]
    sx_up, sy_up = np.cos(theta_up), np.sin(theta_up)
    
    theta_down = spins[i_down, j]
    sx_down, sy_down = np.cos(theta_down), np.sin(theta_down)
    
    theta_left = spins[i, j_left]
    sx_left, sy_left = np.cos(theta_left), np.sin(theta_left)
    
    theta_right = spins[i, j_right]
    sx_right, sy_right = np.cos(theta_right), np.sin(theta_right)
    
    # Calculate energy contribution
    energy = 0.0
    energy -= J * (sx * sx_up + sy * sy_up)
    energy -= J * (sx * sx_down + sy * sy_down)
    energy -= J * (sx * sx_left + sy * sy_left)
    energy -= J * (sx * sx_right + sy * sy_right)
    
    return energy


@njit
def metropolis_sweep(spins, L, beta, J):
    """
    Perform one Monte Carlo sweep (L^2 attempted spin flips) using the Metropolis algorithm.
    
    Args:
        spins: 2D array of spin angles
        L: Linear size of the lattice
        beta: Inverse temperature (1/T)
        J: Coupling constant
        
    Returns:
        None (modifies spins in-place)
    """
    for _ in range(L * L):
        # Choose a random site
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        
        # Calculate current energy contribution
        old_energy = site_energy(spins, i, j, L, J)
        
        # Propose a new spin angle
        new_angle = np.random.uniform(0, 2*np.pi)
        
        # Store the old angle
        old_angle = spins[i, j]
        
        # Temporarily update the spin
        spins[i, j] = new_angle
        
        # Calculate new energy contribution
        new_energy = site_energy(spins, i, j, L, J)
        
        # Calculate energy difference
        delta_E = new_energy - old_energy
        
        # Metropolis acceptance criterion
        if delta_E <= 0 or np.random.random() < np.exp(-beta * delta_E):
            # Accept the move (already updated)
            pass
        else:
            # Reject the move and restore the old angle
            spins[i, j] = old_angle


@njit
def compute_vorticity(spins, L):
    """
    Calculate the vorticity field for the current spin configuration.
    
    Args:
        spins: 2D array of spin angles
        L: Linear size of the lattice
        
    Returns:
        2D array of vorticity per plaquette
    """
    vortices = np.zeros((L, L))
    
    for i in range(L):
        for j in range(L):
            # Spins at four corners of a plaquette (counter-clockwise)
            theta1 = spins[i, j]
            theta2 = spins[i, (j + 1) % L]
            theta3 = spins[(i + 1) % L, (j + 1) % L]
            theta4 = spins[(i + 1) % L, j]
            
            # Calculate angle differences (modulo 2π)
            dtheta1 = (theta2 - theta1 + np.pi) % (2 * np.pi) - np.pi
            dtheta2 = (theta3 - theta2 + np.pi) % (2 * np.pi) - np.pi
            dtheta3 = (theta4 - theta3 + np.pi) % (2 * np.pi) - np.pi
            dtheta4 = (theta1 - theta4 + np.pi) % (2 * np.pi) - np.pi

            # Sum angle differences; divide by 2π to get winding number
            winding = (dtheta1 + dtheta2 + dtheta3 + dtheta4) / (2 * np.pi)
            vortices[i, j] = np.round(winding)  # Should be 0, ±1 in practice
            
    return vortices


@njit
def calculate_stiffness_components(spins, L, J):
    """
    Calculate the components needed for the ensemble average of spin stiffness
    for a single configuration. Calculates sums for both x and y directions.
    
    Args:
        spins: 2D array of spin angles
        L: Linear size of the lattice
        J: Coupling constant
        
    Returns:
        tuple: (sum_cos_x, sum_sin_x, sum_cos_y, sum_sin_y)
               Components are J * sum(cos(delta_theta)) and J * sum(sin(delta_theta))
               for x and y bonds respectively.
    """
    sum_cos_x = 0.0
    sum_sin_x = 0.0
    sum_cos_y = 0.0
    sum_sin_y = 0.0
    N = L * L # Number of sites
    
    for i in range(L):
        for j in range(L):
            theta_current = spins[i, j]

            # X-direction bond: (i,j) to (i, j+1)
            j_right = (j + 1) % L
            theta_right = spins[i, j_right]
            delta_theta_x = theta_current - theta_right
            sum_cos_x += J * np.cos(delta_theta_x)
            sum_sin_x += J * np.sin(delta_theta_x)

            # Y-direction bond: (i,j) to (i+1, j)
            i_down = (i + 1) % L
            theta_down = spins[i_down, j]
            delta_theta_y = theta_current - theta_down
            sum_cos_y += J * np.cos(delta_theta_y)
            sum_sin_y += J * np.sin(delta_theta_y)

    return sum_cos_x, sum_sin_x, sum_cos_y, sum_sin_y


@njit
def calculate_observables(spins, L, J):
    """
    Calculate standard observables and stiffness components for a single configuration.
    
    Args:
        spins: 2D array of spin angles
        L: Linear size of the lattice
        J: Coupling constant
        
    Returns:
        tuple: (energy_per_site, magnetization, avg_abs_vorticity,
                sum_cos_x, sum_sin_x, sum_cos_y, sum_sin_y)
    """
    N = L * L
    # Energy
    energy = total_energy(spins, L, J) / N
    
    # Magnetization
    mx, my = 0.0, 0.0
    for i in range(L):
        for j in range(L):
            mx += np.cos(spins[i, j])
            my += np.sin(spins[i, j])
    
    mx /= N
    my /= N
    magnetization = np.sqrt(mx*mx + my*my)
    # Also return components needed for susceptibility calculation if preferred
    # (current method calculates <|m|>^2 and <|m|^2> later, which is fine)
    
    # Vorticity
    vortex_field = compute_vorticity(spins, L)
    avg_abs_vorticity = np.mean(np.abs(vortex_field))

    # Stiffness components
    sum_cos_x, sum_sin_x, sum_cos_y, sum_sin_y = calculate_stiffness_components(spins, L, J)

    return (energy, magnetization, avg_abs_vorticity,
            sum_cos_x, sum_sin_x, sum_cos_y, sum_sin_y)


# --- Wolff Cluster Algorithm (Adapted from Reference) ---
@njit(types.void(float64[:,:], int64, float64, float64))
def wolff_sweep(spins, L, beta, J):
    """
    Perform one effective Wolff sweep (L*L cluster attempts) for the XY model,
    using the logic adapted from the user-provided reference implementation.

    Args:
        spins (np.ndarray): 2D array of spin angles (modified in-place).
        L (int): Linear size of the lattice.
        beta (float): Inverse temperature (1/T).
        J (float): Coupling constant.
    """
    num_sites = L * L
    # Cluster marker array, reset for each cluster build attempt
    in_cluster = np.zeros((L, L), dtype=boolean)
    # Pre-allocate stack to max possible size (L*L)
    cluster_stack = np.empty((L * L, 2), dtype=np.int64)

    # Perform num_sites cluster build attempts
    for _ in range(num_sites):
        # 1. Choose random reflection angle from [0, 2*pi) - CORRECTED range
        phi_reflect = np.random.uniform(0, 2.0 * np.pi)

        # 2. Choose random initial site
        i0, j0 = np.random.randint(0, L), np.random.randint(0, L)

        # 3. Initialize stack and cluster markers for *this* cluster build
        in_cluster[:] = False # Reset for this cluster
        stack_idx = 0 # Reset stack pointer
        head = 0      # Reset queue head pointer

        # Add initial site to stack and mark
        cluster_stack[stack_idx] = (i0, j0)
        stack_idx += 1
        in_cluster[i0, j0] = True

        # 4. Grow and flip the cluster simultaneously (BFS)
        while head < stack_idx:
            i, j = cluster_stack[head]
            head += 1

            # Store original angle before flipping
            current_theta = spins[i, j]
            # Reflect the spin *as it's processed*
            spins[i, j] = (2.0 * phi_reflect - current_theta) % (2.0 * np.pi)

            # Neighbors coordinates
            neighbors_coords = [
                ((i - 1 + L) % L, j),
                ((i + 1) % L, j),
                (i, (j - 1 + L) % L),
                (i, (j + 1) % L)
            ]

            # Check neighbors for adding to cluster
            for ni, nj in neighbors_coords:
                if not in_cluster[ni, nj]:
                    neighbor_theta = spins[ni, nj] # Neighbor's current angle

                    # Calculate energy difference using reference method:
                    # Compare interaction E(orig_i, orig_j) vs E(orig_i, reflected_j)
                    energy_diff = -J * (np.cos(current_theta - neighbor_theta) -
                                       np.cos(current_theta - (2.0 * phi_reflect - neighbor_theta)))

                    # Freezing probability (Reference method, ensuring prob <= 1)
                    # P = 1 - exp(beta * dE). If dE >= 0, P <= 0. If dE < 0, P > 0.
                    prob_arg = beta * energy_diff
                    freeze_prob = 0.0
                    if prob_arg < 0: # Only add if energy diff is negative (favorable flip for neighbor)
                        freeze_prob = 1.0 - np.exp(prob_arg)
                        # Clamp probability just in case of potential float issues, though unlikely here
                        freeze_prob = min(freeze_prob, 1.0)

                    # Add neighbor to cluster?
                    if np.random.random() < freeze_prob:
                        in_cluster[ni, nj] = True
                        # Check for stack overflow before adding
                        if stack_idx < num_sites:
                            cluster_stack[stack_idx] = (ni, nj)
                            stack_idx += 1
                        else:
                             # This case should ideally not be reached if L*L is allocated
                             # but provides robustness. If stack is full, stop adding for this cluster.
                             break # Stop checking neighbors for this site
            # If stack overflow occurred in inner loop, break outer BFS loop too
            if stack_idx >= num_sites and head < stack_idx: # Check if we overflowed but still have items to process
                 # This indicates an issue, possibly log or handle if needed, but for now just stop cluster growth
                 break

# ============================================================================
# Main XY Model class - MODIFIED SECTION
# ============================================================================

class XYModel:
    """
    Implementation of the 2D XY model with Numba-accelerated Monte Carlo sampling.
    Supports both Metropolis and Wolff algorithms.
    """
    def __init__(self, L: int, J: float = 1.0):
        """ Initialize the 2D XY model. """
        self.L = L
        self.J = J
        self.spins = np.random.uniform(0, 2*np.pi, (L, L))
    
    def energy(self) -> float:
        """Calculate the total energy."""
        return total_energy(self.spins, self.L, self.J)
    
    def site_energy(self, i: int, j: int) -> float:
        """Calculate a site's energy."""
        return site_energy(self.spins, i, j, self.L, self.J)
    
    def metropolis_step(self, beta: float) -> None:
        """Perform one Metropolis sweep (L*L attempted flips)."""
        metropolis_sweep(self.spins, self.L, beta, self.J)
    
    def wolff_sweep(self, beta: float) -> None:
        """Perform one effective Wolff sweep using the adapted reference implementation."""
        wolff_sweep(self.spins, self.L, beta, self.J) # Call the global JIT function

    def compute_vorticity(self) -> np.ndarray:
        """Calculate the vorticity field."""
        return compute_vorticity(self.spins, self.L)
    
    def calculate_observables(self) -> Tuple[float, float, float, float, float, float, float]:
        """Calculate standard observables and stiffness components."""
        return calculate_observables(self.spins, self.L, self.J)


# ============================================================================
# Simulation Runner and Post-processing - MINOR CHANGES
# ============================================================================

def find_cross(stiffness: np.ndarray, line: np.ndarray, temperatures: np.ndarray) -> float:
    """
    Find the crossing point between the spin stiffness curve and the 2T/π line,
    which provides an estimate for the BKT transition temperature.
    
    Parameters:
    -----------
    stiffness : np.ndarray
        Spin stiffness values
    line : np.ndarray
        2T/π line values
    temperatures : np.ndarray
        Temperature values
        
    Returns:
    --------
    float
        Estimated transition temperature
    """
    difference = stiffness - line

    # Find index where the sign of the difference changes (i.e., crossing point)
    crossing_indices = np.where(np.diff(np.sign(difference)))[0]

    # If no crossing found, return NaN
    if len(crossing_indices) == 0:
        return np.nan

    # Get the exact crossing point using linear interpolation
    idx = crossing_indices[0]  # Use first crossing if multiple exist
    x0, x1 = temperatures[idx], temperatures[idx+1]
    y0, y1 = difference[idx], difference[idx+1]
    # Linear interpolation to find more precise x where crossing occurs
    x_cross = x0 - y0 * (x1 - x0) / (y1 - y0)
    
    return x_cross


def run_single_temperature(params: Tuple[SimulationParameters, float, int, str]) -> Tuple[float, Dict[str, Any]]:
    """
    Run simulation for a single temperature point using the specified algorithm.
    Calculates stiffness correctly from ensemble averages.
    
    Parameters:
    -----------
    params : Tuple
        (simulation_params, temperature, temp_index, algorithm)
        
    Returns:
    --------
    Tuple[float, Dict]
        (temperature, results dictionary including correct stiffness)
    """
    sim_params, temperature, temp_idx, algorithm = params
    L = sim_params.L
    sweeps = sim_params.sweeps
    thermalize_sweeps = sim_params.thermalize_sweeps
    N = L * L
    model = XYModel(L, J=sim_params.J)
    beta = 1.0 / temperature
    
    # --- Thermalization phase ---
    thermal_desc = f"L={L}, T={temperature:.4f} ({algorithm}): Thermalization"
    if algorithm == 'metropolis':
        for _ in tqdm(range(thermalize_sweeps), desc=thermal_desc, leave=False, disable=temp_idx > 0):
        model.metropolis_step(beta)
    elif algorithm == 'wolff':
        for _ in tqdm(range(thermalize_sweeps), desc=thermal_desc, leave=False, disable=temp_idx > 0):
            model.wolff_sweep(beta)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # --- Data collection phase ---
    energies = []
    magnetizations = []
    vorticities = []
    # Lists to store stiffness components for averaging
    sum_cos_x_list = []
    sum_sin_x_list = []
    sum_cos_y_list = []
    sum_sin_y_list = []


    collection_desc = f"L={L}, T={temperature:.4f} ({algorithm}): Data collection"
    for sweep_num in tqdm(range(sweeps), desc=collection_desc, leave=False, disable=temp_idx > 0):
        if algorithm == 'metropolis':
        model.metropolis_step(beta)
        elif algorithm == 'wolff':
            model.wolff_sweep(beta)
        
        # Collect observables and stiffness components
        energy, mag, vorticity, scx, ssx, scy, ssy = model.calculate_observables()
        energies.append(energy)
        magnetizations.append(mag)
        vorticities.append(vorticity)
        sum_cos_x_list.append(scx)
        sum_sin_x_list.append(ssx)
        sum_cos_y_list.append(scy)
        sum_sin_y_list.append(ssy)

    # Convert lists to numpy arrays for efficient calculation
    energies_arr = np.array(energies)
    magnetizations_arr = np.array(magnetizations)
    vorticities_arr = np.array(vorticities)
    sum_cos_x_arr = np.array(sum_cos_x_list)
    sum_sin_x_arr = np.array(sum_sin_x_list)
    sum_cos_y_arr = np.array(sum_cos_y_list)
    sum_sin_y_arr = np.array(sum_sin_y_list)

    # --- Calculate final averages and errors ---
    n_meas = len(energies_arr) # Number of measurements
    if n_meas < 2: # Need at least 2 points for std dev
         # Handle case with few measurements (e.g., return NaN or raise error)
         # For simplicity, returning NaNs here. Consider logging a warning.
         logger.warning(f"L={L}, T={temperature:.4f}: Too few measurements ({n_meas}) for error calculation.")
         stiffness_avg = np.nan
         stiffness_err = np.nan
         energy_avg = np.mean(energies_arr) if n_meas > 0 else np.nan
         energy_err = np.nan
         energy_sq_avg = np.mean(energies_arr**2) if n_meas > 0 else np.nan
         mag_avg = np.mean(magnetizations_arr) if n_meas > 0 else np.nan
         mag_err = np.nan
         mag_sq_for_chi = np.mean(magnetizations_arr**2) if n_meas > 0 else np.nan
         vorticity_avg = np.mean(vorticities_arr) if n_meas > 0 else np.nan

    else:
        # Calculate stiffness using the correct formula (average over x and y directions)
        # rho_s_x = <sum_cos_x>/N - beta * <(sum_sin_x)^2>/N
        # rho_s_y = <sum_cos_y>/N - beta * <(sum_sin_y)^2>/N
        avg_sum_cos_x = np.mean(sum_cos_x_arr)
        avg_sum_sin_x_sq = np.mean(sum_sin_x_arr**2)
        rho_s_x = avg_sum_cos_x / N - beta * avg_sum_sin_x_sq / N

        avg_sum_cos_y = np.mean(sum_cos_y_arr)
        avg_sum_sin_y_sq = np.mean(sum_sin_y_arr**2)
        rho_s_y = avg_sum_cos_y / N - beta * avg_sum_sin_y_sq / N

        # Average stiffness over directions (assuming isotropy)
        stiffness_avg = 0.5 * (rho_s_x + rho_s_y)

        # --- Error calculation for stiffness (using Jackknife for simplicity) ---
        # Note: Simple std error propagation is complex due to averages of products/squares.
        # Jackknife or Bootstrap is more robust. Here's a basic Jackknife:
        jk_stiffness_estimates = []
        for i in range(n_meas):
            jk_scx = np.delete(sum_cos_x_arr, i)
            jk_ssx = np.delete(sum_sin_x_arr, i)
            jk_scy = np.delete(sum_cos_y_arr, i)
            jk_ssy = np.delete(sum_sin_y_arr, i)

            jk_avg_scx = np.mean(jk_scx)
            jk_avg_ssx_sq = np.mean(jk_ssx**2)
            jk_rho_x = jk_avg_scx / N - beta * jk_avg_ssx_sq / N

            jk_avg_scy = np.mean(jk_scy)
            jk_avg_ssy_sq = np.mean(jk_ssy**2)
            jk_rho_y = jk_avg_scy / N - beta * jk_avg_ssy_sq / N

            jk_stiffness_estimates.append(0.5 * (jk_rho_x + jk_rho_y))

        jk_stiffness_estimates = np.array(jk_stiffness_estimates)
        stiffness_err = np.sqrt((n_meas - 1) / n_meas * np.sum((jk_stiffness_estimates - stiffness_avg)**2))

        # Calculate other averages and std errors (standard method)
        energy_avg = np.mean(energies_arr)
        energy_err = np.std(energies_arr, ddof=1) / np.sqrt(n_meas)
        energy_sq_avg = np.mean(energies_arr**2) # Needed for Cv

        mag_avg = np.mean(magnetizations_arr)
        mag_err = np.std(magnetizations_arr, ddof=1) / np.sqrt(n_meas)
        mag_sq_avg = np.mean(magnetizations_arr**2) # <|m|>^2, need <|m|^2> = <mx^2+my^2> for Chi
        # To calculate Chi correctly, need to collect mx, my separately or recalculate |m|^2
        # For now, assuming the original code's calculation of Chi based on <|m|> and <|m|^2>
        # (which itself relied on <|m|> and <|m|^2>) needs verification or recalculation.
        # Let's stick to the structure but note Chi might need adjustment.
        # Calculate <|m|^2> needed for Chi (requires recalculating from spins or collecting mx,my)
        # A simpler (though potentially less accurate for Chi) approach used in the original code
        # was to approximate <|m|^2> with <m_x^2 + m_y^2> (often calculated).
        # Let's retain the original structure for Chi for now, using <|m|> and <|m|^2> approx
        mag_sq_for_chi = np.mean(magnetizations_arr**2) # This is < |m|^2 >

        vorticity_avg = np.mean(vorticities_arr)
        # Vorticity error not typically plotted but can be calculated:
        # vorticity_err = np.std(vorticities_arr, ddof=1) / np.sqrt(n_meas) if n_meas >= 2 else np.nan

    
    # Aggregate results
    results = {
        'energy': energy_avg,
        'energy_error': energy_err,
        'energy_sq': energy_sq_avg, # Mean of squared energy per site
        'magnetization': mag_avg,
        'magnetization_error': mag_err,
        'magnetization_sq': mag_sq_for_chi, # Mean of squared magnetization magnitude
        'vorticity': vorticity_avg,
        'stiffness': stiffness_avg,
        'stiffness_error': stiffness_err,
        'model_state': model if temp_idx % (sim_params.num_points // min(4, sim_params.num_points) + 1) == 0 else None
    }
    
    return temperature, results


def simulate_lattice(sim_params: SimulationParameters, algorithm: str) -> Tuple[SimulationResults, List[Tuple[float, XYModel]]]:
    """
    Simulate the XY model for various temperatures using the specified algorithm.
    Uses corrected stiffness calculation via run_single_temperature.
    
    Parameters:
    -----------
    sim_params : SimulationParameters
        Simulation parameters (including L-dependent sweeps)
    algorithm : str
        'metropolis' or 'wolff'
        
    Returns:
    --------
    tuple
        (SimulationResults, sample_states)
    """
    L = sim_params.L
    temperatures = np.linspace(sim_params.T_min, sim_params.T_max, sim_params.num_points)
    N = L*L
    sim_params.temperatures = temperatures # Store temps in params object
    
    logger.info(f"Simulating XY model on {L}x{L} lattice using {algorithm} algorithm")
    
    # Arrays to store results
    energies = np.zeros(sim_params.num_points)
    energy_errors = np.zeros(sim_params.num_points)
    energy_sq_avg = np.zeros(sim_params.num_points) # Renamed for clarity
    magnetizations = np.zeros(sim_params.num_points)
    mag_errors = np.zeros(sim_params.num_points)
    mag_sq_avg = np.zeros(sim_params.num_points) # Renamed for clarity
    vortices = np.zeros(sim_params.num_points)
    stiffnesses = np.zeros(sim_params.num_points)
    stiffness_errors = np.zeros(sim_params.num_points)
    
    # Sample states
    sample_states = []
    
    start_time = time.time()
    
    params_list = [(sim_params, T, i, algorithm) for i, T in enumerate(temperatures)]

    max_workers = mp.cpu_count()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results_iter = list(tqdm(
            executor.map(run_single_temperature, params_list),
            total=len(params_list),
            desc=f"L={L} ({algorithm}) Progress"
        ))
    
    # Process results
    for T, result_data in results_iter:
        i = np.where(temperatures == T)[0][0]
        energies[i] = result_data['energy']
        energy_errors[i] = result_data['energy_error']
        energy_sq_avg[i] = result_data['energy_sq'] # Store <E_site^2>
        magnetizations[i] = result_data['magnetization']
        mag_errors[i] = result_data['magnetization_error']
        mag_sq_avg[i] = result_data['magnetization_sq'] # Store <|m|^2>
        vortices[i] = result_data['vorticity']
        stiffnesses[i] = result_data['stiffness']
        stiffness_errors[i] = result_data['stiffness_error']

        if result_data['model_state'] is not None:
            sample_states.append((T, result_data['model_state']))
    
    elapsed = time.time() - start_time
    logger.info(f"Simulation for L={L} ({algorithm}) completed in {elapsed:.1f} seconds")
    
    # Calculate derived quantities
    beta = 1.0 / temperatures # Use array of betas

    # Specific Heat C_v / N = beta^2 * N * (<E_site^2> - <E_site>^2)
    specific_heat = beta**2 * N * (energy_sq_avg - energies**2)

    # Susceptibility Chi / N = beta * N * (<|m|^2> - <|m|>^2)
    # Note: Using <|m|^2> approximation here. For better accuracy,
    # calculate <m_x^2 + m_y^2> explicitly during sampling.
    susceptibility = beta * N * (mag_sq_avg - magnetizations**2)

    # Calculate BKT transition temperature using the *corrected* stiffness
    line = 2/np.pi * temperatures
    T_BKT = find_cross(stiffnesses, line, temperatures)
    logger.info(f"L={L} ({algorithm}): Estimated BKT T_BKT ≈ {T_BKT:.4f}")
    
    # Store results
    sim_results = SimulationResults(
        temperatures=temperatures,
        energies=energies,
        energy_errors=energy_errors,
        magnetizations=magnetizations,
        mag_errors=mag_errors,
        specific_heat=specific_heat,
        susceptibility=susceptibility,
        vortices=vortices,
        stiffnesses=stiffnesses,
        stiffness_errors=stiffness_errors, # Now correctly calculated
        T_BKT=T_BKT
    )
    
    return sim_results, sample_states

# ============================================================================
# Visualization and Main Script - NO CHANGES NEEDED FOR PLOTS
# ============================================================================

def visualize_plots(results_dict: Dict[int, SimulationResults], output_dir: str, algorithm: str) -> None:
    """
    Create comparison plots for different lattice sizes.
    
    Parameters:
    -----------
    results_dict : Dict[int, SimulationResults]
        Dictionary mapping lattice sizes to simulation results
    output_dir : str
        Directory to save plot files
    algorithm : str
        Algorithm used ('metropolis' or 'wolff')
    """
    plt.style.use('ggplot')
    lattice_sizes = sorted(results_dict.keys())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(lattice_sizes)))
    
    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 16), dpi=150)
    
    # Energy vs Temperature
    ax = axes[0, 0]
    for i, L in enumerate(lattice_sizes):
        results = results_dict[L]
        ax.errorbar(
            results.temperatures, results.energies, yerr=results.energy_errors,
            marker='o', markersize=4, linestyle='-', linewidth=1.5,
            color=colors[i], label=f'L = {L}'
        )
    ax.set_xlabel('Temperature (T)', fontsize=12)
    ax.set_ylabel('Energy per site', fontsize=12)
    ax.set_title('Energy vs Temperature', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Magnetization vs Temperature
    ax = axes[0, 1]
    for i, L in enumerate(lattice_sizes):
        results = results_dict[L]
        ax.errorbar(
            results.temperatures, results.magnetizations, yerr=results.mag_errors,
            marker='o', markersize=4, linestyle='-', linewidth=1.5,
            color=colors[i], label=f'L = {L}'
        )
    ax.set_xlabel('Temperature (T)', fontsize=12)
    ax.set_ylabel('Magnetization', fontsize=12)
    ax.set_title('Magnetization vs Temperature', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Specific Heat vs Temperature
    ax = axes[1, 0]
    for i, L in enumerate(lattice_sizes):
        results = results_dict[L]
        ax.plot(
            results.temperatures, results.specific_heat,
            marker='o', markersize=4, linestyle='-', linewidth=1.5,
            color=colors[i], label=f'L = {L}'
        )
        # Consider adding error bars for Cv if calculable (e.g., via Jackknife on energy_sq)
    ax.set_xlabel('Temperature (T)', fontsize=12)
    ax.set_ylabel('Specific Heat per Site', fontsize=12) # Updated label
    ax.set_title('Specific Heat vs Temperature', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Susceptibility vs Temperature
    ax = axes[1, 1]
    for i, L in enumerate(lattice_sizes):
        results = results_dict[L]
        ax.plot(
            results.temperatures, results.susceptibility,
            marker='o', markersize=4, linestyle='-', linewidth=1.5,
            color=colors[i], label=f'L = {L}'
        )
        # Consider adding error bars for Chi if calculable
    ax.set_xlabel('Temperature (T)', fontsize=12)
    ax.set_ylabel('Susceptibility per Site', fontsize=12) # Updated label
    ax.set_title('Susceptibility vs Temperature', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Vortex Density vs Temperature
    ax = axes[2, 0]
    for i, L in enumerate(lattice_sizes):
        results = results_dict[L]
        ax.plot(
            results.temperatures, results.vortices,
            marker='o', markersize=4, linestyle='-', linewidth=1.5,
            color=colors[i], label=f'L = {L}'
        )
    ax.set_xlabel('Temperature (T)', fontsize=12)
    ax.set_ylabel('Vortex Density', fontsize=12)
    ax.set_title('Vortex Density vs Temperature', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Stiffness vs Temperature
    ax = axes[2, 1]
    T_plot = None # Keep track of temperatures for the theoretical line
    for i, L in enumerate(lattice_sizes):
        results = results_dict[L]
        T_plot = results.temperatures # Store last temperature array
        # Plot stiffness with error bars
        ax.errorbar(
            results.temperatures, results.stiffnesses, yerr=results.stiffness_errors,
            marker='o', markersize=4, linestyle='-', linewidth=1.5,
            color=colors[i], label=f'L = {L}',
            elinewidth=1, capsize=3, alpha=0.8
        )
        # Faint line connecting markers underneath error bars
        # ax.plot(results.temperatures, results.stiffnesses, marker='', linestyle='-', color=colors[i], alpha=0.5)

    # Ensure we have temperatures to plot the theoretical line
    if T_plot is not None:
        line = 2/np.pi * T_plot
        ax.plot(T_plot, line, 'k--', label=r'$\frac{2}{\pi}T$', linewidth=1.5)
    else:
        logger.warning("Could not plot theoretical 2T/pi line for stiffness.")

    
    # Add transition temperature markers
    for i, L in enumerate(lattice_sizes):
        results = results_dict[L]
        if results.T_BKT is not None and not np.isnan(results.T_BKT):
            ax.axvline(results.T_BKT, color=colors[i], linestyle=':', alpha=0.7)
            # Adjust text position slightly to avoid overlap
            text_y_pos = 0.1 + 0.05 * (i % 4) # Simple vertical staggering
            ax.text(
                results.T_BKT, text_y_pos, f'T_BKT(L={L})={results.T_BKT:.3f}',
                color=colors[i], rotation=90, fontsize=8,
                verticalalignment='bottom'
            )
    
    ax.set_xlabel('Temperature (T)', fontsize=12)
    ax.set_ylabel('Spin Stiffness', fontsize=12)
    ax.set_title('Spin Stiffness vs Temperature with BKT Transition', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0])) # Ensure y=0 is visible if stiffness goes negative
    
    # Add a summary of transition temperatures in a text box
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    textstr = "BKT Transition Temperature Estimates:\n"
    valid_tbkt_count = 0
    for L in lattice_sizes:
        results = results_dict[L]
        if results.T_BKT is not None and not np.isnan(results.T_BKT):
        textstr += f"L = {L}: T_BKT = {results.T_BKT:.4f}\n"
            valid_tbkt_count += 1
        else:
            textstr += f"L = {L}: T_BKT = NaN\n"
    textstr += f"Theoretical: T_BKT ≈ 0.8935"
    # Place text box relative to figure, adjust position if needed
    fig.text(0.5, 0.01, textstr, fontsize=12, bbox=props, ha='center', va='bottom')

    plt.tight_layout(rect=[0, 0.05, 1, 0.98]) # Adjust rect to make space for text
    main_title = f'XY Model Simulation Results ({algorithm.capitalize()} Algorithm)'
    fig.suptitle(main_title, y=0.995) # Add main title slightly higher
    plot_filename = f'xy_model_comparison_{algorithm}.png'
    plt.savefig(os.path.join(output_dir, plot_filename), dpi=300, bbox_inches='tight')
    plt.close(fig)


def visualize_spins(samples_dict: Dict[int, List[Tuple[float, XYModel]]], output_dir: str, algorithm: str) -> None:
    """
    Visualize spin configurations for different lattice sizes and temperatures.
    
    Parameters:
    -----------
    samples_dict : Dict[int, List[Tuple[float, XYModel]]]
        Dictionary mapping lattice sizes to lists of (temperature, model) tuples
    output_dir : str
        Directory to save plot files
    algorithm : str
        Algorithm name for filename
    """
    for L, sample_states in samples_dict.items():
        if not sample_states:
            continue
            
        num_samples = len(sample_states)
        fig, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 5), dpi=150, squeeze=False) # Ensure axes is 2D

        for idx, (temp, model) in enumerate(sample_states):
            ax = axes[0, idx]
            # Create mesh grid for the arrows
            x, y = np.meshgrid(np.arange(0, L), np.arange(0, L))
            
            # Get the spin components
            u = np.cos(model.spins)
            v = np.sin(model.spins)
            
            # Plot the arrows
            ax.quiver(
                x, y, u, v,
                pivot='mid',                # Arrows centered on grid points
                scale=25,                   # Adjust based on arrow length
                scale_units='width',        # Arrow length relative to figure width
                width=0.005,                # Thickness of arrow shaft
                headwidth=4,                # Width of arrowhead
                headlength=5,               # Length of arrowhead
                headaxislength=4,           # How far arrowhead extends back
                color='black'               # Color of arrows
            )
            
            # Calculate and overlay vortices
            vortices = model.compute_vorticity()
            for i_lat in range(L):
                for j_lat in range(L):
                    vort_val = vortices[i_lat, j_lat]
                    if vort_val > 0.5: # Check for +1 vortex
                        ax.plot(j_lat, i_lat, 'ro', markersize=4, alpha=0.7)  # Positive vortex
                    elif vort_val < -0.5: # Check for -1 vortex
                        ax.plot(j_lat, i_lat, 'bo', markersize=4, alpha=0.7)  # Negative vortex

            ax.set_title(f'L = {L}, T = {temp:.4f}')
            ax.set_xlim(-1, L)
            ax.set_ylim(-1, L)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        spin_filename = f'xy_model_spins_L{L}_{algorithm}.png'
        try:
            plt.savefig(os.path.join(output_dir, spin_filename), dpi=300)
        except Exception as e:
            logger.error(f"Failed to save spin visualization {spin_filename}: {e}")
        plt.close(fig)


def create_output_dir(output_dir: str) -> str:
    """Create output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    return output_dir


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="XY Model Simulation with Kosterlitz-Thouless Transition Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--output-dir", type=str, default="xy_model_results",
        help="Directory to save results and plots"
    )
    parser.add_argument(
        "--t-min", type=float, default=0.7,
        help="Minimum temperature for simulation"
    )
    parser.add_argument(
        "--t-max", type=float, default=1.1,
        help="Maximum temperature for simulation"
    )
    parser.add_argument(
        "--num-points", type=int, default=15,
        help="Number of temperature points to simulate"
    )
    parser.add_argument(
        "--sweeps", type=int, default=2000, # Base sweeps for L=10 Metropolis / effective sweeps for Wolff
        help="Base number of Monte Carlo sweeps per temperature (scales with L)"
    )
    parser.add_argument(
        "--thermalize", type=int, default=1000, # Base thermalization
        help="Base number of Monte Carlo sweeps for thermalization (scales with L)"
    )
    parser.add_argument(
        "--lattice-sizes", type=int, nargs="+", default=[10, 20, 30, 40],
        help="List of lattice sizes to simulate"
    )
    parser.add_argument(
        "--j", type=float, default=1.0,
        help="Coupling constant (J > 0 for ferromagnetic)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--algorithm", type=str, choices=['metropolis', 'wolff'], default='wolff',
        help="Monte Carlo update algorithm to use"
    )
    parser.add_argument(
        "--disable-jit", action="store_true",
        help="Disable Numba JIT compilation (for debugging)"
    )
    
    return parser.parse_args()


def print_performance_report() -> None:
    """Print a performance report about the Numba JIT compilation."""
    try:
        # Measure performance of key JIT functions
        L = 20
        spins = np.random.uniform(0, 2*np.pi, (L, L))
        beta = 1.0
        J = 1.0
        
        # Warm up JIT
        _ = total_energy(spins, L, J)
        _ = compute_vorticity(spins, L)
        # Stiffness components requires beta, but calculate_observables doesn't pass it
        # Let's warm up the component calculator directly
        _ = calculate_stiffness_components(spins, L, J)
        
        # Measure energy calculation
        t0 = time.time()
        repeats = 1000
        for _ in range(repeats):
            _ = total_energy(spins, L, J)
        energy_time = (time.time() - t0) / repeats
        
        # Measure metropolis step
        t0 = time.time()
        repeats = 10
        for _ in range(repeats):
            metropolis_sweep(spins, L, beta, J)
        sweep_time_metro = (time.time() - t0) / repeats

        # Measure Wolff effective sweep
        t0 = time.time()
        repeats = 2 # Wolff sweeps can be long, use fewer repeats
        for _ in range(repeats):
             wolff_sweep(spins, L, beta, J)
        sweep_time_wolff = (time.time() - t0) / repeats

        
        # Measure vorticity calculation
        t0 = time.time()
        repeats = 100
        for _ in range(repeats):
            _ = compute_vorticity(spins, L)
        vorticity_time = (time.time() - t0) / repeats
        
        logger.info("\nPerformance report (estimates based on L=%d):" % L)
        logger.info(f"  Energy calculation time: {energy_time*1000:.2f} ms per call")
        logger.info(f"  Vorticity calculation time: {vorticity_time*1000:.2f} ms per call")
        logger.info(f"  Metropolis sweep time: {sweep_time_metro*1000:.2f} ms per sweep (L^2 attempts)")
        logger.info(f"  Wolff sweep time: {sweep_time_wolff*1000:.2f} ms per effective sweep (L^2 cluster builds)")
        
    except Exception as e:
        logger.warning(f"Could not generate full performance report: {e}")


def main() -> None:
    """Main function to run the simulation."""
    args = parse_arguments()
    
    if args.seed is not None:
        np.random.seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")
    
    if args.disable_jit:
        logger.warning("JIT compilation disabled - performance will be significantly reduced")
        numba.config.DISABLE_JIT = True
    else:
        print_performance_report() # Print performance report unless JIT disabled
    
    output_dir = create_output_dir(args.output_dir)
    logger.info(f"Using algorithm: {args.algorithm}")
    
    results_dict = {}
    samples_dict = {}
    
    # --- Simulation Loop --- 
    for L in args.lattice_sizes:
        base_thermalize = args.thermalize
        base_sweeps = args.sweeps
        current_thermalize_sweeps = base_thermalize
        current_sweeps = base_sweeps

        # Apply L-dependent scaling only for Metropolis
        if args.algorithm == 'metropolis':
            if L <= 10: scale_factor = 1
            elif L <= 20: scale_factor = 2
            elif L <= 30: scale_factor = 5
            else: scale_factor = 10 # For L=40 and above
            current_thermalize_sweeps = base_thermalize * scale_factor
            current_sweeps = base_sweeps * scale_factor
            logger.info(f"L={L} (Metropolis): Applying scale factor {scale_factor}. Sweeps: {current_thermalize_sweeps} therm, {current_sweeps} meas.")
        else:
            # For Wolff, use the base sweeps for all L by default
            logger.info(f"L={L} (Wolff): Using base sweeps. Sweeps: {current_thermalize_sweeps} therm, {current_sweeps} meas.")

        sim_params = SimulationParameters(
            L=L, T_min=args.t_min, T_max=args.t_max, num_points=args.num_points,
            sweeps=current_sweeps, thermalize_sweeps=current_thermalize_sweeps, J=args.j
            # Note: sim_params.temperatures will be set in simulate_lattice
        )

        results, sample_states = simulate_lattice(sim_params, args.algorithm)

        results_dict[L] = results
        samples_dict[L] = sample_states
    
    # --- Visualization --- 
    visualize_plots(results_dict, output_dir, args.algorithm)
    visualize_spins(samples_dict, output_dir, args.algorithm)

    # --- Summary --- 
    # Extract L and T_BKT for finite-size scaling
    L_values = np.array(sorted(results_dict.keys()))
    T_bkt_values = np.array([results_dict[L].T_BKT for L in L_values])
    # Get stiffness errors near T_BKT to estimate T_BKT error (optional, crude)
    # A proper T_BKT error requires error propagation from stiffness fit
    T_bkt_errors = np.array([results_dict[L].stiffness_errors[np.abs(results_dict[L].temperatures - results_dict[L].T_BKT).argmin()] if results_dict[L].T_BKT is not None and not np.isnan(results_dict[L].T_BKT) else np.nan for L in L_values])

    # Filter out NaN values for fitting
    valid_mask = ~np.isnan(T_bkt_values) & (L_values > 1) # Ensure L > 1 for log
    if np.sum(valid_mask) >= 3: # Need at least 3 points to fit 2 parameters reliably
        L_fit = L_values[valid_mask]
        T_fit = T_bkt_values[valid_mask]
        # Optional: Use T_bkt_errors as sigma for weighted fit if reliable
        # T_err_fit = T_bkt_errors[valid_mask]

        try:
            popt, pcov = curve_fit(scaling_func_log_sq, L_fit, T_fit,
                                   p0=[0.9, 1.0]) # Initial guess: T_inf=0.9, a=1.0
                                   # sigma=T_err_fit, absolute_sigma=True) # Optional weighted fit

            T_inf_fit = popt[0]
            T_inf_err = np.sqrt(pcov[0, 0])
            logger.info(f"Finite-size scaling fit: T_BKT(inf) = {T_inf_fit:.4f} +/- {T_inf_err:.4f}")

            # --- Plot Finite-Size Scaling --- 
            # Pass errors if you trust them, otherwise pass None
            visualize_finite_size_scaling(L_values, T_bkt_values, None, # Pass T_bkt_errors if desired
                                        popt, pcov, output_dir, args.algorithm)

        except RuntimeError as e:
            logger.error(f"Finite-size scaling curve_fit failed: {e}")
        except Exception as e:
            logger.error(f"Error during finite-size scaling analysis: {e}")
    else:
        logger.warning(f"Skipping finite-size scaling fit: Need at least 3 valid T_BKT(L) points (found {np.sum(valid_mask)})." )

    print(f"\nSummary of Kosterlitz-Thouless Transition Temperatures ({args.algorithm.capitalize()} Algorithm):")
    print("------------------------------------------------------")
    for L in sorted(results_dict.keys()): # Sort by L for consistent output
        t_bkt_val = results_dict[L].T_BKT
        if t_bkt_val is not None and not np.isnan(t_bkt_val):
            print(f"Lattice size L = {L:2d}: T_BKT = {t_bkt_val:.4f}")
        else:
            print(f"Lattice size L = {L:2d}: T_BKT = NaN")
    print("Theoretical value: T_BKT ≈ 0.8935")
    print(f"\nResults and plots saved to: {output_dir}")

# ============================================================================
# Finite-Size Scaling Function
# ============================================================================

def scaling_func_log_sq(L, T_inf, a):
    """Scaling function T(L) = T_inf + a / (ln(L))^2."""
    # Add small epsilon to avoid log(0) or division by zero if L=1 is ever used
    # Also handle potential non-positive L values passed erroneously
    L = np.array(L)
    valid_L = L > 1
    result = np.full_like(L, np.nan, dtype=float)
    if np.any(valid_L):
      logL = np.log(L[valid_L])
      result[valid_L] = T_inf + a / (logL**2)
    return result

def visualize_finite_size_scaling(L_values: np.ndarray, T_bkt_values: np.ndarray, T_bkt_errors: Optional[np.ndarray],
                                popt: np.ndarray, pcov: np.ndarray,
                                output_dir: str, algorithm: str) -> None:
    """
    Plot T_BKT(L) vs 1/(ln L)^2 and the linear fit to extrapolate T_BKT(inf).
    """
    T_inf, a = popt
    try:
        T_inf_err = np.sqrt(pcov[0, 0])
        a_err = np.sqrt(pcov[1, 1])
    except (IndexError, ValueError):
        T_inf_err = np.nan
        a_err = np.nan

    # Calculate x-axis values: 1 / (ln L)^2
    # Ensure L > 1 for log
    valid_mask = L_values > 1
    if not np.any(valid_mask):
        logger.warning("No lattice sizes > 1 found for finite-size scaling plot.")
        return
    
    L_plot = L_values[valid_mask]
    T_plot = T_bkt_values[valid_mask]
    if T_bkt_errors is not None:
        T_err_plot = T_bkt_errors[valid_mask]
    else:
        T_err_plot = None
       
    x_plot = 1.0 / (np.log(L_plot)**2)
    
    # Generate points for the fitted line
    x_fit_line = np.linspace(min(x_plot) * 0.9, max(x_plot) * 1.1, 100)
    # Need to map x_fit_line back to L for scaling_func, or use T = T_inf + a * x
    T_fit_line = T_inf + a * x_fit_line

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

    # Plot data points with error bars if available
    if T_err_plot is not None:
         ax.errorbar(x_plot, T_plot, yerr=T_err_plot, fmt='o', markersize=5,
                     capsize=3, label='Simulation Data $T_{BKT}(L)$')
    else:
        ax.plot(x_plot, T_plot, 'o', markersize=5, label='Simulation Data $T_{BKT}(L)$')
        
    # Plot the linear fit
    ax.plot(x_fit_line, T_fit_line, 'r--', 
            label=f'Fit: $T = T_\infty + a / (\ln L)^2$\\\\ $T_\infty = {T_inf:.4f} \pm {T_inf_err:.4f}$')

    ax.set_xlabel(r'$1 / (\ln L)^2$', fontsize=12)
    ax.set_ylabel(r'$T_{BKT}(L)$', fontsize=12)
    ax.set_title(f'Finite-Size Scaling of $T_{BKT}$ ({algorithm.capitalize()} Algorithm)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.5)

    # Add text for extrapolated value near y-intercept
    ax.text(0.05, 0.95, f'Extrapolated $T_{{BKT}}(\infty) = {T_inf:.4f} \pm {T_inf_err:.4f}$',
            transform=ax.transAxes, fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plot_filename = f'xy_model_fss_{algorithm}.png'
    plt.savefig(os.path.join(output_dir, plot_filename), dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    main() 