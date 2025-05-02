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
from numba import jit, njit, prange, float64, int64, boolean

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
# JIT-compiled computational kernels
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


@njit(float64[:,:](float64[:,:], int64))
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


@njit(float64(float64[:,:], float64, int64, float64))
def spin_stiffness(spins, beta, L, J):
    """
    Compute the spin stiffness (helicity modulus) for the current configuration.
    
    Args:
        spins: 2D array of spin angles
        beta: Inverse temperature (1/T)
        L: Linear size of the lattice
        J: Coupling constant
        
    Returns:
        float: Spin stiffness value
    """
    # Sum over x-direction bonds (horizontal)
    sum_cos = 0.0
    sum_sin = 0.0
    
    for i in range(L):
        for j in range(L):
            # Right neighbor (x-direction)
            j_right = (j + 1) % L
            delta_theta = spins[i, j] - spins[i, j_right]
            sum_cos += np.cos(delta_theta)
            sum_sin += np.sin(delta_theta)
    
    # Average per bond
    N_bonds = L * L
    avg_cos = sum_cos / N_bonds
    avg_sin_squared = (sum_sin ** 2) / N_bonds
    
    # Spin stiffness formula
    rho_s = (J * avg_cos) - (J * beta * avg_sin_squared)
    
    return rho_s


@njit
def calculate_observables(spins, beta, L, J):
    """
    Calculate all observables for a single configuration.
    
    Args:
        spins: 2D array of spin angles
        beta: Inverse temperature (1/T)
        L: Linear size of the lattice
        J: Coupling constant
        
    Returns:
        tuple: (energy_per_site, magnetization, vorticity, stiffness)
    """
    # Energy
    energy = total_energy(spins, L, J) / (L * L)
    
    # Magnetization
    mx, my = 0.0, 0.0
    for i in range(L):
        for j in range(L):
            mx += np.cos(spins[i, j])
            my += np.sin(spins[i, j])
    
    mx /= (L * L)
    my /= (L * L)
    magnetization = np.sqrt(mx*mx + my*my)
    
    # Vorticity
    vortex_field = compute_vorticity(spins, L)
    vorticity = np.mean(np.abs(vortex_field))
    
    # Spin stiffness
    stiffness = spin_stiffness(spins, beta, L, J)
    
    return energy, magnetization, vorticity, stiffness


# ============================================================================
# Main XY Model class
# ============================================================================

class XYModel:
    """
    Implementation of the 2D XY model with Numba-accelerated Monte Carlo sampling.
    
    The XY model consists of 2D rotors on a lattice, with nearest-neighbor
    ferromagnetic interactions. Each rotor is characterized by an angle 
    between 0 and 2π.
    """
    
    def __init__(self, L: int, J: float = 1.0):
        """
        Initialize the 2D XY model.
        
        Parameters:
        -----------
        L : int
            Linear size of the lattice (L x L)
        J : float, optional
            Coupling constant (J > 0 for ferromagnetic), default=1.0
        """
        self.L = L
        self.J = J
        # Initialize spins randomly between 0 and 2π
        self.spins = np.random.uniform(0, 2*np.pi, (L, L))
    
    def energy(self) -> float:
        """Calculate the total energy using the JIT-compiled function."""
        return total_energy(self.spins, self.L, self.J)
    
    def site_energy(self, i: int, j: int) -> float:
        """Calculate a site's energy using the JIT-compiled function."""
        return site_energy(self.spins, i, j, self.L, self.J)
    
    def metropolis_step(self, beta: float) -> None:
        """Perform one Monte Carlo step using the JIT-compiled function."""
        metropolis_sweep(self.spins, self.L, beta, self.J)
    
    def compute_vorticity(self) -> np.ndarray:
        """Calculate the vorticity field using the JIT-compiled function."""
        return compute_vorticity(self.spins, self.L)
    
    def spin_stiffness(self, beta: float) -> float:
        """Calculate the spin stiffness using the JIT-compiled function."""
        return spin_stiffness(self.spins, beta, self.L, self.J)
    
    def calculate_observables(self, beta: float) -> Tuple[float, float, float, float]:
        """Calculate all observables using the JIT-compiled function."""
        return calculate_observables(self.spins, beta, self.L, self.J)


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


def run_single_temperature(params: Tuple[SimulationParameters, float, int]) -> Tuple[float, Dict[str, Any]]:
    """
    Run simulation for a single temperature point.
    
    Parameters:
    -----------
    params : Tuple
        (simulation_params, temperature, temp_index)
        
    Returns:
    --------
    Tuple[float, Dict]
        (temperature, results dictionary)
    """
    sim_params, temperature, temp_idx = params
    L = sim_params.L
    sweeps = sim_params.sweeps
    thermalize_sweeps = sim_params.thermalize_sweeps
    
    # Create model
    model = XYModel(L, J=sim_params.J)
    beta = 1.0 / temperature
    
    # Thermalization phase
    for _ in tqdm(range(thermalize_sweeps), desc=f"L={L}, T={temperature:.4f}: Thermalization", 
                 leave=False, disable=temp_idx > 0):
        model.metropolis_step(beta)
    
    # Data collection phase
    energies = []
    magnetizations = []
    vortices = []
    stiffnesses = []
    
    for _ in tqdm(range(sweeps), desc=f"L={L}, T={temperature:.4f}: Data collection", 
                 leave=False, disable=temp_idx > 0):
        model.metropolis_step(beta)
        
        # Collect all observables with a single call
        energy, mag, vorticity, stiffness = model.calculate_observables(beta)
        energies.append(energy)
        magnetizations.append(mag)
        vortices.append(vorticity)
        stiffnesses.append(stiffness)
    
    # Aggregate results
    results = {
        'energy': np.mean(energies),
        'energy_error': np.std(energies) / np.sqrt(sweeps),
        'energy_sq': np.mean(np.array(energies)**2),
        'magnetization': np.mean(magnetizations),
        'magnetization_error': np.std(magnetizations) / np.sqrt(sweeps),
        'magnetization_sq': np.mean(np.array(magnetizations)**2),
        'vorticity': np.mean(vortices),
        'stiffness': np.mean(stiffnesses),
        'stiffness_error': np.std(stiffnesses) / np.sqrt(sweeps),
        'model_state': model if temp_idx % (len(sim_params.temperatures) // min(4, len(sim_params.temperatures))) == 0 else None
    }
    
    return temperature, results


def simulate_lattice(sim_params: SimulationParameters) -> Tuple[SimulationResults, List[Tuple[float, XYModel]]]:
    """
    Simulate the XY model for various temperatures on a lattice of given size.
    
    Parameters:
    -----------
    sim_params : SimulationParameters
        Simulation parameters
        
    Returns:
    --------
    tuple
        (SimulationResults, sample_states)
    """
    L = sim_params.L
    temperatures = np.linspace(sim_params.T_min, sim_params.T_max, sim_params.num_points)
    sim_params.temperatures = temperatures  # Store for later reference
    
    logger.info(f"Simulating XY model on {L}x{L} lattice with {sim_params.num_points} temperature points")
    
    # Arrays to store results
    energies = np.zeros(sim_params.num_points)
    energy_errors = np.zeros(sim_params.num_points)
    energy_sq = np.zeros(sim_params.num_points)
    magnetizations = np.zeros(sim_params.num_points)
    mag_errors = np.zeros(sim_params.num_points)
    mag_sq = np.zeros(sim_params.num_points)
    vortices = np.zeros(sim_params.num_points)
    stiffnesses = np.zeros(sim_params.num_points)
    stiffness_errors = np.zeros(sim_params.num_points)
    
    # Sample states to visualize
    sample_states = []
    
    # Run simulations
    start_time = time.time()
    
    # Create parameter tuples for each temperature
    params_list = [(sim_params, T, i) for i, T in enumerate(temperatures)]
    
    # Use process pool to parallelize across temperatures
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        results = list(tqdm(
            executor.map(run_single_temperature, params_list),
            total=len(params_list),
            desc=f"L={L} Progress"
        ))
    
    # Process results
    for T, result in results:
        i = np.where(temperatures == T)[0][0]
        
        energies[i] = result['energy']
        energy_errors[i] = result['energy_error']
        energy_sq[i] = result['energy_sq']
        magnetizations[i] = result['magnetization']
        mag_errors[i] = result['magnetization_error']
        mag_sq[i] = result['magnetization_sq']
        vortices[i] = result['vorticity']
        stiffnesses[i] = result['stiffness']
        stiffness_errors[i] = result['stiffness_error']
        
        # Save sample state if available
        if result['model_state'] is not None:
            sample_states.append((T, result['model_state']))
    
    elapsed = time.time() - start_time
    logger.info(f"Simulation for lattice size L={L} completed in {elapsed:.1f} seconds")
    
    # Calculate derived quantities
    beta = 1.0 / temperatures[-1]  # Use the last temperature's beta for calculations
    specific_heat = beta**2 * (energy_sq - energies**2) * L**2
    susceptibility = beta * L**2 * (mag_sq - magnetizations**2)
    
    # Calculate BKT transition temperature
    line = 2/np.pi * temperatures
    T_BKT = find_cross(stiffnesses, line, temperatures)
    logger.info(f"Lattice L={L}: Estimated BKT transition temperature T_BKT ≈ {T_BKT:.4f}")
    
    # Store results
    results = SimulationResults(
        temperatures=temperatures,
        energies=energies,
        energy_errors=energy_errors,
        magnetizations=magnetizations,
        mag_errors=mag_errors,
        specific_heat=specific_heat,
        susceptibility=susceptibility,
        vortices=vortices,
        stiffnesses=stiffnesses,
        stiffness_errors=stiffness_errors,
        T_BKT=T_BKT
    )
    
    return results, sample_states


def visualize_plots(results_dict: Dict[int, SimulationResults], output_dir: str) -> None:
    """
    Create comparison plots for different lattice sizes.
    
    Parameters:
    -----------
    results_dict : Dict[int, SimulationResults]
        Dictionary mapping lattice sizes to simulation results
    output_dir : str
        Directory to save plot files
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
    ax.set_xlabel('Temperature (T)', fontsize=12)
    ax.set_ylabel('Specific Heat', fontsize=12)
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
    ax.set_xlabel('Temperature (T)', fontsize=12)
    ax.set_ylabel('Susceptibility', fontsize=12)
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
    for i, L in enumerate(lattice_sizes):
        results = results_dict[L]
        line = 2/np.pi * results.temperatures
        ax.plot(
            results.temperatures, results.stiffnesses,
            marker='o', markersize=4, linestyle='-', linewidth=1.5,
            color=colors[i], label=f'L = {L}'
        )
        ax.errorbar(
            results.temperatures, results.stiffnesses, yerr=results.stiffness_errors,
            fmt='none', ecolor=colors[i], alpha=0.3
        )
    
    ax.plot(results.temperatures, line, 'k--', label=r'$\frac{2}{\pi}T$', linewidth=1.5)
    
    # Add transition temperature markers
    for i, L in enumerate(lattice_sizes):
        results = results_dict[L]
        if not np.isnan(results.T_BKT):
            ax.axvline(results.T_BKT, color=colors[i], linestyle=':', alpha=0.7)
            ax.text(
                results.T_BKT, 0.1, f'T_BKT(L={L})={results.T_BKT:.3f}',
                color=colors[i], rotation=90, fontsize=8
            )
    
    ax.set_xlabel('Temperature (T)', fontsize=12)
    ax.set_ylabel('Spin Stiffness', fontsize=12)
    ax.set_title('Spin Stiffness vs Temperature with BKT Transition', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add a summary of transition temperatures in a text box
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    textstr = "BKT Transition Temperature Estimates:\n"
    for L in lattice_sizes:
        results = results_dict[L]
        textstr += f"L = {L}: T_BKT = {results.T_BKT:.4f}\n"
    textstr += f"Theoretical: T_BKT ≈ 0.8935"
    fig.text(0.5, 0.01, textstr, fontsize=12, bbox=props, ha='center')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(os.path.join(output_dir, 'xy_model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def visualize_spins(samples_dict: Dict[int, List[Tuple[float, XYModel]]], output_dir: str) -> None:
    """
    Visualize spin configurations for different lattice sizes and temperatures.
    
    Parameters:
    -----------
    samples_dict : Dict[int, List[Tuple[float, XYModel]]]
        Dictionary mapping lattice sizes to lists of (temperature, model) tuples
    output_dir : str
        Directory to save plot files
    """
    for L, sample_states in samples_dict.items():
        if not sample_states:
            continue
            
        fig, axes = plt.subplots(1, len(sample_states), figsize=(5*len(sample_states), 5), dpi=150)
        
        # Handle case with just one temperature
        if len(sample_states) == 1:
            axes = [axes]
            
        for i, (temp, model) in enumerate(sample_states):
            # Create mesh grid for the arrows
            x, y = np.meshgrid(np.arange(0, L), np.arange(0, L))
            
            # Get the spin components
            u = np.cos(model.spins)
            v = np.sin(model.spins)
            
            # Plot the arrows
            axes[i].quiver(
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
                    if vortices[i_lat, j_lat] > 0:
                        axes[i].plot(j_lat, i_lat, 'ro', markersize=4, alpha=0.7)  # Positive vortex
                    elif vortices[i_lat, j_lat] < 0:
                        axes[i].plot(j_lat, i_lat, 'bo', markersize=4, alpha=0.7)  # Negative vortex
            
            axes[i].set_title(f'L = {L}, T = {temp:.4f}')
            axes[i].set_xlim(-1, L)
            axes[i].set_ylim(-1, L)
            axes[i].set_aspect('equal')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'xy_model_spins_L{L}.png'), dpi=300)
        plt.close()


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
        "--sweeps", type=int, default=2000,
        help="Number of Monte Carlo sweeps per temperature for measurements"
    )
    parser.add_argument(
        "--thermalize", type=int, default=1000,
        help="Number of Monte Carlo sweeps for thermalization"
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
        _ = spin_stiffness(spins, beta, L, J)
        
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
        sweep_time = (time.time() - t0) / repeats
        
        # Measure vorticity calculation
        t0 = time.time()
        repeats = 100
        for _ in range(repeats):
            _ = compute_vorticity(spins, L)
        vorticity_time = (time.time() - t0) / repeats
        
        logger.info("\nPerformance report:")
        logger.info(f"Energy calculation: {energy_time*1000:.2f} ms")
        logger.info(f"Monte Carlo sweep: {sweep_time*1000:.2f} ms")
        logger.info(f"Vorticity calculation: {vorticity_time*1000:.2f} ms")
        logger.info(f"Estimated time per {L}x{L} lattice sweep: {sweep_time*1000:.2f} ms")
        
    except Exception as e:
        logger.warning(f"Could not generate performance report: {e}")


def main() -> None:
    """Main function to run the simulation."""
    # Parse arguments
    args = parse_arguments()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")
    
    # Disable JIT if requested
    if args.disable_jit:
        logger.warning("JIT compilation disabled - performance will be significantly reduced")
        numba.config.DISABLE_JIT = True
    else:
        # Print performance report
        print_performance_report()
    
    # Create output directory
    output_dir = create_output_dir(args.output_dir)
    
    # Store results for each lattice size
    results_dict = {}
    samples_dict = {}
    
    # Run simulations for each lattice size
    for L in args.lattice_sizes:
        # Determine L-dependent sweep counts
        # Treat args values as baseline for L=10 (or smallest L)
        base_thermalize = args.thermalize
        base_sweeps = args.sweeps

        if L <= 10:
            scale_factor = 1
        elif L <= 20:
            scale_factor = 2
        elif L <= 30:
            scale_factor = 5
        else: # L > 30 (e.g., L=40)
            scale_factor = 10

        current_thermalize_sweeps = base_thermalize * scale_factor
        current_sweeps = base_sweeps * scale_factor

        logger.info(f"L={L}: Using {current_thermalize_sweeps} thermalization sweeps and {current_sweeps} measurement sweeps.")

        # Create simulation parameters with adjusted sweeps
        sim_params = SimulationParameters(
            L=L,
            T_min=args.t_min,
            T_max=args.t_max,
            num_points=args.num_points,
            sweeps=current_sweeps, # Use adjusted value
            thermalize_sweeps=current_thermalize_sweeps, # Use adjusted value
            J=args.j
        )
        
        # Run simulation
        results, sample_states = simulate_lattice(sim_params)
        
        # Store results
        results_dict[L] = results
        samples_dict[L] = sample_states
    
    # Create comparative visualizations
    visualize_plots(results_dict, output_dir)
    visualize_spins(samples_dict, output_dir)
    
    # Print summary
    print("\nSummary of Kosterlitz-Thouless Transition Temperatures:")
    print("------------------------------------------------------")
    for L in args.lattice_sizes:
        print(f"Lattice size L = {L:2d}: T_BKT = {results_dict[L].T_BKT:.4f}")
    print("Theoretical value: T_BKT ≈ 0.8935")
    print("\nResults and plots saved to:", output_dir)


if __name__ == "__main__":
    main() 