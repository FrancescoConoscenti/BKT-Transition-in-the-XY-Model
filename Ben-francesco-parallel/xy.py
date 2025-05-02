#!/usr/bin/env python3
"""
2D XY Model Simulation with Kosterlitz-Thouless Phase Transition Analysis

This module implements a high-performance Monte Carlo simulation of the classical
XY model on a 2D lattice, specifically designed to investigate the 
Kosterlitz-Thouless phase transition. The implementation uses vectorized
operations where possible to maximize computational efficiency.
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
from typing import List, Tuple, Dict, Optional, Any
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

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


class XYModel:
    """
    Implementation of the 2D XY model with efficient Monte Carlo sampling.
    
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
        """
        Calculate the total energy of the system.
        
        Returns:
        --------
        float
            Total energy of the system
        """
        # Vectorized energy calculation
        sx = np.cos(self.spins)
        sy = np.sin(self.spins)
        
        # Calculate interactions with neighbors (with periodic boundary)
        energy = 0.0
        
        # Horizontal neighbors
        sx_right = np.roll(sx, -1, axis=1)
        sy_right = np.roll(sy, -1, axis=1)
        energy -= self.J * np.sum(sx * sx_right + sy * sy_right)
        
        # Vertical neighbors
        sx_down = np.roll(sx, -1, axis=0)
        sy_down = np.roll(sy, -1, axis=0)
        energy -= self.J * np.sum(sx * sx_down + sy * sy_down)
        
        return energy
    
    def site_energy(self, i: int, j: int) -> float:
        """
        Calculate the energy contribution of a single site.
        
        Parameters:
        -----------
        i, j : int
            Site coordinates
            
        Returns:
        --------
        float
            Energy contribution of the site
        """
        # Get the spin angles
        spin = self.spins[i, j]
        sx, sy = np.cos(spin), np.sin(spin)
        
        # Get neighbor indices with periodic boundary conditions
        i_up = (i - 1) % self.L
        i_down = (i + 1) % self.L
        j_left = (j - 1) % self.L
        j_right = (j + 1) % self.L
        
        # Get neighbor spin components
        neighbors = [
            (np.cos(self.spins[i_up, j]), np.sin(self.spins[i_up, j])),
            (np.cos(self.spins[i_down, j]), np.sin(self.spins[i_down, j])),
            (np.cos(self.spins[i, j_left]), np.sin(self.spins[i, j_left])),
            (np.cos(self.spins[i, j_right]), np.sin(self.spins[i, j_right]))
        ]
        
        # Calculate energy contribution
        energy = 0.0
        for nx, ny in neighbors:
            energy -= self.J * (sx * nx + sy * ny)
            
        return energy
    
    def metropolis_step(self, beta: float) -> None:
        """
        Perform one Monte Carlo step per site using Metropolis algorithm.
        
        Parameters:
        -----------
        beta : float
            Inverse temperature (1/T)
        """
        for _ in range(self.L * self.L):
            # Choose a random site
            i, j = np.random.randint(0, self.L, 2)
            
            # Calculate current energy contribution
            old_energy = self.site_energy(i, j)
            
            # Propose a new spin angle
            new_angle = np.random.uniform(0, 2*np.pi)
            
            # Store the old angle
            old_angle = self.spins[i, j]
            
            # Temporarily update the spin
            self.spins[i, j] = new_angle
            
            # Calculate new energy contribution
            new_energy = self.site_energy(i, j)
            
            # Calculate energy difference
            delta_E = new_energy - old_energy
            
            # Metropolis acceptance criterion
            if delta_E <= 0 or np.random.random() < np.exp(-beta * delta_E):
                # Accept the move (already updated)
                pass
            else:
                # Reject the move and restore the old angle
                self.spins[i, j] = old_angle
    
    def compute_vorticity(self) -> np.ndarray:
        """
        Calculate the vorticity field for the current spin configuration.
        
        Returns:
        --------
        np.ndarray
            2D array of vorticity per plaquette
        """
        vortices = np.zeros((self.L, self.L))
        for i in range(self.L):
            for j in range(self.L):
                # Spins at four corners of a plaquette (counter-clockwise)
                theta = [
                    self.spins[i, j],
                    self.spins[i, (j + 1) % self.L],
                    self.spins[(i + 1) % self.L, (j + 1) % self.L],
                    self.spins[(i + 1) % self.L, j]
                ]
                # Calculate angle differences (modulo 2π)
                dtheta = [(theta[n + 1] - theta[n] + np.pi) % (2 * np.pi) - np.pi for n in range(3)]
                dtheta.append((theta[0] - theta[3] + np.pi) % (2 * np.pi) - np.pi)

                # Sum angle differences; divide by 2π to get winding number
                winding = np.sum(dtheta) / (2 * np.pi)
                vortices[i, j] = np.round(winding)  # Should be 0, ±1 in practice
        return vortices
    
    def spin_stiffness(self, beta: float) -> float:
        """
        Compute the spin stiffness (helicity modulus) for the current configuration.
        
        Parameters:
        -----------
        beta : float
            Inverse temperature (1/T)
            
        Returns:
        --------
        float
            Spin stiffness value
        """
        L = self.L
        J = self.J
        
        # Sum over x-direction bonds (horizontal)
        sum_cos = 0.0
        sum_sin = 0.0
        
        for i in range(L):
            for j in range(L):
                # Right neighbor (x-direction)
                j_right = (j + 1) % L
                delta_theta = self.spins[i, j] - self.spins[i, j_right]
                sum_cos += np.cos(delta_theta)
                sum_sin += np.sin(delta_theta)
        
        # Average per bond
        N_bonds = L * L
        avg_cos = sum_cos / N_bonds
        avg_sin_squared = (sum_sin ** 2) / N_bonds
        
        # Spin stiffness formula
        rho_s = (J * avg_cos) - (J * beta * avg_sin_squared)
        
        return rho_s


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
    for _ in tqdm(range(thermalize_sweeps), desc=f"L={L}, T={temperature:.2f}: Thermalization", 
                 leave=False, disable=temp_idx > 0):
        model.metropolis_step(beta)
    
    # Data collection phase
    energies = []
    magnetizations = []
    vortices = []
    stiffnesses = []
    
    for _ in tqdm(range(sweeps), desc=f"L={L}, T={temperature:.2f}: Data collection", 
                 leave=False, disable=temp_idx > 0):
        model.metropolis_step(beta)
        
        # Collect energy
        energy = model.energy() / (L * L)
        energies.append(energy)
        
        # Calculate magnetization
        mx = np.mean(np.cos(model.spins))
        my = np.mean(np.sin(model.spins))
        m = np.sqrt(mx**2 + my**2)
        magnetizations.append(m)

        # Calculate vorticity
        vortex = model.compute_vorticity()
        v = np.mean(np.abs(vortex))
        vortices.append(v)

        # Calculate spin stiffness
        stiffnesses.append(model.spin_stiffness(beta))
    
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


def simulate_lattice(sim_params: SimulationParameters) -> SimulationResults:
    """
    Simulate the XY model for various temperatures on a lattice of given size.
    
    Parameters:
    -----------
    sim_params : SimulationParameters
        Simulation parameters
        
    Returns:
    --------
    SimulationResults
        Consolidated simulation results
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
            
            axes[i].set_title(f'L = {L}, T = {temp:.2f}')
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
        "--lattice-sizes", type=int, nargs="+", default=[10, 15, 20, 25],
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
    
    return parser.parse_args()


def main() -> None:
    """Main function to run the simulation."""
    # Parse arguments
    args = parse_arguments()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")
    
    # Create output directory
    output_dir = create_output_dir(args.output_dir)
    
    # Store results for each lattice size
    results_dict = {}
    samples_dict = {}
    
    # Run simulations for each lattice size
    for L in args.lattice_sizes:
        # Create simulation parameters
        sim_params = SimulationParameters(
            L=L,
            T_min=args.t_min,
            T_max=args.t_max,
            num_points=args.num_points,
            sweeps=args.sweeps,
            thermalize_sweeps=args.thermalize,
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
    