#!/usr/bin/env python3
"""
2D XY Model with Kosterlitz-Thouless Phase Transition
High-performance Monte Carlo simulation using JIT compilation
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
import os
import sys
import argparse
from tqdm import tqdm
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numba
from numba import jit, njit, prange, float64, int64, boolean, types
from scipy.optimize import curve_fit

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# JIT-compiled computational kernels
# ============================================================================

@njit(float64(float64[:,:], int64, float64))
def total_energy(spins, L, J):
    """Calculate total energy with nearest-neighbor interactions"""
    energy = 0.0
    for i in range(L):
        for j in range(L):
            theta = spins[i, j]
            sx, sy = np.cos(theta), np.sin(theta)
            
            j_right = (j + 1) % L
            theta_right = spins[i, j_right]
            sx_right, sy_right = np.cos(theta_right), np.sin(theta_right)
            
            i_down = (i + 1) % L
            theta_down = spins[i_down, j]
            sx_down, sy_down = np.cos(theta_down), np.sin(theta_down)
            
            energy -= J * (sx * sx_right + sy * sy_right)
            energy -= J * (sx * sx_down + sy * sy_down)
            
    return energy

@njit(float64(float64[:,:], int64, int64, int64, float64))
def site_energy(spins, i, j, L, J):
    """Calculate energy contribution from site (i,j)"""
    theta = spins[i, j]
    sx, sy = np.cos(theta), np.sin(theta)
    
    i_up = (i - 1) % L
    i_down = (i + 1) % L
    j_left = (j - 1) % L
    j_right = (j + 1) % L
    
    theta_up = spins[i_up, j]
    sx_up, sy_up = np.cos(theta_up), np.sin(theta_up)
    
    theta_down = spins[i_down, j]
    sx_down, sy_down = np.cos(theta_down), np.sin(theta_down)
    
    theta_left = spins[i, j_left]
    sx_left, sy_left = np.cos(theta_left), np.sin(theta_left)
    
    theta_right = spins[i, j_right]
    sx_right, sy_right = np.cos(theta_right), np.sin(theta_right)
    
    energy = 0.0
    energy -= J * (sx * sx_up + sy * sy_up)
    energy -= J * (sx * sx_down + sy * sy_down)
    energy -= J * (sx * sx_left + sy * sy_left)
    energy -= J * (sx * sx_right + sy * sy_right)
    
    return energy

@njit
def metropolis_sweep(spins, L, beta, J):
    """Perform L²  Metropolis spin updates"""
    for _ in range(L * L):
        i, j = np.random.randint(0, L), np.random.randint(0, L)
        old_energy = site_energy(spins, i, j, L, J)
        new_angle = np.random.uniform(0, 2*np.pi)
        old_angle = spins[i, j]
        spins[i, j] = new_angle
        new_energy = site_energy(spins, i, j, L, J)
        delta_E = new_energy - old_energy
        if delta_E > 0 and np.random.random() >= np.exp(-beta * delta_E):
            spins[i, j] = old_angle

@njit
def compute_vorticity(spins, L):
    """Calculate vorticity field (winding numbers) for all plaquettes"""
    vortices = np.zeros((L, L))
    
    for i in range(L):
        for j in range(L):
            theta1 = spins[i, j]
            theta2 = spins[i, (j + 1) % L]
            theta3 = spins[(i + 1) % L, (j + 1) % L]
            theta4 = spins[(i + 1) % L, j]
            
            dtheta1 = (theta2 - theta1 + np.pi) % (2 * np.pi) - np.pi
            dtheta2 = (theta3 - theta2 + np.pi) % (2 * np.pi) - np.pi
            dtheta3 = (theta4 - theta3 + np.pi) % (2 * np.pi) - np.pi
            dtheta4 = (theta1 - theta4 + np.pi) % (2 * np.pi) - np.pi

            winding = (dtheta1 + dtheta2 + dtheta3 + dtheta4) / (2 * np.pi)
            vortices[i, j] = np.round(winding)
            
    return vortices

@njit
def calculate_stiffness_components(spins, L, J):
    """Calculate components for spin stiffness: J⟨cos(Δθ)⟩ and J⟨sin(Δθ)⟩²"""
    sum_cos_x, sum_sin_x, sum_cos_y, sum_sin_y = 0.0, 0.0, 0.0, 0.0
    
    for i in range(L):
        for j in range(L):
            theta_current = spins[i, j]

            j_right = (j + 1) % L
            theta_right = spins[i, j_right]
            delta_theta_x = theta_current - theta_right
            sum_cos_x += J * np.cos(delta_theta_x)
            sum_sin_x += J * np.sin(delta_theta_x)

            i_down = (i + 1) % L
            theta_down = spins[i_down, j]
            delta_theta_y = theta_current - theta_down
            sum_cos_y += J * np.cos(delta_theta_y)
            sum_sin_y += J * np.sin(delta_theta_y)

    return sum_cos_x, sum_sin_x, sum_cos_y, sum_sin_y

@njit 
def calculate_correlation(spins, L):
        """ Calculate the spin-spin correlation function """

        max_dist = L // 2
        correlation = np.zeros(max_dist + 1)
        count = np.zeros(max_dist + 1 )
    
        for i in range(L):
            for j in range(L):
                spin_i = spins[i, j]
            
                # Loop over all other spins
                for k in range(L):
                    for l in range(L):
                        # Calculate distance with PBC
                        dist_x = min(abs(i - k), L - abs(i - k))
                        dist_y = min(abs(j - l), L - abs(j - l))
                        dist = int(round(np.sqrt(dist_x**2 + dist_y**2)))
                    
                        if dist <= max_dist:
                            spin_j = spins[k, l]
                            # Calculate correlation
                            correlation[dist] += np.cos(spin_i - spin_j)
                            count[dist] += 1
    
        # Normalize by counts
        mask = count > 0
        correlation[mask] /= count[mask]
    
        return correlation

def estimate_correlation_decay_parameters(correlation, T, L, fit_range=None):
        """
        Estimate the critical exponent η and correlation length ξ from G(r)
        """
        distances = np.arange(L // 2 + 1)
        mask = distances > 0
        r = distances[mask]
        G = correlation[mask]

        eps = 0.15  # Temperature window around 0.95

        try:
            if T < 0.95 + eps:  # Pure algebraic decay
                def correlation_function(r, A, eta):
                    return A * r**(-eta)
                p0 = [1.0, 0.1]
                popt, _ = curve_fit(correlation_function, r, G, p0=p0)
                A, eta = popt
                xi = float('nan')  # No exponential cutoff
            else:  # Pure exponential decay
                def correlation_function(r, A, xi):
                    return A * np.exp(-r/xi)
                p0 = [1.0, L / 10]
                popt, _ = curve_fit(correlation_function, r, G, p0=p0)
                A, xi = popt
                eta = float('nan') # No algebraic component

        except:
            print("Fitting failed!")
            A, eta, xi = float('nan'), float('nan'), float('nan')

        return eta, xi, A


@njit
def calculate_observables(spins, L, J):
    """Calculate energy, magnetization, vorticity, and stiffness components"""
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
    
    # Vorticity
    vortex_field = compute_vorticity(spins, L)
    avg_abs_vorticity = np.mean(np.abs(vortex_field))

    # Stiffness components
    sum_cos_x, sum_sin_x, sum_cos_y, sum_sin_y = calculate_stiffness_components(spins, L, J)

    # Correlation function
    correlation = calculate_correlation(spins, L) 

    return (energy, magnetization, avg_abs_vorticity,
            sum_cos_x, sum_sin_x, sum_cos_y, sum_sin_y, correlation)

@njit(types.void(float64[:,:], int64, float64, float64))
def wolff_sweep(spins, L, beta, J):
    """Perform L² Wolff cluster updates"""
    num_sites = L * L
    in_cluster = np.zeros((L, L), dtype=boolean)
    cluster_stack = np.empty((L * L, 2), dtype=np.int64)

    for _ in range(num_sites):
        phi_reflect = np.random.uniform(0, 2.0 * np.pi)
        i0, j0 = np.random.randint(0, L), np.random.randint(0, L)

        in_cluster[:] = False
        stack_idx = 0
        head = 0

        cluster_stack[stack_idx] = (i0, j0)
        stack_idx += 1
        in_cluster[i0, j0] = True

        while head < stack_idx:
            i, j = cluster_stack[head]
            head += 1

            current_theta = spins[i, j]
            spins[i, j] = (2.0 * phi_reflect - current_theta) % (2.0 * np.pi)

            neighbors_coords = [
                ((i - 1 + L) % L, j),
                ((i + 1) % L, j),
                (i, (j - 1 + L) % L),
                (i, (j + 1) % L)
            ]

            for ni, nj in neighbors_coords:
                if not in_cluster[ni, nj]:
                    neighbor_theta = spins[ni, nj]
                    energy_diff = -J * (np.cos(current_theta - neighbor_theta) -
                                       np.cos(current_theta - (2.0 * phi_reflect - neighbor_theta)))
                    prob_arg = beta * energy_diff
                    freeze_prob = 0.0
                    if prob_arg < 0:
                        freeze_prob = 1.0 - np.exp(prob_arg)
                        freeze_prob = min(freeze_prob, 1.0)

                    if np.random.random() < freeze_prob:
                        in_cluster[ni, nj] = True
                        if stack_idx < num_sites:
                            cluster_stack[stack_idx] = (ni, nj)
                            stack_idx += 1
                        else:
                            break
            if stack_idx >= num_sites and head < stack_idx:
                break

# ============================================================================
# Main XY Model class
# ============================================================================

class XYModel:
    """2D XY model with Metropolis and Wolff update algorithms"""
    
    def __init__(self, L, J=1.0):
        self.L = L
        self.J = J
        self.spins = np.random.uniform(0, 2*np.pi, (L, L))
    
    def energy(self):
        return total_energy(self.spins, self.L, self.J)
    
    def site_energy(self, i, j):
        return site_energy(self.spins, i, j, self.L, self.J)
    
    def metropolis_step(self, beta):
        metropolis_sweep(self.spins, self.L, beta, self.J)
    
    def wolff_sweep(self, beta):
        wolff_sweep(self.spins, self.L, beta, self.J)

    def compute_vorticity(self):
        return compute_vorticity(self.spins, self.L)
    
    def calculate_observables(self):
        """Calculate all observables for the current configuration"""
        return calculate_observables(self.spins, self.L, self.J)

# ============================================================================
# Simulation functions
# ============================================================================

def find_cross(stiffness, line, temperatures):
    """Find crossing of stiffness with 2T/π line to estimate T_BKT"""
    difference = stiffness - line
    crossing_indices = np.where(np.diff(np.sign(difference)))[0]
    
    if len(crossing_indices) == 0:
        return np.nan

    idx = crossing_indices[0]
    x0, x1 = temperatures[idx], temperatures[idx+1]
    y0, y1 = difference[idx], difference[idx+1]
    x_cross = x0 - y0 * (x1 - x0) / (y1 - y0)
    
    return x_cross

def find_eta_cross(etas: np.ndarray, temperatures: np.ndarray) -> float:
    target_eta = 0.25
    for i in range(len(etas) - 1):
        if (etas[i] - target_eta) * (etas[i+1] - target_eta) < 0:
            # Linear interpolation between temperatures[i] and temperatures[i+1]
            t1, t2 = temperatures[i], temperatures[i+1]
            e1, e2 = etas[i], etas[i+1]
            slope = (e2 - e1) / (t2 - t1)
            x_cross = t1 + (target_eta - e1) / slope
            return x_cross
    raise ValueError("No crossing found within the given range.")

def xi_fit_func(T, A, b, T_BKT):
    return A * np.exp(b / np.sqrt(T / T_BKT - 1))
    

def fit_xi(xis: np.ndarray, temperatures: np.ndarray, T_BKT_guess: float) -> tuple[float, tuple[np.ndarray, np.ndarray]] | None:
    # Clean and filter data
    mask = ~np.isnan(temperatures) & ~np.isnan(xis) & np.isfinite(temperatures) & np.isfinite(xis)
    T_fit = temperatures[mask]
    xi_fit = xis[mask]

    # Require at least 3 points
    if len(T_fit) < 3:
        return None

    try:
        # Fit correlation length
        popt, _ = curve_fit(xi_fit_func, T_fit, xi_fit, p0=(1.0, 1.0, T_BKT_guess), maxfev=10000)
        T_BKT_xi = popt[-1]

        # Generate smooth fit curve
        T_dense = np.linspace(T_fit.min(), T_fit.max(), 300)
        xi_pred = xi_fit_func(T_dense, *popt)

        return T_BKT_xi, (T_dense, xi_pred)

    except RuntimeError:
        print("Fit failed")
        return None
        


def run_single_temperature(params):
    """Run simulation for a single temperature"""
    sim_params, temperature, temp_idx, algorithm = params
    L = sim_params['L']
    sweeps = sim_params['sweeps']
    thermalize_sweeps = sim_params['thermalize_sweeps']
    N = L * L
    model = XYModel(L, J=sim_params['J'])
    beta = 1.0 / temperature
    
    # Thermalization
    therm_disable = temp_idx > 0
    for _ in tqdm(range(thermalize_sweeps), desc=f"L={L}, T={temperature:.4f}: Therm", 
                  leave=False, disable=therm_disable):
        getattr(model, f"{algorithm}_step" if algorithm == 'metropolis' else f"{algorithm}_sweep")(beta)
    
    # Data collection
    data = {
        'energies': [],
        'magnetizations': [],
        'vorticities': [],
        'sum_cos_x': [],
        'sum_sin_x': [],
        'sum_cos_y': [],
        'sum_sin_y': [],
        'correlations': []
    }
    
    for _ in tqdm(range(sweeps), desc=f"L={L}, T={temperature:.4f}: Data", 
                  leave=False, disable=therm_disable):
        getattr(model, f"{algorithm}_step" if algorithm == 'metropolis' else f"{algorithm}_sweep")(beta)
        energy, mag, vorticity, scx, ssx, scy, ssy, correlation = model.calculate_observables()
        
        data['energies'].append(energy)
        data['magnetizations'].append(mag)
        data['vorticities'].append(vorticity)
        data['sum_cos_x'].append(scx)
        data['sum_sin_x'].append(ssx)
        data['sum_cos_y'].append(scy)
        data['sum_sin_y'].append(ssy)
        data['correlations'].append(correlation)
       

    # Convert to arrays
    for key in data:
        data[key] = np.array(data[key])

    # Calculate averages and errors
    n_meas = len(data['energies'])
    if n_meas < 2:
        results = {
            'energy': np.mean(data['energies']) if n_meas > 0 else np.nan,
            'energy_error': np.nan,
            'energy_sq': np.mean(data['energies']**2) if n_meas > 0 else np.nan,
            'magnetization': np.mean(data['magnetizations']) if n_meas > 0 else np.nan,
            'magnetization_error': np.nan,
            'magnetization_sq': np.mean(data['magnetizations']**2) if n_meas > 0 else np.nan,
            'magnetization_quad': np.mean(data['magnetizations']**4) if n_meas > 0 else np.nan,
            'vorticity': np.mean(data['vorticities']) if n_meas > 0 else np.nan,
            'stiffness': np.nan,
            'stiffness_error': np.nan,
            'eta': np.nan,
            'eta_error' : np.nan,
            'xi': np.nan,
            'xi_error' : np.nan
        }
    else:
        # Calculate stiffness: ρs = ⟨J cos(Δθ)⟩ - βJ²⟨sin(Δθ)⟩²
        avg_sum_cos_x = np.mean(data['sum_cos_x'])
        avg_sum_sin_x_sq = np.mean(data['sum_sin_x']**2)
        rho_s_x = avg_sum_cos_x / N - beta * avg_sum_sin_x_sq / N

        avg_sum_cos_y = np.mean(data['sum_cos_y'])
        avg_sum_sin_y_sq = np.mean(data['sum_sin_y']**2)
        rho_s_y = avg_sum_cos_y / N - beta * avg_sum_sin_y_sq / N

        stiffness_avg = 0.5 * (rho_s_x + rho_s_y)

        # Jackknife error estimation for stiffness
        jk_estimates = []
        for i in range(n_meas):
            jk_scx = np.delete(data['sum_cos_x'], i)
            jk_ssx = np.delete(data['sum_sin_x'], i)
            jk_scy = np.delete(data['sum_cos_y'], i)
            jk_ssy = np.delete(data['sum_sin_y'], i)

            jk_avg_scx = np.mean(jk_scx)
            jk_avg_ssx_sq = np.mean(jk_ssx**2)
            jk_rho_x = jk_avg_scx / N - beta * jk_avg_ssx_sq / N

            jk_avg_scy = np.mean(jk_scy)
            jk_avg_ssy_sq = np.mean(jk_ssy**2)
            jk_rho_y = jk_avg_scy / N - beta * jk_avg_ssy_sq / N

            jk_estimates.append(0.5 * (jk_rho_x + jk_rho_y))

        jk_estimates = np.array(jk_estimates)
        stiffness_err = np.sqrt((n_meas - 1) / n_meas * np.sum((jk_estimates - stiffness_avg)**2))

        eta, xi, _ = estimate_correlation_decay_parameters(np.mean(data['correlations'], axis = 0), temperature, L)
        # Jackknife error for eta and xi
        
        eta_jk = []
        xi_jk = []

        for i in range(n_meas):
            jk_corrs = np.delete(data['correlations'], i)
            jk_corr_avg = np.mean(jk_corrs, axis=0)
    
            try:
                # Use a more robust fitting approach for jackknife subsamples
                eta_i, xi_i, _ = estimate_correlation_decay_parameters(jk_corr_avg, temperature, L)
                eta_jk.append(eta_i)
                xi_jk.append(xi_i)
            except:
                # Skip this jackknife sample if fitting fails
                logger.debug(f"Jackknife fitting failed for sample {i} at T={temperature}")
                continue

        # Filter out any NaN values that might have slipped through
        eta_jk = np.array([e for e in eta_jk if not np.isnan(e)])
        xi_jk = np.array([x for x in xi_jk if not np.isnan(x)])

        # Only calculate errors if we have enough valid jackknife samples
        if len(eta_jk) > 1:
            eta_err = np.sqrt((n_meas - 1) / n_meas * np.sum((eta_jk - eta)**2))
        else:
            eta_err = np.nan

        if len(xi_jk) > 1:
            xi_err = np.sqrt((n_meas - 1) / n_meas * np.sum((xi_jk - xi)**2))
        else:
            xi_err = np.nan
        
        

        results = {
            'energy': np.mean(data['energies']),
            'energy_error': np.std(data['energies'], ddof=1) / np.sqrt(n_meas),
            'energy_sq': np.mean(data['energies']**2),
            'magnetization': np.mean(data['magnetizations']),
            'magnetization_error': np.std(data['magnetizations'], ddof=1) / np.sqrt(n_meas),
            'magnetization_sq': np.mean(data['magnetizations']**2),
            'magnetization_quad': np.mean(data['magnetizations']**4),
            'vorticity': np.mean(data['vorticities']),
            'stiffness': stiffness_avg,
            'stiffness_error': stiffness_err,
            'eta': eta,
            'eta_error': eta_err,
            'xi': xi,
            'xi_error': xi_err
        }
    
    return temperature, results

def simulate_lattice(sim_params, algorithm):
    """Run simulation across temperature range for a given lattice size"""
    L = sim_params['L']
    temperatures = np.linspace(sim_params['T_min'], sim_params['T_max'], sim_params['num_points'])
    N = L * L
    sim_params['temperatures'] = temperatures
    
    logger.info(f"Simulating XY model on {L}x{L} lattice using {algorithm} algorithm")
    
    # Arrays to store results
    results = {
        'temperatures': temperatures,
        'energies': np.zeros(sim_params['num_points']),
        'energy_errors': np.zeros(sim_params['num_points']),
        'energy_sq_avg': np.zeros(sim_params['num_points']),
        'magnetizations': np.zeros(sim_params['num_points']),
        'mag_errors': np.zeros(sim_params['num_points']),
        'mag_sq_avg': np.zeros(sim_params['num_points']),
        'mag_quad_avg': np.zeros(sim_params['num_points']),
        'vortices': np.zeros(sim_params['num_points']),
        'stiffnesses': np.zeros(sim_params['num_points']),
        'stiffness_errors': np.zeros(sim_params['num_points']),
        'etas': np.zeros(sim_params['num_points']),
        'eta_errors': np.zeros(sim_params['num_points']),
        'xis': np.zeros(sim_params['num_points']),
        'xi_errors': np.zeros(sim_params['num_points'])
        
    }
    
    start_time = time.time()
    
    params_list = [(sim_params, T, i, algorithm) for i, T in enumerate(temperatures)]

    max_workers = mp.cpu_count()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        sim_results = list(tqdm(
            executor.map(run_single_temperature, params_list),
            total=len(params_list),
            desc=f"L={L} ({algorithm}) Progress"
        ))
    
    # Process results
    for T, result_data in sim_results:
        i = np.where(temperatures == T)[0][0]
        results['energies'][i] = result_data['energy']
        results['energy_errors'][i] = result_data['energy_error']
        results['energy_sq_avg'][i] = result_data['energy_sq']
        results['magnetizations'][i] = result_data['magnetization']
        results['mag_errors'][i] = result_data['magnetization_error']
        results['mag_sq_avg'][i] = result_data['magnetization_sq']
        results['mag_quad_avg'][i] = result_data['magnetization_quad']
        results['vortices'][i] = result_data['vorticity']
        results['stiffnesses'][i] = result_data['stiffness']
        results['stiffness_errors'][i] = result_data['stiffness_error']
        results['etas'][i] = result_data['eta']
        results['eta_errors'][i] = result_data['eta_error']
        results['xis'][i] = result_data['xi']
        results['xi_errors'][i] = result_data['xi_error']
    

    elapsed = time.time() - start_time
    logger.info(f"Simulation for L={L} ({algorithm}) completed in {elapsed:.1f} seconds")
    
    # Calculate derived quantities
    beta = 1.0 / temperatures
    results['specific_heat'] = beta**2 * N * (results['energy_sq_avg'] - results['energies']**2)
    results['susceptibility'] = beta * N * (results['mag_sq_avg'] - results['magnetizations']**2)
    # Calculate Binder cumulant: U = 1 - <M^4> / (3 <M^2>^2)
    results['binder_cumulant'] = 1.0 - results['mag_quad_avg'] / (3.0 * results['mag_sq_avg']**2)

    # Calculate BKT transition temperature
    line = 2/np.pi * temperatures
    results['T_BKT'] = find_cross(results['stiffnesses'], line, temperatures)
    results['T_BKT_eta'] = find_eta_cross(results['etas'], temperatures)
    results['T_BKT_xi'], results['xi_fit_curve'] = fit_xi(results['xis'], temperatures, results['T_BKT_eta'] )
    logger.info(f"L={L} ({algorithm}): Estimated BKT temperature from stiffness criterion: T_BKT ≈ {results['T_BKT']:.4f}")
    logger.info(f"L={L} ({algorithm}): Estimated BKT temperature from η = 1/4 criterion (spin-spin correlation): T_BKT ≈ {results['T_BKT_eta']:.4f}")


    return results

# ============================================================================
# Visualization functions
# ============================================================================

def visualize_plots(results_dict, output_dir, algorithm):
    """Create comparison plots for different lattice sizes"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('ggplot')
    lattice_sizes = sorted(results_dict.keys())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(lattice_sizes)))
    
    # Plot properties for each subplot
    plot_props = {
        (0, 0): {'y': 'energies', 'yerr': 'energy_errors', 'title': 'Energy vs Temperature', 'ylabel': 'Energy per site', 'filename': 'energy'},
        (0, 1): {'y': 'magnetizations', 'yerr': 'mag_errors', 'title': 'Magnetization vs Temperature', 'ylabel': 'Magnetization', 'filename': 'magnetization'},
        (0, 2): {'y': 'specific_heat', 'title': 'Specific Heat vs Temperature', 'ylabel': 'Specific Heat per Site', 'filename': 'specific_heat'},
        (1, 0): {'y': 'susceptibility', 'title': 'Susceptibility vs Temperature', 'ylabel': 'Susceptibility per Site', 'filename': 'susceptibility'},
        (1, 1): {'y': 'binder_cumulant', 'title': 'Binder Cumulant vs Temperature', 'ylabel': 'Binder cumulant', 'filename': 'binder_cumulant'},
        (1, 2): {'y': 'vortices', 'title': 'Vortex Density vs Temperature', 'ylabel': 'Vortex Density', 'filename': 'vortices'},
        (2, 0): {'y': 'stiffnesses', 'yerr': 'stiffness_errors', 'title': 'Spin Stiffness vs Temperature with BKT Transition', 'ylabel': 'Spin Stiffness', 'filename': 'stiffness'},
        (2, 1): {'y': 'etas', 'yerr': 'eta_errors', 'title': 'η Exponent vs Temperature ', 'ylabel': 'Correlation Decay Exponent η', 'filename': 'eta'},
        (2, 2): {'y': 'xis', 'yerr': 'xi_errors', 'title': 'Correlation Length ξ vs Temperature', 'ylabel': 'Correlation Length ξ', 'filename': 'xi'}
    }
    
    # First, create the combined figure
    fig, axes = plt.subplots(3, 3, figsize=(20, 25), dpi=150)
    
    # Plot each subplot in the combined figure
    for (row, col), props in plot_props.items():
        ax = axes[row, col]
        for i, L in enumerate(lattice_sizes):
            results = results_dict[L]
            if 'yerr' in props:
                ax.errorbar(
                    results['temperatures'], results[props['y']], yerr=results[props.get('yerr')],
                    marker='o', markersize=4, linestyle='-', linewidth=1.5,
                    color=colors[i], label=f'L = {L}'
                )
            else:
                ax.plot(
                    results['temperatures'], results[props['y']],
                    marker='o', markersize=4, linestyle='-', linewidth=1.5,
                    color=colors[i], label=f'L = {L}'
                )
        
        # Add theoretical line for stiffness plot
        if (row, col) == (2, 0):
            T_plot = results_dict[lattice_sizes[-1]]['temperatures']
            line = 2/np.pi * T_plot
            ax.plot(T_plot, line, 'k--', label=r'$\frac{2}{\pi}T$', linewidth=1.5)
            
            # Add transition temperature markers with labels in legend
            for i, L in enumerate(lattice_sizes):
                results = results_dict[L]
                if not np.isnan(results['T_BKT']):
                    ax.axvline(results['T_BKT'], color=colors[i], linestyle=':', alpha=0.7, 
                              label=f'T_BKT(L={L})={results["T_BKT"]:.3f}')
            
            # Use a more compact legend for the stiffness plot
            ax.legend(fontsize=8, loc='upper right', ncol=2)

        elif (row,col) == (2, 1):
            ax.axhline(0.25, color='black', linestyle='--', linewidth=1.5, label=r'$\eta = \frac{1}{4}$')

            # Add transition temperature markers with labels in legend
            for i, L in enumerate(lattice_sizes):
                results = results_dict[L]
                if not np.isnan(results['T_BKT_eta']):
                    ax.axvline(results['T_BKT_eta'], color=colors[i], linestyle=':', alpha=0.7, 
                              label=f'T_BKT(L={L})={results["T_BKT_eta"]:.3f}')
            
            # Use a more compact legend for the stiffness plot
            ax.legend(fontsize=8, loc='upper right', ncol=2)
        
        elif (row,col) == (2,2):
    
            for i, L in enumerate(lattice_sizes):
                results = results_dict[L]
                T_dense , xi_pred = results['xi_fit_curve']
                ax.plot(T_dense, xi_pred, '--', color=colors[i], label= f"Fit L={L}, $T_{{BKT}}$={results['T_BKT_xi']:.3f}")
            ax.legend(fontsize=8, loc='upper right', ncol=2)
            
        else:
            ax.legend(fontsize=10)

        
        ax.set_xlabel('Temperature (T)', fontsize=12)
        ax.set_ylabel(props['ylabel'], fontsize=12)
        ax.set_title(props['title'], fontsize=14)
        ax.grid(True, alpha=0.3)
    
    # Add summary text box
    textstr = "BKT Transition Temperature Estimates:\n"
    for L in lattice_sizes:
        results = results_dict[L]
        # From stiffness
        if not np.isnan(results['T_BKT']):
            textstr += f"L = {L}: T_BKT (stiffness) = {results['T_BKT']:.4f}\n"
        else:
           textstr += f"L = {L}: T_BKT (stiffness) = NaN\n"
        # From η exponent
        if not np.isnan(results['T_BKT_eta']):
           textstr += f"L = {L}: T_BKT (η = 1/4) = {results['T_BKT_eta']:.4f}\n"
        else:
           textstr += f"L = {L}: T_BKT (η = 1/4) = NaN\n"
        # From xi exponent
        if not np.isnan(results['T_BKT_xi']):
           textstr += f"L = {L}: T_BKT (xi) = {results['T_BKT_xi']:.4f}\n"
        else:
           textstr += f"L = {L}: T_BKT (xi) = NaN\n"

    textstr += f"Theoretical: T_BKT ≈ 0.8935"
    fig.text(0.5, 0.01, textstr, fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5), ha='center')

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    fig.suptitle(f'XY Model Simulation Results ({algorithm.capitalize()} Algorithm)', y=0.995)
    plt.savefig(os.path.join(output_dir, f'xy_model_comparison_{algorithm}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Create and save individual plots
    for (row, col), props in plot_props.items():
        plt.figure(figsize=(8, 6), dpi=150)
        
        for i, L in enumerate(lattice_sizes):
            results = results_dict[L]
            if 'yerr' in props:
                plt.errorbar(
                    results['temperatures'], results[props['y']], yerr=results[props.get('yerr')],
                    marker='o', markersize=5, linestyle='-', linewidth=1.5,
                    color=colors[i], label=f'L = {L}'
                )
            else:
                plt.plot(
                    results['temperatures'], results[props['y']],
                    marker='o', markersize=5, linestyle='-', linewidth=1.5,
                    color=colors[i], label=f'L = {L}'
                )
        
        # Add theoretical line for stiffness plot
        if props['filename'] == 'stiffness':
            T_plot = results_dict[lattice_sizes[-1]]['temperatures']
            line = 2/np.pi * T_plot
            plt.plot(T_plot, line, 'k--', label=r'$\frac{2}{\pi}T$', linewidth=1.5)
            
            # Add transition temperature markers with labels in legend
            for i, L in enumerate(lattice_sizes):
                results = results_dict[L]
                if not np.isnan(results['T_BKT']):
                    plt.axvline(results['T_BKT'], color=colors[i], linestyle=':', alpha=0.7,
                               label=f'T_BKT(L={L})={results["T_BKT"]:.3f}')
            
            # Use a more compact legend for the stiffness plot
            plt.legend(fontsize=9, loc='upper right', ncol=2)
        else:
            plt.legend(fontsize=10, loc='best')


        # Add theoretical line for eta plot
        if props['filename'] == 'eta':
            T_plot = results_dict[lattice_sizes[-1]]['temperatures']
            plt.axhline(0.25, color='black', linestyle='--', linewidth=1.5, label=r'$\eta = \frac{1}{4}$')
            
            # Add transition temperature markers with labels in legend
            for i, L in enumerate(lattice_sizes):
                results = results_dict[L]
                if not np.isnan(results['T_BKT_eta']):
                    plt.axvline(results['T_BKT_eta'], color=colors[i], linestyle=':', alpha=0.7,
                               label=f'T_BKT(L={L})={results["T_BKT_eta"]:.3f}')
            
            # Use a more compact legend for the stiffness plot
            plt.legend(fontsize=9, loc='upper right', ncol=2)
        else:
            plt.legend(fontsize=10, loc='best')


        # Fit xi plot
        if props['filename'] == 'xi':
            for i, L in enumerate(lattice_sizes):
                results = results_dict[L]
                T_dense , xi_pred = results['xi_fit_curve']
                plt.plot(T_dense, xi_pred, '--', color=colors[i], label= f"Fit L={L}, $T_{{BKT}}$={results['T_BKT_xi']:.3f}")
            plt.legend(fontsize=8, loc='upper right', ncol=2)
            
        else:
            plt.legend(fontsize=10, loc='best')


        
        plt.xlabel('Temperature (T)', fontsize=12)
        plt.ylabel(props['ylabel'], fontsize=12)
        plt.title(props['title'], fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the individual plot
        plt.savefig(os.path.join(output_dir, f'xy_model_{props["filename"]}_{algorithm}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def scaling_func_log_sq(L, T_inf, a):
    """Scaling function T(L) = T_inf + a/(ln L)²"""
    L = np.array(L)
    valid_L = L > 1
    result = np.full_like(L, np.nan, dtype=float)
    if np.any(valid_L):
        logL = np.log(L[valid_L])
        result[valid_L] = T_inf + a / (logL**2)
    return result

def visualize_finite_size_scaling(L_values, T_bkt_values, T_bkt_errors, popt, pcov, output_dir, algorithm, method):
    """Plot finite-size scaling analysis of T_BKT(L)"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    T_inf, a = popt
    T_inf_err = np.sqrt(pcov[0, 0]) if pcov.size > 0 else np.nan
    
    valid_mask = L_values > 1
    L_plot = L_values[valid_mask]
    T_plot = T_bkt_values[valid_mask]
    x_plot = 1.0 / (np.log(L_plot)**2)
    
    x_fit_line = np.linspace(min(x_plot) * 0.9, max(x_plot) * 1.1, 100)
    T_fit_line = T_inf + a * x_fit_line

    plt.figure(figsize=(8, 6), dpi=120)
    if T_bkt_errors is not None:
        plt.errorbar(x_plot, T_plot, yerr=T_bkt_errors[valid_mask], fmt='o', markersize=5,
                    capsize=3, label='Simulation Data $T_{cross}(L)$')
    else:
        plt.plot(x_plot, T_plot, 'o', markersize=5, label='Simulation Data $T_{cross}(L)$')
        
    plt.plot(x_fit_line, T_fit_line, 'r--', 
            label=f'Fit: $T = T_\\infty + a / (\\ln L)^2$ ($T_\\infty = {T_inf:.4f} \\pm {T_inf_err:.4f}$)')

    plt.xlabel(r'$1 / (\ln L)^2$', fontsize=12)
    plt.ylabel(r'$T_{cross}(L)$', fontsize=12)
    plt.title(f'Finite-Size Scaling of Crossing Temperature ({algorithm.capitalize()} Algorithm, {method.capitalize()} Method)', fontsize=14)
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.5)
    plt.text(0.05, 0.95, f'Extrapolated $T(\\infty) = {T_inf:.4f} \\pm {T_inf_err:.4f}$',
            transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'xy_model_fss_{algorithm}_{method}.png'), dpi=300)
    plt.close()

# ============================================================================
# Main script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="XY Model Simulation", 
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--output-dir", type=str, default="xy_model_results")
    parser.add_argument("--t-min", type=float, default=0.7)
    parser.add_argument("--t-max", type=float, default=1.5)
    parser.add_argument("--num-points", type=int, default=20)
    parser.add_argument("--sweeps", type=int, default=2000)
    parser.add_argument("--thermalize", type=int, default=2000)
    parser.add_argument("--lattice-sizes", type=int, nargs="+", default=[20,30,40,50])
    parser.add_argument("--j", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--algorithm", type=str, choices=['metropolis', 'wolff'], default='wolff')
    parser.add_argument("--disable-jit", action="store_true")
    
    args = parser.parse_args()
    
    if args.seed is not None:
        np.random.seed(args.seed)
    
    if args.disable_jit:
        numba.config.DISABLE_JIT = True
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    results_dict = {}
    
    for L in args.lattice_sizes:
        # Scale sweeps for Metropolis (but not for Wolff)
        thermalize_sweeps = args.thermalize
        sweeps = args.sweeps
        
        if args.algorithm == 'metropolis':
            scale_factor = 1 if L <= 10 else (2 if L <= 20 else (5 if L <= 30 else 10))
            thermalize_sweeps *= scale_factor
            sweeps *= scale_factor
        
        sim_params = {
            'L': L,
            'T_min': args.t_min,
            'T_max': args.t_max,
            'num_points': args.num_points,
            'sweeps': sweeps,
            'thermalize_sweeps': thermalize_sweeps,
            'J': args.j
        }

        results = simulate_lattice(sim_params, args.algorithm)
        results_dict[L] = results
    
    # Visualization
    visualize_plots(results_dict, args.output_dir, args.algorithm)

    # Finite-size scaling analysis
    L_values = np.array(sorted(results_dict.keys()))
    T_bkt_values = np.array([results_dict[L]['T_BKT'] for L in L_values])
    T_bkt_eta_values = np.array([results_dict[L]['T_BKT_eta'] for L in L_values])
    T_bkt_xi_values = np.array([results_dict[L]['T_BKT_xi'] for L in L_values])
    
    # Filter out NaN values for fitting
    valid_mask = ~np.isnan(T_bkt_values) & (L_values > 1)
    if np.sum(valid_mask) >= 3:
        L_fit = L_values[valid_mask]
        T_fit = T_bkt_values[valid_mask]
        
        try:
            popt, pcov = curve_fit(scaling_func_log_sq, L_fit, T_fit, p0=[0.9, 1.0])
            T_inf_fit = popt[0]
            T_inf_err = np.sqrt(pcov[0, 0])
            logger.info(f"Finite-size scaling fit (stiffness): T_BKT(inf) = {T_inf_fit:.4f} +/- {T_inf_err:.4f}")
            
            visualize_finite_size_scaling(L_values, T_bkt_values, None, popt, pcov, args.output_dir, args.algorithm, method = "stiffness")
        except Exception as e:
            logger.error(f"Finite-size scaling analysis failed: {e}")
    else:
        logger.warning(f"Skipping finite-size scaling fit: Need at least 3 valid T_BKT(L) points")

    # Filter out NaN values for fitting
    valid_mask = ~np.isnan(T_bkt_eta_values) & (L_values > 1)
    if np.sum(valid_mask) >= 3:
        L_fit = L_values[valid_mask]
        T_fit = T_bkt_eta_values[valid_mask]
        
        try:
            popt, pcov = curve_fit(scaling_func_log_sq, L_fit, T_fit, p0=[0.9, 0.8])
            T_inf_fit = popt[0]
            T_inf_err = np.sqrt(pcov[0, 0])
            logger.info(f"Finite-size scaling fit (eta): T_BKT(inf) = {T_inf_fit:.4f} +/- {T_inf_err:.4f}")
            
            visualize_finite_size_scaling(L_values, T_bkt_eta_values, None, popt, pcov, args.output_dir, args.algorithm, method = "eta")
        except Exception as e:
            logger.error(f"Finite-size scaling analysis failed: {e}")
    else:
        logger.warning(f"Skipping finite-size scaling fit: Need at least 3 valid T_BKT(L) points")


        # Filter out NaN values for fitting
    valid_mask = ~np.isnan(T_bkt_xi_values) & (L_values > 1)
    if np.sum(valid_mask) >= 3:
        L_fit = L_values[valid_mask]
        T_fit = T_bkt_xi_values[valid_mask]
        
        try:
            popt, pcov = curve_fit(scaling_func_log_sq, L_fit, T_fit, p0=[0.9, 0.8])
            T_inf_fit = popt[0]
            T_inf_err = np.sqrt(pcov[0, 0])
            logger.info(f"Finite-size scaling fit (xi): T_BKT(inf) = {T_inf_fit:.4f} +/- {T_inf_err:.4f}")
            
            visualize_finite_size_scaling(L_values, T_bkt_xi_values, None, popt, pcov, args.output_dir, args.algorithm, method = "xi")
        except Exception as e:
            logger.error(f"Finite-size scaling analysis failed: {e}")
    else:
        logger.warning(f"Skipping finite-size scaling fit: Need at least 3 valid T_BKT(L) points")

    # Print summary
    print(f"\nSummary of Kosterlitz-Thouless Transition Temperatures ({args.algorithm.capitalize()} Algorithm):")
    print("------------------------------------------------------")
    for L in sorted(results_dict.keys()):
        t_bkt_val = results_dict[L]['T_BKT']
        t_bkt_eta_val = results_dict[L]['T_BKT_eta']
        t_bkt_xi_val = results_dict[L]['T_BKT_xi']
        if not np.isnan(t_bkt_val):
            print(f"Lattice size L = {L:2d}: T_BKT = {t_bkt_val:.4f}")
        else:
            print(f"Lattice size L = {L:2d}: T_BKT = NaN")

        if not np.isnan(t_bkt_eta_val):
            print(f"Lattice size L = {L:2d}: T_BKT (eta) = {t_bkt_eta_val:.4f}")
        else:
            print(f"Lattice size L = {L:2d}: T_BKT (eta) = NaN")

        if not np.isnan(t_bkt_xi_val):
            print(f"Lattice size L = {L:2d}: T_BKT (xi) = {t_bkt_xi_val:.4f}")
        else:
            print(f"Lattice size L = {L:2d}: T_BKT (xi) = NaN")
    print("Theoretical value: T_BKT ≈ 0.8935")
    print(f"\nResults and plots saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 