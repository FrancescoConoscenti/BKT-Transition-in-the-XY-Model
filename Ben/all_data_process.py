#!/usr/bin/env python3
"""
Data analysis module for the 2D XY model simulation.

This script processes the raw output from the Monte Carlo simulations,
calculating thermodynamic quantities and their error estimates using
jackknife resampling. It computes energy, specific heat, magnetization,
susceptibility, Binder cumulant, correlation length, spin stiffness,
and vortex density.
"""
from __future__ import division
import numpy as np
import time
import sys
import os
from numba import jit


# Fitting functions (rarely used but kept for compatibility)
def func(x, c, a):
    """Exponential decay function with offset for curve fitting."""
    return (1-a) * np.exp(-c*x) + a


def func2(x, c):
    """Simple exponential decay function for curve fitting."""
    return np.exp(-c*x)


# Jackknife error analysis functions
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def jackBlocks(original_list, num_of_blocks, length_of_blocks):
    """
    Create blocks of data for jackknife analysis.
    
    Args:
        original_list: Original data array
        num_of_blocks: Number of blocks to create
        length_of_blocks: Length of each block
        
    Returns:
        Array of block averages
    """
    block_list = np.zeros(num_of_blocks)
    length_of_blocks = int(length_of_blocks)
    for i in range(num_of_blocks):
        block_list[i] = (1/length_of_blocks)*np.sum(original_list[i*(length_of_blocks) : (i + 1)*(length_of_blocks)])
    return block_list


@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def JackknifeError(blocks, length_of_blocks):
    """
    Calculate mean and error using jackknife resampling.
    
    Args:
        blocks: Array of block averages
        length_of_blocks: Length of each block
        
    Returns:
        Tuple of (mean, error)
    """
    #blocks is already O_(B,n)
    blocks = np.array(blocks)
    N_B = len(blocks)
    avg = np.sum(blocks)/N_B
    #length_of_blocks is k
    N_J = N_B*length_of_blocks #is basically N
    jack_block = (1/(N_J - length_of_blocks))*(N_J*np.ones(N_B)*avg - length_of_blocks*blocks)
    bar_o_j = np.sum(jack_block)/N_B
    error_sq = ((N_B - 1)/N_B)*np.sum((jack_block - bar_o_j*np.ones(N_B))**2)

    return avg, np.sqrt(error_sq)


@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def JackknifeErrorFromFullList(original_list, num_of_blocks, length_of_blocks):
    """
    Calculate mean and error using jackknife resampling directly from data.
    
    Args:
        original_list: Original data array
        num_of_blocks: Number of blocks to create
        length_of_blocks: Length of each block
        
    Returns:
        Tuple of (mean, error)
    """
    blocks = jackBlocks(original_list, num_of_blocks, length_of_blocks)

    #blocks is already O_(B,n)
    N_B = len(blocks)
    avg = np.sum(blocks)/N_B
    #length_of_blocks is k
    N_J = N_B*length_of_blocks #is basically N
    jack_block = (1/(N_J - length_of_blocks))*(N_J*np.ones(N_B)*avg - length_of_blocks*blocks)
    bar_o_j = np.sum(jack_block)/N_B
    error_sq = ((N_B - 1)/N_B)*np.sum((jack_block - bar_o_j*np.ones(N_B))**2)

    return avg, np.sqrt(error_sq)


# Autocorrelation analysis functions
def binning_level(values):
    """
    Bin values pairwise to create a coarser time series.
    
    Args:
        values: Time series data
        
    Returns:
        Binned time series with half the length
    """
    n = len(values) // 2
    new_values = np.zeros(n)
    for i in range(n):
        new_values[i] = (values[2*i] + values[2*i+1]) / 2
    return new_values


def autocorrelation(data, k_max):
    """
    Calculate autocorrelation function for time lags up to k_max.
    
    Args:
        data: Time series data
        k_max: Maximum lag to calculate
        
    Returns:
        Tuple of (correlation values, mean of data)
    """
    n = len(data)
    correlations = np.zeros(k_max)
    
    for lag in range(k_max):
        # Calculate correlation at this lag
        sum_corr = 0.0
        count = 0
        
        for i in range(n - lag):
            sum_corr += data[i] * data[i + lag]
            count += 1
            
        correlations[lag] = sum_corr / count
    
    # Calculate mean
    data_mean = np.sum(data) / n
    
    return correlations, data_mean


def autocorrelation_with_errors(data, k_max):
    """
    Calculate autocorrelation function with error estimates.
    
    Args:
        data: Time series data
        k_max: Maximum lag to calculate
        
    Returns:
        Tuple of (correlation values, error estimates)
    """
    n = len(data)
    correlations = np.zeros(k_max)
    errors = np.zeros(k_max)
    
    for lag in range(k_max):
        # Calculate correlation at this lag with sample values for error estimation
        corr_samples = []
        count = 0
        
        for i in range(n - lag):
            corr_samples.append(data[i] * data[i + lag])
            count += 1
            
        # Calculate mean and standard deviation
        mean_corr = np.sum(corr_samples) / count
        std_corr = np.std(corr_samples)
        
        correlations[lag] = mean_corr
        errors[lag] = std_corr
    
    return correlations, errors


def calculate_normalized_autocorrelation(data, lag_max, avg_length, error_length):
    """
    Calculate normalized autocorrelation function with error estimates.
    
    Args:
        data: Time series data
        lag_max: Maximum lag to calculate
        avg_length: Length of data chunks to average over
        error_length: Number of chunks for error analysis
        
    Returns:
        Tuple of (normalized correlation function, error estimates)
    """
    # Calculate number of samples available at each lag
    samples_per_lag = np.array([(avg_length - i) for i in range(lag_max)])
    
    # Collect correlation data for each chunk
    correlations = []
    means = []

    for i in range(error_length):
        chunk = data[avg_length*i:avg_length*(i+1)]
        corr, mean = autocorrelation(chunk, lag_max)
        correlations.append(corr)
        means.append(mean)
        
    correlations = np.array(correlations)
    means = np.array(means)
    
    # Jackknife analysis for mean
    mean_avg, mean_err = JackknifeError(means, error_length)
    
    # Jackknife analysis for correlations at each lag
    avg_correlations = np.zeros(lag_max)
    err_correlations = np.zeros(lag_max)
    
    for m in range(lag_max):
        avg_correlations[m], err_correlations[m] = JackknifeError(
            correlations[:, m], samples_per_lag[m])
    
    # Calculate normalized autocorrelation
    denom = avg_correlations[0] - mean_avg**2
    numer = avg_correlations - mean_avg**2
    
    norm_corr = numer / denom
    
    # Error propagation
    denom_err = np.sqrt(err_correlations[0]**2 + (2*np.abs(mean_avg)*mean_err)**2)
    numer_err = np.sqrt(err_correlations**2 + (2*np.abs(mean_avg)*mean_err)**2)
    
    norm_corr_err = np.abs(norm_corr) * np.sqrt(
        (numer_err/numer)**2 + (denom_err/denom)**2)

    return norm_corr, norm_corr_err


def calculate_autocorrelation_simple(data, k_max):
    """
    Simplified method to calculate normalized autocorrelation function.
    
    Args:
        data: Time series data
        k_max: Maximum lag to calculate
        
    Returns:
        Tuple of (normalized correlation function, error estimates)
    """
    # Calculate raw correlations with errors
    correlations, corr_errors = autocorrelation_with_errors(data, k_max)

    # Calculate mean and standard error
    data_mean = np.mean(data)
    data_err = np.std(data) / np.sqrt(len(data))
    
    # Calculate normalized autocorrelation
    denom = correlations[0] - data_mean**2
    numer = correlations - data_mean**2
    
    norm_corr = numer / denom
    
    # Error propagation
    denom_err = np.sqrt(corr_errors[0]**2 + (2*np.abs(data_mean)*data_err)**2)
    numer_err = np.sqrt(corr_errors**2 + (2*np.abs(data_mean)*data_err)**2)
    
    norm_corr_err = np.abs(norm_corr) * np.sqrt(
        (numer_err/numer)**2 + (denom_err/denom)**2)
    
    return norm_corr, norm_corr_err


def main():
    """Process simulation data and calculate thermodynamic observables."""
    # Parse command line arguments
    N = int(sys.argv[1])  # Linear system size
    factor_print = 10000  # Scaling factor for filenames
    
    # Input and output directories
    name_dir = f'testL={N}'
    where_to_save = './'
    output_dir = f'{name_dir}finalData'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(where_to_save + output_dir):
        os.mkdir(where_to_save + output_dir)

    # Load simulation parameters
    sim_params = np.loadtxt(f'{where_to_save}{name_dir}/variables.data')
    length_box = int(sim_params[0])   # Number of MC steps per bin
    number_box = int(sim_params[1])   # Number of measurement bins
    temperatures = sim_params[3:]     # List of temperatures
    
    # Save parameters to output directory
    np.savetxt(f'{where_to_save}{output_dir}/variables.data', sim_params)

    # Number of temperature points
    num_temps = len(temperatures)

    print()
    print('Starting Analysis')
    print()

    # Initialize arrays for various thermodynamic quantities
    # Each array has data values in first half, error bars in second half
    energy = np.zeros(2 * num_temps)
    specific_heat = np.zeros(2 * num_temps)
    order_cumulant = np.zeros(2 * num_temps)
    order_param = np.zeros(2 * num_temps)
    order_param_alt = np.zeros(2 * num_temps)
    susceptibility = np.zeros(2 * num_temps)
    susceptibility_alt = np.zeros(2 * num_temps)
    binder_cumulant = np.zeros(2 * num_temps)
    correlation_length = np.zeros(2 * num_temps)
    spin_stiffness = np.zeros(2 * num_temps)
    fourth_order = np.zeros(2 * num_temps)
    vorticity = np.zeros(2 * num_temps)

    # Energy histogram data
    num_bins_hist = int(number_box * length_box / 20)
    energy_hist = np.zeros((num_temps, num_bins_hist))
    energy_hist_edges = np.zeros((num_temps, num_bins_hist + 1))
    
    # Process data for each temperature
    for t_idx in range(num_temps):
        # Load raw data for this temperature
        temp = temperatures[t_idx]
        filename = f'{where_to_save}{name_dir}/outputatT={int(temp*factor_print):05d}.data'
        data = np.loadtxt(filename)
        
        # Extract and normalize raw data
        # Column 0: Energy, Columns 1-2: Order parameter (complex), Column 3: Correlation
        # Columns 4-5: Stiffness terms, Column 6: Vorticity
        e_raw = data[:, 0] / N**2
        m_raw = (data[:, 1] + 1j * data[:, 2]) / N**2
        corr_xi = data[:, 3] / N**4
        corr_xi_0 = (data[:, 1]**2 + data[:, 2]**2) / N**4
        
        # Derived quantities for energy
        e_squared = e_raw**2
        e_fourth = e_squared**2
        
        # Derived quantities for magnetization
        m_real = np.real(m_raw)
        m_imag = np.imag(m_raw)
        m_abs = np.abs(m_raw)
        m_squared = m_abs**2
        m_fourth = m_squared**2
        
        # Calculate means and errors using jackknife
        e_avg, e_err = JackknifeErrorFromFullList(e_raw, number_box, length_box)
        e2_avg, e2_err = JackknifeErrorFromFullList(e_squared, number_box, length_box)
        e4_avg, e4_err = JackknifeErrorFromFullList(e_fourth, number_box, length_box)
        
        m_real_avg, m_real_err = JackknifeErrorFromFullList(m_real, number_box, length_box)
        m_imag_avg, m_imag_err = JackknifeErrorFromFullList(m_imag, number_box, length_box)
        m_abs_avg, m_abs_err = JackknifeErrorFromFullList(m_abs, number_box, length_box)
        m2_avg, m2_err = JackknifeErrorFromFullList(m_squared, number_box, length_box)
        m4_avg, m4_err = JackknifeErrorFromFullList(m_fourth, number_box, length_box)
        
        xi_avg, xi_err = JackknifeErrorFromFullList(corr_xi, number_box, length_box)
        xi0_avg, xi0_err = JackknifeErrorFromFullList(corr_xi_0, number_box, length_box)

        # Store energy and errors
        energy[t_idx] = e_avg
        energy[num_temps + t_idx] = e_err
        
        # Calculate specific heat: C_v = (<E²> - <E>²)/T²
        T = temperatures[t_idx]
        T_squared = T**2
        specific_heat[t_idx] = (e2_avg - e_avg**2) / T_squared
        specific_heat[num_temps + t_idx] = np.sqrt(e2_err**2 + (2*e_avg*e_err)**2) / T_squared
        
        # Calculate energy cumulant: (<E⁴>/<E²>²) - 1
        e_cumulant = e4_avg / e2_avg**2 - 1
        order_cumulant[t_idx] = e_cumulant
        order_cumulant[num_temps + t_idx] = np.abs(e_cumulant) * np.sqrt(
            (e4_err/e4_avg)**2 + (2*e2_err/e2_avg)**2)

        # Calculate order parameter: |<M>| = sqrt(<Re M>² + <Im M>²)
        m_op_squared = m_real_avg**2 + m_imag_avg**2
        m_op_err = np.sqrt((2*m_real_avg*m_real_err)**2 + (2*m_imag_avg*m_imag_err)**2)
        
        order_param[t_idx] = np.sqrt(m_op_squared)
        order_param[num_temps + t_idx] = 0.5 * m_op_err / np.sqrt(m_op_squared)
        
        # Alternative order parameter: <|M|>
        order_param_alt[t_idx] = m_abs_avg
        order_param_alt[num_temps + t_idx] = m_abs_err
        
        # Calculate susceptibility: (<|M|²> - <|M|>²)/T
        susceptibility[t_idx] = (m2_avg - m_abs_avg**2) / T
        susceptibility[num_temps + t_idx] = np.sqrt(m2_err**2 + (2*m_abs_avg*m_abs_err)**2) / T
        
        # Alternative susceptibility: <|M|²>/T
        susceptibility_alt[t_idx] = m2_avg / T
        susceptibility_alt[num_temps + t_idx] = m2_err / T
        
        # Binder cumulant: 1 - <|M|⁴>/(3<|M|²>²)
        binder = 1.0 - m4_avg / (3.0 * m2_avg**2)
        binder_cumulant[t_idx] = binder
        binder_cumulant[num_temps + t_idx] = (1/3) * np.abs(m4_avg/m2_avg**2) * np.sqrt(
            (m4_err/m4_avg)**2 + (2*m2_err/m2_avg)**2)
        
        # Calculate correlation length
        factor = 1.0 / (2.0 * np.sin(np.pi/N))**2
        xi_ratio = (xi0_avg / xi_avg) - 1.0
        
        correlation_length[t_idx] = factor * xi_ratio
        correlation_length[num_temps + t_idx] = factor * np.sqrt(
            (xi0_err/xi0_avg)**2 + (xi_err/xi_avg)**2)
        
        # Create energy histogram
        e_min = np.min(e_raw)
        e_max = np.max(e_raw)
        e_padding = 0.01 * (e_max - e_min)
        e_range = (e_min + e_padding, e_max - e_padding)
        
        hist, edges = np.histogram(e_raw, num_bins_hist, range=e_range)
        energy_hist[t_idx] = hist
        energy_hist_edges[t_idx] = edges
        
        # Calculate spin stiffness related quantities
        stiff_h = data[:, 4] / N**2
        stiff_i = data[:, 5] / N**2
        stiff_i_squared = stiff_i**2
        stiff_i_fourth = stiff_i_squared**2

        # Calculate means and errors for stiffness
        stiff_h_avg, stiff_h_err = JackknifeErrorFromFullList(stiff_h, number_box, length_box)
        stiff_i_avg, stiff_i_err = JackknifeErrorFromFullList(stiff_i, number_box, length_box)
        stiff_i2_avg, stiff_i2_err = JackknifeErrorFromFullList(stiff_i_squared, number_box, length_box)
        stiff_i4_avg, stiff_i4_err = JackknifeErrorFromFullList(stiff_i_fourth, number_box, length_box)

        # Calculate spin stiffness: ρ_s = <H> - (N²/T)(<I²> - <I>²)
        stiffness = stiff_h_avg - (N**2/T) * (stiff_i2_avg - stiff_i_avg**2)
        
        spin_stiffness[t_idx] = stiffness
        spin_stiffness[num_temps + t_idx] = np.sqrt(
            stiff_h_err**2 + 
            ((N**2/T) * stiff_i2_err)**2 + 
            ((N**2/T) * 2 * stiff_i_avg * stiff_i_err)**2
        )

        # Calculate Y²
        y_squared = (stiff_h - (N**2/T) * (stiff_i_squared - stiff_i**2))**2
        y2_avg, y2_err = JackknifeErrorFromFullList(y_squared, number_box, length_box)

        # Calculate fourth-order term
        fourth_order[t_idx] = -4 * stiffness + 3 * (
            stiff_h_avg - (N**2/T) * (y2_avg - stiffness**2)
        ) + 2 * (N**6/T**3) * stiff_i4_avg
        # Note: error calculation for fourth-order term is complex and omitted for brevity
        
        # Calculate vorticity
        vort_raw = data[:, 6]
        vort_avg, vort_err = JackknifeErrorFromFullList(vort_raw, number_box, length_box)
        
        vorticity[t_idx] = vort_avg / N**2
        vorticity[num_temps + t_idx] = vort_err / N**2
    
    # Save all thermodynamic data
    np.savetxt(f'{where_to_save}{output_dir}/thermo_output.data',
               np.column_stack([energy, specific_heat, order_cumulant,
                               order_param, order_param_alt, susceptibility,
                               susceptibility_alt, binder_cumulant, correlation_length,
                               spin_stiffness, fourth_order, vorticity]))

    print()
    print('Done with Scaling Analysis')
    print()

    # Optional: Calculate autocorrelation functions (disabled by default)
    if False:
        # Length of autocorrelation analysis
        k_max = length_box

        for t_idx in range(num_temps):
            temp = temperatures[t_idx]
            filename = f'{where_to_save}{name_dir}/outputatT={int(temp*factor_print):05d}.data'
            data = np.loadtxt(filename)
            
            # Autocorrelation for energy
            energy_data = data[:, 0] / N**2
            auto_energy, auto_energy_err = calculate_autocorrelation_simple(energy_data, k_max)
            
            # Autocorrelation for magnetization
            mag_data = np.abs(data[:, 1] + 1j * data[:, 2]) / N**2
            auto_mag, auto_mag_err = calculate_autocorrelation_simple(mag_data, k_max)
            
            # Combine results
            auto_results = np.column_stack([
                np.concatenate([auto_energy, auto_energy_err]),
                np.concatenate([auto_mag, auto_mag_err])
            ])
            
            # Save autocorrelation results
            np.savetxt(
                f'{where_to_save}{output_dir}/Autocorr_outputatT={int(temp*factor_print):05d}.data',
                auto_results
            )
            
            # Copy configurations to output directory
            config_data = np.loadtxt(
                f'{where_to_save}{name_dir}/configatT={int(temp*factor_print):05d}.data'
            )
            np.savetxt(
                f'{where_to_save}{output_dir}/configatT={int(temp*factor_print):05d}.data',
                config_data
            )

        print()
        print('Done with Autocorrelation Analysis')
        print()


if __name__ == '__main__':
    main()
