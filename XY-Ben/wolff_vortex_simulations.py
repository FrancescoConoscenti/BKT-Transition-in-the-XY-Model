import numpy as np
import matplotlib.pyplot as plt
from XY_model import XYSystem
import os
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
from matplotlib.patches import Circle, Rectangle
from scipy.interpolate import RectBivariateSpline
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.optimize import linear_sum_assignment
from matplotlib.lines import Line2D
import time

def calculate_vorticity(system):
    """Calculate vorticity for each plaquette in the system."""
    L = system.width
    vorticity = np.zeros((L, L))
    
    # Get the spin configuration as a 2D array of angles
    spins_2d = system.list2matrix(system.spin_config)
    
    # Calculate vorticity for each plaquette
    for i in range(L):
        for j in range(L):
            # Get angles of the 4 corners of each plaquette (clockwise or counterclockwise)
            theta1 = spins_2d[i, j]
            theta2 = spins_2d[i, (j+1)%L]
            theta3 = spins_2d[(i+1)%L, (j+1)%L]
            theta4 = spins_2d[(i+1)%L, j]
            
            # Calculate phase differences (considering periodic boundaries)
            dtheta1 = (theta2 - theta1 + np.pi) % (2*np.pi) - np.pi
            dtheta2 = (theta3 - theta2 + np.pi) % (2*np.pi) - np.pi
            dtheta3 = (theta4 - theta3 + np.pi) % (2*np.pi) - np.pi
            dtheta4 = (theta1 - theta4 + np.pi) % (2*np.pi) - np.pi
            
            # Sum of phase differences around plaquette
            total_phase = dtheta1 + dtheta2 + dtheta3 + dtheta4
            
            # Convert to vorticity (-1, 0, 1)
            vorticity[i, j] = np.round(total_phase / (2*np.pi))
    
    return vorticity

def filter_vortices(vorticity, min_distance=3.0, remove_boundary=True, boundary_size=3):
    """Filter out vortices that are too close to each other or likely noise.
    
    Parameters:
    -----------
    vorticity : array
        The vorticity field
    min_distance : float
        Minimum distance between vortices to be considered separate
    remove_boundary : bool
        Whether to remove vortices near the boundary
    boundary_size : int
        Size of boundary region to exclude vortices from
    """
    L = vorticity.shape[0]
    
    # Identify vortices and antivortices
    vortex_y, vortex_x = np.where(vorticity > 0.5)
    antivortex_y, antivortex_x = np.where(vorticity < -0.5)
    
    # Function to remove vortices near the boundary
    def filter_boundary(y_coords, x_coords):
        if not remove_boundary:
            return y_coords, x_coords
            
        mask = ((y_coords >= boundary_size) & 
                (y_coords < L - boundary_size) & 
                (x_coords >= boundary_size) & 
                (x_coords < L - boundary_size))
                
        return y_coords[mask], x_coords[mask]
    
    # Remove boundary vortices if requested
    vortex_y, vortex_x = filter_boundary(vortex_y, vortex_x)
    antivortex_y, antivortex_x = filter_boundary(antivortex_y, antivortex_x)
    
    # Function to cluster nearby vortices
    def cluster_points(points_y, points_x, threshold):
        if len(points_y) == 0:
            return [], []
            
        # Create points array
        points = np.column_stack((points_y, points_x))
        
        # Calculate pairwise distances
        periodic_dist = np.zeros((len(points), len(points)))
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                y1, x1 = points[i]
                y2, x2 = points[j]
                
                # Account for periodic boundaries
                dx = min(abs(x1 - x2), L - abs(x1 - x2))
                dy = min(abs(y1 - y2), L - abs(y1 - y2))
                dist = np.sqrt(dx**2 + dy**2)
                periodic_dist[i, j] = periodic_dist[j, i] = dist
        
        # Flatten the distance matrix for use with scipy
        condensed_dist = []
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                condensed_dist.append(periodic_dist[i, j])
        
        if len(condensed_dist) == 0:  # Only one point
            return points_y, points_x
            
        # Perform hierarchical clustering
        Z = linkage(np.array(condensed_dist), method='single')
        clusters = fcluster(Z, threshold, criterion='distance')
        
        # Find cluster centers
        unique_clusters = np.unique(clusters)
        cluster_y = []
        cluster_x = []
        
        for cluster_id in unique_clusters:
            mask = clusters == cluster_id
            cluster_indices = np.where(mask)[0]
            
            # Calculate center of the cluster
            cy = np.mean([points_y[i] for i in cluster_indices])
            cx = np.mean([points_x[i] for i in cluster_indices])
            
            cluster_y.append(cy)
            cluster_x.append(cx)
        
        return np.array(cluster_y), np.array(cluster_x)
    
    # Cluster nearby vortices
    clustered_vortex_y, clustered_vortex_x = cluster_points(vortex_y, vortex_x, min_distance)
    clustered_antivortex_y, clustered_antivortex_x = cluster_points(antivortex_y, antivortex_x, min_distance)
    
    # Return filtered vortex locations
    return (clustered_vortex_y, clustered_vortex_x), (clustered_antivortex_y, clustered_antivortex_x)

def identify_vortex_pairs(vortex_positions, antivortex_positions, system_width, max_pair_distance=7.0):
    """Identify vortex-antivortex pairs using a greedy nearest-neighbor approach."""
    vortex_y, vortex_x = vortex_positions
    antivortex_y, antivortex_x = antivortex_positions
    
    # If no vortices or antivortices, return empty list
    if len(vortex_y) == 0 or len(antivortex_y) == 0:
        return []
    
    # Create arrays of positions
    vortices = [(y, x) for y, x in zip(vortex_y, vortex_x)]
    antivortices = [(y, x) for y, x in zip(antivortex_y, antivortex_x)]
    
    # Track which vortices and antivortices are paired
    paired_vortices = set()
    paired_antivortices = set()
    pairs = []
    
    # Function to calculate distance with periodic boundaries
    def periodic_distance(pos1, pos2):
        y1, x1 = pos1
        y2, x2 = pos2
        dx = min(abs(x1 - x2), system_width - abs(x1 - x2))
        dy = min(abs(y1 - y2), system_width - abs(y1 - y2))
        return np.sqrt(dx**2 + dy**2)
    
    # Greedy approach: for each vortex, find the closest unpaired antivortex
    for v_idx, vortex in enumerate(vortices):
        if vortex in paired_vortices:
            continue
            
        # Find distances to all unpaired antivortices
        distances = []
        available_indices = []
        
        for av_idx, antivortex in enumerate(antivortices):
            if antivortex in paired_antivortices:
                continue
                
            dist = periodic_distance(vortex, antivortex)
            if dist <= max_pair_distance:
                distances.append(dist)
                available_indices.append(av_idx)
        
        # If there are available antivortices within range, pair with the closest
        if distances:
            closest_idx = available_indices[np.argmin(distances)]
            closest_dist = min(distances)
            
            # Add the pair
            pairs.append((vortex, antivortices[closest_idx], closest_dist))
            paired_vortices.add(vortex)
            paired_antivortices.add(antivortices[closest_idx])
    
    return pairs

def get_pairing_distance(temperature):
    """Get appropriate pairing distance based on temperature.
    
    The pairing distance should reflect the physical correlation length,
    which diverges exponentially as T→T_KT from above and 
    follows power-law decay below T_KT.
    """
    # BKT transition temperature
    T_KT = 0.89
    
    if temperature < 0.7:  # Deep in ordered phase
        return 10.0  # Large pairing distance
    elif temperature < T_KT:  # Near but below transition
        return 8.0
    elif temperature < 1.0:  # Just above transition
        return 6.0
    elif temperature < 2.0:  # Well above transition
        return 4.0
    else:  # Deep in disordered phase
        return 3.0

def plot_vortices(system, temperature, save_path):
    """Plot the spin configuration with vortices highlighted."""
    # Calculate vorticity
    vorticity = calculate_vorticity(system)
    
    # Apply small smoothing to the vorticity field
    smoothed_vorticity = gaussian_filter(vorticity, sigma=0.5)
    
    # Filter vortices to keep only significant ones
    # Use stronger filtering for high temperatures to reduce noise
    filter_distance = 2.5 if temperature < 0.9 else 3.0
    remove_boundary = temperature > 2.0  # Only remove boundary vortices at high T
    
    (vortex_y, vortex_x), (antivortex_y, antivortex_x) = filter_vortices(
        vorticity, 
        min_distance=filter_distance,
        remove_boundary=remove_boundary
    )
    
    # Identify vortex-antivortex pairs using temperature-dependent criteria
    max_pair_dist = get_pairing_distance(temperature)
        
    vortex_pairs = identify_vortex_pairs(
        (vortex_y, vortex_x), 
        (antivortex_y, antivortex_x), 
        system.width,
        max_pair_distance=max_pair_dist
    )
    
    # Calculate pairing statistics
    num_vortices = len(vortex_y)
    num_antivortices = len(antivortex_y)
    num_pairs = len(vortex_pairs)
    pairing_ratio = num_pairs / max(1, (num_vortices + num_antivortices) / 2)
    
    # Track which vortices and antivortices are paired
    paired_vortices = set()
    paired_antivortices = set()
    
    for (vy, vx), (avy, avx), _ in vortex_pairs:
        paired_vortices.add((vy, vx))
        paired_antivortices.add((avy, avx))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot the spin configuration
    config_matrix = system.list2matrix(system.spin_config)
    X, Y = np.meshgrid(np.arange(0, system.width), np.arange(0, system.width))
    U = np.cos(config_matrix)
    V = np.sin(config_matrix)
    
    # Create vortex mask with larger influence radius
    vortex_mask = np.zeros_like(vorticity, dtype=float)
    antivortex_mask = np.zeros_like(vorticity, dtype=float)
    
    # Use larger radius for better visualization
    vortex_radius = 3.5
    
    # Create distance-based shading for vortices
    for y, x in zip(vortex_y, vortex_x):
        for i in range(system.width):
            for j in range(system.width):
                # Calculate minimum distance (accounting for periodic boundaries)
                dx = min(abs(x - j), system.width - abs(x - j))
                dy = min(abs(y - i), system.width - abs(y - i))
                dist = np.sqrt(dx**2 + dy**2)
                
                # Shade with intensity decreasing with distance - stronger falloff
                if dist < vortex_radius:
                    vortex_mask[i, j] = max(vortex_mask[i, j], 1.0 * np.exp(-dist**2/(vortex_radius)))
    
    # Same for antivortices
    for y, x in zip(antivortex_y, antivortex_x):
        for i in range(system.width):
            for j in range(system.width):
                dx = min(abs(x - j), system.width - abs(x - j))
                dy = min(abs(y - i), system.width - abs(y - i))
                dist = np.sqrt(dx**2 + dy**2)
                if dist < vortex_radius:
                    antivortex_mask[i, j] = max(antivortex_mask[i, j], 1.0 * np.exp(-dist**2/(vortex_radius)))
    
    # Combined mask (positive for vortices, negative for antivortices)
    combined_mask = vortex_mask - antivortex_mask
    
    # Plot the spin configuration with shaded vortices (first subplot)
    im = ax1.imshow(combined_mask, origin='lower', cmap='coolwarm', 
                   alpha=0.6, vmin=-1, vmax=1, interpolation='bilinear')
    
    # Sample the spin field to reduce arrow density - makes plot clearer
    sample_rate = 3  # Display every 3rd arrow in each direction
    X_sampled = X[::sample_rate, ::sample_rate]
    Y_sampled = Y[::sample_rate, ::sample_rate]
    U_sampled = U[::sample_rate, ::sample_rate]
    V_sampled = V[::sample_rate, ::sample_rate]
    
    # Add arrows for spin directions - smaller and less dense for better readability
    ax1.quiver(X_sampled, Y_sampled, U_sampled, V_sampled, 
               units='width', scale=42, color='black', width=0.008, 
               headwidth=4, headlength=4)
    
    ax1.set_title(f'Spin Configuration with Vortices at T={temperature}', fontsize=14)
    ax1.set_aspect('equal')
    ax1.set_xlim(-0.5, system.width-0.5)
    ax1.set_ylim(-0.5, system.width-0.5)
    
    # Add dots for vortex centers (red for vortices, blue for antivortices) - larger dots
    ax1.scatter(vortex_x, vortex_y, color='red', s=100, marker='o', label='Vortex', zorder=3, edgecolor='black')
    ax1.scatter(antivortex_x, antivortex_y, color='blue', s=100, marker='o', label='Antivortex', zorder=3, edgecolor='black')
    
    # Function to check if a pair spans across the periodic boundary
    def is_boundary_crossing(pos1, pos2, system_width):
        y1, x1 = pos1
        y2, x2 = pos2
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        # If the direct distance is more than half the system size, it's crossing the boundary
        return dx > system_width/2 or dy > system_width/2
    
    # Draw cyan lines connecting vortex-antivortex pairs in the first plot
    boundary_pairs = []  # Track pairs that cross boundaries
    normal_pairs = []    # Track normal pairs
    
    if vortex_pairs:
        for (vy, vx), (avy, avx), dist in vortex_pairs:
            # Check if this pair crosses the periodic boundary
            if is_boundary_crossing((vy, vx), (avy, avx), system.width):
                # Use dotted line for boundary-crossing pairs
                line = ax1.plot([vx, avx], [vy, avy], 'c--', linewidth=2.5, alpha=0.9, zorder=2, dashes=(4, 3))[0]
                boundary_pairs.append(line)
            else:
                # Use solid line for normal pairs
                line = ax1.plot([vx, avx], [vy, avy], 'c-', linewidth=3.0, alpha=0.9, zorder=2)[0]
                normal_pairs.append(line)
    
    # Add magenta square boxes around unpaired vortices and antivortices in the first plot
    square_size = 1.5  # Size of the square box
    unpaired_vortex_boxes = []
    unpaired_antivortex_boxes = []
    
    # Add boxes around unpaired vortices
    for i, (y, x) in enumerate(zip(vortex_y, vortex_x)):
        if (y, x) not in paired_vortices:
            rect = Rectangle((x-square_size/2, y-square_size/2), square_size, square_size, 
                          linewidth=2, edgecolor='magenta', facecolor='none', zorder=4)
            ax1.add_patch(rect)
            unpaired_vortex_boxes.append(rect)
    
    # Add boxes around unpaired antivortices
    for i, (y, x) in enumerate(zip(antivortex_y, antivortex_x)):
        if (y, x) not in paired_antivortices:
            rect = Rectangle((x-square_size/2, y-square_size/2), square_size, square_size, 
                          linewidth=2, edgecolor='magenta', facecolor='none', zorder=4)
            ax1.add_patch(rect)
            unpaired_antivortex_boxes.append(rect)
    
    # Create legend entries
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markeredgecolor='black', 
              markersize=10, label='Vortex'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markeredgecolor='black', 
              markersize=10, label='Antivortex'),
    ]
    
    # Add lines to legend
    if normal_pairs:
        legend_elements.append(Line2D([0], [0], color='cyan', lw=3, label='Vortex-Antivortex Pair'))
    if boundary_pairs:
        legend_elements.append(Line2D([0], [0], color='cyan', lw=2.5, linestyle='--', dashes=(4, 3), 
                               label='Pair (across boundary)'))
    
    # Add magenta box to legend if unpaired vortices exist
    if unpaired_vortex_boxes or unpaired_antivortex_boxes:
        legend_elements.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='none', 
                               markeredgecolor='magenta', markersize=10, label='Unpaired (Free)'))
    
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Add colorbar for vortex regions
    cbar1 = plt.colorbar(im, ax=ax1)
    cbar1.set_ticks([-1, 0, 1])
    cbar1.set_ticklabels(['Antivortex', 'No vortex', 'Vortex'])
    
    # Second subplot: Vortex field visualization with streamlines
    # Use a very light background
    ax2.imshow(np.zeros((system.width, system.width)), cmap='gray', 
               alpha=0.05, vmin=0, vmax=1, extent=(-0.5, system.width-0.5, -0.5, system.width-0.5))
    
    # Draw larger vortex and antivortex regions
    for y, x in zip(vortex_y, vortex_x):
        circle = Circle((x, y), radius=vortex_radius, alpha=0.3, fc='red', ec='darkred')
        ax2.add_patch(circle)
    
    # Add shaded regions for antivortices (blue)
    for y, x in zip(antivortex_y, antivortex_x):
        circle = Circle((x, y), radius=vortex_radius, alpha=0.3, fc='blue', ec='darkblue')
        ax2.add_patch(circle)
    
    # Add the center dots - larger and with outline
    ax2.scatter(vortex_x, vortex_y, color='red', s=120, marker='o', label='Vortex', zorder=3, edgecolor='black')
    ax2.scatter(antivortex_x, antivortex_y, color='blue', s=120, marker='o', label='Antivortex', zorder=3, edgecolor='black')
    
    # Draw cyan lines connecting vortex-antivortex pairs with varying thickness based on distance
    for (vy, vx), (avy, avx), dist in vortex_pairs:
        # Use thickness that varies with distance - thicker for closer pairs
        linewidth = max(2.5, 4.0 - dist/3.0)  # Even thicker lines
        
        # Check if this pair crosses the periodic boundary
        if is_boundary_crossing((vy, vx), (avy, avx), system.width):
            # Use dotted line for boundary-crossing pairs
            ax2.plot([vx, avx], [vy, avy], 'c--', linewidth=linewidth, alpha=1.0, zorder=2, dashes=(4, 3))
        else:
            # Use solid line for normal pairs
            ax2.plot([vx, avx], [vy, avy], 'c-', linewidth=linewidth, alpha=1.0, zorder=2)
    
    # Add magenta square boxes around unpaired vortices and antivortices in second plot
    # Add boxes around unpaired vortices
    for i, (y, x) in enumerate(zip(vortex_y, vortex_x)):
        if (y, x) not in paired_vortices:
            rect = Rectangle((x-square_size/2, y-square_size/2), square_size, square_size, 
                          linewidth=2, edgecolor='magenta', facecolor='none', zorder=4)
            ax2.add_patch(rect)
    
    # Add boxes around unpaired antivortices
    for i, (y, x) in enumerate(zip(antivortex_y, antivortex_x)):
        if (y, x) not in paired_antivortices:
            rect = Rectangle((x-square_size/2, y-square_size/2), square_size, square_size, 
                          linewidth=2, edgecolor='magenta', facecolor='none', zorder=4)
            ax2.add_patch(rect)
    
    # Show field lines to represent the spin pattern - more refined streamlines
    streamplot_x = np.linspace(0, system.width-1, system.width*3)  # Higher resolution
    streamplot_y = np.linspace(0, system.width-1, system.width*3)
    sX, sY = np.meshgrid(streamplot_x, streamplot_y)
    
    # Interpolate U and V for streamplot
    u_interp = RectBivariateSpline(np.arange(system.width), np.arange(system.width), U)
    v_interp = RectBivariateSpline(np.arange(system.width), np.arange(system.width), V)
    
    # Apply periodic boundary conditions
    sU = u_interp.ev(sY % system.width, sX % system.width)
    sV = v_interp.ev(sY % system.width, sX % system.width)
    
    # Better streamplot for clearer visualization
    ax2.streamplot(sX, sY, sU, sV, density=1.8, color='black', linewidth=1.0, arrowsize=1.2)
    
    # Count unbound vortices
    num_unbound_vortices = num_vortices + num_antivortices - 2 * num_pairs
    
    # Show pairing statistics in the title
    paired_pct = 100 * num_pairs / max(1, (num_vortices + num_antivortices)/2)
    ax2.set_title(f'Vortex and Antivortex Centers at T={temperature}\n' +
                 f'{num_pairs} pairs ({paired_pct:.1f}% paired), {num_unbound_vortices} free', fontsize=14)
    ax2.set_xlim(-0.5, system.width-0.5)
    ax2.set_ylim(-0.5, system.width-0.5)
    ax2.set_aspect('equal')
    
    # Create legend for second plot
    legend_elements2 = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markeredgecolor='black', 
              markersize=10, label='Vortex'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markeredgecolor='black', 
              markersize=10, label='Antivortex'),
    ]
    
    # Add lines to legend
    if normal_pairs:
        legend_elements2.append(Line2D([0], [0], color='cyan', lw=3, label='Vortex-Antivortex Pair'))
    if boundary_pairs:
        legend_elements2.append(Line2D([0], [0], color='cyan', lw=2.5, linestyle='--', dashes=(4, 3), 
                               label='Pair (across boundary)'))
    
    # Add magenta box to legend if unpaired vortices exist
    if unpaired_vortex_boxes or unpaired_antivortex_boxes:
        legend_elements2.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='none', 
                               markeredgecolor='magenta', markersize=10, label='Unpaired (Free)'))
    
    ax2.legend(handles=legend_elements2, loc='upper right', fontsize=12)
    
    # Add info about temperature regime
    T_KT = 0.89  # BKT transition temperature
    
    temp_info = "Low-T (ordered)"
    if abs(temperature - T_KT) < 0.05:
        temp_info = "Near BKT transition (T_KT ≈ 0.89)"
    elif temperature > T_KT:
        temp_info = "High-T (disordered)"
    
    plt.figtext(0.5, 0.01, f'Algorithm: Wolff, System size: {system.width}x{system.width}, {temp_info}', 
                ha='center', fontsize=12)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    
    # Close the plot to free memory
    plt.close(fig)
    
    return {
        'num_vortices': num_vortices,
        'num_antivortices': num_antivortices,
        'num_pairs': num_pairs,
        'pairing_ratio': pairing_ratio,
        'num_unbound': num_unbound_vortices
    }

def initialize_ordered_system(width, temperature):
    """Initialize a system with ordered spins for low temperatures."""
    if temperature < 1.0:
        # Create system with all spins aligned (ordered state)
        system = XYSystem(temperature=temperature, width=width, algorithm="wolff")
        # Set all spins to same angle (fully ordered state)
        system.spin_config = np.zeros(system.num_spins)
        print(f"Initialized ordered state for T={temperature}")
    else:
        # For high temperatures, random initialization is fine (default)
        system = XYSystem(temperature=temperature, width=width, algorithm="wolff")
        print(f"Initialized random state for T={temperature}")
    
    return system

def get_equilibration_time(temperature, base_sweeps=10000):
    """Get temperature-dependent equilibration time.
    
    The equilibration time should be much longer near the transition.
    """
    # BKT transition temperature
    T_KT = 0.89
    
    # Increase equilibration near the transition, especially just above it
    if abs(temperature - T_KT) < 0.05:  # Very near transition
        return base_sweeps * 5  # Much longer equilibration
    elif temperature > T_KT and temperature < 1.0:  # Just above transition
        return base_sweeps * 3
    elif temperature < T_KT and temperature > 0.8:  # Just below transition
        return base_sweeps * 2
    else:  # Far from transition
        return base_sweeps

def run_simulation(width=50, temperature=1.0, base_equilibration_sweeps=20000, 
                   measurement_sweeps=500, save_path=None):
    """Run a simulation at the given temperature and save the vortex plot."""
    
    print(f"\n{'='*60}")
    print(f"STARTING SIMULATION FOR T = {temperature:.4f}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Initialize the system (ordered for low T, random for high T)
    system = initialize_ordered_system(width, temperature)
    
    # Get temperature-dependent equilibration time
    actual_equil_sweeps = get_equilibration_time(temperature, base_equilibration_sweeps)
    
    # Equilibrate the system
    print(f"Equilibrating for {actual_equil_sweeps} sweeps...")
    system.equilibrate(max_nsweeps=actual_equil_sweeps, show=False)
    
    # Do additional sweeps for measurement
    print(f"Performing {measurement_sweeps} measurement sweeps...")
    for _ in range(measurement_sweeps):
        system.sweep()
    
    # Plot and save
    stats = {}
    if save_path:
        stats = plot_vortices(system, temperature, save_path)
        print(f"Vortex statistics for T={temperature}:")
        print(f"  Vortices: {stats['num_vortices']}, Antivortices: {stats['num_antivortices']}")
        print(f"  Paired: {stats['num_pairs']}, Pairing ratio: {stats['pairing_ratio']:.2f}")
        print(f"  Free vortices: {stats['num_unbound']}")
    
    # Print total execution time
    elapsed_time = time.time() - start_time
    print(f"Simulation for T={temperature} completed in {elapsed_time:.1f} seconds")
    
    return system, stats

def main():
    # Create output directory if it doesn't exist
    output_dir = "vortex_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # BKT transition temperature (approximate)
    T_KT = 0.89
    
    # Temperatures to simulate - focus on a tighter range around the BKT transition
    temperatures = [
        0.5,      # Deep in ordered phase
        0.7,      # Well below transition
        0.80,     # Below transition
        0.85,     # Just below transition
        0.87,     # Very close to transition (below)
        0.89,     # At the transition (T_KT)
        0.91,     # Very close to transition (above)
        0.93,     # Just above transition
        0.95,     # Above transition
        1.0       # Well above transition
    ]
    
    # Store statistics for comparison
    all_stats = {}
    
    # Run simulations with increased equilibration time
    for T in temperatures:
        save_path = os.path.join(output_dir, f"vortices_T{T:.2f}.png")
        
        # Use 100x100 system for high-resolution simulations
        _, stats = run_simulation(
            width=100,  # Large system size for better statistics
            temperature=T, 
            base_equilibration_sweeps=20000,  # Increased equilibration for larger system
            measurement_sweeps=1000,
            save_path=save_path
        )
        
        all_stats[T] = stats
    
    # Print summary of pairing ratios
    print("\nSummary of pairing ratios:")
    for T in sorted(all_stats.keys()):
        print(f"T={T:.2f}: Paired={all_stats[T]['num_pairs']}, " + 
              f"Free={all_stats[T]['num_unbound']}, " + 
              f"Pairing ratio={all_stats[T]['pairing_ratio']:.2f}")
    
    print("All simulations completed!")
    
    # Create a summary plot of statistics vs temperature with three subplots
    fig = plt.figure(figsize=(18, 6))
    
    temps = sorted(all_stats.keys())
    ratios = [all_stats[T]['pairing_ratio'] for T in temps]
    free_counts = [all_stats[T]['num_unbound'] for T in temps]
    pair_counts = [all_stats[T]['num_pairs'] for T in temps]
    
    # Plot 1: Pairing ratio
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(temps, ratios, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=T_KT, color='r', linestyle='--', label=f'T_KT = {T_KT}')
    ax1.set_xlabel('Temperature (T)', fontsize=12)
    ax1.set_ylabel('Pairing Ratio', fontsize=12)
    ax1.set_title('Vortex-Antivortex Pairing Ratio vs Temperature', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Free vortex count
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(temps, free_counts, 'ro-', linewidth=2, markersize=8)
    ax2.axvline(x=T_KT, color='r', linestyle='--', label=f'T_KT = {T_KT}')
    ax2.set_xlabel('Temperature (T)', fontsize=12)
    ax2.set_ylabel('Number of Free Vortices', fontsize=12)
    ax2.set_title('Free Vortex Count vs Temperature', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Number of vortex-antivortex pairs
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(temps, pair_counts, 'go-', linewidth=2, markersize=8)
    ax3.axvline(x=T_KT, color='r', linestyle='--', label=f'T_KT = {T_KT}')
    ax3.set_xlabel('Temperature (T)', fontsize=12)
    ax3.set_ylabel('Number of Vortex-Antivortex Pairs', fontsize=12)
    ax3.set_title('Bound Pair Count vs Temperature', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pairing_statistics.png"), dpi=200)
    plt.close()

if __name__ == "__main__":
    main() 