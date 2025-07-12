import numpy as np
from scipy.stats import hypergeom, binom
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import FuncFormatter

def probability_of_detecting_defects(prod_rate, defective_rate, average_check_time=6.03, num_of_cores=6, total_time=3600, fast=False):
    """
    Calculate the probability of detecting defects in a batch given production and defective rates.
    
    Args:
        prod_rate: Production rate (items/second)
        defective_rate: Defective rate (%)
        average_check_time: Average time to check one item (seconds)
        num_of_cores: Number of parallel processing cores
        total_time: Total time window for analysis (seconds)
    
    Returns:
        Probability of detecting defects given that defects are present.

    Explanation:
        the probability of detecting 1 or more defects given that 1 or more defects are present.
        this is given by sum of the hypergeometric distributions of each amount of possible defects multiplied by the probability of that amount of defects occurring. 
    """
    
    # Calculate check rate
    check_rate = num_of_cores / average_check_time

    # If we can check everything we check everything
    if check_rate >= prod_rate:
        return 1.0
    
    # Calculate total items produced in the given time
    n_total = math.ceil(prod_rate * total_time)
    
    
    # Calculate number of items checked
    n_checked = min(math.floor(check_rate * total_time), n_total)
    


    if fast: # to speed up calculations, range is limited to statistically significant range
        n_expected_defective = defective_rate * n_total
        std_dev = math.sqrt(n_total * defective_rate * (1 - defective_rate)) #standard deviation of the binomial distribution
        min_defects = max(1, int(n_expected_defective - 3 * std_dev)) #max is one because we there is at least one defect detected since thiis is a conditional probability
        max_defects = min(n_total, int(n_expected_defective + 3 * std_dev))
        defect_range = np.arange(min_defects, max_defects + 1)
    else:
        defect_range = np.arange(1, n_total + 1)  # Full range of defects of interest

    # Calculate probabilities of faults and detection
    P_faults = binom.pmf(defect_range, n_total, defective_rate)
    
    # Vectorized calculation of detection probabilities
    P_detection = 1 - hypergeom.pmf(0, n_total, defect_range, n_checked)
    
    # Overall probability of detection (weighted sum) AND conditional probability forula applied
    # P(A|B) = P(A and B) / P(B)
    prob_detect = np.sum(P_faults * P_detection) / (1 - binom.pmf(0, n_total, defective_rate))
    
    return prob_detect


def plot_detection_heatmap(prod_rates, defective_rates, average_check_time=6.03, num_of_cores=6, 
                          total_time=3600, fast=True, show_contours=True, figsize=(12, 8)):
    """
    Create a 2D heatmap showing detection probability for given production rates and defective rates.
    
    Args:
        prod_rates: List or array of production rates (items/second)
        defective_rates: List or array of defective rates (proportion, e.g., 0.001 = 0.1%)
        average_check_time: Average time to check one item (seconds)
        num_of_cores: Number of parallel processing cores
        total_time: Total time window for analysis (seconds)
        fast: Use fast calculation method
        show_contours: Whether to overlay contour lines
        figsize: Figure size tuple
        
    Returns:
        prob_matrix: 2D array of detection probabilities
    """
    
    # Convert to numpy arrays if needed
    prod_rates = np.array(prod_rates)
    defective_rates = np.array(defective_rates)
    
    # Create meshgrid
    P, D = np.meshgrid(prod_rates, defective_rates)
    
    # Initialize probability matrix
    prob_matrix = np.zeros_like(P)
    
    print(f"Calculating detection probabilities for {len(prod_rates)} x {len(defective_rates)} grid...")
    
    # Calculate detection probabilities
    total_calculations = len(prod_rates) * len(defective_rates)
    calculation_count = 0
    
    for i, def_rate in enumerate(defective_rates):
        for j, prod_rate in enumerate(prod_rates):
            prob = probability_of_detecting_defects(
                prod_rate=prod_rate,
                defective_rate=def_rate,
                average_check_time=average_check_time,
                num_of_cores=num_of_cores,
                total_time=total_time,
                fast=fast
            )
            prob_matrix[i, j] = prob
            calculation_count += 1
            
            # Progress indicator
            if calculation_count % 50 == 0:
                print(f"Progress: {calculation_count}/{total_calculations} ({100*calculation_count/total_calculations:.1f}%)")
    
    # Create the plot
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Create heatmap
    # im = plt.imshow(
    #     prob_matrix,
    #     extent=[prod_rates.min(), prod_rates.max(), defective_rates.min(), defective_rates.max()],
    #     aspect='auto', origin='lower', cmap='viridis', interpolation='bilinear'
    # )

    X, Y = np.meshgrid(prod_rates, defective_rates)
    im = plt.pcolormesh(X, Y, prob_matrix, cmap='viridis', shading='auto')

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Detection Probability', fontsize=12)

    # Format colorbar ticks as percentages

    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x*100:.0f}%"))

    # Add contours if requested
    if show_contours:
        contour_levels = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        CS = plt.contour(P, D, prob_matrix, levels=contour_levels, colors='black', alpha=0.8, linewidths=1.5)
        plt.clabel(CS, inline=True, fontsize=10, fmt=lambda x: f"{x*100:.0f}% confidence")

    # Formatting
    plt.xlabel('Production Rate (items/second)', fontsize=12)
    plt.ylabel('Defective Rate', fontsize=12)
    plt.title('Detection Probability Heatmap', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Set x and y ticks to the specified values and label them as is
    tick_step = 20 
    x_ticks = prod_rates[::tick_step]
    y_ticks = defective_rates[::tick_step]
    if x_ticks[-1] != prod_rates[-1]:
        x_ticks = np.append(x_ticks, prod_rates[-1])
    if y_ticks[-1] != defective_rates[-1]:
        y_ticks = np.append(y_ticks, defective_rates[-1])

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{x:.2f}" for x in x_ticks], rotation=45, ha='right')
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y*100:.3f}%" for y in y_ticks])

    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n=== HEATMAP SUMMARY ===")
    print(f"Production Rate Range: {prod_rates.min():.1f} - {prod_rates.max():.1f} items/second")
    print(f"Defective Rate Range: {defective_rates.min():.4f} - {defective_rates.max():.4f} ({defective_rates.min()*100:.2f}% - {defective_rates.max()*100:.2f}%)")
    print(f"Detection Probability Range: {prob_matrix.min():.4f} - {prob_matrix.max():.4f}")
    print(f"Mean Detection Probability: {prob_matrix.mean():.4f}")
    
    return prob_matrix


def plot_detection_3d_surface(prod_rates, defective_rates, average_check_time=6.03, num_of_cores=6, 
                             total_time=3600, fast=True, show_contours=True, figsize=(14, 10)):
    """
    Create a 3D surface plot showing detection probability for given production rates and defective rates.
    
    Args:
        prod_rates: List or array of production rates (items/second)
        defective_rates: List or array of defective rates (proportion, e.g., 0.001 = 0.1%)
        average_check_time: Average time to check one item (seconds)
        num_of_cores: Number of parallel processing cores
        total_time: Total time window for analysis (seconds)
        fast: Use fast calculation method
        show_contours: Whether to show contour projections
        figsize: Figure size tuple
        
    Returns:
        prob_matrix: 2D array of detection probabilities
    """
    
    # Convert to numpy arrays if needed
    prod_rates = np.array(prod_rates)
    defective_rates = np.array(defective_rates)
    
    # Create meshgrid
    P, D = np.meshgrid(prod_rates, defective_rates)
    
    # Initialize probability matrix
    prob_matrix = np.zeros_like(P)
    
    print(f"Calculating detection probabilities for {len(prod_rates)} x {len(defective_rates)} 3D surface...")
    
    # Calculate detection probabilities
    total_calculations = len(prod_rates) * len(defective_rates)
    calculation_count = 0
    
    for i, def_rate in enumerate(defective_rates):
        for j, prod_rate in enumerate(prod_rates):
            prob = probability_of_detecting_defects(
                prod_rate=prod_rate,
                defective_rate=def_rate,
                average_check_time=average_check_time,
                num_of_cores=num_of_cores,
                total_time=total_time,
                fast=fast
            )
            prob_matrix[i, j] = prob
            calculation_count += 1
            
            # Progress indicator
            if calculation_count % 50 == 0:
                print(f"Progress: {calculation_count}/{total_calculations} ({100*calculation_count/total_calculations:.1f}%)")
    
    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create 3D surface
    surf = ax.plot_surface(P, D*100, prob_matrix, cmap='viridis', alpha=0.8, 
                          linewidth=0, antialiased=True)
    
    # Add contour projections if requested
    if show_contours:
        # Project contours on the bottom (z=0)
        contour_levels = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        ax.contour(P, D*100, prob_matrix, zdir='z', offset=0, 
                  levels=contour_levels, cmap='viridis', alpha=0.6)
        
        # Project contours on back walls
        ax.contour(P, D*100, prob_matrix, zdir='x', offset=prod_rates.min(), 
                  levels=contour_levels, cmap='viridis', alpha=0.3)
        ax.contour(P, D*100, prob_matrix, zdir='y', offset=defective_rates.max()*100, 
                  levels=contour_levels, cmap='viridis', alpha=0.3)
    
    # Formatting
    ax.set_xlabel('Production Rate (items/second)', fontsize=12)
    ax.set_ylabel('Defective Rate (%)', fontsize=12)
    ax.set_zlabel('Detection Probability', fontsize=12)
    ax.set_title('3D Detection Probability Surface', fontsize=14)
    
    # Set z-axis limits for better visualization
    ax.set_zlim(0, 1)
    
    # Add colorbar
    cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Detection Probability', fontsize=12)
    
    # Improve viewing angle
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n=== 3D SURFACE SUMMARY ===")
    print(f"Production Rate Range: {prod_rates.min():.1f} - {prod_rates.max():.1f} items/second")
    print(f"Defective Rate Range: {defective_rates.min():.4f} - {defective_rates.max():.4f} ({defective_rates.min()*100:.2f}% - {defective_rates.max()*100:.2f}%)")
    print(f"Detection Probability Range: {prob_matrix.min():.4f} - {prob_matrix.max():.4f}")
    print(f"Mean Detection Probability: {prob_matrix.mean():.4f}")
    
    return prob_matrix



if __name__ == "__main__":
    # parameter ranges

    
    # print("=== CREATING HEATMAP ===")
    # production_rates = np.linspace(0.5, 30.0, 50)  
    # defective_rates = np.linspace(0.0001, 0.002, 50)  
    # heatmap_matrix = plot_detection_heatmap(
    #     prod_rates=production_rates,
    #     defective_rates=defective_rates,
    #     average_check_time=6.03,
    #     num_of_cores=6,
    #     total_time=3600,
    #     fast=True,
    #     show_contours=True
    # )

    # print("=== CREATING HEATMAP ===")
    # production_rates = np.linspace(0.5, 30.0, 50)  
    # defective_rates = np.linspace(0.0001, 0.005, 50)  
    # heatmap_matrix = plot_detection_heatmap(
    #     prod_rates=production_rates,
    #     defective_rates=defective_rates,
    #     average_check_time=6.03,
    #     num_of_cores=6,
    #     total_time=3600/4,
    #     fast=True,
    #     show_contours=True
    # )

    # print("=== CREATING HEATMAP ===")
    # production_rates = np.linspace(0.5, 30.0, 50)  
    # defective_rates = np.linspace(0.0001, 0.03, 50)  
    # heatmap_matrix = plot_detection_heatmap(
    #     prod_rates=production_rates,
    #     defective_rates=defective_rates,
    #     average_check_time=2.5,
    #     num_of_cores=6,
    #     total_time=60,
    #     fast=True,
    #     show_contours=True
    # )

    print("=== CREATING HEATMAP ===")
    production_rates = np.linspace(0.5, 30.0, 100)  
    defective_rates = np.linspace(0.0001, 0.005, 100)  
    heatmap_matrix = plot_detection_heatmap(
        prod_rates=production_rates,
        defective_rates=defective_rates,
        average_check_time=2.5,
        num_of_cores=6,
        total_time=60*5, # 5 minutes
        fast=True,
        show_contours=True
    )