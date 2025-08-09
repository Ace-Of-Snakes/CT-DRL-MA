import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Import Container and constants from the separated modules
from simulation.terminal_components.Container import (
    Container, 
    MIN_ACCURACY_DAYS, 
    MAX_HOLDING_DAYS, 
    MIN_ACCURACY_PERCENT, 
    LATE_ACCURACY_PERCENT, 
    PEAK_UNCERTAINTY_DAY
)
from simulation.terminal_components.ContainerFactory import ContainerFactory


def plot_accuracy_function():
    """
    Plot the accuracy function used in Container estimation.
    Shows how prediction accuracy varies with container stay duration.
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Generate stay durations from 0 to MAX_HOLDING_DAYS
    stay_durations = np.arange(0, MAX_HOLDING_DAYS + 1)
    accuracies = []
    
    # Calculate accuracy for each stay duration
    for total_stay in stay_durations:
        # Simulate the accuracy calculation from Container class
        if total_stay <= MIN_ACCURACY_DAYS:
            accuracy = 1.0
        elif total_stay >= MAX_HOLDING_DAYS:
            accuracy = LATE_ACCURACY_PERCENT
        else:
            if total_stay <= PEAK_UNCERTAINTY_DAY:
                progress = (total_stay - MIN_ACCURACY_DAYS) / (PEAK_UNCERTAINTY_DAY - MIN_ACCURACY_DAYS)
                accuracy = 1.0 - (1.0 - MIN_ACCURACY_PERCENT) * progress
            else:
                progress = (total_stay - PEAK_UNCERTAINTY_DAY) / (MAX_HOLDING_DAYS - PEAK_UNCERTAINTY_DAY)
                accuracy = MIN_ACCURACY_PERCENT + (LATE_ACCURACY_PERCENT - MIN_ACCURACY_PERCENT) * progress
        
        accuracies.append(accuracy)
    
    # Plot 1: Accuracy vs Stay Duration
    ax1.plot(stay_durations, accuracies, 'b-', linewidth=2, label='Estimation Accuracy')
    ax1.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Perfect Accuracy')
    ax1.axhline(y=MIN_ACCURACY_PERCENT, color='r', linestyle='--', alpha=0.5, label=f'Minimum Accuracy ({MIN_ACCURACY_PERCENT:.0%})')
    ax1.axhline(y=LATE_ACCURACY_PERCENT, color='orange', linestyle='--', alpha=0.5, label=f'Late Period Accuracy ({LATE_ACCURACY_PERCENT:.0%})')
    ax1.axvline(x=MIN_ACCURACY_DAYS, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(x=PEAK_UNCERTAINTY_DAY, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(x=MAX_HOLDING_DAYS, color='gray', linestyle=':', alpha=0.5)
    
    # Add annotations
    ax1.annotate(f'Perfect accuracy\n(≤{MIN_ACCURACY_DAYS} days)', 
                xy=(MIN_ACCURACY_DAYS/2, 1.0), 
                xytext=(MIN_ACCURACY_DAYS/2, 0.9),
                ha='center',
                arrowprops=dict(arrowstyle='->', alpha=0.5))
    
    ax1.annotate(f'Peak uncertainty\n(day {PEAK_UNCERTAINTY_DAY})', 
                xy=(PEAK_UNCERTAINTY_DAY, MIN_ACCURACY_PERCENT), 
                xytext=(PEAK_UNCERTAINTY_DAY-20, MIN_ACCURACY_PERCENT+0.15),
                ha='center',
                arrowprops=dict(arrowstyle='->', alpha=0.5))
    
    ax1.annotate(f'Long-term storage\n(day {MAX_HOLDING_DAYS})', 
                xy=(MAX_HOLDING_DAYS, LATE_ACCURACY_PERCENT), 
                xytext=(MAX_HOLDING_DAYS-20, LATE_ACCURACY_PERCENT+0.1),
                ha='center',
                arrowprops=dict(arrowstyle='->', alpha=0.5))
    
    ax1.set_xlabel('Container Stay Duration (days)', fontsize=12)
    ax1.set_ylabel('Prediction Accuracy', fontsize=12)
    ax1.set_title('Container Departure Estimation Accuracy vs Stay Duration', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.set_xlim(0, MAX_HOLDING_DAYS)
    ax1.set_ylim(0, 1.1)
    
    # Plot 2: Expected Error Range vs Stay Duration
    ax2.fill_between(stay_durations, 0, 0, label='Error Range', alpha=0.3)
    
    error_ranges = []
    for i, total_stay in enumerate(stay_durations):
        accuracy = accuracies[i]
        # Calculate expected error range (simplified)
        max_error_days = max(7, total_stay * 0.3)
        std_dev = max_error_days * (1.0 - accuracy)
        # 95% confidence interval (approximately 2 standard deviations)
        error_range = 2 * std_dev
        error_ranges.append(error_range)
    
    ax2.fill_between(stay_durations, 
                     [-e for e in error_ranges], 
                     error_ranges, 
                     alpha=0.3, color='red', label='95% Confidence Interval')
    ax2.plot(stay_durations, error_ranges, 'r-', linewidth=1, label='Expected Error (2σ)')
    ax2.plot(stay_durations, [-e for e in error_ranges], 'r-', linewidth=1)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    
    ax2.set_xlabel('Container Stay Duration (days)', fontsize=12)
    ax2.set_ylabel('Estimation Error Range (days)', fontsize=12)
    ax2.set_title('Expected Estimation Error Range vs Stay Duration', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    ax2.set_xlim(0, MAX_HOLDING_DAYS)
    
    plt.tight_layout()
    plt.show()


def test_estimation_distribution():
    """
    Test the estimation accuracy by running multiple simulations
    and plotting the distribution of estimation errors.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Test different stay durations
    test_durations = [2, 10, 30, 60, 120, 160]
    base_date = datetime(2025, 1, 1)
    
    for idx, stay_duration in enumerate(test_durations):
        errors = []
        arrival = base_date
        departure = arrival + timedelta(days=stay_duration)
        
        # Run 1000 simulations for each duration
        for _ in range(1000):
            # Use ContainerFactory to create container
            container = ContainerFactory.create_container(
                container_id=f"TEST_{idx}_{_}",
                container_type="FEU",
                direction="Import",
                goods_type="Regular",
                arrival_date=arrival,
                departure_date=departure
            )
            
            # Calculate error in days
            error = (container.estimated_departure - departure).total_seconds() / 86400.0
            errors.append(error)
        
        # Plot histogram
        ax = axes[idx]
        
        # Filter out any extreme outliers for better visualization
        filtered_errors = [e for e in errors if abs(e) < stay_duration]
        
        ax.hist(filtered_errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', label='True Departure')
        
        # Calculate statistics
        mean_error = np.mean(filtered_errors)
        std_error = np.std(filtered_errors)
        
        ax.set_title(f'Stay Duration: {stay_duration} days\nMean Error: {mean_error:.1f} ± {std_error:.1f} days', 
                    fontsize=10)
        ax.set_xlabel('Estimation Error (days)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set reasonable x-axis limits
        if stay_duration <= 10:
            ax.set_xlim(-10, 10)
        else:
            ax.set_xlim(-stay_duration*0.5, stay_duration*0.5)
    
    plt.suptitle('Distribution of Estimation Errors for Different Stay Durations\n(1000 simulations each)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def test_estimation_convergence():
    """
    Test how estimation accuracy improves as departure approaches.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    base_date = datetime(2025, 1, 1)
    stay_duration = 30  # 30-day stay
    arrival = base_date
    departure = arrival + timedelta(days=stay_duration)
    
    # Test at different points in time
    days_elapsed = []
    mean_errors = []
    std_errors = []
    
    for day in range(0, stay_duration + 1, 2):
        current_date = arrival + timedelta(days=day)
        errors = []
        
        # Run 500 simulations for each time point
        for _ in range(500):
            # Use ContainerFactory to create container
            container = ContainerFactory.create_container(
                container_id=f"TEST_{day}_{_}",
                container_type="FEU",
                direction="Import",
                goods_type="Regular",
                arrival_date=arrival,
                departure_date=departure
            )
            
            # Update estimation at current date
            container.update_estimation(current_date)
            error = (container.estimated_departure - departure).days
            errors.append(error)
        
        days_elapsed.append(day)
        mean_errors.append(np.mean(errors))
        std_errors.append(np.std(errors))
    
    # Plot 1: Mean error over time
    ax1.plot(days_elapsed, mean_errors, 'b-', linewidth=2, label='Mean Error')
    ax1.fill_between(days_elapsed, 
                     [m - s for m, s in zip(mean_errors, std_errors)],
                     [m + s for m, s in zip(mean_errors, std_errors)],
                     alpha=0.3, label='±1 Std Dev')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Perfect Estimation')
    ax1.set_xlabel('Days Since Arrival')
    ax1.set_ylabel('Estimation Error (days)')
    ax1.set_title(f'Estimation Error Evolution\n({stay_duration}-day stay)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Standard deviation over time
    ax2.plot(days_elapsed, std_errors, 'g-', linewidth=2)
    ax2.set_xlabel('Days Since Arrival')
    ax2.set_ylabel('Standard Deviation of Error (days)')
    ax2.set_title('Estimation Uncertainty Evolution', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('How Estimation Improves as Departure Approaches', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("Plotting Container Estimation Accuracy Functions...\n")
    
    print("1. Accuracy Function Visualization")
    plot_accuracy_function()
    
    print("\n2. Testing Estimation Distribution")
    test_estimation_distribution()
    
    print("\n3. Testing Estimation Convergence")
    test_estimation_convergence()