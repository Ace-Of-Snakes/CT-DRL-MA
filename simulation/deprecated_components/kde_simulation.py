# simulation/kde_simulation.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from simulation.TerminalConfig import TerminalConfig

def simulate_terminal_day(config=None):
    """
    Simulate a day of terminal operations using KDE models.
    
    Args:
        config: TerminalConfig object (created if None)
        
    Returns:
        DataFrame with simulated arrivals and pickups
    """
    if config is None:
        config = TerminalConfig()
    
    # Set base date
    base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Generate train schedule
    train_schedule = config.generate_train_arrival_schedule(
        n_trains=30,
        base_date=base_date
    )
    
    # Generate container IDs for trains
    container_ids = []
    for i in range(100):
        container_ids.append(f"CONT{i+1:04d}")
    
    # Generate truck pickups
    pickup_schedule = config.generate_truck_pickup_schedule(
        container_ids,
        base_date=base_date
    )
    
    # Create DataFrame with all events
    events = []
    
    # Add train arrivals
    for train_id, planned, realized in train_schedule:
        events.append({
            'event_type': 'train_arrival_planned',
            'id': train_id,
            'time': planned,
            'hour': planned.hour + planned.minute/60
        })
        events.append({
            'event_type': 'train_arrival_realized',
            'id': train_id,
            'time': realized,
            'hour': realized.hour + realized.minute/60,
            'delay': (realized - planned).total_seconds() / 3600
        })
    
    # Add truck pickups
    for container_id, pickup_time in pickup_schedule.items():
        events.append({
            'event_type': 'truck_pickup',
            'id': f"TRK_{container_id}",
            'container_id': container_id,
            'time': pickup_time,
            'hour': pickup_time.hour + pickup_time.minute/60
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(events)
    
    return df

def plot_simulation_results(df):
    """
    Plot simulated terminal operations.
    
    Args:
        df: DataFrame with simulated events
    """
    plt.figure(figsize=(14, 10))
    
    # Plot train arrivals
    train_arrivals = df[df['event_type'] == 'train_arrival_realized']
    plt.subplot(2, 1, 1)
    plt.hist(train_arrivals['hour'], bins=24, alpha=0.7, label='Train Arrivals')
    plt.title('Simulated Train Arrivals')
    plt.xlabel('Hour of Day')
    plt.ylabel('Count')
    plt.xticks(np.arange(0, 25, 3))
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot truck pickups
    truck_pickups = df[df['event_type'] == 'truck_pickup']
    plt.subplot(2, 1, 2)
    plt.hist(truck_pickups['hour'], bins=24, alpha=0.7, label='Truck Pickups')
    plt.title('Simulated Truck Pickups')
    plt.xlabel('Hour of Day')
    plt.ylabel('Count')
    plt.xticks(np.arange(0, 25, 3))
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/simulation_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot train delays
    plt.figure(figsize=(10, 6))
    plt.hist(train_arrivals['delay'], bins=20, alpha=0.7)
    plt.title('Simulated Train Delays')
    plt.xlabel('Delay (hours)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/simulated_train_delays.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Create config and load KDE models
    config = TerminalConfig()
    
    # Simulate a day of terminal operations
    df = simulate_terminal_day(config)
    
    # Plot results
    plot_simulation_results(df)
    
    # Print summary statistics
    print("Terminal Simulation Summary:")
    print(f"Total trains: {len(df[df['event_type'] == 'train_arrival_realized'])}")
    print(f"Total truck pickups: {len(df[df['event_type'] == 'truck_pickup'])}")
    print(f"Average train delay: {df[df['event_type'] == 'train_arrival_realized']['delay'].mean():.2f} hours")
    
    # Export to CSV
    df.to_csv('simulated_terminal_events.csv', index=False)
    print("Simulation results saved to 'simulated_terminal_events.csv'")