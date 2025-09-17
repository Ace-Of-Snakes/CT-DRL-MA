import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List
import json
from datetime import datetime
from simulation.terminal_components.vehicles.Train import Train
from simulation.terminal_components.storage_units.ContainerFactory import ContainerFactory
from simulation.terminal_components.systems.train_tools.TrainLoader import TrainLoader

def run_performance_test():
    """Run comprehensive performance test for TrainLoader component."""
    
    # Initialize components
    factory = ContainerFactory()
    loader = TrainLoader(factory, overgeneration_factor=3.0)
    
    # Load operator list from distribution file
    with open("simulation/data/train_operator_container_type_distribution_import.json", 'r') as f:
        operator_data = json.load(f)
    operators = list(operator_data.keys())[:10]  # Test first 10 operators
    
    # Test parameters
    train_sizes = [5, 10, 20, 29, 40, 50]  # Different wagon counts
    repetitions = 10  # Repetitions for averaging
    
    # Results storage
    results = {
        'timing': {},      # operator -> train_size -> avg_time_ms
        'utilization': {}, # operator -> train_size -> avg_utilization
        'containers': {}   # operator -> train_size -> avg_containers_loaded
    }
    
    print("Running TrainLoader Performance Test")
    print("=" * 50)
    print(f"Operators: {len(operators)}")
    print(f"Train sizes: {train_sizes}")
    print(f"Repetitions: {repetitions}")
    print("=" * 50)
    
    # Run tests
    for operator in operators:
        results['timing'][operator] = []
        results['utilization'][operator] = []
        results['containers'][operator] = []
        
        for wagon_count in train_sizes:
            times = []
            utilizations = []
            container_counts = []
            
            for _ in range(repetitions):
                # Create fresh train
                train = Train(train_id=f"TEST_{wagon_count}", num_wagons=wagon_count)
                
                # Measure loading time
                start = time.perf_counter()
                loaded_train = loader.load_train(train, operator, datetime.now())
                elapsed_ms = (time.perf_counter() - start) * 1000
                
                # Collect metrics
                stats = loader.get_loading_stats(loaded_train)
                times.append(elapsed_ms)
                utilizations.append(stats.avg_utilization * 100)
                container_counts.append(stats.containers_loaded)
            
            # Store averages
            results['timing'][operator].append(np.mean(times))
            results['utilization'][operator].append(np.mean(utilizations))
            results['containers'][operator].append(np.mean(container_counts))
        
        print(f"Completed: {operator}")
    
    # Create visualizations
    create_performance_plots(results, operators, train_sizes)
    
    # Print summary statistics
    print_summary_statistics(results, operators, train_sizes)
    
    return results


def create_performance_plots(results: Dict, operators: List[str], train_sizes: List[int]):
    """Create performance visualization plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('TrainLoader Performance Analysis', fontsize=14, fontweight='bold')
    
    # Plot 1: Loading Time vs Train Size
    ax1 = axes[0, 0]
    for operator in operators[:5]:  # Top 5 for clarity
        ax1.plot(train_sizes, results['timing'][operator], marker='o', label=operator, alpha=0.7)
    ax1.set_xlabel('Number of Wagons')
    ax1.set_ylabel('Loading Time (ms)')
    ax1.set_title('Loading Performance by Operator')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    
    # Plot 2: Average Loading Time
    ax2 = axes[0, 1]
    avg_times = [np.mean([results['timing'][op][i] for op in operators]) 
                 for i in range(len(train_sizes))]
    std_times = [np.std([results['timing'][op][i] for op in operators]) 
                 for i in range(len(train_sizes))]
    ax2.errorbar(train_sizes, avg_times, yerr=std_times, marker='s', 
                 color='darkblue', capsize=5, capthick=2)
    ax2.set_xlabel('Number of Wagons')
    ax2.set_ylabel('Avg Loading Time (ms)')
    ax2.set_title('Average Loading Time (All Operators)')
    ax2.grid(True, alpha=0.3)
    
    # Add linear fit line
    z = np.polyfit(train_sizes, avg_times, 1)
    p = np.poly1d(z)
    ax2.plot(train_sizes, p(train_sizes), "r--", alpha=0.5, 
             label=f'Linear fit: {z[0]:.3f}ms/wagon')
    ax2.legend()
    
    # Plot 3: Utilization Rates
    ax3 = axes[1, 0]
    utilization_matrix = np.array([results['utilization'][op] for op in operators])
    im = ax3.imshow(utilization_matrix, aspect='auto', cmap='RdYlGn', vmin=70, vmax=100)
    ax3.set_xticks(range(len(train_sizes)))
    ax3.set_xticklabels(train_sizes)
    ax3.set_yticks(range(len(operators)))
    ax3.set_yticklabels(operators, fontsize=8)
    ax3.set_xlabel('Number of Wagons')
    ax3.set_ylabel('Operator')
    ax3.set_title('Wagon Utilization Heatmap (%)')
    plt.colorbar(im, ax=ax3)
    
    # Plot 4: Containers Loaded vs Wagon Count
    ax4 = axes[1, 1]
    avg_containers = [np.mean([results['containers'][op][i] for op in operators]) 
                      for i in range(len(train_sizes))]
    theoretical_max = [size * 4 for size in train_sizes]  # Max ~4 20ft containers per wagon
    theoretical_avg = [size * 2.5 for size in train_sizes]  # Avg mix of containers
    
    ax4.plot(train_sizes, avg_containers, marker='o', color='blue', 
             label='Actual Loaded', linewidth=2)
    ax4.plot(train_sizes, theoretical_max, '--', color='gray', 
             label='Theoretical Max', alpha=0.5)
    ax4.plot(train_sizes, theoretical_avg, ':', color='green', 
             label='Expected Average', alpha=0.5)
    ax4.set_xlabel('Number of Wagons')
    ax4.set_ylabel('Containers Loaded')
    ax4.set_title('Container Loading Efficiency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trainloader_performance.png', dpi=150, bbox_inches='tight')
    plt.show()


def print_summary_statistics(results: Dict, operators: List[str], train_sizes: List[int]):
    """Print summary statistics table."""
    
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    # Overall statistics
    all_times = [results['timing'][op][i] 
                 for op in operators for i in range(len(train_sizes))]
    all_utils = [results['utilization'][op][i] 
                 for op in operators for i in range(len(train_sizes))]
    
    print(f"\nOverall Statistics:")
    print(f"  Avg Loading Time: {np.mean(all_times):.2f} ms")
    print(f"  Time per Wagon: {np.mean(all_times) / np.mean(train_sizes):.3f} ms")
    print(f"  Avg Utilization: {np.mean(all_utils):.1f}%")
    print(f"  Min Utilization: {np.min(all_utils):.1f}%")
    print(f"  Max Utilization: {np.max(all_utils):.1f}%")
    
    # Performance scaling
    print(f"\nScaling Analysis:")
    for i, size in enumerate(train_sizes):
        avg_time = np.mean([results['timing'][op][i] for op in operators])
        avg_util = np.mean([results['utilization'][op][i] for op in operators])
        avg_cont = np.mean([results['containers'][op][i] for op in operators])
        print(f"  {size:2d} wagons: {avg_time:6.2f}ms, "
              f"{avg_util:5.1f}% util, {avg_cont:5.1f} containers")
    
    # Operator comparison (top 5)
    print(f"\nTop 5 Operators by Performance (29 wagons):")
    idx_29 = train_sizes.index(29) if 29 in train_sizes else -1
    if idx_29 >= 0:
        op_performance = [(op, results['timing'][op][idx_29], 
                          results['utilization'][op][idx_29]) 
                         for op in operators]
        op_performance.sort(key=lambda x: x[1])  # Sort by time
        
        for i, (op, time, util) in enumerate(op_performance[:5], 1):
            print(f"  {i}. {op:15s}: {time:6.2f}ms, {util:5.1f}% utilization")


if __name__ == "__main__":
    results = run_performance_test()