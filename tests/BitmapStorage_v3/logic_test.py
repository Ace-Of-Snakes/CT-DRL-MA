#!/usr/bin/env python3
"""
Comprehensive test suite for OptimizedSlotTierBitmapYard
Tests stacking rules, visualizes yard state, and benchmarks proximity search performance.
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from typing import List, Tuple

# Add repository root to path for imports
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.append(repo_root)

# Import classes from repository
from simulation.terminal_components.Container import Container, ContainerFactory

# Import our optimized yard (assuming it's saved in the same directory)
# Note: In practice, this would be imported from the appropriate module
from simulation.terminal_components.BitmapYard3 import OptimizedSlotTierBitmapYard

def create_test_containers() -> List[Container]:
    """Create a diverse set of test containers for stacking rule validation."""
    containers = []
    
    # Regular containers with different priorities
    containers.extend([
        ContainerFactory.create_container(f"REG{i:03d}", "TWEU", "Import", "Regular", weight=15000 + i*1000)
        for i in range(10)
    ])
    
    containers.extend([
        ContainerFactory.create_container(f"FEU{i:03d}", "FEU", "Export", "Regular", weight=20000 + i*1000) 
        for i in range(5)
    ])
    
    # High priority containers (lower numbers = higher priority)
    high_priority_containers = [
        ContainerFactory.create_container("HPR001", "TWEU", "Import", "Regular", weight=18000),
        ContainerFactory.create_container("HPR002", "TWEU", "Import", "Regular", weight=16000),
    ]
    for container in high_priority_containers:
        container.priority = 25  # High priority
    containers.extend(high_priority_containers)
    
    # Low priority containers
    low_priority_containers = [
        ContainerFactory.create_container("LPR001", "TWEU", "Import", "Regular", weight=22000),
        ContainerFactory.create_container("LPR002", "TWEU", "Import", "Regular", weight=24000),
    ]
    for container in low_priority_containers:
        container.priority = 150  # Low priority
    containers.extend(low_priority_containers)
    
    # Reefer containers
    containers.extend([
        ContainerFactory.create_container(f"REF{i:03d}", "TWEU", "Import", "Reefer", weight=19000 + i*500)
        for i in range(3)
    ])
    
    # Dangerous goods
    containers.extend([
        ContainerFactory.create_container(f"DG{i:03d}", "FEU", "Export", "Dangerous", weight=21000 + i*800)
        for i in range(3)
    ])
    
    # Special containers
    containers.extend([
        ContainerFactory.create_container("TRL001", "Trailer", "Export", "Regular", weight=14000),
        ContainerFactory.create_container("SB001", "Swap Body", "Export", "Regular", weight=12000),
        ContainerFactory.create_container("THEU001", "THEU", "Import", "Regular", weight=23000),
        ContainerFactory.create_container("FFEU001", "FFEU", "Export", "Regular", weight=28000),
    ])
    
    return containers

def test_stacking_rules(yard, containers: List[Container]):
    """Test and demonstrate stacking rules with various container combinations."""
    print("\n" + "="*60)
    print("TESTING STACKING RULES")
    print("="*60)
    
    # Test 1: Valid stacking by weight (heavier below lighter)
    print("\n--- Test 1: Weight-based stacking ---")
    heavy_container = containers[0]  # REG000, 15000kg
    light_container = containers[1]  # REG001, 16000kg
    heavy_container.weight = 25000
    light_container.weight = 20000
    
    # Place heavy container first
    success1 = yard.add_container(0, 5, 1, 1, heavy_container)
    print(f"Placing heavy container (25000kg) at A5.1-T1: {'✓' if success1 else '✗'}")
    
    # Try to stack light container on top
    success2 = yard.add_container(0, 5, 1, 2, light_container) 
    print(f"Stacking light container (20000kg) at A5.1-T2: {'✓' if success2 else '✗'}")
    
    # Test 2: Invalid stacking (heavier on lighter) 
    print("\n--- Test 2: Invalid weight stacking ---")
    very_heavy = containers[2]
    very_heavy.weight = 30000
    success3 = yard.add_container(0, 5, 1, 3, very_heavy)
    print(f"Trying to stack very heavy container (30000kg) on top: {'✓' if success3 else '✗ (Expected failure)'}")
    
    # Test 3: Priority-based stacking
    print("\n--- Test 3: Priority-based stacking ---")
    high_priority = [c for c in containers if hasattr(c, 'priority') and c.priority < 50][0]
    low_priority = [c for c in containers if hasattr(c, 'priority') and c.priority > 100][0]
    
    # Place low priority first
    success4 = yard.add_container(0, 6, 1, 1, low_priority)
    print(f"Placing low priority container (P={low_priority.priority}) at A6.1-T1: {'✓' if success4 else '✗'}")
    
    # Try to stack high priority on top (should work if weight is compatible)
    if high_priority.weight <= low_priority.weight:
        success5 = yard.add_container(0, 6, 1, 2, high_priority)
        print(f"Stacking high priority container (P={high_priority.priority}) on top: {'✓' if success5 else '✗'}")
    else:
        print(f"High priority container too heavy for stacking (weight compatibility)")
    
    # Test 4: Container type compatibility
    print("\n--- Test 4: Container type stacking ---")
    tweu1 = [c for c in containers if c.container_type == 'TWEU'][0]
    tweu2 = [c for c in containers if c.container_type == 'TWEU'][1]
    
    # Same type should stack (if other rules allow)
    success6 = yard.add_container(0, 7, 1, 1, tweu1)
    success7 = yard.add_container(0, 7, 1, 2, tweu2)
    print(f"Stacking TWEU on TWEU: {'✓' if success6 and success7 else '✗'}")
    
    # Test 5: Special area constraints
    print("\n--- Test 5: Special area constraints ---")
    reefer = [c for c in containers if c.goods_type == 'Reefer'][0]
    dangerous = [c for c in containers if c.goods_type == 'Dangerous'][0]
    
    # Try to place reefer in reefer area (should work)
    success8 = yard.add_container(0, 1, 1, 1, reefer)  # Row A, Bay 1 should be reefer area
    print(f"Placing reefer in reefer area (A1.1-T1): {'✓' if success8 else '✗'}")
    
    # Try to place reefer in non-reefer area (should fail)
    success9 = yard.add_container(0, 10, 1, 1, reefer)  # Row A, Bay 10 should NOT be reefer area
    print(f"Placing reefer in non-reefer area (A10.1-T1): {'✓' if success9 else '✗ (Expected failure)'}")
    
    # Test 6: Multi-slot containers
    print("\n--- Test 6: Multi-slot container placement ---")
    feu = [c for c in containers if c.container_type == 'FEU'][0]
    theu = [c for c in containers if c.container_type == 'THEU'][0] 
    ffeu = [c for c in containers if c.container_type == 'FFEU'][0]
    
    # FEU takes full bay (4 slots)
    success10 = yard.add_container(0, 15, 1, 1, feu)
    print(f"Placing FEU (full bay) at A15.1-T1: {'✓' if success10 else '✗'}")
    
    # THEU takes 3 slots
    success11 = yard.add_container(0, 16, 1, 1, theu)
    print(f"Placing THEU (3 slots) at A16.1-T1: {'✓' if success11 else '✗'}")
    
    # FFEU takes 5 slots (spans bays)
    success12 = yard.add_container(0, 20, 1, 1, ffeu)
    print(f"Placing FFEU (5 slots, spans bays) at A20.1-T1: {'✓' if success12 else '✗'}")
    
    # Test 7: Invalid slot starts
    print("\n--- Test 7: Invalid starting slots ---")
    tweu_wrong_slot = containers[3]
    success13 = yard.add_container(0, 25, 2, 1, tweu_wrong_slot)  # TWEU can't start at slot 2
    print(f"Placing TWEU at slot 2 (invalid): {'✓' if success13 else '✗ (Expected failure)'}")
    
    success14 = yard.add_container(0, 25, 3, 1, tweu_wrong_slot)  # TWEU can start at slot 3
    print(f"Placing TWEU at slot 3 (valid): {'✓' if success14 else '✗'}")

def benchmark_proximity_performance(yard, containers: List[Container]):
    """Benchmark proximity search performance with different n values."""
    print("\n" + "="*60)
    print("PROXIMITY SEARCH PERFORMANCE BENCHMARK")
    print("="*60)
    
    # Place some containers for testing
    test_positions = [
        (0, 10, 1, 1), (0, 15, 1, 1), (0, 20, 1, 1),
        (1, 10, 1, 1), (1, 15, 1, 1), (1, 20, 1, 1),
        (2, 10, 1, 1), (2, 15, 1, 1), (2, 20, 1, 1)
    ]
    
    # Place containers at test positions
    for i, (row_idx, bay, slot, tier) in enumerate(test_positions):
        if i < len(containers):
            yard.add_container(row_idx, bay, slot, tier, containers[i])
    
    print(f"Placed {len(test_positions)} containers for testing")
    
    # Test different proximity values
    n_values = [1, 3, 5, 7, 10, 15, 20]
    times_per_n = []
    moves_per_n = []
    
    print(f"\nTesting proximity search with n values: {n_values}")
    print("n\tTime (ms)\tMoves Found\tMoves/sec")
    print("-" * 45)
    
    for n in n_values:
        # Time multiple runs for accuracy
        times = []
        total_moves = 0
        num_runs = 50
        
        for run in range(num_runs):
            start_time = time.perf_counter()
            
            # Calculate moves for all test positions
            run_moves = 0
            for row_idx, bay, slot, tier in test_positions:
                moves = yard.calc_possible_moves_vectorized(row_idx, bay, slot, tier, n)
                run_moves += len(moves)
            
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
            
            if run == 0:  # Count moves only once
                total_moves = run_moves
        
        avg_time = np.mean(times)
        moves_per_sec = (total_moves * num_runs) / (np.sum(times) / 1000) if np.sum(times) > 0 else 0
        
        times_per_n.append(avg_time)
        moves_per_n.append(total_moves)
        
        print(f"{n}\t{avg_time:.2f}\t\t{total_moves}\t\t{moves_per_sec:.0f}")
    
    return n_values, times_per_n, moves_per_n

def visualize_yard_state(yard):
    """Create visualizations of the current yard state."""
    print("\n" + "="*60)
    print("GENERATING YARD VISUALIZATIONS")
    print("="*60)
    
    # Create a simplified visualization since we're focusing on testing
    # Rather than implementing full visualization here, we'll show key metrics
    
    container_count = yard.get_container_count()
    occupancy_rate = yard.get_occupancy_rate()
    
    print(f"Total containers in yard: {container_count}")
    print(f"Occupancy rate: {occupancy_rate:.1%}")
    
    # Count containers by type
    type_counts = {}
    for bit_idx, container in yard.container_by_bit_idx.items():
        container_type = container.container_type
        type_counts[container_type] = type_counts.get(container_type, 0) + 1
    
    print("\nContainer types in yard:")
    for container_type, count in sorted(type_counts.items()):
        print(f"  {container_type}: {count}")
    
    # Show stacking statistics
    stacked_positions = 0
    max_stack_height = 0
    
    for row_idx in range(yard.num_rows):
        for bay in range(1, yard.num_bays + 1):
            for slot in range(1, yard.slots_per_bay + 1):
                stack_height = 0
                for tier in range(1, yard.max_tier_height + 1):
                    if yard.is_position_occupied(row_idx, bay, slot, tier):
                        stack_height = tier
                
                if stack_height > 1:
                    stacked_positions += 1
                max_stack_height = max(max_stack_height, stack_height)
    
    print(f"\nStacking statistics:")
    print(f"  Positions with stacks: {stacked_positions}")
    print(f"  Maximum stack height: {max_stack_height}")

def visualize_performance_results(n_values, times_per_n, moves_per_n):
    """Create performance visualization plots."""
    print("\n" + "="*60)
    print("CREATING PERFORMANCE PLOTS")
    print("="*60)
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Time vs N
    ax1.plot(n_values, times_per_n, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Proximity Range (n)')
    ax1.set_ylabel('Average Time (ms)')
    ax1.set_title('Search Time vs Proximity Range')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Add time annotations
    for i, (n, t) in enumerate(zip(n_values, times_per_n)):
        ax1.annotate(f'{t:.1f}ms', (n, t), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8)
    
    # Plot 2: Moves found vs N
    ax2.plot(n_values, moves_per_n, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Proximity Range (n)')
    ax2.set_ylabel('Total Moves Found')
    ax2.set_title('Moves Found vs Proximity Range')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Efficiency (Moves per ms)
    efficiency = [moves / time_ms for moves, time_ms in zip(moves_per_n, times_per_n)]
    ax3.plot(n_values, efficiency, 'ro-', linewidth=2, markersize=8)
    ax3.set_xlabel('Proximity Range (n)')
    ax3.set_ylabel('Moves per ms')
    ax3.set_title('Search Efficiency')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('proximity_performance.png', dpi=300, bbox_inches='tight')
    print("Performance plots saved as 'proximity_performance.png'")
    plt.show()

def test_batch_operations(yard, containers: List[Container]):
    """Test batch operation performance."""
    print("\n" + "="*60)
    print("TESTING BATCH OPERATIONS")
    print("="*60)
    
    # Place containers for batch testing
    test_positions = []
    for i in range(20):
        row_idx = i % yard.num_rows
        bay = 30 + (i // yard.num_rows)
        slot = 1
        tier = 1
        
        if i < len(containers):
            success = yard.add_container(row_idx, bay, slot, tier, containers[i])
            if success:
                test_positions.append((row_idx, bay, slot, tier))
    
    print(f"Placed {len(test_positions)} containers for batch testing")
    
    # Test batch move calculation
    start_time = time.perf_counter()
    batch_results = yard.batch_calc_possible_moves(test_positions, n=5)
    end_time = time.perf_counter()
    
    total_moves = sum(len(moves) for moves in batch_results.values())
    batch_time = (end_time - start_time) * 1000
    
    print(f"Batch calculation: {total_moves} total moves in {batch_time:.2f}ms")
    print(f"Average: {len(test_positions) / (batch_time/1000):.0f} containers/sec")
    
    # Compare with individual calculations
    start_time = time.perf_counter()
    individual_total = 0
    for pos in test_positions:
        moves = yard.calc_possible_moves_vectorized(*pos, n=5)
        individual_total += len(moves)
    end_time = time.perf_counter()
    
    individual_time = (end_time - start_time) * 1000
    
    print(f"Individual calculation: {individual_total} total moves in {individual_time:.2f}ms")
    print(f"Speedup: {individual_time/batch_time:.1f}x")

def main():
    """Main test function."""
    print("OptimizedSlotTierBitmapYard Comprehensive Test Suite")
    print("=" * 60)
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create optimized yard
    yard = OptimizedSlotTierBitmapYard(
        num_rows=5,
        num_bays=58,
        max_tier_height=5,
        special_areas={
            'reefer': [('A', 1, 5), ('E', 54, 58)],  # Reefer areas at ends
            'dangerous': [('C', 28, 32)],            # Dangerous goods in middle
            'trailer': [('E', 1, 58)],               # Trailers in row E
            'swap_body': [('E', 1, 58)]              # Swap bodies in row E
        },
        device=device
    )
    
    print(f"Created yard: {yard}")
    
    # Create test containers
    print("\nCreating test containers...")
    containers = create_test_containers()
    print(f"Created {len(containers)} test containers")
    
    # Run tests
    try:
        # Test 1: Stacking rules
        test_stacking_rules(yard, containers)
        
        # Test 2: Visualize current state
        visualize_yard_state(yard)
        
        # Test 3: Benchmark proximity performance
        n_values, times_per_n, moves_per_n = benchmark_proximity_performance(yard, containers)
        
        # Test 4: Test batch operations
        test_batch_operations(yard, containers)
        
        # Test 5: Create performance visualizations
        visualize_performance_results(n_values, times_per_n, moves_per_n)
        
        # Final statistics
        print("\n" + "="*60)
        print("FINAL TEST SUMMARY")
        print("="*60)
        print(f"Yard state: {yard}")
        print(f"Peak performance: {min(times_per_n):.2f}ms for n={n_values[times_per_n.index(min(times_per_n))]}")
        print(f"Max moves found: {max(moves_per_n)} for n={n_values[moves_per_n.index(max(moves_per_n))]}")
        
        # Memory efficiency test
        cache_size = len(yard.mask_cache) + len(yard.proximity_cache)
        print(f"Cache entries created: {cache_size}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTest suite completed!")

if __name__ == "__main__":
    main()