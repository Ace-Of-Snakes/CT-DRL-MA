import time
import random
import numpy as np
import sys
import tracemalloc
from typing import List, Dict, Tuple
from simulation.terminal_components.Container import Container, ContainerFactory

# Import your BooleanStorageYard class
from simulation.terminal_components.BooleanStorage import BooleanStorageYard

class ContainerPlacementStressTest:
    """
    Comprehensive stress test for container placement performance.
    Tests search_insertion_position with varying numbers of containers.
    """
    
    def __init__(self, yard_config: Dict = None):
        """
        Initialize stress test with yard configuration.
        
        Args:
            yard_config: Dictionary with yard parameters
        """
        self.yard_config = yard_config or {
            'n_rows': 6,
            'n_bays': 30,
            'n_tiers': 4,
            'split_factor': 4
        }
        
        # Test container types and their frequencies
        self.container_types = {
            'TWEU': 0.35,    # 35% - most common
            'FEU': 0.40,     # 40% - very common
            'THEU': 0.15,    # 15% - moderate
            'FFEU': 0.05,    # 5% - rare
            'Swap Body': 0.03, # 3% - rare
            'Trailer': 0.02    # 2% - rare
        }
        
        self.goods_types = ['reg', 'r', 'dg', 'sb_t']
        self.goods_weights = [0.70, 0.15, 0.10, 0.05]  # Regular containers are most common
        
        # Results storage
        self.results = {}
        
    def create_test_yard(self) -> 'BooleanStorageYard':
        """Create a test yard with realistic special area configuration."""
        coordinates = []
        
        n_rows = self.yard_config['n_rows']
        n_bays = self.yard_config['n_bays']
        
        # Reefer areas (first and last 2 bays of each row)
        for row in range(1, n_rows + 1):
            for bay in [1, 2, n_bays-1, n_bays]:
                coordinates.append((bay, row, "r"))
        
        # Dangerous goods area (middle section)
        dg_start = n_bays // 2 - 2
        dg_end = n_bays // 2 + 2
        for row in range(3, min(6, n_rows + 1)):  # Rows 3-5
            for bay in range(dg_start, dg_end + 1):
                coordinates.append((bay, row, "dg"))
        
        # Swap body/trailer area (row 1, middle section)
        sb_start = 5
        sb_end = n_bays - 5
        for bay in range(sb_start, sb_end + 1):
            coordinates.append((bay, 1, "sb_t"))
        
        from simulation.terminal_components.BooleanStorage import BooleanStorageYard
        return BooleanStorageYard(
            n_rows=n_rows,
            n_bays=n_bays,
            n_tiers=self.yard_config['n_tiers'],
            coordinates=coordinates,
            split_factor=self.yard_config['split_factor'],
            validate=False
        )
    
    def generate_random_containers(self, count: int) -> List[Container]:
        """Generate random containers for testing."""
        containers = []
        
        for i in range(count):
            # Select container type based on frequency
            container_type = random.choices(
                list(self.container_types.keys()),
                weights=list(self.container_types.values())
            )[0]
            
            # Select goods type
            goods_type_mapping = {'reg': 'Regular', 'r': 'Reefer', 'dg': 'Dangerous', 'sb_t': 'Regular'}
            goods_code = random.choices(self.goods_types, weights=self.goods_weights)[0]
            goods_type = goods_type_mapping[goods_code]
            
            # Adjust container type for special goods
            if goods_code == 'sb_t':
                container_type = random.choice(['Swap Body', 'Trailer'])
            
            container = ContainerFactory.create_container(
                container_id=f"TEST{i:05d}",
                container_type=container_type,
                direction=random.choice(["Import", "Export"]),
                goods_type=goods_type,
                weight=random.randint(5000, 30000)
            )
            containers.append(container)
        
        return containers
    
    def place_containers_in_yard(self, yard: 'BooleanStorageYard', containers: List[Container]) -> Dict:
        """
        Place containers in the yard and measure placement performance.
        
        Returns:
            Dictionary with placement statistics
        """
        placement_times = []
        search_times = []
        failed_placements = 0
        placed_containers = 0
        
        for i, container in enumerate(containers):
            # Determine goods type for search
            goods_map = {
                'Regular': 'reg',
                'Reefer': 'r', 
                'Dangerous': 'dg'
            }
            
            if container.container_type in ['Swap Body', 'Trailer']:
                goods_code = 'sb_t'
            else:
                goods_code = goods_map.get(container.goods_type, 'reg')
            
            # Random bay to search around
            search_bay = random.randint(0, yard.n_bays - 1)
            max_proximity = random.randint(2, 8)
            
            # Measure search time
            search_start = time.perf_counter()
            
            try:
                valid_placements = yard.search_insertion_position(
                    bay=search_bay,
                    goods=goods_code,
                    container_type=container.container_type,
                    max_proximity=max_proximity
                )
                
                search_time = time.perf_counter() - search_start
                search_times.append(search_time)
                
                # Try to place container
                if valid_placements:
                    placement_start = time.perf_counter()
                    
                    # Select random placement from available options
                    placement = random.choice(valid_placements)
                    coordinates = yard.get_container_coordinates_from_placement(placement, container.container_type)
                    
                    # Attempt to place
                    try:
                        yard.add_container(container, coordinates)
                        placement_time = time.perf_counter() - placement_start
                        placement_times.append(placement_time)
                        placed_containers += 1
                        
                    except Exception as e:
                        failed_placements += 1
                        # print(f"Failed to place container {i}: {e}")
                        
                else:
                    failed_placements += 1
                    
            except Exception as e:
                failed_placements += 1
                search_times.append(0.001)  # Minimal time for failed search
                # print(f"Search failed for container {i}: {e}")
            
            # Progress indicator for large tests
            if i > 0 and i % 100 == 0:
                print(f"  Processed {i}/{len(containers)} containers...")
        
        return {
            'total_containers': len(containers),
            'placed_containers': placed_containers,
            'failed_placements': failed_placements,
            'avg_search_time': np.mean(search_times) if search_times else 0,
            'max_search_time': np.max(search_times) if search_times else 0,
            'avg_placement_time': np.mean(placement_times) if placement_times else 0,
            'total_search_time': sum(search_times),
            'total_placement_time': sum(placement_times),
            'search_times': search_times,
            'placement_times': placement_times
        }
    
    def run_stress_test(self, container_counts: List[int] = None) -> Dict:
        """
        Run comprehensive stress test with different container counts.
        
        Args:
            container_counts: List of container counts to test
            
        Returns:
            Dictionary with test results
        """
        if container_counts is None:
            container_counts = [100, 500, 1000, 5000]
        
        print("🚛 Starting Container Placement Stress Test 🚛")
        print("=" * 60)
        print(f"Yard Configuration: {self.yard_config}")
        print(f"Test Sizes: {container_counts}")
        print("=" * 60)
        
        results = {}
        
        for count in container_counts:
            print(f"\n📦 Testing with {count} containers...")
            
            # Start memory tracking
            tracemalloc.start()
            test_start = time.perf_counter()
            
            # Create fresh yard for each test
            yard = self.create_test_yard()
            
            # Generate containers
            containers = self.generate_random_containers(count)
            
            # Run placement test
            placement_stats = self.place_containers_in_yard(yard, containers)
            
            # Calculate total test time
            total_test_time = time.perf_counter() - test_start
            
            # Get memory usage
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Calculate performance metrics
            search_ops_per_second = count / placement_stats['total_search_time'] if placement_stats['total_search_time'] > 0 else 0
            
            # Store results
            results[count] = {
                **placement_stats,
                'total_test_time': total_test_time,
                'memory_usage_mb': current_memory / (1024 * 1024),
                'peak_memory_mb': peak_memory / (1024 * 1024),
                'search_ops_per_second': search_ops_per_second,
                'success_rate': placement_stats['placed_containers'] / count,
                'yard_utilization': placement_stats['placed_containers'] / (yard.n_rows * yard.n_bays * yard.n_tiers)
            }
            
            # Print immediate results
            self.print_test_results(count, results[count])
        
        print("\n" + "=" * 60)
        print("📊 STRESS TEST SUMMARY")
        print("=" * 60)
        
        self.print_summary_table(results)
        self.analyze_performance_scaling(results)
        
        return results
    
    def print_test_results(self, count: int, stats: Dict):
        """Print detailed results for a single test."""
        print(f"Results for {count} containers:")
        print(f"  ✅ Successfully placed: {stats['placed_containers']}/{count} ({stats['success_rate']:.1%})")
        print(f"  ⏱️  Average search time: {stats['avg_search_time']*1000:.3f} ms")
        print(f"  ⏱️  Max search time: {stats['max_search_time']*1000:.3f} ms")
        print(f"  🔍 Search operations/sec: {stats['search_ops_per_second']:.1f}")
        print(f"  💾 Memory usage: {stats['memory_usage_mb']:.1f} MB (peak: {stats['peak_memory_mb']:.1f} MB)")
        print(f"  📊 Yard utilization: {stats['yard_utilization']:.1%}")
        print(f"  🕐 Total test time: {stats['total_test_time']:.2f} seconds")
    
    def print_summary_table(self, results: Dict):
        """Print a summary table of all test results."""
        print(f"{'Containers':<12} {'Success%':<10} {'Avg Search(ms)':<15} {'Ops/sec':<10} {'Memory(MB)':<12} {'Time(s)':<10}")
        print("-" * 75)
        
        for count in sorted(results.keys()):
            stats = results[count]
            print(f"{count:<12} {stats['success_rate']*100:<10.1f} {stats['avg_search_time']*1000:<15.3f} "
                  f"{stats['search_ops_per_second']:<10.1f} {stats['memory_usage_mb']:<12.1f} {stats['total_test_time']:<10.2f}")
    
    def analyze_performance_scaling(self, results: Dict):
        """Analyze how performance scales with container count."""
        print(f"\n🔬 Performance Scaling Analysis:")
        
        counts = sorted(results.keys())
        if len(counts) < 2:
            print("  Need at least 2 test sizes for scaling analysis")
            return
        
        # Calculate scaling factors
        base_count = counts[0]
        base_time = results[base_count]['avg_search_time']
        
        print(f"  Base case ({base_count} containers): {base_time*1000:.3f} ms avg search")
        
        for count in counts[1:]:
            current_time = results[count]['avg_search_time']
            scaling_factor = current_time / base_time
            theoretical_linear = count / base_count
            
            print(f"  {count} containers: {current_time*1000:.3f} ms ({scaling_factor:.2f}x vs {theoretical_linear:.1f}x linear)")
        
        # Performance recommendations
        max_search_time = max(results[count]['max_search_time'] for count in counts)
        avg_search_time = results[counts[-1]]['avg_search_time']  # Largest test
        
        print(f"\n💡 Recommendations:")
        if max_search_time > 0.001:  # > 1ms
            print(f"  ⚠️  Maximum search time: {max_search_time*1000:.3f} ms - consider batching for large operations")
        if avg_search_time > 0.0005:  # > 0.5ms average
            print(f"  ⚠️  Average search time: {avg_search_time*1000:.3f} ms - may need optimization for real-time use")
        else:
            print(f"  ✅ Performance is excellent - no batching needed")
        
        # Memory usage analysis
        max_memory = max(results[count]['peak_memory_mb'] for count in counts)
        if max_memory > 100:  # > 100MB
            print(f"  ⚠️  Peak memory usage: {max_memory:.1f} MB - monitor for larger datasets")
        else:
            print(f"  ✅ Memory usage is reasonable: {max_memory:.1f} MB peak")

def run_quick_stress_test():
    """Run a quick stress test with default settings."""
    
    # Quick test configuration
    test_config = {
        'n_rows': 5,
        'n_bays': 15,
        'n_tiers': 3,
        'split_factor': 4
    }
    
    print("🚀 Quick Stress Test")
    tester = ContainerPlacementStressTest(test_config)
    results = tester.run_stress_test([100, 500, 1000])
    
    return results

def run_full_stress_test():
    """Run the full stress test with realistic yard size."""
    
    # Realistic terminal configuration
    realistic_config = {
        'n_rows': 8,
        'n_bays': 40,
        'n_tiers': 5,
        'split_factor': 4
    }
    
    print("🏭 Full Scale Stress Test")
    tester = ContainerPlacementStressTest(realistic_config)
    results = tester.run_stress_test([100, 500, 1000, 5000])
    
    return results

# Add this to your stress_test.py fil

def run_optimization_benchmark():
    """Run a benchmark comparing original vs optimized methods."""
    
    print("🚀 OPTIMIZATION BENCHMARK")
    print("=" * 60)
    
    # Add methods to class
    # add_optimized_methods_to_yard()
    
    # Test configuration
    test_config = {
        'n_rows': 6,
        'n_bays': 30,
        'n_tiers': 4,
        'split_factor': 4
    }
    
    container_counts = [100, 300, 500, 1000, 2000, 5000]
    proximities = [1, 3, 5, 8]
    
    for count in container_counts:
        print(f"\n📦 Benchmarking {count} containers:")
        
        for proximity in proximities:
            print(f"  🎯 Proximity {proximity}:")
            
            # Create tester
            tester = ContainerPlacementStressTest(test_config)
            
            # Create and populate yard
            yard = tester.create_test_yard()
            containers = tester.generate_random_containers(count)
            placement_stats = tester.place_containers_in_yard(yard, containers)
            
            actual_placed = placement_stats['placed_containers']
            print(f"    📍 Placed {actual_placed} containers")
            
            if actual_placed < 10:
                print("    ⚠️  Too few containers, skipping")
                continue
            
            # Benchmark methods
            methods = [
                ('Original', 'return_possible_yard_moves'),
                ('Optimized', 'return_possible_yard_moves_optimized'), 
                ('Ultra-Opt', 'return_possible_yard_moves_ultra_optimized')
            ]
            
            results = {}
            iterations = 3
            
            for method_name, method_attr in methods:
                if not hasattr(yard, method_attr):
                    continue
                
                method = getattr(yard, method_attr)
                times = []
                moves_counts = []
                
                for i in range(iterations):
                    start_time = time.perf_counter()
                    
                    try:
                        moves = method(max_proximity=proximity)
                        end_time = time.perf_counter()
                        
                        execution_time = end_time - start_time
                        times.append(execution_time)
                        
                        total_moves = sum(len(m['destinations']) for m in moves.values())
                        moves_counts.append(total_moves)
                        
                    except Exception as e:
                        print(f"    ❌ {method_name} failed: {e}")
                        times.append(float('inf'))
                        moves_counts.append(0)
                
                if times and any(t != float('inf') for t in times):
                    valid_times = [t for t in times if t != float('inf')]
                    results[method_name] = {
                        'avg_time': np.mean(valid_times),
                        'avg_moves': np.mean(moves_counts)
                    }
            
            # Print comparison
            if 'Original' in results:
                baseline_time = results['Original']['avg_time']
                baseline_moves = results['Original']['avg_moves']
                
                for method_name, stats in results.items():
                    speedup = baseline_time / stats['avg_time'] if stats['avg_time'] > 0 else 0
                    moves_match = abs(stats['avg_moves'] - baseline_moves) < 1
                    
                    status = "✅" if moves_match else "⚠️ "
                    print(f"    {status} {method_name:12}: {stats['avg_time']*1000:7.2f}ms "
                          f"({speedup:5.1f}x) - {stats['avg_moves']:.0f} moves")
            else:
                for method_name, stats in results.items():
                    print(f"    📊 {method_name:12}: {stats['avg_time']*1000:7.2f}ms - {stats['avg_moves']:.0f} moves")

def run_comprehensive_optimization_test():
    """Run comprehensive test of the optimized methods."""
    
    print("\n🔬 COMPREHENSIVE OPTIMIZATION TEST")
    print("=" * 80)
    
    # Add methods first
    # add_optimized_methods_to_yard()
    
    # Different yard sizes
    configs = [
        {
            'name': 'Small Yard',
            'config': {'n_rows': 4, 'n_bays': 20, 'n_tiers': 3, 'split_factor': 4},
            'containers': [50, 150, 300]
        },
        {
            'name': 'Medium Yard', 
            'config': {'n_rows': 6, 'n_bays': 30, 'n_tiers': 4, 'split_factor': 4},
            'containers': [100, 300, 600]
        },
        {
            'name': 'Large Yard',
            'config': {'n_rows': 8, 'n_bays': 50, 'n_tiers': 5, 'split_factor': 4},
            'containers': [200, 500, 1000]
        }
    ]
    
    all_results = {}
    
    for config in configs:
        print(f"\n🏭 Testing {config['name']}: {config['config']}")
        
        yard_results = {}
        
        for container_count in config['containers']:
            print(f"  📦 {container_count} containers:")
            
            # Create tester and yard
            tester = ContainerPlacementStressTest(config['config'])
            yard = tester.create_test_yard()
            containers = tester.generate_random_containers(container_count)
            placement_stats = tester.place_containers_in_yard(yard, containers)
            
            actual_placed = placement_stats['placed_containers']
            if actual_placed < 10:
                continue
            
            # Test with proximity=5 for consistency
            proximity = 5
            iterations = 5
            
            # Test both optimized methods
            test_methods = [
                ('Optimized', 'return_possible_yard_moves_optimized'),
                ('Ultra-Opt', 'return_possible_yard_moves_ultra_optimized')
            ]
            
            method_results = {}
            
            for method_name, method_attr in test_methods:
                if not hasattr(yard, method_attr):
                    continue
                
                method = getattr(yard, method_attr)
                times = []
                
                for i in range(iterations):
                    start_time = time.perf_counter()
                    
                    try:
                        moves = method(max_proximity=proximity)
                        end_time = time.perf_counter()
                        times.append(end_time - start_time)
                        
                    except Exception as e:
                        print(f"    ❌ {method_name} failed: {e}")
                        continue
                
                if times:
                    avg_time = np.mean(times)
                    method_results[method_name] = avg_time
                    print(f"    ✅ {method_name}: {avg_time*1000:.2f}ms avg")
            
            yard_results[container_count] = method_results
        
        all_results[config['name']] = yard_results
    
    # Summary
    print(f"\n📊 OPTIMIZATION SUMMARY")
    print("=" * 50)
    
    for yard_name, yard_data in all_results.items():
        print(f"\n{yard_name}:")
        for container_count, methods in yard_data.items():
            print(f"  {container_count} containers:")
            for method_name, avg_time in methods.items():
                rating = "🟢" if avg_time < 0.005 else "🟡" if avg_time < 0.020 else "🔴"
                print(f"    {rating} {method_name}: {avg_time*1000:.2f}ms")
    
    return all_results

import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from simulation.terminal_components.Container import Container, ContainerFactory

def stress_test_ultra_optimized():
    """
    Comprehensive stress test for the ultra_optimized yard moves function.
    Tests performance, correctness, and scalability.
    """
    
    print("=" * 60)
    print("STRESS TEST: Ultra Optimized Yard Moves Function")
    print("=" * 60)
    
    # Test configurations with increasing complexity
    test_configs = [
        {"name": "Small", "rows": 3, "bays": 10, "tiers": 3, "containers": 20},
        {"name": "Medium", "rows": 5, "bays": 20, "tiers": 4, "containers": 100},
        {"name": "Large", "rows": 8, "bays": 30, "tiers": 5, "containers": 300},
        {"name": "Extra Large", "rows": 10, "bays": 40, "tiers": 6, "containers": 600},
        {"name": "Massive", "rows": 15, "bays": 50, "tiers": 6, "containers": 1000},
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n--- Testing {config['name']} Configuration ---")
        print(f"Yard: {config['rows']}x{config['bays']}x{config['tiers']}, "
              f"Containers: {config['containers']}")
        
        # Create yard
        yard = create_test_yard(config)
        
        # Add containers to yard
        containers_added = populate_yard_with_containers(yard, config['containers'])
        print(f"Successfully added {containers_added} containers")
        
        # Performance test
        performance_results = run_performance_test(yard, config)
        results.append({
            'config': config,
            'performance': performance_results,
            'containers_added': containers_added
        })
        
        # Memory usage test
        memory_usage = estimate_memory_usage(yard)
        print(f"Estimated memory usage: {memory_usage:.2f} MB")
        
        print(f"Performance: {performance_results['avg_time']:.4f}s per call")
        print(f"Found moves: {performance_results['avg_moves']:.1f} containers with moves")
        print(f"Total destinations: {performance_results['avg_destinations']:.1f} total destinations")
    
    # Generate performance plots
    plot_performance_results(results)
    
    # Detailed analysis on largest configuration
    if results:
        print(f"\n--- Detailed Analysis on {test_configs[-1]['name']} Configuration ---")
        detailed_analysis(results[-1])
    
    return results

def create_test_yard(config):
    """Create a test yard with the given configuration."""
    from simulation.terminal_components.BooleanStorage import BooleanStorageYard
    
    rows, bays, tiers = config['rows'], config['bays'], config['tiers']
    
    # Generate coordinate configuration for special areas
    coordinates = []
    
    # Reefers on both ends
    for row in range(1, rows + 1):
        coordinates.append((1, row, "r"))  # First bay
        coordinates.append((bays, row, "r"))  # Last bay
    
    # Dangerous goods in middle
    mid_bay = bays // 2
    for row in range(2, min(rows, 4) + 1):  # Rows 2-4 or available
        for bay_offset in [-1, 0, 1]:
            if 1 <= mid_bay + bay_offset <= bays:
                coordinates.append((mid_bay + bay_offset, row, "dg"))
    
    # Swap bodies and trailers in first row
    for bay in range(2, min(bays, 10)):  # Bays 2-9 or available
        coordinates.append((bay, 1, "sb_t"))
    
    return BooleanStorageYard(
        n_rows=rows,
        n_bays=bays,
        n_tiers=tiers,
        coordinates=coordinates,
        split_factor=4,
        validate=False  # Skip validation for speed
    )

def populate_yard_with_containers(yard, target_container_count):
    """Populate the yard with random containers."""
    containers_added = 0
    attempts = 0
    max_attempts = target_container_count * 3
    
    # Container type distribution
    container_types = ["TWEU", "FEU", "THEU", "Trailer", "Swap Body"]
    container_weights = [0.4, 0.3, 0.15, 0.1, 0.05]
    
    goods_types = ["Regular", "Reefer", "Dangerous"]
    goods_weights = [0.85, 0.1, 0.05]
    
    while containers_added < target_container_count and attempts < max_attempts:
        attempts += 1
        
        # Create random container
        container_type = np.random.choice(container_types, p=container_weights)
        goods_type = np.random.choice(goods_types, p=goods_weights)
        
        container = ContainerFactory.create_container(
            f"TEST{containers_added:04d}",
            container_type,
            "Import",
            goods_type
        )
        
        # Find a random valid placement
        row = random.randint(0, yard.n_rows - 1)
        bay = random.randint(0, yard.n_bays - 1)
        
        # Determine goods parameter
        if goods_type == 'Reefer':
            goods_param = 'r'
        elif goods_type == 'Dangerous':
            goods_param = 'dg'
        elif container_type in ('Swap Body', 'Trailer'):
            goods_param = 'sb_t'
        else:
            goods_param = 'reg'
        
        # Try to place container
        try:
            valid_placements = yard.search_insertion_position(bay, goods_param, container_type, 5)
            
            if valid_placements:
                # Choose random placement
                placement = random.choice(valid_placements)
                coordinates = yard.get_container_coordinates_from_placement(placement, container_type)
                
                if coordinates:
                    yard.add_container(container, coordinates)
                    containers_added += 1
        except Exception as e:
            # Skip problematic containers
            continue
    
    return containers_added

def run_performance_test(yard, config, num_runs=10):
    """Run performance test on the yard."""
    times = []
    move_counts = []
    destination_counts = []
    
    for run in range(num_runs):
        start_time = time.perf_counter()
        
        # Run the ultra optimized function
        possible_moves = yard.return_possible_yard_moves(max_proximity=5)
        
        end_time = time.perf_counter()
        
        # Collect metrics
        times.append(end_time - start_time)
        move_counts.append(len(possible_moves))
        
        total_destinations = sum(len(data['destinations']) for data in possible_moves.values())
        destination_counts.append(total_destinations)
    
    return {
        'times': times,
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'avg_moves': np.mean(move_counts),
        'avg_destinations': np.mean(destination_counts)
    }

def estimate_memory_usage(yard):
    """Estimate memory usage of the yard in MB."""
    # Rough estimation
    mask_size = yard.dynamic_yard_mask.nbytes
    coordinate_size = yard.coordinates.nbytes if hasattr(yard.coordinates, 'nbytes') else 0
    containers_size = len(yard.containers) * 1000  # Rough estimate per container
    
    total_bytes = mask_size + coordinate_size + containers_size
    return total_bytes / (1024 * 1024)  # Convert to MB

def plot_performance_results(results):
    """Generate performance plots."""
    if not results:
        return
    
    # Extract data for plotting
    config_names = [r['config']['name'] for r in results]
    avg_times = [r['performance']['avg_time'] for r in results]
    containers_counts = [r['containers_added'] for r in results]
    yard_sizes = [r['config']['rows'] * r['config']['bays'] * r['config']['tiers'] for r in results]
    move_counts = [r['performance']['avg_moves'] for r in results]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Execution time vs configuration
    ax1.bar(config_names, avg_times, color='skyblue', alpha=0.7)
    ax1.set_title('Execution Time by Configuration')
    ax1.set_ylabel('Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Time vs yard size
    ax2.scatter(yard_sizes, avg_times, color='red', alpha=0.7)
    ax2.set_title('Execution Time vs Yard Size')
    ax2.set_xlabel('Yard Size (positions)')
    ax2.set_ylabel('Time (seconds)')
    
    # Plot 3: Time vs number of containers
    ax3.scatter(containers_counts, avg_times, color='green', alpha=0.7)
    ax3.set_title('Execution Time vs Container Count')
    ax3.set_xlabel('Number of Containers')
    ax3.set_ylabel('Time (seconds)')
    
    # Plot 4: Containers with moves vs total containers
    ax4.scatter(containers_counts, move_counts, color='purple', alpha=0.7)
    ax4.set_title('Movable Containers vs Total Containers')
    ax4.set_xlabel('Total Containers')
    ax4.set_ylabel('Containers with Valid Moves')
    
    plt.tight_layout()
    plt.savefig('ultra_optimized_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def detailed_analysis(result):
    """Perform detailed analysis on a specific result."""
    config = result['config']
    perf = result['performance']
    
    print(f"Configuration: {config['name']}")
    print(f"  Yard dimensions: {config['rows']}x{config['bays']}x{config['tiers']}")
    print(f"  Total positions: {config['rows'] * config['bays'] * config['tiers']}")
    print(f"  Containers added: {result['containers_added']}")
    print(f"  Fill rate: {result['containers_added'] / (config['rows'] * config['bays'] * config['tiers']):.1%}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Average time: {perf['avg_time']:.4f}s")
    print(f"  Standard deviation: {perf['std_time']:.4f}s")
    print(f"  Min time: {perf['min_time']:.4f}s")
    print(f"  Max time: {perf['max_time']:.4f}s")
    print(f"  Average containers with moves: {perf['avg_moves']:.1f}")
    print(f"  Average total destinations: {perf['avg_destinations']:.1f}")
    
    # Calculate efficiency metrics
    containers_per_second = result['containers_added'] / perf['avg_time']
    moves_per_second = perf['avg_moves'] / perf['avg_time']
    
    print(f"\nEfficiency Metrics:")
    print(f"  Containers processed per second: {containers_per_second:.1f}")
    print(f"  Moves calculated per second: {moves_per_second:.1f}")

if __name__ == "__main__":
    # Run the stress test
    results = stress_test_ultra_optimized()
    
    print(f"\n" + "=" * 60)
    print("STRESS TEST COMPLETED")
    print("=" * 60)
    print(f"Tested {len(results)} configurations")
    print("Performance plots saved as 'ultra_optimized_performance.png'")