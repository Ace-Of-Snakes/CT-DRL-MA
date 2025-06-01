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

# if __name__ == "__main__":
#     # Run both quick and full tests
#     print("Starting Container Placement Performance Tests...\n")
    
#     # Quick test first
#     quick_results = run_quick_stress_test()
    
#     print("\n" + "="*80 + "\n")
    
#     # Full test if quick test performs well
#     max_search_time = max(quick_results[count]['max_search_time'] for count in quick_results.keys())
    
#     if max_search_time < 0.01:  # Less than 10ms max search time
#         print("✅ Quick test performance is good - proceeding with full test...")
#         full_results = run_full_stress_test()
#     else:
#         print(f"⚠️  Quick test shows search times up to {max_search_time*1000:.1f}ms")
#         print("Consider optimization before running full scale test")

if __name__ == "__main__":
    print("🚀 ENHANCED CONTAINER PLACEMENT PERFORMANCE TESTS")
    print("=" * 80)
    
    # Run optimization benchmark first
    run_optimization_benchmark()
    
    print("\n" + "="*80)
    
    # Run comprehensive optimization test
    optimization_results = run_comprehensive_optimization_test()
    
    print(f"\n💡 FINAL RECOMMENDATIONS:")
    
    # Analyze results and provide recommendations
    all_times = []
    for yard_data in optimization_results.values():
        for methods in yard_data.values():
            all_times.extend(methods.values())
    
    if all_times:
        min_time = min(all_times)
        max_time = max(all_times)
        avg_time = np.mean(all_times)
        
        print(f"  ⏱️  Optimized performance range: {min_time*1000:.2f} - {max_time*1000:.2f}ms")
        print(f"  📊 Average optimized time: {avg_time*1000:.2f}ms")
        
        if avg_time < 0.005:  # < 5ms
            print(f"  🎉 EXCELLENT: Suitable for real-time interactive applications")
        elif avg_time < 0.020:  # < 20ms  
            print(f"  ✅ GOOD: Suitable for most terminal operations")
        else:
            print(f"  ⚠️  MODERATE: Consider further optimization for high-frequency use")
    
    print(f"  🚀 Use 'Ultra-Optimized' method for best performance")
    print(f"  📝 Consider proximity <= 8 for optimal speed/quality balance")