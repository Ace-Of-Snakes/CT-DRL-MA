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

if __name__ == "__main__":
    # Run both quick and full tests
    print("Starting Container Placement Performance Tests...\n")
    
    # Quick test first
    quick_results = run_quick_stress_test()
    
    print("\n" + "="*80 + "\n")
    
    # Full test if quick test performs well
    max_search_time = max(quick_results[count]['max_search_time'] for count in quick_results.keys())
    
    if max_search_time < 0.01:  # Less than 10ms max search time
        print("✅ Quick test performance is good - proceeding with full test...")
        full_results = run_full_stress_test()
    else:
        print(f"⚠️  Quick test shows search times up to {max_search_time*1000:.1f}ms")
        print("Consider optimization before running full scale test")