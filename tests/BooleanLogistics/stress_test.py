import time
import random
import numpy as np
import sys
import tracemalloc
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta

# Import required components
from simulation.terminal_components.Container import Container, ContainerFactory
from simulation.terminal_components.Train import Train
from simulation.terminal_components.Truck import Truck
from simulation.terminal_components.BooleanStorage import BooleanStorageYard
from simulation.terminal_components.BooleanLogistics import BooleanLogistics

class BoolLogisticsStressTest:
    """
    Comprehensive stress test for BooleanLogistics class.
    Tests performance, scalability, and correctness under various load conditions.
    """
    
    def __init__(self, logistics_config: Dict = None):
        """
        Initialize stress test with logistics configuration.
        
        Args:
            logistics_config: Dictionary with logistics parameters
        """
        self.logistics_config = logistics_config or {
            'n_rows': 15,
            'n_railtracks': 6,
            'split_factor': 4,
            'yard_config': {
                'n_rows': 15,
                'n_bays': 40,
                'n_tiers': 5,
                'split_factor': 4
            }
        }
        
        # Test scenarios configuration
        self.train_config = {
            'min_wagons': 3,
            'max_wagons': 12,
            'container_fill_rate': 0.6,  # 60% of wagons have containers
            'pickup_request_rate': 0.4   # 40% of wagons have pickup requests
        }
        
        self.truck_config = {
            'container_fill_rate': 0.3,  # 30% of trucks arrive with containers
            'pickup_request_rate': 0.7   # 70% of trucks want to pick up containers
        }
        
        # Container type distributions
        self.container_types = {
            'TWEU': 0.35,
            'FEU': 0.40, 
            'THEU': 0.15,
            'FFEU': 0.05,
            'Swap Body': 0.03,
            'Trailer': 0.02
        }
        
        self.goods_types = {
            'Regular': 0.75,
            'Reefer': 0.15,
            'Dangerous': 0.10
        }
        
        # Results storage
        self.results = {}
        
    def create_test_logistics_system(self) -> Tuple[BooleanLogistics, BooleanStorageYard]:
        """Create a test logistics system with associated yard."""
        
        # Create yard first
        yard_config = self.logistics_config['yard_config']
        coordinates = self._generate_special_coordinates(
            yard_config['n_rows'], 
            yard_config['n_bays']
        )
        
        yard = BooleanStorageYard(
            n_rows=yard_config['n_rows'],
            n_bays=yard_config['n_bays'], 
            n_tiers=yard_config['n_tiers'],
            coordinates=coordinates,
            split_factor=yard_config['split_factor'],
            validate=False
        )
        
        # Create logistics system
        logistics = BooleanLogistics(
            n_rows=self.logistics_config['n_rows'],
            n_railtracks=self.logistics_config['n_railtracks'],
            split_factor=self.logistics_config['split_factor'],
            yard=yard,
            validate=False
        )
        
        return logistics, yard
    
    def _generate_special_coordinates(self, n_rows: int, n_bays: int) -> List[Tuple]:
        """Generate realistic special area coordinates."""
        coordinates = []
        
        # Reefers at both ends
        for row in range(1, n_rows + 1):
            coordinates.extend([
                (1, row, "r"), (2, row, "r"),
                (n_bays-1, row, "r"), (n_bays, row, "r")
            ])
        
        # Dangerous goods in middle section
        dg_start = n_bays // 3
        dg_end = 2 * n_bays // 3
        for row in range(3, min(n_rows, 6) + 1):
            for bay in range(dg_start, dg_end + 1, 3):
                coordinates.append((bay, row, "dg"))
        
        # Swap bodies and trailers near parking
        for bay in range(5, min(n_bays - 5, 25)):
            coordinates.append((bay, 1, "sb_t"))
        
        return coordinates
    
    def generate_random_trains(self, count: int, container_pool: List[Container] = None) -> List[Train]:
        """Generate random trains with realistic configurations."""
        trains = []
        container_pool = container_pool or []
        container_idx = 0
        
        for i in range(count):
            # Random number of wagons
            num_wagons = random.randint(
                self.train_config['min_wagons'],
                self.train_config['max_wagons']
            )
            
            train = Train(f"TRAIN_{i:04d}", num_wagons=num_wagons)
            
            # Set departure time (random future time)
            departure_hours = random.uniform(2, 24)  # 2-24 hours from now
            train.departure_time = datetime.now() + timedelta(hours=departure_hours)
            
            # Add containers to some wagons
            wagons_with_containers = int(num_wagons * self.train_config['container_fill_rate'])
            wagon_indices = random.sample(range(num_wagons), wagons_with_containers)
            
            for wagon_idx in wagon_indices:
                # Add 1-2 containers per wagon
                containers_per_wagon = random.randint(1, 2)
                for _ in range(containers_per_wagon):
                    if container_idx < len(container_pool):
                        container = container_pool[container_idx]
                        train.wagons[wagon_idx].add_container(container)
                        container_idx += 1
            
            # Add pickup requests to other wagons
            wagons_with_pickups = int(num_wagons * self.train_config['pickup_request_rate'])
            available_wagons = [idx for idx in range(num_wagons) if idx not in wagon_indices]
            
            if available_wagons:
                pickup_wagon_indices = random.sample(
                    available_wagons, 
                    min(wagons_with_pickups, len(available_wagons))
                )
                
                for wagon_idx in pickup_wagon_indices:
                    # Each wagon requests 1-3 containers
                    pickup_count = random.randint(1, 3)
                    for j in range(pickup_count):
                        pickup_id = f"PICKUP_T{i:04d}_W{wagon_idx:02d}_{j:02d}"
                        train.wagons[wagon_idx].add_pickup_container(pickup_id)
            
            trains.append(train)
        
        return trains
    
    def generate_random_trucks(self, count: int, container_pool: List[Container] = None, 
                             pickup_requests: List[str] = None) -> List[Truck]:
        """Generate random trucks with realistic configurations."""
        trucks = []
        container_pool = container_pool or []
        pickup_requests = pickup_requests or []
        container_idx = 0
        pickup_idx = 0
        
        for i in range(count):
            truck = Truck(f"TRUCK_{i:04d}")
            
            # Some trucks arrive with containers (delivery trucks)
            if random.random() < self.truck_config['container_fill_rate']:
                containers_count = random.randint(1, 2)
                for _ in range(containers_count):
                    if container_idx < len(container_pool):
                        container = container_pool[container_idx]
                        truck.add_container(container)
                        container_idx += 1
            
            # Most trucks want to pick up containers
            if random.random() < self.truck_config['pickup_request_rate']:
                pickup_count = random.randint(1, 3)
                for _ in range(pickup_count):
                    if pickup_idx < len(pickup_requests):
                        truck.add_pickup_container_id(pickup_requests[pickup_idx])
                        pickup_idx += 1
                    else:
                        # Generate new pickup request
                        pickup_id = f"PICKUP_TR{i:04d}_{_:02d}"
                        truck.add_pickup_container_id(pickup_id)
            
            trucks.append(truck)
        
        return trucks
    
    def generate_random_containers(self, count: int) -> List[Container]:
        """Generate random containers with realistic distributions."""
        containers = []
        
        for i in range(count):
            # Select container type
            container_type = random.choices(
                list(self.container_types.keys()),
                weights=list(self.container_types.values())
            )[0]
            
            # Select goods type
            goods_type = random.choices(
                list(self.goods_types.keys()),
                weights=list(self.goods_types.values())
            )[0]
            
            # Adjust for special cases
            if container_type in ['Swap Body', 'Trailer']:
                goods_type = 'Regular'  # These don't typically handle special goods
            
            container = ContainerFactory.create_container(
                container_id=f"CONT_{i:05d}",
                container_type=container_type,
                direction=random.choice(["Import", "Export"]),
                goods_type=goods_type,
                weight=random.randint(5000, 30000)
            )
            containers.append(container)
        
        return containers
    
    def populate_yard_with_containers(self, yard: BooleanStorageYard, 
                                    containers: List[Container]) -> int:
        """Populate yard with containers that can be picked up."""
        placed_count = 0
        
        for container in containers:
            # Determine goods type for placement
            goods_map = {
                'Regular': 'reg',
                'Reefer': 'r',
                'Dangerous': 'dg'
            }
            
            if container.container_type in ['Swap Body', 'Trailer']:
                goods_code = 'sb_t'
            else:
                goods_code = goods_map.get(container.goods_type, 'reg')
            
            # Try to place container
            try:
                bay = random.randint(0, yard.n_bays - 1)
                positions = yard.search_insertion_position(
                    bay, goods_code, container.container_type, max_proximity=5
                )
                
                if positions:
                    placement = random.choice(positions)
                    coordinates = yard.get_container_coordinates_from_placement(
                        placement, container.container_type
                    )
                    yard.add_container(container, coordinates)
                    placed_count += 1
            except Exception:
                continue
        
        return placed_count
    
    def run_logistics_operations_test(self, logistics: BooleanLogistics, 
                                    trains: List[Train], trucks: List[Truck]) -> Dict:
        """Run comprehensive logistics operations test."""
        
        print("  🚂 Testing logistics operations...")
        
        results = {
            'queue_operations': {},
            'placement_operations': {},
            'move_finding': {},
            'auto_assignment': {},
            'reorganization': {},
            'tensor_conversion': {}
        }
        
        # Test 1: Queue Operations
        start_time = time.perf_counter()
        
        for train in trains:
            logistics.add_train_to_queue(train)
        
        for truck in trucks:
            logistics.add_truck_to_queue(truck)
        
        queue_time = time.perf_counter() - start_time
        
        results['queue_operations'] = {
            'time': queue_time,
            'trains_queued': len(trains),
            'trucks_queued': len(trucks),
            'trains_in_queue': logistics.trains.size(),
            'trucks_in_queue': logistics.trucks.size()
        }
        
        # Test 2: Vehicle Placement
        start_time = time.perf_counter()
        
        trains_placed = logistics.process_current_trains()
        trucks_placed = logistics.process_current_trucks()
        
        placement_time = time.perf_counter() - start_time
        
        results['placement_operations'] = {
            'time': placement_time,
            'trains_placed': trains_placed,
            'trucks_placed': trucks_placed,
            'trains_success_rate': trains_placed / len(trains) if trains else 0,
            'trucks_success_rate': trucks_placed / len(trucks) if trucks else 0
        }
        
        # Test 3: Auto Assignment
        start_time = time.perf_counter()
        
        assignments = logistics.auto_assign_trucks()
        
        assignment_time = time.perf_counter() - start_time
        
        results['auto_assignment'] = {
            'time': assignment_time,
            'assignments_made': len(assignments),
            'total_container_assignments': sum(len(v) for v in assignments.values())
        }
        
        # Test 4: Move Finding (Critical Operation) - Compare optimized vs original
        move_times = []
        move_times_optimized = []
        move_counts = []
        
        for i in range(5):  # Multiple runs for averaging
            # Test optimized version
            start_time = time.perf_counter()
            moves = logistics.find_moves_optimized()
            end_time = time.perf_counter()
            move_times_optimized.append(end_time - start_time)
            
            # Test original version for comparison (if available)
            try:
                start_time = time.perf_counter()
                # Temporarily use the old method name if it exists
                if hasattr(logistics, 'find_moves_original'):
                    moves_orig = logistics.find_moves_original()
                else:
                    moves_orig = moves  # Use optimized as fallback
                end_time = time.perf_counter()
                move_times.append(end_time - start_time)
            except:
                move_times.append(move_times_optimized[-1])  # Use optimized time as fallback
            
            move_counts.append(len(moves))
        
        # Calculate performance improvement
        avg_time_optimized = np.mean(move_times_optimized)
        avg_time_original = np.mean(move_times)
        speedup = avg_time_original / avg_time_optimized if avg_time_optimized > 0 else 1.0
        
        results['move_finding'] = {
            'times': move_times,
            'times_optimized': move_times_optimized,
            'avg_time': avg_time_optimized,  # Use optimized time as primary metric
            'avg_time_original': avg_time_original,
            'speedup': speedup,
            'max_time': np.max(move_times_optimized),
            'min_time': np.min(move_times_optimized),
            'std_time': np.std(move_times_optimized),
            'avg_moves_found': np.mean(move_counts),
            'moves_per_second': np.mean(move_counts) / avg_time_optimized if avg_time_optimized > 0 else 0
        }
        
        # Test 5: Reorganization
        start_time = time.perf_counter()
        
        reorganization_summary = logistics.reorganize_logistics()
        
        reorganization_time = time.perf_counter() - start_time
        
        results['reorganization'] = {
            'time': reorganization_time,
            'summary': reorganization_summary.tolist() if hasattr(reorganization_summary, 'tolist') else reorganization_summary
        }
        
        # Test 6: Tensor Conversion (Critical for DRL)
        tensor_times = {}
        tensor_shapes = {}
        
        # Test each tensor conversion
        tensor_operations = [
            ('rail_state', lambda: logistics.get_rail_state_tensor()),
            ('parking_state', lambda: logistics.get_parking_state_tensor()),
            ('train_properties', lambda: logistics.get_train_properties_tensor()),
            ('truck_properties', lambda: logistics.get_truck_properties_tensor()),
            ('queue_state', lambda: logistics.get_queue_state_tensor())
        ]
        
        for name, tensor_func in tensor_operations:
            start_time = time.perf_counter()
            
            try:
                tensor = tensor_func()
                end_time = time.perf_counter()
                
                tensor_times[name] = end_time - start_time
                
                if hasattr(tensor, 'shape'):
                    tensor_shapes[name] = tensor.shape
                elif isinstance(tensor, dict):
                    tensor_shapes[name] = {k: v.shape for k, v in tensor.items()}
                else:
                    tensor_shapes[name] = str(type(tensor))
                    
            except Exception as e:
                tensor_times[name] = float('inf')
                tensor_shapes[name] = f"Error: {e}"
        
        results['tensor_conversion'] = {
            'times': tensor_times,
            'shapes': tensor_shapes,
            'total_time': sum(t for t in tensor_times.values() if t != float('inf'))
        }
        
        return results
    
    def run_stress_test(self, test_configs: List[Dict] = None) -> Dict:
        """
        Run comprehensive stress test with different configurations.
        
        Args:
            test_configs: List of test configurations
            
        Returns:
            Dictionary with test results
        """
        
        if test_configs is None:
            test_configs = [
                {
                    'name': 'Small',
                    'trains': 5,
                    'trucks': 15,
                    'containers_in_yard': 50,
                    'logistics_config': {
                        'n_rows': 10,
                        'n_railtracks': 4,
                        'split_factor': 4,
                        'yard_config': {
                            'n_rows': 10,
                            'n_bays': 20,
                            'n_tiers': 3,
                            'split_factor': 4
                        }
                    }
                },
                {
                    'name': 'Medium',
                    'trains': 12,
                    'trucks': 35,
                    'containers_in_yard': 200,
                    'logistics_config': {
                        'n_rows': 15,
                        'n_railtracks': 6,
                        'split_factor': 4,
                        'yard_config': {
                            'n_rows': 15,
                            'n_bays': 35,
                            'n_tiers': 4,
                            'split_factor': 4
                        }
                    }
                },
                {
                    'name': 'Large',
                    'trains': 25,
                    'trucks': 60,
                    'containers_in_yard': 500,
                    'logistics_config': {
                        'n_rows': 20,
                        'n_railtracks': 8,
                        'split_factor': 4,
                        'yard_config': {
                            'n_rows': 20,
                            'n_bays': 50,
                            'n_tiers': 5,
                            'split_factor': 4
                        }
                    }
                },
                {
                    'name': 'Extra Large',
                    'trains': 50,
                    'trucks': 120,
                    'containers_in_yard': 1000,
                    'logistics_config': {
                        'n_rows': 25,
                        'n_railtracks': 10,
                        'split_factor': 4,
                        'yard_config': {
                            'n_rows': 25,
                            'n_bays': 60,
                            'n_tiers': 6,
                            'split_factor': 4
                        }
                    }
                },
                {
                    'name': 'Massive',
                    'trains': 100,
                    'trucks': 250,
                    'containers_in_yard': 2000,
                    'logistics_config': {
                        'n_rows': 30,
                        'n_railtracks': 12,
                        'split_factor': 4,
                        'yard_config': {
                            'n_rows': 30,
                            'n_bays': 80,
                            'n_tiers': 6,
                            'split_factor': 4
                        }
                    }
                }
            ]
        
        print("🚢 BOOLEAN LOGISTICS STRESS TEST 🚢")
        print("=" * 80)
        
        all_results = {}
        
        for config in test_configs:
            print(f"\n🏗️  Testing {config['name']} Configuration")
            print("-" * 50)
            print(f"Trains: {config['trains']}, Trucks: {config['trucks']}")
            print(f"Containers in yard: {config['containers_in_yard']}")
            print(f"Yard: {config['logistics_config']['yard_config']['n_rows']}x"
                  f"{config['logistics_config']['yard_config']['n_bays']}x"
                  f"{config['logistics_config']['yard_config']['n_tiers']}")
            
            # Start memory tracking
            tracemalloc.start()
            test_start_time = time.perf_counter()
            
            # Update configuration
            self.logistics_config = config['logistics_config']
            
            # Create logistics system
            logistics, yard = self.create_test_logistics_system()
            
            # Generate test data
            print("  📦 Generating test data...")
            
            # Generate containers for yard
            yard_containers = self.generate_random_containers(config['containers_in_yard'])
            containers_placed = self.populate_yard_with_containers(yard, yard_containers)
            print(f"    Placed {containers_placed}/{config['containers_in_yard']} containers in yard")
            
            # OPTIMIZATION: Sync yard index after population
            logistics.sync_yard_index()
            print(f"    Synchronized yard index with {len(logistics.yard_container_index)} containers")
            
            # Generate containers for vehicles
            vehicle_containers = self.generate_random_containers(config['trains'] * 4 + config['trucks'])
            
            # Create pickup requests list from containers in yard
            pickup_requests = [c.container_id for c in yard_containers[:min(100, len(yard_containers))]]
            
            # Generate trains and trucks
            trains = self.generate_random_trains(config['trains'], vehicle_containers[:config['trains'] * 3])
            trucks = self.generate_random_trucks(
                config['trucks'], 
                vehicle_containers[config['trains'] * 3:],
                pickup_requests
            )
            
            print(f"    Generated {len(trains)} trains, {len(trucks)} trucks")
            
            # Run logistics operations test
            operations_results = self.run_logistics_operations_test(logistics, trains, trucks)
            
            # Calculate total test time and memory
            total_test_time = time.perf_counter() - test_start_time
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Compile results
            config_results = {
                'config': config,
                'containers_placed_in_yard': containers_placed,
                'operations': operations_results,
                'total_test_time': total_test_time,
                'memory_usage_mb': current_memory / (1024 * 1024),
                'peak_memory_mb': peak_memory / (1024 * 1024)
            }
            
            all_results[config['name']] = config_results
            
            # Print results
            self.print_test_results(config['name'], config_results)
        
        # Generate summary and analysis
        print("\n" + "=" * 80)
        print("📊 STRESS TEST SUMMARY")
        print("=" * 80)
        
        self.print_summary_table(all_results)
        self.analyze_performance_scaling(all_results)
        
        # Generate plots
        self.plot_performance_results(all_results)
        
        return all_results
    
    def print_test_results(self, config_name: str, results: Dict):
        """Print detailed results for a single test configuration."""
        print(f"\n📊 Results for {config_name}:")
        
        ops = results['operations']
        
        # Queue operations
        queue_ops = ops['queue_operations']
        print(f"  Queue Operations:")
        print(f"    ✅ Queued {queue_ops['trains_queued']} trains, {queue_ops['trucks_queued']} trucks")
        print(f"    ⏱️  Time: {queue_ops['time']*1000:.2f}ms")
        
        # Placement operations
        place_ops = ops['placement_operations']
        print(f"  Vehicle Placement:")
        print(f"    🚂 Trains: {place_ops['trains_placed']} placed ({place_ops['trains_success_rate']:.1%} success)")
        print(f"    🚛 Trucks: {place_ops['trucks_placed']} placed ({place_ops['trucks_success_rate']:.1%} success)")
        print(f"    ⏱️  Time: {place_ops['time']*1000:.2f}ms")
        
        # Move finding (most critical)
        move_ops = ops['move_finding']
        speedup_text = f" ({move_ops.get('speedup', 1.0):.1f}x faster)" if 'speedup' in move_ops else ""
        print(f"  Move Finding (CRITICAL):")
        print(f"    🔍 Average: {move_ops['avg_time']*1000:.2f}ms ({move_ops['avg_moves_found']:.1f} moves){speedup_text}")
        print(f"    📈 Max: {move_ops['max_time']*1000:.2f}ms, Min: {move_ops['min_time']*1000:.2f}ms")
        print(f"    🚀 Moves/sec: {move_ops['moves_per_second']:.1f}")
        
        # Tensor conversion
        tensor_ops = ops['tensor_conversion']
        print(f"  Tensor Conversion:")
        print(f"    ⚡ Total time: {tensor_ops['total_time']*1000:.2f}ms")
        
        # Memory usage
        print(f"  💾 Memory: {results['memory_usage_mb']:.1f}MB (peak: {results['peak_memory_mb']:.1f}MB)")
        print(f"  🕐 Total test time: {results['total_test_time']:.2f}s")
    
    def print_summary_table(self, all_results: Dict):
        """Print a summary table of all test results."""
        print(f"{'Config':<12} {'Move Find(ms)':<15} {'Moves/sec':<12} {'Success%':<10} {'Memory(MB)':<12} {'Total(s)':<10}")
        print("-" * 85)
        
        for config_name, results in all_results.items():
            move_ops = results['operations']['move_finding']
            place_ops = results['operations']['placement_operations']
            
            avg_success = (place_ops['trains_success_rate'] + place_ops['trucks_success_rate']) / 2
            
            print(f"{config_name:<12} {move_ops['avg_time']*1000:<15.2f} {move_ops['moves_per_second']:<12.1f} "
                  f"{avg_success*100:<10.1f} {results['peak_memory_mb']:<12.1f} {results['total_test_time']:<10.2f}")
    
    def analyze_performance_scaling(self, all_results: Dict):
        """Analyze how performance scales with system size."""
        print(f"\n🔬 Performance Scaling Analysis:")
        
        configs = list(all_results.keys())
        if len(configs) < 2:
            print("  Need at least 2 configurations for scaling analysis")
            return
        
        # Extract scaling data
        vehicle_counts = []
        move_times = []
        
        for config_name in configs:
            results = all_results[config_name]
            config = results['config']
            
            total_vehicles = config['trains'] + config['trucks']
            avg_move_time = results['operations']['move_finding']['avg_time']
            
            vehicle_counts.append(total_vehicles)
            move_times.append(avg_move_time)
        
        # Analyze scaling
        base_vehicles = vehicle_counts[0]
        base_time = move_times[0]
        
        print(f"  Base case ({base_vehicles} vehicles): {base_time*1000:.2f}ms move finding")
        
        for i in range(1, len(configs)):
            vehicles = vehicle_counts[i]
            time_taken = move_times[i]
            
            scaling_factor = time_taken / base_time
            theoretical_linear = vehicles / base_vehicles
            
            efficiency = theoretical_linear / scaling_factor if scaling_factor > 0 else 0
            
            print(f"  {vehicles} vehicles: {time_taken*1000:.2f}ms ({scaling_factor:.2f}x vs {theoretical_linear:.1f}x linear, {efficiency:.2f} efficiency)")
        
        # Performance recommendations
        max_move_time = max(move_times)
        latest_move_time = move_times[-1]
        
        print(f"\n💡 Performance Recommendations:")
        
        if max_move_time > 0.050:  # > 50ms
            print(f"  ⚠️  Maximum move finding time: {max_move_time*1000:.1f}ms - may impact real-time operations")
        elif max_move_time > 0.020:  # > 20ms
            print(f"  ⚠️  Move finding time: {max_move_time*1000:.1f}ms - monitor for larger systems")
        else:
            print(f"  ✅ Move finding performance excellent: {max_move_time*1000:.1f}ms maximum")
        
        if latest_move_time > 0.100:  # > 100ms for largest config
            print(f"  🔴 Large system performance: {latest_move_time*1000:.1f}ms - consider optimization")
        elif latest_move_time > 0.050:  # > 50ms
            print(f"  🟡 Large system performance: {latest_move_time*1000:.1f}ms - acceptable but monitor")
        else:
            print(f"  🟢 Large system performance: {latest_move_time*1000:.1f}ms - excellent scalability")
    
    def plot_performance_results(self, all_results: Dict):
        """Generate performance plots."""
        if len(all_results) < 2:
            print("  Not enough data points for plotting")
            return
        
        try:
            # Extract data for plotting
            config_names = list(all_results.keys())
            vehicle_counts = []
            move_times = []
            memory_usage = []
            success_rates = []
            moves_per_second = []
            
            for config_name in config_names:
                results = all_results[config_name]
                config = results['config']
                
                vehicle_counts.append(config['trains'] + config['trucks'])
                move_times.append(results['operations']['move_finding']['avg_time'] * 1000)  # Convert to ms
                memory_usage.append(results['peak_memory_mb'])
                
                place_ops = results['operations']['placement_operations']
                avg_success = (place_ops['trains_success_rate'] + place_ops['trucks_success_rate']) / 2
                success_rates.append(avg_success * 100)
                
                moves_per_second.append(results['operations']['move_finding']['moves_per_second'])
            
            # Create plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Move finding time vs vehicle count
            ax1.plot(vehicle_counts, move_times, 'b-o', linewidth=2, markersize=8)
            ax1.set_title('Move Finding Performance vs Vehicle Count', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Total Vehicles (Trains + Trucks)')
            ax1.set_ylabel('Average Move Finding Time (ms)')
            ax1.grid(True, alpha=0.3)
            
            # Add performance threshold lines
            ax1.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='20ms threshold')
            ax1.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50ms threshold')
            ax1.legend()
            
            # Plot 2: Memory usage vs vehicle count
            ax2.plot(vehicle_counts, memory_usage, 'g-o', linewidth=2, markersize=8)
            ax2.set_title('Memory Usage vs Vehicle Count', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Total Vehicles (Trains + Trucks)')
            ax2.set_ylabel('Peak Memory Usage (MB)')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Success rate vs vehicle count
            ax3.plot(vehicle_counts, success_rates, 'r-o', linewidth=2, markersize=8)
            ax3.set_title('Vehicle Placement Success Rate', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Total Vehicles (Trains + Trucks)')
            ax3.set_ylabel('Average Success Rate (%)')
            ax3.set_ylim(0, 100)
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Moves per second vs vehicle count
            ax4.plot(vehicle_counts, moves_per_second, 'm-o', linewidth=2, markersize=8)
            ax4.set_title('Move Detection Throughput', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Total Vehicles (Trains + Trucks)')
            ax4.set_ylabel('Moves Detected per Second')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout(pad=3.0)
            plt.savefig('bool_logistics_stress_test_results.png', dpi=300, bbox_inches='tight')
            print(f"\n📈 Performance plots saved as 'bool_logistics_stress_test_results.png'")
            
        except Exception as e:
            print(f"  ⚠️  Could not generate plots: {e}")

def run_quick_logistics_stress_test():
    """Run a quick stress test with small configurations."""
    print("🚀 Quick BoolLogistics Stress Test")
    
    quick_configs = [
        {
            'name': 'Tiny',
            'trains': 3,
            'trucks': 8,
            'containers_in_yard': 25,
            'logistics_config': {
                'n_rows': 8,
                'n_railtracks': 3,
                'split_factor': 4,
                'yard_config': {
                    'n_rows': 8,
                    'n_bays': 15,
                    'n_tiers': 3,
                    'split_factor': 4
                }
            }
        },
        {
            'name': 'Small',
            'trains': 6,
            'trucks': 18,
            'containers_in_yard': 75,
            'logistics_config': {
                'n_rows': 12,
                'n_railtracks': 4,
                'split_factor': 4,
                'yard_config': {
                    'n_rows': 12,
                    'n_bays': 25,
                    'n_tiers': 4,
                    'split_factor': 4
                }
            }
        }
    ]
    
    tester = BoolLogisticsStressTest()
    results = tester.run_stress_test(quick_configs)
    return results

def run_full_logistics_stress_test():
    """Run the full stress test with all configurations."""
    print("🏭 Full Scale BoolLogistics Stress Test")
    
    tester = BoolLogisticsStressTest()
    results = tester.run_stress_test()  # Uses default configs
    return results

def run_realistic_terminal_simulation():
    """Run a realistic terminal simulation with complex scenarios."""
    print("🌊 REALISTIC TERMINAL SIMULATION 🌊")
    print("=" * 60)
    
    # Realistic port terminal configuration
    realistic_config = {
        'name': 'Realistic Port',
        'trains': 40,
        'trucks': 85,
        'containers_in_yard': 800,
        'logistics_config': {
            'n_rows': 20,
            'n_railtracks': 8,
            'split_factor': 4,
            'yard_config': {
                'n_rows': 20,
                'n_bays': 60,
                'n_tiers': 5,
                'split_factor': 4
            }
        }
    }
    
    tester = BoolLogisticsStressTest()
    
    # Run multiple iterations to test consistency
    print("Running 3 iterations to test consistency...")
    
    iteration_results = []
    for i in range(3):
        print(f"\n🔄 Iteration {i+1}/3")
        results = tester.run_stress_test([realistic_config])
        iteration_results.append(results['Realistic Port']['operations']['move_finding']['avg_time'])
    
    # Analyze consistency
    avg_time = np.mean(iteration_results)
    std_time = np.std(iteration_results)
    
    print(f"\n📊 Consistency Analysis:")
    print(f"  Average move finding time: {avg_time*1000:.2f}ms")
    print(f"  Standard deviation: {std_time*1000:.2f}ms")
    print(f"  Coefficient of variation: {(std_time/avg_time)*100:.1f}%")
    
    if (std_time/avg_time) < 0.1:
        print(f"  ✅ Excellent consistency - performance is stable")
    elif (std_time/avg_time) < 0.2:
        print(f"  ⚠️  Moderate consistency - some variation in performance")
    else:
        print(f"  🔴 Poor consistency - performance varies significantly")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='BoolLogistics Stress Test')
    parser.add_argument('--quick', action='store_true', help='Run quick test')
    parser.add_argument('--full', action='store_true', help='Run full test')
    parser.add_argument('--realistic', action='store_true', help='Run realistic simulation')
    
    args = parser.parse_args()
    
    if args.quick:
        results = run_quick_logistics_stress_test()
    elif args.realistic:
        run_realistic_terminal_simulation()
    else:
        results = run_full_logistics_stress_test()
    
    print(f"\n" + "=" * 80)
    print("🎯 BOOLLOGISTICS STRESS TEST COMPLETED")
    print("=" * 80)
    print("🚀 PERFORMANCE OPTIMIZATIONS APPLIED:")
    print("  ✅ O(1) yard container lookup (eliminates 4D search)")
    print("  ✅ Cached yard positions by container type")
    print("  ✅ Vectorized move detection (reduces nested loops)")
    print("  ✅ Single-pass vehicle container collection")
    print("  ✅ Fast set intersection for pickup requests")
    print("")
    print("Key performance indicators:")
    print("  🟢 < 20ms move finding: Excellent for real-time DRL")
    print("  🟡 20-50ms move finding: Acceptable for most applications")  
    print("  🔴 > 50ms move finding: May need optimization")
    print("\nRecommendation: Monitor move finding performance as it's critical for DRL agent response time.")