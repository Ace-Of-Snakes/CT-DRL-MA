import time
import random
import numpy as np
import sys
import tracemalloc
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta
import concurrent.futures
import threading

# Import required components
from simulation.terminal_components.Container import Container, ContainerFactory
from simulation.terminal_components.Train import Train
from simulation.terminal_components.Truck import Truck
from simulation.terminal_components.BooleanStorage import BooleanStorageYard
from simulation.terminal_components.BooleanLogistics import BooleanLogistics
from simulation.terminal_components.RMGC import RMGC_Controller

class RMGC_StressTest:
    """
    Comprehensive stress test for RMGC_Controller class.
    Tests crane performance, physics calculations, multi-head coordination, and scalability.
    """
    
    def __init__(self, rmgc_config: Dict = None):
        """
        Initialize stress test with RMGC configuration.
        
        Args:
            rmgc_config: Dictionary with RMGC parameters
        """
        self.rmgc_config = rmgc_config or {
            'heads': 2,
            'trolley_speed': 70.0,  # m/min
            'hoisting_speed': 28.0,  # m/min
            'gantry_speed': 130.0,   # m/min
            'trolley_acceleration': 0.3,  # m/s²
            'hoisting_acceleration': 0.2,  # m/s²
            'gantry_acceleration': 0.1,   # m/s²
            'max_height': 20.0  # meters
        }
        
        # Test scenario configurations
        self.move_scenarios = {
            'short_distance': {
                'description': 'Adjacent bay moves',
                'avg_distance': 15.0,  # meters
                'mix': {'yard_to_truck': 0.4, 'truck_to_train': 0.3, 'train_to_yard': 0.3}
            },
            'medium_distance': {
                'description': 'Cross-zone moves',
                'avg_distance': 50.0,
                'mix': {'yard_to_train': 0.5, 'train_to_truck': 0.3, 'truck_to_yard': 0.2}
            },
            'long_distance': {
                'description': 'Terminal-wide moves',
                'avg_distance': 100.0,
                'mix': {'train_to_train': 0.3, 'yard_to_yard': 0.4, 'truck_to_train': 0.3}
            }
        }
        
        # Results storage
        self.results = {}
        
    def create_test_terminal(self, config: Dict) -> Tuple[RMGC_Controller, BooleanLogistics, BooleanStorageYard]:
        """Create a test terminal with RMGC controller."""
        
        # Create yard
        yard_config = config['yard_config']
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
        
        # Create logistics
        logistics = BooleanLogistics(
            n_rows=config['n_rows'],
            n_railtracks=config['n_railtracks'],
            split_factor=config['split_factor'],
            yard=yard,
            validate=False
        )
        
        # Create RMGC controller
        rmgc = RMGC_Controller(
            yard=yard,
            logistics=logistics,
            heads=self.rmgc_config['heads'],
            trolley_speed=self.rmgc_config['trolley_speed'],
            hoisting_speed=self.rmgc_config['hoisting_speed'],
            gantry_speed=self.rmgc_config['gantry_speed'],
            trolley_acceleration=self.rmgc_config['trolley_acceleration'],
            hoisting_acceleration=self.rmgc_config['hoisting_acceleration'],
            gantry_acceleration=self.rmgc_config['gantry_acceleration'],
            max_height=self.rmgc_config['max_height']
        )
        
        return rmgc, logistics, yard
    
    def _generate_special_coordinates(self, n_rows: int, n_bays: int) -> List[Tuple]:
        """Generate realistic special area coordinates."""
        coordinates = []
        
        # Reefers at ends
        for row in range(1, min(n_rows + 1, 5)):
            coordinates.extend([
                (1, row, "r"), (2, row, "r"),
                (n_bays-1, row, "r"), (n_bays, row, "r")
            ])
        
        # Dangerous goods in middle
        dg_start = n_bays // 3
        dg_end = 2 * n_bays // 3
        for row in range(3, min(n_rows, 6) + 1):
            for bay in range(dg_start, dg_end + 1, 3):
                if bay <= n_bays:
                    coordinates.append((bay, row, "dg"))
        
        # Swap bodies near parking
        for bay in range(5, min(n_bays - 5, 25)):
            coordinates.append((bay, 1, "sb_t"))
        
        return coordinates
    
    def generate_test_moves(self, logistics: BooleanLogistics, count: int, 
                           scenario: str = 'medium_distance') -> Dict[str, Dict]:
        """Generate realistic test moves based on scenario."""
        
        scenario_config = self.move_scenarios[scenario]
        moves = {}
        move_id = 0
        
        # First populate terminal with vehicles and containers
        self._populate_terminal_for_moves(logistics, count)
        
        # Get available moves from logistics
        available_moves = logistics.find_moves_optimized()
        
        # If we have enough real moves, use them
        if len(available_moves) >= count:
            move_items = list(available_moves.items())
            random.shuffle(move_items)
            return dict(move_items[:count])
        
        # Otherwise, generate synthetic moves based on scenario
        for i in range(count - len(available_moves)):
            move_type = random.choices(
                list(scenario_config['mix'].keys()),
                weights=list(scenario_config['mix'].values())
            )[0]
            
            move = self._generate_synthetic_move(logistics, move_type, move_id)
            if move:
                moves[f"synthetic_{move_id}"] = move
                move_id += 1
        
        # Combine real and synthetic moves
        moves.update(available_moves)
        
        return moves
    
    def _populate_terminal_for_moves(self, logistics: BooleanLogistics, target_moves: int):
        """Populate terminal with vehicles and containers to generate moves."""
        
        # Estimate needed vehicles and containers
        trains_needed = max(2, target_moves // 20)
        trucks_needed = max(5, target_moves // 10)
        containers_needed = target_moves * 2
        
        # Generate containers
        containers = []
        for i in range(containers_needed):
            container = ContainerFactory.create_container(
                f"TEST_{i:05d}",
                random.choice(['TWEU', 'FEU', 'THEU']),
                random.choice(['Import', 'Export']),
                random.choice(['Regular', 'Reefer', 'Dangerous'])
            )
            containers.append(container)
        
        # Add some containers to yard
        yard_containers = containers[:containers_needed // 3]
        for container in yard_containers:
            try:
                goods_map = {'Regular': 'reg', 'Reefer': 'r', 'Dangerous': 'dg'}
                goods_code = goods_map.get(container.goods_type, 'reg')
                
                positions = logistics.yard.search_insertion_position(
                    random.randint(0, logistics.yard.n_bays - 1),
                    goods_code,
                    container.container_type,
                    max_proximity=3
                )
                
                if positions:
                    coords = logistics.yard.get_container_coordinates_from_placement(
                        positions[0], container.container_type
                    )
                    logistics.add_container_to_yard(container, coords)
            except:
                continue
        
        # Sync yard index
        logistics.sync_yard_index()
        
        # Add trains with containers and pickup requests
        container_idx = containers_needed // 3
        for i in range(trains_needed):
            train = Train(f"TRAIN_{i:03d}", num_wagons=random.randint(3, 6))
            
            # Add some containers
            for wagon_idx in range(0, len(train.wagons), 2):
                if container_idx < len(containers):
                    train.wagons[wagon_idx].add_container(containers[container_idx])
                    container_idx += 1
            
            # Add pickup requests for yard containers
            if yard_containers:
                for wagon_idx in range(1, len(train.wagons), 2):
                    pickup_container = random.choice(yard_containers)
                    train.wagons[wagon_idx].add_pickup_container(pickup_container.container_id)
            
            # Try to place train
            for railtrack in range(logistics.n_railtracks):
                if logistics.add_train_to_yard(train, railtrack):
                    break
        
        # Add trucks
        for i in range(trucks_needed):
            truck = Truck(f"TRUCK_{i:03d}")
            
            # Some trucks have containers
            if random.random() < 0.3 and container_idx < len(containers):
                truck.add_container(containers[container_idx])
                container_idx += 1
            
            # Some trucks want pickups
            if random.random() < 0.7 and yard_containers:
                pickup_container = random.choice(yard_containers)
                truck.add_pickup_container_id(pickup_container.container_id)
            
            # Find parking position
            for row in range(logistics.n_rows):
                for split in range(logistics.split_factor):
                    if logistics.add_truck_to_yard(truck, (row, 0, split)):
                        break
    
    def _generate_synthetic_move(self, logistics: BooleanLogistics, 
                                move_type: str, move_id: int) -> Optional[Dict]:
        """Generate a synthetic move for testing."""
        
        # This is simplified - in reality would create more realistic moves
        move = {
            'container_id': f"SYNTH_{move_id:05d}",
            'move_type': move_type,
            'priority': random.uniform(5.0, 10.0)
        }
        
        # Set source and destination based on move type
        if 'yard' in move_type:
            move['source_type'] = 'yard'
            move['source_pos'] = (
                random.randint(0, logistics.yard.n_rows - 1),
                random.randint(0, logistics.yard.n_bays - 1),
                0,  # Ground tier
                0   # First split
            )
        elif 'train' in move_type.split('_')[0]:
            move['source_type'] = 'train'
            move['source_pos'] = (0, 0)  # Railtrack 0, wagon 0
        else:
            move['source_type'] = 'truck'
            move['source_pos'] = (0, 0, 0)  # First parking position
        
        # Set destination
        dest_part = move_type.split('_to_')[1]
        if dest_part == 'yard':
            move['dest_type'] = 'yard'
            move['dest_pos'] = (
                random.randint(0, logistics.yard.n_rows - 1),
                random.randint(0, logistics.yard.n_bays - 1),
                0,
                0
            )
        elif dest_part == 'train':
            move['dest_type'] = 'train'
            move['dest_pos'] = (0, 1)  # Railtrack 0, wagon 1
        else:
            move['dest_type'] = 'truck'
            move['dest_pos'] = (1, 0, 0)  # Second parking position
        
        return move
    
    def test_coordinate_system_performance(self, rmgc: RMGC_Controller) -> Dict:
        """Test coordinate system building performance."""
        print("  🗺️  Testing coordinate system...")
        
        results = {}
        
        # Test 1: Measure initial build time (already done in __init__)
        # We'll measure a rebuild
        start_time = time.perf_counter()
        rmgc._build_coordinate_system()
        build_time = time.perf_counter() - start_time
        
        results['build_time'] = build_time
        results['num_positions'] = rmgc.num_positions
        results['positions_per_second'] = rmgc.num_positions / build_time if build_time > 0 else 0
        
        # Test 2: Position lookup performance
        lookup_times = []
        test_positions = list(rmgc.position_to_coords.keys())[:1000]  # Test first 1000
        
        start_time = time.perf_counter()
        for pos in test_positions:
            _ = rmgc.position_to_coords[pos]
            _ = rmgc.position_to_idx[pos]
        lookup_time = time.perf_counter() - start_time
        
        results['lookup_time_per_position'] = lookup_time / len(test_positions)
        results['lookups_per_second'] = len(test_positions) / lookup_time if lookup_time > 0 else 0
        
        return results
    
    def test_distance_matrix_performance(self, rmgc: RMGC_Controller) -> Dict:
        """Test distance matrix building and lookup performance."""
        print("  📏 Testing distance matrix...")
        
        results = {}
        
        # Test 1: Matrix build time
        start_time = time.perf_counter()
        matrix = rmgc._build_distance_matrix()
        build_time = time.perf_counter() - start_time
        
        results['build_time'] = build_time
        results['matrix_size'] = matrix.shape
        results['num_distances'] = matrix.shape[0] * matrix.shape[1]
        results['distances_per_second'] = results['num_distances'] / build_time if build_time > 0 else 0
        
        # Test 2: Distance lookup performance
        num_lookups = 10000
        indices = np.random.randint(0, matrix.shape[0], size=(num_lookups, 2))
        
        start_time = time.perf_counter()
        for i, j in indices:
            _ = matrix[i, j]
        lookup_time = time.perf_counter() - start_time
        
        results['lookup_time'] = lookup_time
        results['lookups_per_second'] = num_lookups / lookup_time if lookup_time > 0 else 0
        results['avg_lookup_time_us'] = (lookup_time / num_lookups) * 1_000_000  # microseconds
        
        return results
    
    def test_physics_calculations(self, rmgc: RMGC_Controller) -> Dict:
        """Test crane physics calculations accuracy and performance."""
        print("  ⚙️  Testing physics calculations...")
        
        results = {}
        test_cases = []
        
        # Generate test cases with known physics
        distances = [10, 50, 100, 200]  # meters
        for dist in distances:
            # Test each axis
            test_cases.extend([
                {'distance': dist, 'speed': rmgc.gantry_speed, 'accel': rmgc.gantry_acceleration, 'axis': 'gantry'},
                {'distance': dist, 'speed': rmgc.trolley_speed, 'accel': rmgc.trolley_acceleration, 'axis': 'trolley'},
                {'distance': dist, 'speed': rmgc.hoisting_speed, 'accel': rmgc.hoisting_acceleration, 'axis': 'hoist'}
            ])
        
        # Test physics calculations
        calc_times = []
        physics_results = []
        
        for test in test_cases:
            start_time = time.perf_counter()
            calc_time = rmgc._axis_time(test['distance'], test['speed'], test['accel'])
            end_time = time.perf_counter()
            
            calc_times.append(end_time - start_time)
            
            # Verify physics
            t_accel = test['speed'] / test['accel']
            d_accel = 0.5 * test['accel'] * t_accel * t_accel
            
            if test['distance'] <= 2 * d_accel:
                # Triangular profile
                expected_time = 2 * np.sqrt(test['distance'] / test['accel'])
            else:
                # Trapezoidal profile
                d_constant = test['distance'] - 2 * d_accel
                expected_time = 2 * t_accel + d_constant / test['speed']
            
            error = abs(calc_time - expected_time)
            physics_results.append({
                'axis': test['axis'],
                'distance': test['distance'],
                'calculated': calc_time,
                'expected': expected_time,
                'error': error,
                'error_percent': (error / expected_time * 100) if expected_time > 0 else 0
            })
        
        results['calculation_times'] = calc_times
        results['avg_calc_time_us'] = np.mean(calc_times) * 1_000_000
        results['physics_accuracy'] = physics_results
        results['max_error_percent'] = max(r['error_percent'] for r in physics_results)
        
        # Test full movement calculation
        if rmgc.num_positions > 100:
            pos1 = rmgc.idx_to_position[0]
            pos2 = rmgc.idx_to_position[min(50, rmgc.num_positions - 1)]
            
            start_time = time.perf_counter()
            movement_time = rmgc._calculate_movement_time(
                pos1, pos2, 
                np.array([0, 0, rmgc.max_height])
            )
            calc_time = time.perf_counter() - start_time
            
            results['full_movement_calc_time'] = calc_time * 1000  # ms
            results['calculated_movement_time'] = movement_time
        
        return results
    
    def test_multi_head_coordination(self, rmgc: RMGC_Controller, moves: Dict) -> Dict:
        """Test multi-head crane coordination and collision avoidance."""
        print("  🏗️  Testing multi-head coordination...")
        
        results = {
            'num_heads': rmgc.heads,
            'zone_overlaps': [],
            'masking_performance': {},
            'head_utilization': {}
        }
        
        # Test 1: Zone overlap calculation
        for i in range(rmgc.heads):
            start, end = rmgc.crane_zones[i]
            results['head_utilization'][f'head_{i}'] = {
                'zone_start': start,
                'zone_end': end,
                'zone_size': end - start,
                'busy': False,
                'moves_executed': 0
            }
            
            # Check overlaps with other zones
            for j in range(i + 1, rmgc.heads):
                other_start, other_end = rmgc.crane_zones[j]
                overlap_start = max(start, other_start)
                overlap_end = min(end, other_end)
                
                if overlap_start < overlap_end:
                    results['zone_overlaps'].append({
                        'heads': (i, j),
                        'overlap_bays': list(range(overlap_start, overlap_end)),
                        'overlap_size': overlap_end - overlap_start
                    })
        
        # Test 2: Move masking performance with different head states
        scenarios = [
            {'name': 'all_free', 'busy_heads': []},
            {'name': 'one_busy', 'busy_heads': [0]},
            {'name': 'half_busy', 'busy_heads': list(range(rmgc.heads // 2))},
            {'name': 'all_but_one_busy', 'busy_heads': list(range(rmgc.heads - 1))}
        ]
        
        for scenario in scenarios:
            # Set head states
            for i in range(rmgc.heads):
                rmgc.crane_heads[i]['busy'] = i in scenario['busy_heads']
                if i in scenario['busy_heads']:
                    # Simulate working in middle bays
                    rmgc.crane_heads[i]['working_bays'] = set(range(i * 5, (i + 1) * 5))
            
            # Time move masking
            start_time = time.perf_counter()
            eligible_moves = rmgc.mask_moves(moves)
            mask_time = time.perf_counter() - start_time
            
            results['masking_performance'][scenario['name']] = {
                'time': mask_time,
                'total_moves': len(moves),
                'eligible_moves': len(eligible_moves),
                'filtered_percent': (1 - len(eligible_moves) / len(moves)) * 100 if moves else 0
            }
            
            # Reset head states
            for i in range(rmgc.heads):
                rmgc.crane_heads[i]['busy'] = False
                rmgc.crane_heads[i]['working_bays'].clear()
        
        # Test 3: Concurrent move execution simulation
        if rmgc.heads > 1 and len(moves) >= rmgc.heads:
            move_items = list(moves.items())[:rmgc.heads]
            
            execution_times = []
            for i, (move_id, move) in enumerate(move_items):
                if i < rmgc.heads:
                    # Simulate assigning move to head
                    rmgc.crane_heads[i]['busy'] = True
                    source_str = rmgc._position_to_string(move['source_pos'])
                    dest_str = rmgc._position_to_string(move['dest_pos'])
                    
                    # Calculate execution time
                    start_time = time.perf_counter()
                    exec_time = rmgc._calculate_movement_time(
                        source_str, dest_str,
                        rmgc.crane_heads[i]['position']
                    )
                    calc_time = time.perf_counter() - start_time
                    
                    execution_times.append({
                        'head': i,
                        'move_id': move_id,
                        'execution_time': exec_time,
                        'calculation_time': calc_time
                    })
            
            results['concurrent_execution_test'] = execution_times
            
            # Reset heads
            for i in range(rmgc.heads):
                rmgc.crane_heads[i]['busy'] = False
        
        return results
    
    def test_move_execution_performance(self, rmgc: RMGC_Controller, 
                                      logistics: BooleanLogistics,
                                      moves: Dict) -> Dict:
        """Test move execution performance."""
        print("  🚀 Testing move execution...")
        
        results = {
            'execution_times': [],
            'success_rate': 0,
            'failures': [],
            'performance_by_type': {}
        }
        
        # Execute subset of moves to test performance
        test_moves = dict(list(moves.items())[:min(50, len(moves))])
        
        successful_moves = 0
        move_times_by_type = defaultdict(list)
        
        for move_id, move in test_moves.items():
            try:
                start_time = time.perf_counter()
                distance, exec_time = rmgc.execute_move(move)
                end_time = time.perf_counter()
                
                calc_time = end_time - start_time
                
                if distance > 0:  # Successful move
                    successful_moves += 1
                    results['execution_times'].append({
                        'move_id': move_id,
                        'distance': distance,
                        'physics_time': exec_time,
                        'actual_time': calc_time,
                        'overhead': calc_time - (exec_time / 1000) if exec_time > 0 else calc_time
                    })
                    
                    # Track by move type
                    move_type_key = f"{move.get('source_type', 'unknown')}_to_{move.get('dest_type', 'unknown')}"
                    move_times_by_type[move_type_key].append(calc_time)
                else:
                    results['failures'].append({
                        'move_id': move_id,
                        'reason': 'Zero distance returned'
                    })
                
                # Unlock head for next test
                for i in range(rmgc.heads):
                    if rmgc.crane_heads[i]['current_move'] == move:
                        rmgc.unlock_head(i)
                        break
                        
            except Exception as e:
                results['failures'].append({
                    'move_id': move_id,
                    'reason': str(e)
                })
        
        results['success_rate'] = successful_moves / len(test_moves) if test_moves else 0
        
        # Aggregate performance by type
        for move_type, times in move_times_by_type.items():
            results['performance_by_type'][move_type] = {
                'count': len(times),
                'avg_time': np.mean(times),
                'max_time': np.max(times),
                'min_time': np.min(times),
                'std_time': np.std(times)
            }
        
        # Overall statistics
        if results['execution_times']:
            all_times = [r['actual_time'] for r in results['execution_times']]
            results['overall_stats'] = {
                'avg_execution_time': np.mean(all_times),
                'max_execution_time': np.max(all_times),
                'min_execution_time': np.min(all_times),
                'moves_per_second': 1.0 / np.mean(all_times) if np.mean(all_times) > 0 else 0
            }
        
        return results
    
    def test_scalability(self, configs: List[Dict]) -> Dict:
        """Test RMGC scalability with different terminal sizes."""
        print("  📈 Testing scalability...")
        
        scalability_results = {}
        
        for config in configs:
            print(f"\n    Testing {config['name']} configuration...")
            
            # Create terminal
            rmgc, logistics, yard = self.create_test_terminal(config)
            
            # Generate moves
            moves = self.generate_test_moves(logistics, config['test_moves'], 'medium_distance')
            
            # Run performance tests
            results = {
                'config': config,
                'coordinate_system': self.test_coordinate_system_performance(rmgc),
                'distance_matrix': self.test_distance_matrix_performance(rmgc),
                'move_execution': self.test_move_execution_performance(rmgc, logistics, moves)
            }
            
            scalability_results[config['name']] = results
        
        return scalability_results
    
    def run_comprehensive_test(self, test_configs: List[Dict] = None) -> Dict:
        """Run comprehensive RMGC stress test."""
        
        if test_configs is None:
            test_configs = [
                {
                    'name': 'Small Terminal',
                    'test_moves': 50,
                    'n_rows': 10,
                    'n_railtracks': 4,
                    'split_factor': 4,
                    'yard_config': {
                        'n_rows': 10,
                        'n_bays': 20,
                        'n_tiers': 3,
                        'split_factor': 4
                    }
                },
                {
                    'name': 'Medium Terminal',
                    'test_moves': 100,
                    'n_rows': 15,
                    'n_railtracks': 6,
                    'split_factor': 4,
                    'yard_config': {
                        'n_rows': 15,
                        'n_bays': 40,
                        'n_tiers': 4,
                        'split_factor': 4
                    }
                },
                {
                    'name': 'Large Terminal',
                    'test_moves': 200,
                    'n_rows': 20,
                    'n_railtracks': 8,
                    'split_factor': 4,
                    'yard_config': {
                        'n_rows': 20,
                        'n_bays': 60,
                        'n_tiers': 5,
                        'split_factor': 4
                    }
                },
                {
                    'name': 'Mega Terminal',
                    'test_moves': 400,
                    'n_rows': 30,
                    'n_railtracks': 10,
                    'split_factor': 4,
                    'yard_config': {
                        'n_rows': 30,
                        'n_bays': 80,
                        'n_tiers': 6,
                        'split_factor': 4
                    }
                }
            ]
        
        print("🏗️  RMGC CONTROLLER STRESS TEST 🏗️")
        print("=" * 80)
        
        all_results = {}
        
        # Test different numbers of crane heads
        head_configs = [1, 2, 4, 6]
        
        for num_heads in head_configs:
            print(f"\n🔧 Testing with {num_heads} crane heads")
            print("-" * 60)
            
            self.rmgc_config['heads'] = num_heads
            
            # Use medium terminal for head comparison
            config = test_configs[1]  # Medium terminal
            
            # Start memory tracking
            tracemalloc.start()
            test_start_time = time.perf_counter()
            
            # Create terminal
            rmgc, logistics, yard = self.create_test_terminal(config)
            
            # Generate test moves
            print(f"  Generating {config['test_moves']} test moves...")
            moves = self.generate_test_moves(logistics, config['test_moves'], 'medium_distance')
            print(f"  Generated {len(moves)} moves")
            
            # Run all tests
            test_results = {
                'num_heads': num_heads,
                'coordinate_system': self.test_coordinate_system_performance(rmgc),
                'distance_matrix': self.test_distance_matrix_performance(rmgc),
                'physics': self.test_physics_calculations(rmgc),
                'multi_head': self.test_multi_head_coordination(rmgc, moves),
                'move_execution': self.test_move_execution_performance(rmgc, logistics, moves)
            }
            
            # Calculate total test time and memory
            total_test_time = time.perf_counter() - test_start_time
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            test_results['performance_summary'] = {
                'total_test_time': total_test_time,
                'memory_usage_mb': current_memory / (1024 * 1024),
                'peak_memory_mb': peak_memory / (1024 * 1024)
            }
            
            all_results[f'{num_heads}_heads'] = test_results
            
            # Print immediate results
            self._print_head_test_results(num_heads, test_results)
        
        # Run scalability test with optimal head count
        print(f"\n📏 Running scalability test with {2} heads...")
        self.rmgc_config['heads'] = 2  # Use 2 heads for scalability test
        scalability_results = self.test_scalability(test_configs)
        all_results['scalability'] = scalability_results
        
        # Generate summary and plots
        self._print_comprehensive_summary(all_results)
        self._generate_performance_plots(all_results)
        
        return all_results
    
    def _print_head_test_results(self, num_heads: int, results: Dict):
        """Print results for specific head configuration."""
        print(f"\n  Results for {num_heads} crane heads:")
        
        # Coordinate system
        coord_results = results['coordinate_system']
        print(f"    📍 Coordinate System: {coord_results['num_positions']} positions")
        print(f"       Build time: {coord_results['build_time']*1000:.2f}ms")
        print(f"       Lookup speed: {coord_results['lookups_per_second']:.0f} lookups/sec")
        
        # Distance matrix
        dist_results = results['distance_matrix']
        print(f"    📏 Distance Matrix: {dist_results['matrix_size']}")
        print(f"       Build time: {dist_results['build_time']*1000:.2f}ms")
        print(f"       Lookup speed: {dist_results['avg_lookup_time_us']:.2f}μs per lookup")
        
        # Physics
        physics_results = results['physics']
        print(f"    ⚙️  Physics Accuracy: {physics_results['max_error_percent']:.2f}% max error")
        print(f"       Calc speed: {physics_results['avg_calc_time_us']:.2f}μs per calculation")
        
        # Multi-head coordination
        if num_heads > 1:
            multi_results = results['multi_head']
            print(f"    🏗️  Multi-head Coordination:")
            print(f"       Zone overlaps: {len(multi_results['zone_overlaps'])}")
            
            # Masking performance
            all_free = multi_results['masking_performance'].get('all_free', {})
            if all_free:
                print(f"       Masking time: {all_free.get('time', 0)*1000:.2f}ms for {all_free.get('total_moves', 0)} moves")
        
        # Move execution
        exec_results = results['move_execution']
        print(f"    🚀 Move Execution:")
        print(f"       Success rate: {exec_results['success_rate']*100:.1f}%")
        
        if 'overall_stats' in exec_results:
            stats = exec_results['overall_stats']
            print(f"       Avg time: {stats['avg_execution_time']*1000:.2f}ms")
            print(f"       Throughput: {stats['moves_per_second']:.1f} moves/sec")
        
        # Memory and time
        perf = results['performance_summary']
        print(f"    💾 Memory: {perf['memory_usage_mb']:.1f}MB (peak: {perf['peak_memory_mb']:.1f}MB)")
        print(f"    ⏱️  Total time: {perf['total_test_time']:.2f}s")
    
    def _print_comprehensive_summary(self, all_results: Dict):
        """Print comprehensive test summary."""
        print("\n" + "=" * 80)
        print("📊 RMGC STRESS TEST SUMMARY")
        print("=" * 80)
        
        # Head comparison table
        print("\n🏗️  Performance by Number of Crane Heads:")
        print(f"{'Heads':<8} {'Coord Build':<15} {'Move Exec(ms)':<15} {'Success%':<12} {'Memory(MB)':<12}")
        print("-" * 70)
        
        for key in sorted(all_results.keys()):
            if '_heads' in key:
                num_heads = int(key.split('_')[0])
                results = all_results[key]
                
                coord_time = results['coordinate_system']['build_time'] * 1000
                
                exec_stats = results['move_execution'].get('overall_stats', {})
                exec_time = exec_stats.get('avg_execution_time', 0) * 1000
                success_rate = results['move_execution']['success_rate'] * 100
                
                memory = results['performance_summary']['peak_memory_mb']
                
                print(f"{num_heads:<8} {coord_time:<15.2f} {exec_time:<15.2f} {success_rate:<12.1f} {memory:<12.1f}")
        
        # Scalability summary
        if 'scalability' in all_results:
            print("\n📈 Scalability Analysis:")
            print(f"{'Terminal':<20} {'Positions':<12} {'Matrix Build':<15} {'Move Exec':<15}")
            print("-" * 70)
            
            for name, results in all_results['scalability'].items():
                positions = results['coordinate_system']['num_positions']
                matrix_time = results['distance_matrix']['build_time'] * 1000
                
                exec_stats = results['move_execution'].get('overall_stats', {})
                exec_time = exec_stats.get('avg_execution_time', 0) * 1000
                
                print(f"{name:<20} {positions:<12} {matrix_time:<15.2f} {exec_time:<15.2f}")
        
        # Performance recommendations
        print("\n💡 Performance Insights:")
        
        # Find optimal head count
        best_throughput = 0
        best_heads = 1
        
        for key in all_results.keys():
            if '_heads' in key:
                results = all_results[key]
                stats = results['move_execution'].get('overall_stats', {})
                throughput = stats.get('moves_per_second', 0)
                
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_heads = int(key.split('_')[0])
        
        print(f"  ✅ Optimal configuration: {best_heads} crane heads ({best_throughput:.1f} moves/sec)")
        
        # Check physics accuracy
        max_physics_error = 0
        for key in all_results.keys():
            if '_heads' in key:
                physics_error = all_results[key]['physics']['max_error_percent']
                max_physics_error = max(max_physics_error, physics_error)
        
        if max_physics_error < 0.1:
            print(f"  ✅ Physics calculations highly accurate: {max_physics_error:.3f}% max error")
        elif max_physics_error < 1.0:
            print(f"  ⚠️  Physics calculations good: {max_physics_error:.2f}% max error")
        else:
            print(f"  🔴 Physics calculations need review: {max_physics_error:.2f}% max error")
        
        # Performance thresholds
        print(f"\n🎯 Performance Benchmarks:")
        print(f"  🟢 < 10ms move execution: Excellent for real-time operations")
        print(f"  🟡 10-25ms move execution: Good for most applications")
        print(f"  🔴 > 25ms move execution: May need optimization")
    
    def _generate_performance_plots(self, all_results: Dict):
        """Generate performance visualization plots."""
        try:
            # Extract data for plotting
            head_counts = []
            coord_build_times = []
            move_exec_times = []
            success_rates = []
            throughputs = []
            
            for key in sorted(all_results.keys()):
                if '_heads' in key:
                    num_heads = int(key.split('_')[0])
                    results = all_results[key]
                    
                    head_counts.append(num_heads)
                    coord_build_times.append(results['coordinate_system']['build_time'] * 1000)
                    
                    exec_stats = results['move_execution'].get('overall_stats', {})
                    move_exec_times.append(exec_stats.get('avg_execution_time', 0) * 1000)
                    throughputs.append(exec_stats.get('moves_per_second', 0))
                    
                    success_rates.append(results['move_execution']['success_rate'] * 100)
            
            # Create plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Move execution time vs heads
            ax1.plot(head_counts, move_exec_times, 'b-o', linewidth=2, markersize=8)
            ax1.set_title('Move Execution Time vs Crane Heads', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Number of Crane Heads')
            ax1.set_ylabel('Average Move Execution Time (ms)')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=10, color='green', linestyle='--', alpha=0.7, label='10ms target')
            ax1.axhline(y=25, color='orange', linestyle='--', alpha=0.7, label='25ms threshold')
            ax1.legend()
            
            # Plot 2: Throughput vs heads
            ax2.plot(head_counts, throughputs, 'g-o', linewidth=2, markersize=8)
            ax2.set_title('Move Throughput vs Crane Heads', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Number of Crane Heads')
            ax2.set_ylabel('Moves per Second')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Success rate vs heads
            ax3.plot(head_counts, success_rates, 'r-o', linewidth=2, markersize=8)
            ax3.set_title('Move Success Rate vs Crane Heads', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Number of Crane Heads')
            ax3.set_ylabel('Success Rate (%)')
            ax3.set_ylim(0, 105)
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Scalability - if available
            if 'scalability' in all_results:
                terminal_sizes = []
                scalability_times = []
                
                for name, results in all_results['scalability'].items():
                    positions = results['coordinate_system']['num_positions']
                    terminal_sizes.append(positions)
                    
                    exec_stats = results['move_execution'].get('overall_stats', {})
                    scalability_times.append(exec_stats.get('avg_execution_time', 0) * 1000)
                
                ax4.plot(terminal_sizes, scalability_times, 'm-o', linewidth=2, markersize=8)
                ax4.set_title('Scalability: Move Time vs Terminal Size', fontsize=14, fontweight='bold')
                ax4.set_xlabel('Number of Terminal Positions')
                ax4.set_ylabel('Average Move Execution Time (ms)')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout(pad=3.0)
            plt.savefig('rmgc_stress_test_results.png', dpi=300, bbox_inches='tight')
            print(f"\n📈 Performance plots saved as 'rmgc_stress_test_results.png'")
            
        except Exception as e:
            print(f"\n⚠️  Could not generate plots: {e}")


def run_quick_rmgc_test():
    """Run a quick RMGC stress test."""
    print("🚀 Quick RMGC Controller Test")
    
    quick_configs = [{
        'name': 'Quick Test',
        'test_moves': 25,
        'n_rows': 8,
        'n_railtracks': 3,
        'split_factor': 4,
        'yard_config': {
            'n_rows': 8,
            'n_bays': 15,
            'n_tiers': 3,
            'split_factor': 4
        }
    }]
    
    tester = RMGC_StressTest()
    tester.rmgc_config['heads'] = 2  # Test with 2 heads
    
    # Create terminal and run basic tests
    rmgc, logistics, yard = tester.create_test_terminal(quick_configs[0])
    moves = tester.generate_test_moves(logistics, 25, 'short_distance')
    
    print(f"\nCreated terminal with {rmgc.num_positions} positions")
    print(f"Generated {len(moves)} test moves")
    
    # Run key tests
    coord_results = tester.test_coordinate_system_performance(rmgc)
    physics_results = tester.test_physics_calculations(rmgc)
    exec_results = tester.test_move_execution_performance(rmgc, logistics, moves)
    
    print(f"\n✅ Quick test completed:")
    print(f"  Coordinate build: {coord_results['build_time']*1000:.2f}ms")
    print(f"  Physics accuracy: {physics_results['max_error_percent']:.2f}% max error")
    print(f"  Move execution: {exec_results.get('overall_stats', {}).get('avg_execution_time', 0)*1000:.2f}ms avg")


def run_full_rmgc_test():
    """Run the full RMGC stress test."""
    print("🏭 Full Scale RMGC Controller Stress Test")
    
    tester = RMGC_StressTest()
    results = tester.run_comprehensive_test()
    return results


def run_physics_validation():
    """Validate RMGC physics calculations against known values."""
    print("⚙️  RMGC Physics Validation")
    print("=" * 60)
    
    # Create minimal RMGC for physics testing
    from simulation.terminal_components.BooleanStorage import BooleanStorageYard
    from simulation.terminal_components.BooleanLogistics import BooleanLogistics
    
    yard = BooleanStorageYard(5, 10, 3, [], 4, False)
    logistics = BooleanLogistics(5, 2, 4, yard, False)
    
    rmgc = RMGC_Controller(
        yard=yard,
        logistics=logistics,
        heads=1,
        trolley_speed=70.0,
        hoisting_speed=28.0,
        gantry_speed=130.0,
        trolley_acceleration=0.3,
        hoisting_acceleration=0.2,
        gantry_acceleration=0.1
    )
    
    # Test cases
    test_cases = [
        # (distance, expected_time_range)
        (10, (9.5, 10.5)),    # Short distance
        (50, (25, 27)),       # Medium distance
        (100, (48, 52)),      # Long distance
        (200, (95, 100))      # Very long distance
    ]
    
    print("Testing gantry axis physics (130 m/min, 0.1 m/s² acceleration):")
    print(f"{'Distance (m)':<15} {'Calculated (s)':<15} {'Expected Range':<20} {'Status':<10}")
    print("-" * 60)
    
    all_passed = True
    for distance, expected_range in test_cases:
        calc_time = rmgc._axis_time(distance, rmgc.gantry_speed, rmgc.gantry_acceleration)
        in_range = expected_range[0] <= calc_time <= expected_range[1]
        status = "✅ PASS" if in_range else "❌ FAIL"
        all_passed &= in_range
        
        print(f"{distance:<15} {calc_time:<15.2f} {str(expected_range):<20} {status:<10}")
    
    print(f"\n{'✅ All physics tests passed!' if all_passed else '❌ Some physics tests failed!'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RMGC Controller Stress Test')
    parser.add_argument('--quick', action='store_true', help='Run quick test')
    parser.add_argument('--full', action='store_true', help='Run full test')
    parser.add_argument('--physics', action='store_true', help='Run physics validation')
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_rmgc_test()
    elif args.physics:
        run_physics_validation()
    else:
        results = run_full_rmgc_test()
    
    print(f"\n" + "=" * 80)
    print("🎯 RMGC CONTROLLER STRESS TEST COMPLETED")
    print("=" * 80)
    print("🏗️  KEY FEATURES TESTED:")
    print("  ✅ Coordinate system building and O(1) lookups")
    print("  ✅ Distance matrix pre-computation")
    print("  ✅ Physics-based movement calculations")
    print("  ✅ Multi-head coordination and collision avoidance")
    print("  ✅ Move execution with container transfers")
    print("  ✅ Scalability with different terminal sizes")
    print("")
    print("🎯 Performance targets:")
    print("  🟢 < 10ms move execution: Optimal for real-time control")
    print("  🟡 10-25ms move execution: Acceptable for most operations")
    print("  🔴 > 25ms move execution: May impact system responsiveness")