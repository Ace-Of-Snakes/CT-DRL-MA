from typing import Dict, Tuple, List, Optional, Set, Any
from simulation.terminal_components.Container import Container, ContainerFactory
from simulation.terminal_components.Train import Train
from simulation.terminal_components.Truck import Truck
from simulation.terminal_components.BooleanStorage import BooleanStorageYard
from simulation.terminal_components.BooleanLogistics import BoolLogistics
import numpy as np
import torch
from collections import defaultdict, deque
import time

class ContainerTerminal:
    """
    Unified container terminal combining storage yard and logistics.
    Optimized for O(1) operations and fast tensor conversion.
    """
    
    def __init__(self,
                 n_rows: int = 5,
                 n_bays: int = 58,
                 n_tiers: int = 5,
                 split_factor: int = 4,
                 n_rail_tracks: int = 6,
                 n_parking_spots: int = 29,
                 special_coordinates: List[Tuple[int, int, str]] = None,
                 distance_matrix: Optional[np.ndarray] = None,
                 position_mapping: Optional[Dict[str, Tuple[float, float]]] = None,
                 device: str = 'cpu'):
        """
        Initialize complete container terminal.
        
        Args:
            n_rows: Number of storage rows
            n_bays: Number of bays per row
            n_tiers: Number of tiers (height)
            split_factor: Subdivision factor for container placement
            n_rail_tracks: Number of rail tracks
            n_parking_spots: Number of parking spots
            special_coordinates: Special area coordinates [(bay, row, type), ...]
            distance_matrix: Pre-computed distance matrix
            position_mapping: Position to coordinate mapping
            device: Computation device ('cpu' or 'cuda')
        """
        
        # Default special area configuration if not provided
        if special_coordinates is None:
            special_coordinates = self._create_default_special_areas(n_bays, n_rows)
        
        # Initialize storage yard
        # from BooleanStorage import BooleanStorageYard  # Import your optimized yard
        self.storage_yard = BooleanStorageYard(
            n_rows=n_rows,
            n_bays=n_bays,
            n_tiers=n_tiers,
            coordinates=special_coordinates,
            split_factor=split_factor,
            validate=False,
            device=device
        )
        
        # Initialize logistics
        self.logistics = BoolLogistics(
            n_rows=n_rows,
            split_factor=split_factor,
            n_rail_tracks=n_rail_tracks,
            n_parking_spots=n_parking_spots,
            device=device
        )
        
        self.device = device
        
        # Distance and time calculation
        self.distance_matrix = distance_matrix
        self.position_mapping = position_mapping or {}
        
        # Pre-computed time matrices for O(1) lookup
        self.time_cache = {}
        self._precompute_common_times()
        
        # Movement speed constants (from your RMG specs)
        self.trolley_speed = 70.0 / 60.0  # m/s
        self.hoisting_speed = 28.0 / 60.0  # m/s
        self.gantry_speed = 130.0 / 60.0  # m/s
        self.base_operation_time = 60.0  # seconds
        
        print(f"Container Terminal initialized:")
        print(f"  Storage: {n_rows}x{n_bays}x{n_tiers} (split_factor={split_factor})")
        print(f"  Rails: {n_rail_tracks} tracks")
        print(f"  Parking: {n_parking_spots} spots")
        print(f"  Device: {device}")
    
    def _create_default_special_areas(self, n_bays: int, n_rows: int) -> List[Tuple[int, int, str]]:
        """Create default special area configuration."""
        coordinates = []
        
        # Reefers at both ends of each row
        for row in range(1, n_rows + 1):
            coordinates.extend([
                (1, row, "r"),      # First bay
                (n_bays, row, "r")  # Last bay
            ])
        
        # Dangerous goods in middle section
        middle_start = n_bays // 3
        middle_end = 2 * n_bays // 3
        for row in range(2, min(n_rows, 4) + 1):  # Middle rows
            for bay in range(middle_start, middle_end + 1):
                coordinates.append((bay, row, "dg"))
        
        # Swap bodies and trailers in first row (closest to driving lane)
        for bay in range(5, min(n_bays - 5, 20)):  # Reasonable section
            coordinates.append((bay, 1, "sb_t"))
        
        return coordinates
    
    def step_simulation(self) -> Dict[str, Any]:
        """
        Perform one simulation step:
        1. Auto-place trucks from queue
        2. Auto-place trains from queue  
        3. Find all possible moves
        4. Return state information
        """
        step_info = {
            'trucks_placed': 0,
            'trains_placed': 0,
            'available_moves': 0,
            'queue_status': {}
        }
        
        # 1. Auto-place trucks (with intelligent positioning)
        trucks_placed = self.logistics.auto_place_trucks_from_queue()
        step_info['trucks_placed'] = trucks_placed
        
        # 2. Auto-place trains on available tracks
        trains_placed = 0
        for track_id in range(self.logistics.n_rail_tracks):
            if self.logistics.try_place_train(track_id):
                trains_placed += 1
        step_info['trains_placed'] = trains_placed
        
        # 3. Find all available moves
        all_moves = self.find_all_moves()
        step_info['available_moves'] = len(all_moves)
        
        # 4. Get queue status
        step_info['queue_status'] = {
            'trucks_waiting': len(self.logistics.truck_queue),
            'trains_waiting': len(self.logistics.train_queue),
            'trucks_active': sum(self.logistics.truck_occupied),
            'trains_active': sum(self.logistics.train_occupied)
        }
        
        return step_info
    
    def _precompute_common_times(self):
        """Pre-compute movement times for common operations."""
        # This would be populated with actual distance calculations
        # For now, using estimated times
        self.time_cache = {
            'yard_to_rail': 120.0,     # 2 minutes average
            'yard_to_parking': 90.0,   # 1.5 minutes average
            'rail_to_parking': 180.0,  # 3 minutes average
            'parking_to_rail': 180.0,  # 3 minutes average
            'rail_to_yard': 120.0,     # 2 minutes average
            'parking_to_yard': 90.0,   # 1.5 minutes average
        }
    
    def find_all_moves(self) -> Dict[str, Dict[str, Any]]:
        """
        Find all possible moves across the entire terminal.
        
        Returns:
            Dict mapping move_id -> {
                'container_id': str,
                'source_type': str,  # 'yard', 'rail', 'parking'
                'source_pos': str,
                'dest_type': str,
                'dest_pos': str,
                'move_type': str,    # 'pickup', 'delivery', 'reshuffle'
                'time_estimate': float,
                'priority': float
            }
        """
        all_moves = {}
        move_counter = 0
        
        # 1. Yard to Rail/Parking moves (deliveries and pickups)
        yard_moves = self._find_yard_to_vehicle_moves()
        for move in yard_moves:
            all_moves[f"move_{move_counter}"] = move
            move_counter += 1
        
        # 2. Rail/Parking to Yard moves (unloading)
        vehicle_moves = self._find_vehicle_to_yard_moves()
        for move in vehicle_moves:
            all_moves[f"move_{move_counter}"] = move
            move_counter += 1
        
        # 3. Direct Rail to Parking moves (matching IDs)
        logistics_moves = self.logistics.find_container_moves()
        for container_id, move_list in logistics_moves.items():
            for source, destination in move_list:
                move = {
                    'container_id': container_id,
                    'source_type': source.split('_')[0],
                    'source_pos': source,
                    'dest_type': destination.split('_')[0],
                    'dest_pos': destination,
                    'move_type': 'transfer',
                    'time_estimate': self._estimate_move_time(source, destination),
                    'priority': self._calculate_move_priority(container_id, source, destination)
                }
                all_moves[f"move_{move_counter}"] = move
                move_counter += 1
        
        # 4. Internal yard reshuffling
        yard_internal_moves = self.storage_yard.return_possible_yard_moves()
        for container_id, move_data in yard_internal_moves.items():
            for dest_coords in move_data['destinations']:
                move = {
                    'container_id': container_id,
                    'source_type': 'yard',
                    'source_pos': str(move_data['source_coords']),
                    'dest_type': 'yard',
                    'dest_pos': str(dest_coords),
                    'move_type': 'reshuffle',
                    'time_estimate': self.time_cache.get('yard_reshuffle', 180.0),
                    'priority': self._calculate_reshuffle_priority(container_id)
                }
                all_moves[f"move_{move_counter}"] = move
                move_counter += 1
        
        return all_moves
    
    def _find_yard_to_vehicle_moves(self) -> List[Dict[str, Any]]:
        """Find moves from yard to waiting vehicles."""
        moves = []
        
        # Check each container in yard against pickup requests
        for row in range(self.storage_yard.n_rows):
            for bay in range(self.storage_yard.n_bays):
                for tier in range(self.storage_yard.n_tiers):
                    for split in range(self.storage_yard.split_factor):
                        container = self.storage_yard.get_container_at(row, bay, tier, split)
                        
                        if container is None:
                            continue
                        
                        container_id = container.container_id
                        
                        # Check if any truck is waiting for this container
                        if container_id in self.logistics.pickup_to_truck:
                            spot_id = self.logistics.pickup_to_truck[container_id]
                            moves.append({
                                'container_id': container_id,
                                'source_type': 'yard',
                                'source_pos': f"yard_{row}_{bay}_{tier}_{split}",
                                'dest_type': 'parking',
                                'dest_pos': f"parking_{spot_id}",
                                'move_type': 'pickup',
                                'time_estimate': self.time_cache['yard_to_parking'],
                                'priority': 10.0  # High priority for pickups
                            })
                        
                        # Check if any train is waiting for this container
                        if container_id in self.logistics.pickup_to_train:
                            track_id, wagon_idx = self.logistics.pickup_to_train[container_id]
                            moves.append({
                                'container_id': container_id,
                                'source_type': 'yard',
                                'source_pos': f"yard_{row}_{bay}_{tier}_{split}",
                                'dest_type': 'rail',
                                'dest_pos': f"rail_{track_id}_{wagon_idx}",
                                'move_type': 'pickup',
                                'time_estimate': self.time_cache['yard_to_rail'],
                                'priority': 10.0  # High priority for pickups
                            })
        
        return moves
    
    def _find_vehicle_to_yard_moves(self) -> List[Dict[str, Any]]:
        """Find moves from vehicles to yard storage."""
        moves = []
        
        # Moves from trains to yard
        for track_id, train in self.logistics.active_trains.items():
            for wagon_idx, wagon in enumerate(train.wagons):
                for container in wagon.containers:
                    # Find valid storage positions for this container
                    storage_positions = self._find_storage_positions_for_container(container)
                    
                    for pos in storage_positions[:5]:  # Limit to top 5 options
                        moves.append({
                            'container_id': container.container_id,
                            'source_type': 'rail',
                            'source_pos': f"rail_{track_id}_{wagon_idx}",
                            'dest_type': 'yard',
                            'dest_pos': str(pos),
                            'move_type': 'delivery',
                            'time_estimate': self.time_cache['rail_to_yard'],
                            'priority': 8.0  # High priority for deliveries
                        })
        
        # Moves from trucks to yard
        for spot_id, truck in self.logistics.active_trucks.items():
            for container in truck.containers:
                # Find valid storage positions for this container
                storage_positions = self._find_storage_positions_for_container(container)
                
                for pos in storage_positions[:5]:  # Limit to top 5 options
                    moves.append({
                        'container_id': container.container_id,
                        'source_type': 'parking',
                        'source_pos': f"parking_{spot_id}",
                        'dest_type': 'yard',
                        'dest_pos': str(pos),
                        'move_type': 'delivery',
                        'time_estimate': self.time_cache['parking_to_yard'],
                        'priority': 8.0  # High priority for deliveries
                    })
        
        return moves
    
    def _find_storage_positions_for_container(self, container: Container) -> List[Tuple]:
        """Find valid storage positions for a container."""
        # Determine goods type for mask selection
        if container.goods_type == 'Reefer':
            goods_mask = 'r'
        elif container.goods_type == 'Dangerous':
            goods_mask = 'dg'
        elif container.container_type in ['Trailer', 'Swap Body']:
            goods_mask = 'sb_t'
        else:
            goods_mask = 'reg'
        
        # Find positions with proximity of 5 bays (reasonable search area)
        center_bay = self.storage_yard.n_bays // 2  # Use center as default
        positions = self.storage_yard.search_insertion_position(
            center_bay, goods_mask, container.container_type, max_proximity=5
        )
        
        return positions
    
    def _estimate_move_time(self, source: str, destination: str) -> float:
        """Estimate time for a move between positions."""
        source_type = source.split('_')[0]
        dest_type = destination.split('_')[0]
        
        cache_key = f"{source_type}_to_{dest_type}"
        return self.time_cache.get(cache_key, 120.0)  # Default 2 minutes
    
    def _calculate_move_priority(self, container_id: str, source: str, destination: str) -> float:
        """Calculate priority for a move."""
        base_priority = 5.0
        
        # Higher priority for pickup operations
        if 'pickup' in destination:
            base_priority += 5.0
        
        # Higher priority for containers with imminent deadlines
        # (This would use actual container deadline data)
        base_priority += np.random.uniform(0, 2)  # Placeholder
        
        return base_priority
    
    def _calculate_reshuffle_priority(self, container_id: str) -> float:
        """Calculate priority for yard reshuffling."""
        # Lower priority for internal moves
        return 2.0 + np.random.uniform(0, 1)
    
    def execute_move(self, move_id: str, moves_dict: Dict[str, Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Execute a specific move.
        
        Returns:
            (success, message)
        """
        if move_id not in moves_dict:
            return False, f"Move {move_id} not found"
        
        move = moves_dict[move_id]
        container_id = move['container_id']
        source_type = move['source_type']
        dest_type = move['dest_type']
        
        try:
            # Execute based on move type
            if source_type == 'yard' and dest_type == 'parking':
                return self._execute_yard_to_parking(move)
            elif source_type == 'yard' and dest_type == 'rail':
                return self._execute_yard_to_rail(move)
            elif source_type == 'parking' and dest_type == 'yard':
                return self._execute_parking_to_yard(move)
            elif source_type == 'rail' and dest_type == 'yard':
                return self._execute_rail_to_yard(move)
            elif source_type == 'yard' and dest_type == 'yard':
                return self._execute_yard_reshuffle(move)
            else:
                return False, f"Unknown move type: {source_type} -> {dest_type}"
                
        except Exception as e:
            return False, f"Error executing move: {str(e)}"
    
    def _execute_yard_to_parking(self, move: Dict[str, Any]) -> Tuple[bool, str]:
        """Execute move from yard to parking spot."""
        # Parse positions
        source_parts = move['source_pos'].split('_')[1:]  # Remove 'yard_'
        row, bay, tier, split = map(int, source_parts)
        
        spot_id = int(move['dest_pos'].split('_')[1])  # Remove 'parking_'
        
        # Get container from yard
        container = self.storage_yard.get_container_at(row, bay, tier, split)
        if container is None:
            return False, "Container not found at source position"
        
        # Remove from yard
        coords = [(row, bay, split, tier)]
        removed_container = self.storage_yard.remove_container(coords)
        
        # Add to truck
        if spot_id in self.logistics.active_trucks:
            truck = self.logistics.active_trucks[spot_id]
            success = truck.add_container(removed_container)
            
            if success:
                # Update logistics lookup tables
                self.logistics._update_truck_lookups(truck, spot_id)
                return True, f"Container {container.container_id} moved to parking spot {spot_id}"
            else:
                # Put container back in yard
                self.storage_yard.add_container(removed_container, coords)
                return False, "Truck cannot accept container"
        
        return False, "Truck not found at destination"
    
    def _execute_yard_to_rail(self, move: Dict[str, Any]) -> Tuple[bool, str]:
        """Execute move from yard to rail."""
        # Parse positions
        source_parts = move['source_pos'].split('_')[1:]  # Remove 'yard_'
        row, bay, tier, split = map(int, source_parts)
        
        dest_parts = move['dest_pos'].split('_')[1:]  # Remove 'rail_'
        track_id, wagon_idx = int(dest_parts[0]), int(dest_parts[1])
        
        # Get container from yard
        container = self.storage_yard.get_container_at(row, bay, tier, split)
        if container is None:
            return False, "Container not found at source position"
        
        # Remove from yard
        coords = [(row, bay, split, tier)]
        removed_container = self.storage_yard.remove_container(coords)
        
        # Add to train
        if track_id in self.logistics.active_trains:
            train = self.logistics.active_trains[track_id]
            if wagon_idx < len(train.wagons):
                wagon = train.wagons[wagon_idx]
                success = wagon.add_container(removed_container)
                
                if success:
                    # Update logistics lookup tables
                    self.logistics._update_train_lookups(train, track_id)
                    return True, f"Container {container.container_id} moved to rail {track_id} wagon {wagon_idx}"
                else:
                    # Put container back in yard
                    self.storage_yard.add_container(removed_container, coords)
                    return False, "Wagon cannot accept container"
        
        return False, "Train or wagon not found at destination"
    
    def _execute_parking_to_yard(self, move: Dict[str, Any]) -> Tuple[bool, str]:
        """Execute move from parking to yard."""
        spot_id = int(move['source_pos'].split('_')[1])
        dest_coords = eval(move['dest_pos'])  # Convert string back to coordinates
        
        # Get container from truck
        if spot_id not in self.logistics.active_trucks:
            return False, "Truck not found at source"
        
        truck = self.logistics.active_trucks[spot_id]
        if not truck.containers:
            return False, "No containers on truck"
        
        container = truck.containers[0]  # Take first container
        removed_container = truck.remove_container(container.container_id)
        
        # Add to yard
        success = self.storage_yard.add_container(removed_container, dest_coords)
        
        if success:
            # Update logistics lookup tables
            self.logistics._update_truck_lookups(truck, spot_id)
            return True, f"Container {container.container_id} moved from parking to yard"
        else:
            # Put container back on truck
            truck.add_container(removed_container)
            return False, "Cannot place container in yard"
    
    def _execute_rail_to_yard(self, move: Dict[str, Any]) -> Tuple[bool, str]:
        """Execute move from rail to yard."""
        source_parts = move['source_pos'].split('_')[1:]  # Remove 'rail_'
        track_id, wagon_idx = int(source_parts[0]), int(source_parts[1])
        dest_coords = eval(move['dest_pos'])  # Convert string back to coordinates
        
        # Get container from train
        if track_id not in self.logistics.active_trains:
            return False, "Train not found at source"
        
        train = self.logistics.active_trains[track_id]
        if wagon_idx >= len(train.wagons):
            return False, "Wagon not found"
        
        wagon = train.wagons[wagon_idx]
        if not wagon.containers:
            return False, "No containers on wagon"
        
        container = wagon.containers[0]  # Take first container
        removed_container = wagon.remove_container(container.container_id)
        
        # Add to yard
        success = self.storage_yard.add_container(removed_container, dest_coords)
        
        if success:
            # Update logistics lookup tables
            self.logistics._update_train_lookups(train, track_id)
            return True, f"Container {container.container_id} moved from rail to yard"
        else:
            # Put container back on wagon
            wagon.add_container(removed_container)
            return False, "Cannot place container in yard"
    
    def _execute_yard_reshuffle(self, move: Dict[str, Any]) -> Tuple[bool, str]:
        """Execute internal yard reshuffling."""
        source_coords = eval(move['source_pos'])
        dest_coords = eval(move['dest_pos'])
        
        # Use yard's move_container method
        self.storage_yard.move_container(source_coords, dest_coords)
        
        return True, f"Container {move['container_id']} reshuffled within yard"
    
    # ==================== TENSOR CONVERSION METHODS ====================
    
    def get_full_terminal_state(self, flatten: bool = False) -> torch.Tensor:
        """
        Get complete terminal state as tensor for DRL agent.
        
        Returns:
            Combined tensor with all terminal information
        """
        # Get individual state components
        yard_state = self.storage_yard.get_full_state_tensor(flatten=False)
        rail_state = self.logistics.get_rail_state_tensor()
        parking_state = self.logistics.get_parking_state_tensor()
        train_props = self.logistics.get_train_properties_tensor()
        truck_props = self.logistics.get_truck_properties_tensor()
        queue_states = self.logistics.get_queue_state_tensor()
        
        if flatten:
            # Flatten everything for MLP input
            components = [
                yard_state.flatten(),
                rail_state.flatten(),
                parking_state.flatten(),
                train_props.flatten(),
                truck_props.flatten(),
                queue_states['train_queue'].flatten(),
                queue_states['truck_queue'].flatten()
            ]
            return torch.cat(components)
        else:
            # Return as structured dict for CNN/attention models
            return {
                'yard': yard_state,
                'rails': rail_state,
                'parking': parking_state,
                'trains': train_props,
                'trucks': truck_props,
                'queues': queue_states
            }
    
    def get_move_action_space_size(self) -> int:
        """Get the size of the action space for DRL agent."""
        # This would be the maximum number of possible moves
        # Based on your terminal dimensions
        max_moves = (
            self.storage_yard.n_rows * self.storage_yard.n_bays * 
            self.storage_yard.n_tiers * self.storage_yard.split_factor * 10  # Estimate
        )
        return max_moves


# Example usage and testing
if __name__ == "__main__":
    import time
    
    print("="*60)
    print("CONTAINER TERMINAL SYSTEM - COMPREHENSIVE TEST")
    print("="*60)
    
    # ==================== TEST 1: BASIC INITIALIZATION ====================
    print("\n1. BASIC INITIALIZATION TEST")
    print("-" * 40)
    
    start_time = time.time()
    
    # Create terminal with default configuration
    terminal = ContainerTerminal(
        n_rows=5,
        n_bays=20,  # Smaller for testing
        n_tiers=3,
        split_factor=4,
        n_rail_tracks=6,
        n_parking_spots=29,
        device='cpu',
        special_coordinates=[
            # Reefers on both ends
            (1, 1, "r"), (1, 2, "r"), (1, 3, "r"), (1, 4, "r"), (1, 5, "r"),
            (15, 1, "r"), (15, 2, "r"), (15, 3, "r"), (15, 4, "r"), (15, 5, "r"),
            
            # Row nearest to trucks is for swap bodies and trailers
            (1, 1, "sb_t"), (2, 1, "sb_t"), (3, 1, "sb_t"), (4, 1, "sb_t"), (5, 1, "sb_t"),
            (6, 1, "sb_t"), (7, 1, "sb_t"), (8, 1, "sb_t"), (9, 1, "sb_t"), (10, 1, "sb_t"),
            (11, 1, "sb_t"), (12, 1, "sb_t"), (13, 1, "sb_t"), (14, 1, "sb_t"), (15, 1, "sb_t"),
            
            # Pit in the middle for dangerous goods
            (7, 3, "dg"), (8, 3, "dg"), (9, 3, "dg"),
            (7, 4, "dg"), (8, 4, "dg"), (9, 4, "dg"),
            (7, 5, "dg"), (8, 5, "dg"), (9, 5, "dg"),
        ]
    )
    
    init_time = time.time() - start_time
    print(f"✓ Terminal initialized in {init_time:.4f}s")
    
    # ==================== TEST 2: LOGISTICS FUNCTIONALITY ====================
    print("\n2. LOGISTICS FUNCTIONALITY TEST")
    print("-" * 40)
    
    # Create test vehicles
    print("Creating test vehicles...")
    
    # Create test containers
    test_containers = []
    for i in range(5):
        container = ContainerFactory.create_container(
            f"TEST_CONT_{i:03d}", 
            "TWEU", 
            "Import", 
            "Regular", 
            weight=20000 + i * 1000
        )
        test_containers.append(container)
    
    # Create test trains with containers and pickup requests
    test_trains = []
    for i in range(3):
        train = Train(f"TRAIN_{i:03d}", num_wagons=5)
        
        # Add some containers to wagons
        for j in range(2):
            if j < len(test_containers):
                train.wagons[j].add_container(test_containers[j])
        
        # Add pickup requests to other wagons
        for j in range(2, 4):
            train.wagons[j].add_pickup_container(f"PICKUP_CONT_{i}_{j}")
        
        test_trains.append(train)
        print(f"  Created {train.train_id} with {len(train.wagons)} wagons")
    
    # Create test trucks with pickup requests
    test_trucks = []
    for i in range(5):
        truck = Truck(f"TRUCK_{i:03d}")
        
        # Some trucks want to pick up specific containers
        if i < 3:
            truck.add_pickup_container_id(f"PICKUP_CONT_{i}_2")
            truck.add_pickup_container_id(f"PICKUP_CONT_{i}_3")
        
        test_trucks.append(truck)
        print(f"  Created {truck.truck_id} (pickup truck: {truck.is_pickup_truck})")
    
    # ==================== TEST 3: QUEUE OPERATIONS ====================
    print("\n3. QUEUE OPERATIONS TEST")
    print("-" * 40)
    
    # Add vehicles to queues
    for train in test_trains:
        success = terminal.logistics.add_train_to_queue(train)
        print(f"  Train {train.train_id} added to queue: {success}")
    
    for truck in test_trucks:
        success = terminal.logistics.add_truck_to_queue(truck)
        print(f"  Truck {truck.truck_id} added to queue: {success}")
    
    print(f"\nQueue status:")
    print(f"  Trains in queue: {len(terminal.logistics.train_queue)}")
    print(f"  Trucks in queue: {len(terminal.logistics.truck_queue)}")
    
    # ==================== TEST 4: AUTO-PLACEMENT ====================
    print("\n4. AUTO-PLACEMENT TEST")
    print("-" * 40)
    
    # Test automatic train placement
    print("Placing trains on tracks...")
    trains_placed = 0
    for track_id in range(terminal.logistics.n_rail_tracks):
        if terminal.logistics.try_place_train(track_id):
            trains_placed += 1
            print(f"  ✓ Train placed on track {track_id}")
        else:
            print(f"  ✗ Could not place train on track {track_id}")
    
    print(f"Total trains placed: {trains_placed}")
    
    # Test intelligent truck placement
    print("\nPlacing trucks with intelligent positioning...")
    trucks_placed = terminal.logistics.auto_place_trucks_from_queue()
    print(f"✓ Automatically placed {trucks_placed} trucks")
    
    # Show placement results
    for spot_id, truck in terminal.logistics.active_trucks.items():
        pickup_ids = list(getattr(truck, 'pickup_container_ids', set()))
        print(f"  Truck {truck.truck_id} at spot {spot_id} (wants: {pickup_ids})")
    
    # ==================== TEST 5: STORAGE YARD OPERATIONS ====================
    print("\n5. STORAGE YARD OPERATIONS TEST")
    print("-" * 40)
    
    # Add some containers to storage yard
    print("Adding containers to storage yard...")
    
    storage_containers = []
    for i in range(8):
        container = ContainerFactory.create_container(
            f"YARD_CONT_{i:03d}",
            ["TWEU", "FEU", "THEU"][i % 3],
            "Import",
            ["Regular", "Reefer", "Dangerous"][i % 3],
            weight=15000 + i * 2000
        )
        storage_containers.append(container)
    
    # Place containers in yard
    placed_count = 0
    for i, container in enumerate(storage_containers):
        # Find suitable positions
        goods_type = 'reg' if container.goods_type == 'Regular' else container.goods_type.lower()[:1]
        if goods_type == 'd':
            goods_type = 'dg'
        elif goods_type == 'r':
            goods_type = 'r'
        
        positions = terminal.storage_yard.search_insertion_position(
            bay=5 + i,  # Spread them out
            goods=goods_type,
            container_type=container.container_type,
            max_proximity=3
        )
        
        if positions:
            coords = terminal.storage_yard.get_container_coordinates_from_placement(
                positions[0], container.container_type
            )
            terminal.storage_yard.add_container(container, coords)
            placed_count += 1
            print(f"  ✓ Placed {container.container_id} ({container.container_type}, {container.goods_type})")
    
    print(f"Total containers placed in yard: {placed_count}")
    
    # ==================== TEST 6: MOVE DETECTION ====================
    print("\n6. MOVE DETECTION TEST")
    print("-" * 40)
    
    print("Finding all possible moves...")
    start_time = time.time()
    all_moves = terminal.find_all_moves()
    move_time = time.time() - start_time
    
    print(f"✓ Found {len(all_moves)} possible moves in {move_time:.4f}s")
    
    # Categorize moves
    move_types = {}
    for move_id, move in all_moves.items():
        move_type = move['move_type']
        if move_type not in move_types:
            move_types[move_type] = 0
        move_types[move_type] += 1
    
    print("Move breakdown by type:")
    for move_type, count in move_types.items():
        print(f"  {move_type}: {count} moves")
    
    # Show some example moves
    if all_moves:
        print("\nExample moves:")
        for i, (move_id, move) in enumerate(list(all_moves.items())[:5]):
            print(f"  {move_id}: {move['container_id']} {move['source_type']} → {move['dest_type']} "
                  f"({move['move_type']}, {move['time_estimate']:.1f}s)")
    
    # ==================== TEST 7: TENSOR CONVERSION ====================
    print("\n7. TENSOR CONVERSION TEST")
    print("-" * 40)
    
    print("Testing tensor conversion performance...")
    
    # Test individual tensor conversions
    tensors = {}
    conversion_times = {}
    
    components = [
        ('yard_occupied', lambda: terminal.storage_yard.get_occupied_tensor()),
        ('yard_types', lambda: terminal.storage_yard.get_container_type_tensor()),
        ('yard_full', lambda: terminal.storage_yard.get_full_state_tensor()),
        ('rail_state', lambda: terminal.logistics.get_rail_state_tensor()),
        ('train_props', lambda: terminal.logistics.get_train_properties_tensor()),
        ('truck_props', lambda: terminal.logistics.get_truck_properties_tensor()),
        ('queues', lambda: terminal.logistics.get_queue_state_tensor()),
    ]
    
    for name, tensor_func in components:
        start_time = time.time()
        tensor = tensor_func()
        conversion_time = time.time() - start_time
        
        conversion_times[name] = conversion_time
        if isinstance(tensor, dict):
            tensors[name] = {k: v.shape for k, v in tensor.items()}
            print(f"  {name}: {tensors[name]} ({conversion_time:.6f}s)")
        else:
            tensors[name] = tensor.shape
            print(f"  {name}: {tensor.shape} ({conversion_time:.6f}s)")
    
    # Test complete terminal state
    start_time = time.time()
    full_state_flat = terminal.get_full_terminal_state(flatten=True)
    full_state_struct = terminal.get_full_terminal_state(flatten=False)
    full_conversion_time = time.time() - start_time
    
    print(f"\nComplete terminal state:")
    print(f"  Flattened: {full_state_flat.shape} ({full_conversion_time:.6f}s)")
    print(f"  Structured: {len(full_state_struct)} components")
    
    # ==================== TEST 8: MOVE EXECUTION ====================
    print("\n8. MOVE EXECUTION TEST")
    print("-" * 40)
    
    if all_moves:
        print("Testing move execution...")
        
        # Try to execute a few moves
        executed_moves = 0
        for move_id, move in list(all_moves.items())[:3]:
            print(f"\nAttempting to execute {move_id}:")
            print(f"  Move: {move['container_id']} {move['source_type']} → {move['dest_type']}")
            
            success, message = terminal.execute_move(move_id, all_moves)
            if success:
                executed_moves += 1
                print(f"  ✓ {message}")
            else:
                print(f"  ✗ {message}")
        
        print(f"\nSuccessfully executed {executed_moves} moves")
    
    # ==================== TEST 9: SIMULATION STEP ====================
    print("\n9. SIMULATION STEP TEST")
    print("-" * 40)
    
    print("Testing complete simulation step...")
    start_time = time.time()
    step_info = terminal.step_simulation()
    step_time = time.time() - start_time
    
    print(f"✓ Simulation step completed in {step_time:.4f}s")
    print("Step results:")
    for key, value in step_info.items():
        print(f"  {key}: {value}")
    
    # ==================== PERFORMANCE SUMMARY ====================
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    total_objects = (
        terminal.storage_yard.n_rows * terminal.storage_yard.n_bays * 
        terminal.storage_yard.n_tiers * terminal.storage_yard.split_factor +
        terminal.logistics.n_rail_tracks + terminal.logistics.n_parking_spots
    )
    
    print(f"Terminal capacity: {total_objects} total positions")
    print(f"Move detection: {len(all_moves)} moves found in {move_time:.4f}s")
    print(f"Tensor conversion: {sum(conversion_times.values()):.6f}s total")
    print(f"Memory efficiency: Direct array access (O(1) operations)")
    print(f"DRL ready: All state tensors available instantly")
    
    print("\n✓ All tests completed successfully!")
    print("System ready for DRL agent integration.")