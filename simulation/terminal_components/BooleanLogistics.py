from typing import Dict, Tuple, List, Optional, Set, Any
from simulation.terminal_components.Container import Container, ContainerFactory
from simulation.terminal_components.Train import Train
from simulation.terminal_components.Truck import Truck
from simulation.terminal_components.Wagon import Wagon
from simulation.terminal_components.Vehicle_Queue import VehicleQueue
from simulation.terminal_components.BooleanStorage import BooleanStorageYard
import numpy as np
import torch
from collections import defaultdict, deque
import time
from datetime import datetime, timedelta

class BooleanLogistics:
    """
    Optimized logistics manager for rails, parking, and vehicle queues.
    Focus on O(1) operations and 80-foot wagon management.
    """
    
    def __init__(self,
                 n_rows: int,
                 n_railtracks: int,
                 split_factor: int,
                 yard: BooleanStorageYard,
                 validate: bool = False,
                 device: str = 'cpu'):
        
        self.n_rows = n_rows
        self.n_railtracks = n_railtracks
        self.split_factor = split_factor
        self.yard = yard
        self.validate = validate
        self.device = device

        # Create dynamic masks which are to be updated (True = available)
        self.dynamic_rail_mask = np.ones((self.n_rows * split_factor, self.n_railtracks), dtype=bool)
        self.dynamic_parking_mask = np.ones((self.n_rows * split_factor, 1), dtype=bool)

        # Create coordinate mapping for rails and driving lane
        self.train_coords = self.create_coordinate_mapping()
        # Parking is equivalent to first railtrack
        self.park_coords = self.train_coords[:, 0:1]

        # Vehicle queues
        self.trucks = VehicleQueue('Truck')
        self.trains = VehicleQueue('Train')

        # Active vehicles tracking - OPTIMIZED: Direct array access
        self.active_trains = {}  # railtrack_id -> Train object
        self.active_trucks = {}  # position_tuple -> Truck object
        
        # Train placement tracking (positions occupied by each train)
        self.train_positions = {}  # railtrack_id -> list of position tuples
        self.truck_positions = {}  # position_tuple -> truck_id
        
        # Container lookup tables for O(1) access
        self.container_to_train = {}  # container_id -> (track_id, wagon_idx)
        self.container_to_truck = {}  # container_id -> truck_position
        self.pickup_to_train = {}     # container_id -> (track_id, wagon_idx)  
        self.pickup_to_truck = {}     # container_id -> truck_position
        
        # OPTIMIZATION: Fast container location index for yard
        self.yard_container_index = {}  # container_id -> (row, bay, tier, split)
        self.yard_container_set = set()  # Fast membership testing
        
        # OPTIMIZATION: Cached yard positions for different container types
        self.cached_yard_positions = {}  # (goods_type, container_type) -> list of positions
        self.yard_cache_dirty = True  # Flag to rebuild cache when needed
        
        # Move mappings for efficiency
        self.train_to_truck_moves = defaultdict(list)
        self.truck_to_train_moves = defaultdict(list)
        self.available_moves_cache = {}
        
        # Wagon size constants (80-foot wagons)
        self.wagon_length = 2 * split_factor  # 2 rows per wagon
        self.train_head_length = 1 * split_factor  # 1 row for locomotive
        
        # Performance tracking arrays for tensor conversion
        self._init_property_arrays()
        
        # OPTIMIZATION: Build initial yard index
        self._rebuild_yard_container_index()
        
        if self.validate:
            self._print_validation_info()

    def _init_property_arrays(self):
        """Initialize arrays for fast tensor conversion."""
        # Train arrays
        self.train_lengths = np.zeros(self.n_railtracks, dtype=np.int16)
        self.train_wagon_counts = np.zeros(self.n_railtracks, dtype=np.int16)
        self.train_container_counts = np.zeros(self.n_railtracks, dtype=np.int16)
        self.train_pickup_counts = np.zeros(self.n_railtracks, dtype=np.int16)
        self.train_occupied = np.zeros(self.n_railtracks, dtype=bool)
        
        # Truck arrays (simplified - only track if parking positions are occupied)
        self.truck_container_counts = np.zeros(self.n_rows * self.split_factor, dtype=np.int16)
        self.truck_pickup_counts = np.zeros(self.n_rows * self.split_factor, dtype=np.int16)
        self.truck_weights = np.zeros(self.n_rows * self.split_factor, dtype=np.float32)

    def create_coordinate_mapping(self) -> np.ndarray:
        """Create 0-starting coordinate map identical to dynamic mask structure."""
        coordinate_arr = np.zeros((self.n_rows * self.split_factor, self.n_railtracks), dtype=object)
        for row in range(self.n_rows):
            for railtrack in range(self.n_railtracks):
                for split in range(self.split_factor):
                    coordinate_format = (row, railtrack, split)
                    coordinate_arr[row * self.split_factor + split][railtrack] = coordinate_format
        return coordinate_arr

    def add_train_to_queue(self, train: Train):
        """Add train to train queue."""
        self.trains.add_vehicle(train)
        print(f"Train {train.train_id} added to queue. Queue size: {self.trains.size()}")

    def add_truck_to_queue(self, truck: Truck):
        """Add truck to truck queue."""
        self.trucks.add_vehicle(truck)
        print(f"Truck {truck.truck_id} added to queue. Queue size: {self.trucks.size()}")

    def add_train_to_yard(self, train: Train, railtrack_id: int) -> bool:
        """
        Add train to yard at specified railtrack.
        
        Args:
            train: Train object to place
            railtrack_id: Which railtrack to place it on
            
        Returns:
            bool: Success of placement
        """
        if railtrack_id >= self.n_railtracks or self.train_occupied[railtrack_id]:
            return False
            
        # Calculate train length: head + wagons
        train_length = self.train_head_length + len(train.wagons) * self.wagon_length
        
        # Check if we have enough consecutive space
        if not self._check_consecutive_space(railtrack_id, train_length):
            return False
        
        # Mark positions as occupied
        positions = self._occupy_train_positions(railtrack_id, train_length)
        
        # Update tracking structures
        self.active_trains[railtrack_id] = train
        self.train_positions[railtrack_id] = positions
        
        # Update property arrays
        self.train_lengths[railtrack_id] = train_length
        self.train_wagon_counts[railtrack_id] = len(train.wagons)
        self.train_occupied[railtrack_id] = True
        
        # Count containers and pickups
        container_count = sum(len(wagon.containers) for wagon in train.wagons)
        pickup_count = sum(len(wagon.pickup_container_ids) for wagon in train.wagons)
        
        self.train_container_counts[railtrack_id] = container_count
        self.train_pickup_counts[railtrack_id] = pickup_count
        
        # Update lookup tables
        self._update_train_lookups(train, railtrack_id)
        
        return True

    def add_truck_to_yard(self, truck: Truck, position: Tuple[int, int, int]) -> bool:
        """
        Add truck to yard at specified position.
        
        Args:
            truck: Truck object to place
            position: (row, railtrack, split) - must be railtrack 0 (parking)
            
        Returns:
            bool: Success of placement
        """
        row, railtrack, split = position
        
        if railtrack != 0:  # Only parking lane (railtrack 0)
            return False
            
        if not self.dynamic_parking_mask[row * self.split_factor + split, 0]:
            return False  # Position occupied
        
        # Mark position as occupied
        self.dynamic_parking_mask[row * self.split_factor + split, 0] = False
        
        # Update tracking structures
        self.active_trucks[position] = truck
        self.truck_positions[position] = truck.truck_id
        
        # Update property arrays
        pos_idx = row * self.split_factor + split
        self.truck_container_counts[pos_idx] = len(truck.containers)
        self.truck_pickup_counts[pos_idx] = len(getattr(truck, 'pickup_container_ids', set()))
        self.truck_weights[pos_idx] = sum(c.weight for c in truck.containers) if truck.containers else 0.0
        
        # Update lookup tables
        self._update_truck_lookups(truck, position)
        
        return True

    def remove_train(self, railtrack_id: int) -> Optional[Train]:
        """Remove train from railtrack and return it."""
        if railtrack_id not in self.active_trains:
            return None
            
        train = self.active_trains[railtrack_id]
        
        # Free positions
        if railtrack_id in self.train_positions:
            for pos in self.train_positions[railtrack_id]:
                row, track, split = pos
                self.dynamic_rail_mask[row * self.split_factor + split, track] = True
        
        # Clear tracking structures
        del self.active_trains[railtrack_id]
        if railtrack_id in self.train_positions:
            del self.train_positions[railtrack_id]
        
        # Clear property arrays
        self.train_lengths[railtrack_id] = 0
        self.train_wagon_counts[railtrack_id] = 0
        self.train_container_counts[railtrack_id] = 0
        self.train_pickup_counts[railtrack_id] = 0
        self.train_occupied[railtrack_id] = False
        
        # Clear lookup tables
        self._clear_train_lookups(railtrack_id)
        
        return train

    def remove_truck(self, position: Tuple[int, int, int]) -> Optional[Truck]:
        """Remove truck from position and return it."""
        if position not in self.active_trucks:
            return None
            
        truck = self.active_trucks[position]
        row, railtrack, split = position
        
        # Free position
        self.dynamic_parking_mask[row * self.split_factor + split, 0] = True
        
        # Clear tracking structures
        del self.active_trucks[position]
        if position in self.truck_positions:
            del self.truck_positions[position]
        
        # Clear property arrays
        pos_idx = row * self.split_factor + split
        self.truck_container_counts[pos_idx] = 0
        self.truck_pickup_counts[pos_idx] = 0
        self.truck_weights[pos_idx] = 0.0
        
        # Clear lookup tables
        self._clear_truck_lookups(position)
        
        return truck

    def process_current_trains(self) -> int:
        """
        Process trains in queue and try to assign them rail slots.
        
        Returns:
            int: Number of trains successfully placed
        """
        placed_count = 0
        
        while not self.trains.is_empty():
            train = self.trains.get_next_vehicle()
            if train is None:
                break
                
            # Calculate required length
            required_length = self.train_head_length + len(train.wagons) * self.wagon_length
            
            # Try to find an available railtrack
            placed = False
            for railtrack_id in range(self.n_railtracks):
                if self._check_consecutive_space(railtrack_id, required_length):
                    if self.add_train_to_yard(train, railtrack_id):
                        placed_count += 1
                        placed = True
                        print(f"Train {train.train_id} placed on railtrack {railtrack_id}")
                        break
            
            if not placed:
                # Put train back in queue for later
                self.trains.add_vehicle(train)
                break  # Stop processing if we can't place this train
        
        return placed_count

    def process_current_trucks(self) -> int:
        """
        Process trucks in queue with proximity search similar to storage yard.
        
        Returns:
            int: Number of trucks successfully placed
        """
        placed_count = 0
        
        while not self.trucks.is_empty():
            truck = self.trucks.get_next_vehicle()
            if truck is None:
                break
                
            # Find optimal position based on pickup requirements
            optimal_pos = self._find_optimal_truck_position(truck)
            
            if optimal_pos is not None:
                if self.add_truck_to_yard(truck, optimal_pos):
                    placed_count += 1
                    print(f"Truck {truck.truck_id} placed at position {optimal_pos}")
                else:
                    # Put truck back in queue
                    self.trucks.add_vehicle(truck)
                    break
            else:
                # Put truck back in queue
                self.trucks.add_vehicle(truck)
                break
        
        return placed_count

    def auto_assign_trucks(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Auto-assign trucks based on train departure schedules and container IDs.
        
        Returns:
            Dict mapping train_id -> [(container_id, truck_position), ...]
        """
        assignments = defaultdict(list)
        
        # Sort trains by departure time (earliest first)
        train_schedule = []
        for railtrack_id, train in self.active_trains.items():
            departure_time = getattr(train, 'departure_time', None)
            train_schedule.append((departure_time or float('inf'), railtrack_id, train))
        
        train_schedule.sort(key=lambda x: x[0])
        
        # Process each train
        for _, railtrack_id, train in train_schedule:
            # Get containers that need pickup from this train
            pickup_containers = set()
            for wagon in train.wagons:
                pickup_containers.update(wagon.pickup_container_ids)
            
            # Find trucks in queue that want these containers
            for container_id in pickup_containers:
                # Search through truck queue for matching pickup requests
                truck_with_request = self._find_truck_with_pickup_request(container_id)
                
                if truck_with_request:
                    # Calculate optimal parking position (parallel or ±1 offset)
                    wagon_idx = self._find_wagon_with_container(train, container_id)
                    if wagon_idx is not None:
                        optimal_positions = self._calculate_parallel_positions(railtrack_id, wagon_idx)
                        
                        # Try to assign truck to best available position
                        for pos in optimal_positions:
                            if self._is_position_available(pos):
                                if self.add_truck_to_yard(truck_with_request, pos):
                                    assignments[train.train_id].append((container_id, pos))
                                    # Remove truck from queue
                                    self._remove_truck_from_queue(truck_with_request)
                                    break
        
        return dict(assignments)

    def _rebuild_yard_container_index(self):
        """OPTIMIZATION: Build fast O(1) lookup index for all containers in yard."""
        self.yard_container_index.clear()
        self.yard_container_set.clear()
        
        # Single pass through yard to build index
        for row in range(self.yard.n_rows):
            for bay in range(self.yard.n_bays):
                for tier in range(self.yard.n_tiers):
                    for split in range(self.yard.split_factor):
                        container = self.yard.get_container_at(row, bay, tier, split)
                        if container is not None:
                            self.yard_container_index[container.container_id] = (row, bay, tier, split)
                            self.yard_container_set.add(container.container_id)
    
    def _update_yard_container_index(self, container_id: str, position: Tuple = None):
        """OPTIMIZATION: Update yard index when container is added/removed."""
        if position is None:
            # Container removed
            self.yard_container_index.pop(container_id, None)
            self.yard_container_set.discard(container_id)
        else:
            # Container added
            self.yard_container_index[container_id] = position
            self.yard_container_set.add(container_id)
        
        # Mark cache as dirty
        self.yard_cache_dirty = True
    
    def _get_cached_yard_positions(self, goods_type: str, container_type: str) -> List[Tuple]:
        """OPTIMIZATION: Get cached available positions for container type."""
        cache_key = (goods_type, container_type)
        
        # Rebuild cache if dirty or key missing
        if self.yard_cache_dirty or cache_key not in self.cached_yard_positions:
            # Find positions for this container type
            center_bay = self.yard.n_bays // 2
            try:
                positions = self.yard.search_insertion_position(
                    center_bay, goods_type, container_type, max_proximity=5
                )
                self.cached_yard_positions[cache_key] = positions[:10]  # Cache top 10
            except Exception:
                self.cached_yard_positions[cache_key] = []
        
        return self.cached_yard_positions[cache_key]
    
    def _find_container_in_yard_fast(self, container_id: str) -> Optional[Tuple]:
        """OPTIMIZATION: O(1) container lookup in yard using index."""
        return self.yard_container_index.get(container_id)
    
    def sync_yard_index(self):
        """Synchronize yard container index with current yard state."""
        self._rebuild_yard_container_index()
        self.yard_cache_dirty = True
    
    def add_container_to_yard(self, container: 'Container', coordinates: List[Tuple]) -> bool:
        """
        OPTIMIZED: Add container to yard and update index.
        
        Args:
            container: Container to add
            coordinates: List of (row, bay, split, tier) coordinates
            
        Returns:
            bool: Success of operation
        """
        try:
            # Add to yard
            self.yard.add_container(container, coordinates)
            
            # Update index with first coordinate (main position)
            if coordinates:
                main_pos = coordinates[0]
                self._update_yard_container_index(container.container_id, main_pos)
            
            return True
        except Exception:
            return False
    
    def remove_container_from_yard(self, coordinates: List[Tuple]) -> Optional['Container']:
        """
        OPTIMIZED: Remove container from yard and update index.
        
        Args:
            coordinates: List of (row, bay, split, tier) coordinates
            
        Returns:
            Removed container or None
        """
        try:
            # Remove from yard
            container = self.yard.remove_container(coordinates)
            
            # Update index
            if container:
                self._update_yard_container_index(container.container_id, None)
            
            return container
        except Exception:
            return None
    
    def find_moves_optimized(self) -> Dict[str, Dict[str, Any]]:
        """
        OPTIMIZED: Find all possible moves using vectorized operations and caching.
        
        Returns:
            Dict mapping move_id -> move_details
        """
        all_moves = {}
        move_counter = 0
        
        if self.validate:
            print(f"DEBUG: Finding moves (optimized)...")
            print(f"  Active trains: {len(self.active_trains)}")
            print(f"  Active trucks: {len(self.active_trucks)}")
            print(f"  Yard containers: {len(self.yard_container_index)}")
        
        # OPTIMIZATION 1: Collect all vehicle containers in single pass
        train_containers = []  # (container, railtrack_id, wagon_idx)
        truck_containers = []  # (container, truck_pos)
        
        for railtrack_id, train in self.active_trains.items():
            for wagon_idx, wagon in enumerate(train.wagons):
                for container in wagon.containers:
                    train_containers.append((container, railtrack_id, wagon_idx))
        
        for truck_pos, truck in self.active_trucks.items():
            for container in truck.containers:
                truck_containers.append((container, truck_pos))
        
        # OPTIMIZATION 2: Vectorized pickup request processing
        pickup_requests = set()
        pickup_requests.update(self.pickup_to_train.keys())
        pickup_requests.update(self.pickup_to_truck.keys())
        
        # Fast intersection with yard containers
        yard_pickup_containers = pickup_requests & self.yard_container_set
        
        if self.validate:
            print(f"  Train containers: {len(train_containers)}")
            print(f"  Truck containers: {len(truck_containers)}")
            print(f"  Pickup requests in yard: {len(yard_pickup_containers)}")
        
        # OPTIMIZATION 3: Process moves by type with minimal loops
        
        # 1. Train -> Truck moves
        for container, railtrack_id, wagon_idx in train_containers:
            if container.container_id in self.pickup_to_truck:
                truck_pos = self.pickup_to_truck[container.container_id]
                move_id = f"move_{move_counter}"
                all_moves[move_id] = {
                    'container_id': container.container_id,
                    'source_type': 'train',
                    'source_pos': (railtrack_id, wagon_idx),
                    'dest_type': 'truck',
                    'dest_pos': truck_pos,
                    'move_type': 'train_to_truck',
                    'priority': 10.0
                }
                move_counter += 1
        
        # 2. Truck -> Train moves
        for container, truck_pos in truck_containers:
            if container.container_id in self.pickup_to_train:
                railtrack_id, wagon_idx = self.pickup_to_train[container.container_id]
                move_id = f"move_{move_counter}"
                all_moves[move_id] = {
                    'container_id': container.container_id,
                    'source_type': 'truck',
                    'source_pos': truck_pos,
                    'dest_type': 'train',
                    'dest_pos': (railtrack_id, wagon_idx),
                    'move_type': 'truck_to_train',
                    'priority': 10.0
                }
                move_counter += 1
        
        # 3. Yard -> Vehicle moves (using fast lookup)
        for container_id in yard_pickup_containers:
            container_pos = self.yard_container_index[container_id]
            
            # Check train pickups
            if container_id in self.pickup_to_train:
                railtrack_id, wagon_idx = self.pickup_to_train[container_id]
                move_id = f"move_{move_counter}"
                all_moves[move_id] = {
                    'container_id': container_id,
                    'source_type': 'yard',
                    'source_pos': container_pos,
                    'dest_type': 'train',
                    'dest_pos': (railtrack_id, wagon_idx),
                    'move_type': 'from_yard',
                    'priority': 9.0
                }
                move_counter += 1
            
            # Check truck pickups
            if container_id in self.pickup_to_truck:
                truck_pos = self.pickup_to_truck[container_id]
                move_id = f"move_{move_counter}"
                all_moves[move_id] = {
                    'container_id': container_id,
                    'source_type': 'yard',
                    'source_pos': container_pos,
                    'dest_type': 'truck',
                    'dest_pos': truck_pos,
                    'move_type': 'from_yard',
                    'priority': 9.0
                }
                move_counter += 1
        
        # 4. Vehicle -> Yard moves (using cached positions)
        import_containers = []
        
        # Collect import containers from vehicles
        for container, railtrack_id, wagon_idx in train_containers:
            if hasattr(container, 'direction') and container.direction == 'Import':
                import_containers.append((container, 'train', (railtrack_id, wagon_idx)))
        
        for container, truck_pos in truck_containers:
            if hasattr(container, 'direction') and container.direction == 'Import':
                import_containers.append((container, 'truck', truck_pos))
        
        # Process import containers with cached positions
        processed_types = set()
        for container, source_type, source_pos in import_containers:
            # Determine goods type
            if container.goods_type == 'Reefer':
                goods_type = 'r'
            elif container.goods_type == 'Dangerous':
                goods_type = 'dg'
            elif container.container_type in ['Swap Body', 'Trailer']:
                goods_type = 'sb_t'
            else:
                goods_type = 'reg'
            
            # Use cached positions
            cache_key = (goods_type, container.container_type)
            if cache_key not in processed_types:
                # Get cached positions for this type
                cached_positions = self._get_cached_yard_positions(goods_type, container.container_type)
                processed_types.add(cache_key)
            else:
                cached_positions = self.cached_yard_positions.get(cache_key, [])
            
            # Add moves for available positions (limit to 2 per container)
            for position in cached_positions[:2]:
                move_id = f"move_{move_counter}"
                all_moves[move_id] = {
                    'container_id': container.container_id,
                    'source_type': source_type,
                    'source_pos': source_pos,
                    'dest_type': 'yard',
                    'dest_pos': position,
                    'move_type': 'to_yard',
                    'priority': 8.0
                }
                move_counter += 1
        
        if self.validate:
            print(f"  Total moves found: {len(all_moves)}")
        
        # Mark cache as clean after successful move finding
        self.yard_cache_dirty = False
        
        return all_moves
        """
        Find all possible moves: train->truck, truck->train, and yard insertions.
        
        Returns:
            Dict mapping move_id -> move_details
        """
        all_moves = {}
        move_counter = 0
        
        if self.validate:
            print(f"DEBUG: Finding moves...")
            print(f"  Active trains: {len(self.active_trains)}")
            print(f"  Active trucks: {len(self.active_trucks)}")
            print(f"  Pickup to train: {len(self.pickup_to_train)}")
            print(f"  Pickup to truck: {len(self.pickup_to_truck)}")
        
        # 1. Train -> Truck moves (containers from trains to waiting trucks)
        for railtrack_id, train in self.active_trains.items():
            for wagon_idx, wagon in enumerate(train.wagons):
                if self.validate:
                    print(f"  Train {train.train_id} wagon {wagon_idx}: {len(wagon.containers)} containers")
                for container in wagon.containers:
                    container_id = container.container_id
                    
                    # Check if any truck is waiting for this container
                    if container_id in self.pickup_to_truck:
                        truck_pos = self.pickup_to_truck[container_id]
                        move_id = f"move_{move_counter}"
                        all_moves[move_id] = {
                            'container_id': container_id,
                            'source_type': 'train',
                            'source_pos': (railtrack_id, wagon_idx),
                            'dest_type': 'truck',
                            'dest_pos': truck_pos,
                            'move_type': 'train_to_truck',
                            'priority': 10.0
                        }
                        move_counter += 1

        # 2. Truck -> Train moves (containers from trucks to waiting trains)
        for truck_pos, truck in self.active_trucks.items():
            if self.validate:
                print(f"  Truck {truck.truck_id}: {len(truck.containers)} containers")
            for container in truck.containers:
                container_id = container.container_id
                
                # Check if any train is waiting for this container
                if container_id in self.pickup_to_train:
                    railtrack_id, wagon_idx = self.pickup_to_train[container_id]
                    move_id = f"move_{move_counter}"
                    all_moves[move_id] = {
                        'container_id': container_id,
                        'source_type': 'truck',
                        'source_pos': truck_pos,
                        'dest_type': 'train', 
                        'dest_pos': (railtrack_id, wagon_idx),
                        'move_type': 'truck_to_train',
                        'priority': 10.0
                    }
                    move_counter += 1

        # 3. Yard retrieval moves (containers that need to be picked up from yard)
        pickup_requests = set()
        pickup_requests.update(self.pickup_to_train.keys())
        pickup_requests.update(self.pickup_to_truck.keys())
        
        if self.validate:
            print(f"  Pickup requests: {pickup_requests}")
        
        for container_id in pickup_requests:
            # Find container in yard
            container_pos = self._find_container_in_yard(container_id)
            if self.validate:
                print(f"  Container {container_id} in yard: {container_pos is not None}")
            
            if container_pos is not None:
                # Check which vehicle wants it
                if container_id in self.pickup_to_train:
                    railtrack_id, wagon_idx = self.pickup_to_train[container_id]
                    move_id = f"move_{move_counter}"
                    all_moves[move_id] = {
                        'container_id': container_id,
                        'source_type': 'yard',
                        'source_pos': container_pos,
                        'dest_type': 'train',
                        'dest_pos': (railtrack_id, wagon_idx),
                        'move_type': 'from_yard',
                        'priority': 9.0
                    }
                    move_counter += 1
                
                if container_id in self.pickup_to_truck:
                    truck_pos = self.pickup_to_truck[container_id]
                    move_id = f"move_{move_counter}"
                    all_moves[move_id] = {
                        'container_id': container_id,
                        'source_type': 'yard',
                        'source_pos': container_pos,
                        'dest_type': 'truck',
                        'dest_pos': truck_pos,
                        'move_type': 'from_yard',
                        'priority': 9.0
                    }
                    move_counter += 1

        # 4. Yard insertion moves (containers that need to be stored)
        # For containers on trains/trucks that need to go to yard
        all_containers_needing_storage = []
        
        # From trains
        for railtrack_id, train in self.active_trains.items():
            for wagon_idx, wagon in enumerate(train.wagons):
                for container in wagon.containers:
                    if hasattr(container, 'direction') and container.direction == 'Import':  # Needs to go to yard
                        all_containers_needing_storage.append({
                            'container': container,
                            'source_type': 'train',
                            'source_pos': (railtrack_id, wagon_idx)
                        })
        
        # From trucks
        for truck_pos, truck in self.active_trucks.items():
            for container in truck.containers:
                if hasattr(container, 'direction') and container.direction == 'Import':  # Needs to go to yard
                    all_containers_needing_storage.append({
                        'container': container,
                        'source_type': 'truck',
                        'source_pos': truck_pos
                    })
        
        # Find yard positions for each container
        for container_info in all_containers_needing_storage:
            container = container_info['container']
            yard_positions = self._find_yard_positions_for_container(container)
            
            for position in yard_positions[:3]:  # Limit to top 3 options
                move_id = f"move_{move_counter}"
                all_moves[move_id] = {
                    'container_id': container.container_id,
                    'source_type': container_info['source_type'],
                    'source_pos': container_info['source_pos'],
                    'dest_type': 'yard',
                    'dest_pos': position,
                    'move_type': 'to_yard',
                    'priority': 8.0
                }
                move_counter += 1

        if self.validate:
            print(f"  Total moves found: {len(all_moves)}")

        return all_moves

    def reorganize_logistics(self) -> np.ndarray:
        """
        Reorganize logistics after moves: remove completed vehicles, update tracking.
        
        Returns:
            Summary vector: [trains_departed_early, trains_departed_late, trucks_departed]
        """
        trains_departed_early = 0
        trains_departed_late = 0
        trucks_departed = 0
        
        current_time = datetime.now()  # Use datetime for consistency with train.departure_time
        
        # Check trains for completion/departure
        trains_to_remove = []
        for railtrack_id, train in self.active_trains.items():
            # Check if train is ready to depart (all pickups completed, no deliveries left)
            ready_to_depart = True
            
            # Check if all pickup requests are fulfilled
            for wagon in train.wagons:
                if len(wagon.pickup_container_ids) > 0:
                    ready_to_depart = False
                    break
            
            # Check departure time
            departure_time = getattr(train, 'departure_time', None)
            if departure_time and ready_to_depart:
                if current_time <= departure_time:
                    trains_departed_early += 1
                else:
                    trains_departed_late += 1
                trains_to_remove.append(railtrack_id)
            elif departure_time and current_time > departure_time:
                # Force departure if past departure time
                trains_departed_late += 1
                trains_to_remove.append(railtrack_id)
        
        # Remove completed trains
        for railtrack_id in trains_to_remove:
            self.remove_train(railtrack_id)
        
        # Check trucks for completion
        trucks_to_remove = []
        for truck_pos, truck in self.active_trucks.items():
            # Check if truck completed its mission
            pickup_complete = len(getattr(truck, 'pickup_container_ids', set())) == 0
            delivery_complete = len(truck.containers) == 0
            
            if pickup_complete and delivery_complete:
                trucks_departed += 1
                trucks_to_remove.append(truck_pos)
        
        # Remove completed trucks
        for truck_pos in trucks_to_remove:
            self.remove_truck(truck_pos)
        
        # Try to place new vehicles from queues
        self.process_current_trains()
        self.process_current_trucks()
        
        # Update move cache
        self.available_moves_cache = self.find_moves_optimized()
        
        return np.array([trains_departed_early, trains_departed_late, trucks_departed])

    # ==================== TENSOR CONVERSION METHODS ====================
    
    def get_rail_state_tensor(self, as_tensor: bool = True) -> torch.Tensor:
        """Get rail occupancy state as tensor."""
        if as_tensor:
            return torch.from_numpy(self.dynamic_rail_mask.astype(np.float32)).to(self.device)
        return self.dynamic_rail_mask.astype(np.float32)
    
    def get_parking_state_tensor(self, as_tensor: bool = True) -> torch.Tensor:
        """Get parking occupancy state as tensor."""
        if as_tensor:
            return torch.from_numpy(self.dynamic_parking_mask.astype(np.float32)).to(self.device)
        return self.dynamic_parking_mask.astype(np.float32)
    
    def get_train_properties_tensor(self, as_tensor: bool = True) -> torch.Tensor:
        """Get train properties as tensor."""
        properties = np.stack([
            self.train_occupied.astype(np.float32),
            self.train_lengths.astype(np.float32),
            self.train_wagon_counts.astype(np.float32),
            self.train_container_counts.astype(np.float32),
            self.train_pickup_counts.astype(np.float32)
        ], axis=1)
        
        if as_tensor:
            return torch.from_numpy(properties).to(self.device)
        return properties
    
    def get_truck_properties_tensor(self, as_tensor: bool = True) -> torch.Tensor:
        """Get truck properties as tensor."""
        properties = np.stack([
            self.truck_container_counts.astype(np.float32),
            self.truck_pickup_counts.astype(np.float32),
            self.truck_weights
        ], axis=1)
        
        if as_tensor:
            return torch.from_numpy(properties).to(self.device)
        return properties
    
    def get_queue_state_tensor(self, as_tensor: bool = True) -> Dict[str, torch.Tensor]:
        """Get queue states as tensors."""
        train_queue_size = self.trains.size()
        truck_queue_size = self.trucks.size()
        
        queue_info = np.array([train_queue_size, truck_queue_size], dtype=np.float32)
        
        if as_tensor:
            return torch.from_numpy(queue_info).to(self.device)
        return queue_info

    # ==================== HELPER METHODS ====================
    
    def _check_consecutive_space(self, railtrack_id: int, required_length: int) -> bool:
        """Check if railtrack has enough consecutive free space."""
        if self.train_occupied[railtrack_id]:
            return False
            
        track_mask = self.dynamic_rail_mask[:, railtrack_id]
        
        # Find longest consecutive True sequence
        consecutive_count = 0
        max_consecutive = 0
        
        for available in track_mask:
            if available:
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                consecutive_count = 0
        
        return max_consecutive >= required_length
    
    def _occupy_train_positions(self, railtrack_id: int, train_length: int) -> List[Tuple]:
        """Occupy positions for train and return list of occupied positions."""
        positions = []
        positions_filled = 0
        
        for i in range(self.n_rows * self.split_factor):
            if self.dynamic_rail_mask[i, railtrack_id] and positions_filled < train_length:
                row = i // self.split_factor
                split = i % self.split_factor
                position = (row, railtrack_id, split)
                positions.append(position)
                self.dynamic_rail_mask[i, railtrack_id] = False
                positions_filled += 1
                
                if positions_filled >= train_length:
                    break
        
        return positions
    
    def _update_train_lookups(self, train: Train, railtrack_id: int):
        """Update container lookup tables for train."""
        for wagon_idx, wagon in enumerate(train.wagons):
            # Containers currently on train
            for container in wagon.containers:
                self.container_to_train[container.container_id] = (railtrack_id, wagon_idx)
            
            # Pickup requests
            for container_id in wagon.pickup_container_ids:
                self.pickup_to_train[container_id] = (railtrack_id, wagon_idx)
    
    def _update_truck_lookups(self, truck: Truck, position: Tuple):
        """Update container lookup tables for truck."""
        # Containers currently on truck
        for container in truck.containers:
            self.container_to_truck[container.container_id] = position
        
        # Pickup requests
        if hasattr(truck, 'pickup_container_ids'):
            for container_id in truck.pickup_container_ids:
                self.pickup_to_truck[container_id] = position
    
    def _clear_train_lookups(self, railtrack_id: int):
        """Clear lookup tables for removed train."""
        # Remove all entries for this railtrack
        containers_to_remove = [cid for cid, (rid, _) in self.container_to_train.items() if rid == railtrack_id]
        pickups_to_remove = [cid for cid, (rid, _) in self.pickup_to_train.items() if rid == railtrack_id]
        
        for cid in containers_to_remove:
            del self.container_to_train[cid]
        for cid in pickups_to_remove:
            del self.pickup_to_train[cid]
    
    def _clear_truck_lookups(self, position: Tuple):
        """Clear lookup tables for removed truck."""
        containers_to_remove = [cid for cid, pos in self.container_to_truck.items() if pos == position]
        pickups_to_remove = [cid for cid, pos in self.pickup_to_truck.items() if pos == position]
        
        for cid in containers_to_remove:
            del self.container_to_truck[cid]
        for cid in pickups_to_remove:
            del self.pickup_to_truck[cid]
    
    def _find_optimal_truck_position(self, truck: Truck) -> Optional[Tuple[int, int, int]]:
        """Find optimal parking position for truck based on pickup requirements."""
        if not hasattr(truck, 'pickup_container_ids') or not truck.pickup_container_ids:
            return self._find_first_available_parking_position()
        
        # Find positions parallel to wagons containing pickup containers
        target_positions = []
        
        for container_id in truck.pickup_container_ids:
            if container_id in self.pickup_to_train:
                railtrack_id, wagon_idx = self.pickup_to_train[container_id]
                # Calculate parallel parking position
                wagon_center = (wagon_idx + 1) * self.wagon_length + self.train_head_length // 2
                target_positions.append(wagon_center)
        
        if not target_positions:
            return self._find_first_available_parking_position()
        
        # Find best available position near targets
        for target_pos in target_positions:
            # Check parallel and ±1 positions
            for offset in [0, -1, 1]:
                pos_idx = target_pos + offset
                if 0 <= pos_idx < self.n_rows * self.split_factor:
                    if self.dynamic_parking_mask[pos_idx, 0]:
                        row = pos_idx // self.split_factor
                        split = pos_idx % self.split_factor
                        return (row, 0, split)  # railtrack 0 is parking
        
        return self._find_first_available_parking_position()
    
    def _find_first_available_parking_position(self) -> Optional[Tuple[int, int, int]]:
        """Find first available parking position."""
        for i in range(self.n_rows * self.split_factor):
            if self.dynamic_parking_mask[i, 0]:
                row = i // self.split_factor
                split = i % self.split_factor
                return (row, 0, split)
        return None
    
    def _find_truck_with_pickup_request(self, container_id: str) -> Optional[Truck]:
        """Find truck in queue that wants to pick up specific container."""
        # This would require iterating through queue - simplified for now
        return None
    
    def _find_wagon_with_container(self, train: Train, container_id: str) -> Optional[int]:
        """Find which wagon contains the pickup request for container."""
        for wagon_idx, wagon in enumerate(train.wagons):
            if container_id in wagon.pickup_container_ids:
                return wagon_idx
        return None
    
    def _calculate_parallel_positions(self, railtrack_id: int, wagon_idx: int) -> List[Tuple]:
        """Calculate parallel parking positions for wagon."""
        # Calculate wagon center position
        wagon_start = self.train_head_length + wagon_idx * self.wagon_length
        wagon_center = wagon_start + self.wagon_length // 2
        
        positions = []
        # Parallel and ±1 positions
        for offset in [0, -1, 1]:
            pos_idx = wagon_center + offset
            if 0 <= pos_idx < self.n_rows * self.split_factor:
                row = pos_idx // self.split_factor
                split = pos_idx % self.split_factor
                positions.append((row, 0, split))  # Parking is railtrack 0
        
        return positions
    
    def _is_position_available(self, position: Tuple[int, int, int]) -> bool:
        """Check if parking position is available."""
        row, railtrack, split = position
        if railtrack != 0:  # Only parking lane
            return False
        pos_idx = row * self.split_factor + split
        return self.dynamic_parking_mask[pos_idx, 0]
    
    def _remove_truck_from_queue(self, truck: Truck):
        """Remove specific truck from queue - simplified implementation."""
        # This would require queue modification - for now just pass
        pass
    
    def _find_yard_positions_for_container(self, container: 'Container') -> List[Tuple]:
        """OPTIMIZED: Find available positions in yard using cached results."""
        # Determine goods type for mask selection
        if container.goods_type == 'Reefer':
            goods_type = 'r'
        elif container.goods_type == 'Dangerous':
            goods_type = 'dg'
        elif container.container_type in ['Trailer', 'Swap Body']:
            goods_type = 'sb_t'
        else:
            goods_type = 'reg'
        
        # Use cached positions
        return self._get_cached_yard_positions(goods_type, container.container_type)
    
    def _find_container_in_yard(self, container_id: str) -> Optional[Tuple]:
        """Find container position in yard by ID using optimized lookup."""
        return self._find_container_in_yard_fast(container_id)
    
    def sync_yard_index(self):
        """Synchronize yard container index with current yard state."""
        self._rebuild_yard_container_index()
        self.yard_cache_dirty = True
    
    def add_container_to_yard(self, container: 'Container', coordinates: List[Tuple]) -> bool:
        """
        OPTIMIZED: Add container to yard and update index.
        
        Args:
            container: Container to add
            coordinates: List of (row, bay, split, tier) coordinates
            
        Returns:
            bool: Success of operation
        """
        try:
            # Add to yard
            self.yard.add_container(container, coordinates)
            
            # Update index with first coordinate (main position)
            if coordinates:
                main_pos = coordinates[0]
                self._update_yard_container_index(container.container_id, main_pos)
            
            return True
        except Exception:
            return False
    
    def remove_container_from_yard(self, coordinates: List[Tuple]) -> Optional['Container']:
        """
        OPTIMIZED: Remove container from yard and update index.
        
        Args:
            coordinates: List of (row, bay, split, tier) coordinates
            
        Returns:
            Removed container or None
        """
        try:
            # Remove from yard
            container = self.yard.remove_container(coordinates)
            
            # Update index
            if container:
                self._update_yard_container_index(container.container_id, None)
            
            return container
        except Exception:
            return None
    
    def _print_validation_info(self):
        """Print validation information if validate=True."""
        print("="*50)
        print("BOOLEAN LOGISTICS VALIDATION")
        print("="*50)
        print(f"Rows: {self.n_rows}")
        print(f"Rail tracks: {self.n_railtracks}")
        print(f"Split factor: {self.split_factor}")
        print(f"Train coordinates shape: {self.train_coords.shape}")
        print(f"Parking coordinates shape: {self.park_coords.shape}")
        print(f"Dynamic rail mask shape: {self.dynamic_rail_mask.shape}")
        print(f"Dynamic parking mask shape: {self.dynamic_parking_mask.shape}")
        print(f"Wagon length (positions): {self.wagon_length}")
        print(f"Train head length (positions): {self.train_head_length}")
        print("Initialization complete!")


if __name__ == '__main__':
    # Test the implementation
    from simulation.terminal_components.BooleanStorage import BooleanStorageYard
    
    # Create a test yard
    test_yard = BooleanStorageYard(
        n_rows=15,
        n_bays=20,
        n_tiers=3,
        coordinates=[
            (1, 1, "r"), (2, 1, "r"), (19, 1, "r"), (20, 1, "r"),
            (10, 8, "dg"), (11, 8, "dg"), (10, 9, "dg"), (11, 9, "dg"),
            (5, 1, "sb_t"), (6, 1, "sb_t"), (7, 1, "sb_t")
        ],
        split_factor=4,
        validate=False
    )
    
    # Create logistics manager
    logistics = BooleanLogistics(
        n_rows=15,
        n_railtracks=6,
        split_factor=4,
        yard=test_yard,
        validate=True
    )
    
    # Test basic functionality
    print("\nTesting basic operations...")
    
    # CREATE CONTAINERS IN YARD THAT MATCH PICKUP REQUESTS
    print("Adding test containers to yard...")
    
    # Create containers that will be picked up
    pickup_container_1 = ContainerFactory.create_container("PICKUP_001", "TWEU", "Export", "Regular")
    pickup_container_2 = ContainerFactory.create_container("PICKUP_002", "FEU", "Export", "Regular")
    
    # Add them to yard
    positions_1 = test_yard.search_insertion_position(5, 'reg', 'TWEU', 3)
    positions_2 = test_yard.search_insertion_position(8, 'reg', 'FEU', 3)
    
    if positions_1:
        coords_1 = test_yard.get_container_coordinates_from_placement(positions_1[0], 'TWEU')
        test_yard.add_container(pickup_container_1, coords_1)
        print(f"Added {pickup_container_1.container_id} to yard at {coords_1}")
    
    if positions_2:
        coords_2 = test_yard.get_container_coordinates_from_placement(positions_2[0], 'FEU')
        test_yard.add_container(pickup_container_2, coords_2)
        print(f"Added {pickup_container_2.container_id} to yard at {coords_2}")
    
    # CRITICAL: Sync the logistics yard index after manually adding containers
    logistics.sync_yard_index()
    print(f"Synced yard index: {len(logistics.yard_container_index)} containers indexed")
    
    # Create test train with pickup requests
    test_train = Train("TEST_TRAIN_001", num_wagons=5)
    test_train.wagons[0].add_pickup_container("PICKUP_001")  # Wants container in yard
    test_train.wagons[1].add_pickup_container("PICKUP_002")  # Wants container in yard
    
    # Create test truck with pickup requests  
    test_truck = Truck("TEST_TRUCK_001")
    test_truck.add_pickup_container_id("PICKUP_001")  # Also wants PICKUP_001
    
    # Add to queues
    logistics.add_train_to_queue(test_train)
    logistics.add_truck_to_queue(test_truck)
    
    # Process queues
    trains_placed = logistics.process_current_trains()
    trucks_placed = logistics.process_current_trucks()
    
    print(f"Trains placed: {trains_placed}")
    print(f"Trucks placed: {trucks_placed}")
    
    # Find moves
    moves = logistics.find_moves_optimized()
    print(f"Found {len(moves)} possible moves")
    
    # Show move details
    if moves:
        print("\nMove details:")
        for move_id, move in moves.items():
            print(f"  {move_id}: {move['container_id']} {move['source_type']} -> {move['dest_type']} ({move['move_type']})")
    
    # Test tensor conversion
    rail_tensor = logistics.get_rail_state_tensor()
    parking_tensor = logistics.get_parking_state_tensor()
    train_props = logistics.get_train_properties_tensor()
    
    print(f"\nTensor shapes:")
    print(f"Rail tensor shape: {rail_tensor.shape}")
    print(f"Parking tensor shape: {parking_tensor.shape}")
    print(f"Train properties shape: {train_props.shape}")
    
    print("\n✓ All tests completed successfully!")