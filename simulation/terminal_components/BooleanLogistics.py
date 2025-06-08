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
from datetime import datetime, timedelta
import heapq
import bisect

class BooleanLogistics:
    """
    Optimized logistics manager for rails, parking, and vehicle queues.
    Rails now properly span across bays with support for multiple trains per track.
    """
    
    def __init__(self,
                 n_rows: int,
                 n_bays: int,
                 n_railtracks: int,
                 split_factor: int,
                 yard: BooleanStorageYard,
                 validate: bool = False,
                 device: str = 'cuda'):
        
        self.n_rows = n_rows
        self.n_bays = n_bays  # Now properly stored
        self.n_railtracks = n_railtracks
        self.split_factor = split_factor
        self.yard = yard
        self.validate = validate
        self.device = device

        # FIXED: Rails span across BAYS, not rows
        # Shape: (n_railtracks, n_bays * split_factor)
        self.dynamic_rail_mask = np.ones((self.n_railtracks, self.n_bays * split_factor), dtype=bool)
        # Shape: (1, n_bays * split_factor)
        self.dynamic_parking_mask = np.ones((1, self.n_bays * split_factor), dtype=bool)

        # Create coordinate mappings
        self.rail_coords = self._create_rail_coordinate_mapping()
        self.park_coords = self._create_parking_coordinate_mapping()

        # Vehicle queues
        self.trucks = VehicleQueue('Truck')
        self.trains = VehicleQueue('Train')

        # ENHANCED: Track multiple trains per railtrack with sorted order
        self.trains_on_track = defaultdict(list)  # railtrack_id -> sorted list of (departure_time, position_range, train)
        self.train_to_track = {}  # train_id -> railtrack_id
        
        # Active vehicles tracking
        self.active_trucks = {}  # position_tuple -> Truck object
        self.truck_positions = {}  # position_tuple -> truck_id
        
        # Container lookup tables for O(1) access
        self.container_to_train = {}  # container_id -> (train_id, wagon_idx)
        self.container_to_truck = {}  # container_id -> truck_position
        self.pickup_to_train = {}     # container_id -> (train_id, wagon_idx)  
        self.pickup_to_truck = {}     # container_id -> truck_position
        
        # OPTIMIZATION: Fast container location index for yard
        self.yard_container_index = {}  # container_id -> (row, bay, tier, split)
        self.yard_container_set = set()  # Fast membership testing
        
        # OPTIMIZATION: Cached yard positions for different container types
        self.cached_yard_positions = defaultdict(list)
        self.yard_cache_dirty = True
        
        # Move cache
        self.available_moves_cache = {}
        
        # Wagon size constants
        self.wagon_length = 2 * split_factor  # 2 bays per wagon
        self.train_head_length = 1 * split_factor  # 1 bay for locomotive
        
        # Performance tracking arrays
        self._init_property_arrays()
        
        # Build initial yard index
        self._rebuild_yard_container_index()
        
        # Terminal trucks
        self.terminal_trucks = [
            {'id': 0, 'busy': False, 'completion_time': 0.0},
            {'id': 1, 'busy': False, 'completion_time': 0.0}
        ]
        
        if self.validate:
            self._print_validation_info()

    def _init_property_arrays(self):
        """Initialize arrays for fast tensor conversion."""
        # Arrays for each railtrack (now properly sized)
        max_trains_per_track = 10  # Reasonable limit
        self.track_train_counts = np.zeros(self.n_railtracks, dtype=np.int16)
        self.track_container_counts = np.zeros(self.n_railtracks, dtype=np.int16)
        self.track_pickup_counts = np.zeros(self.n_railtracks, dtype=np.int16)
        
        # Truck arrays
        self.truck_container_counts = np.zeros(self.n_bays * self.split_factor, dtype=np.int16)
        self.truck_pickup_counts = np.zeros(self.n_bays * self.split_factor, dtype=np.int16)
        self.truck_weights = np.zeros(self.n_bays * self.split_factor, dtype=np.float32)

    def _create_rail_coordinate_mapping(self) -> np.ndarray:
        """Create coordinate mapping for rails (railtrack, bay, split)."""
        coords = {}
        for railtrack in range(self.n_railtracks):
            for bay in range(self.n_bays):
                for split in range(self.split_factor):
                    pos_idx = bay * self.split_factor + split
                    coords[(railtrack, pos_idx)] = (railtrack, bay, split)
        return coords

    def _create_parking_coordinate_mapping(self) -> np.ndarray:
        """Create coordinate mapping for parking (0, bay, split)."""
        coords = {}
        for bay in range(self.n_bays):
            for split in range(self.split_factor):
                pos_idx = bay * self.split_factor + split
                coords[(0, pos_idx)] = (0, bay, split)  # Parking is "railtrack" 0
        return coords

    def print_masks(self):
        """Print all masks for visual verification."""
        np.set_printoptions(
            threshold=np.inf,
            linewidth=200,
            suppress=True,
            precision=0,
            formatter={'bool': lambda x: '█' if x else '░'}
        )
        
        print("\n" + "="*60)
        print("BOOLEAN LOGISTICS MASKS")
        print("="*60)
        
        print("\nRail Mask (True=Available):")
        print(f"Shape: {self.dynamic_rail_mask.shape}")
        for i, track in enumerate(self.dynamic_rail_mask):
            print(f"Track {i}: {track}")
        
        print("\nParking Mask (True=Available):")
        print(f"Shape: {self.dynamic_parking_mask.shape}")
        print(f"Parking: {self.dynamic_parking_mask[0]}")
        
        # Show train positions
        print("\nTrain Positions:")
        for track_id, trains in self.trains_on_track.items():
            if trains:
                print(f"  Track {track_id}:")
                for dep_time, (start, end), train in trains:
                    print(f"    Train {train.train_id}: positions [{start}-{end}), departs {dep_time}")
        
        print("\nActive Trucks:")
        for pos, truck in self.active_trucks.items():
            print(f"  Position {pos}: Truck {truck.truck_id}")
        
        print("="*60)

    def get_available_terminal_truck(self) -> Optional[int]:
        """Get first available terminal truck ID."""
        for truck in self.terminal_trucks:
            if not truck['busy']:
                return truck['id']
        return None

    def add_train_to_yard(self, train: Train, railtrack_id: int) -> bool:
        """
        Add train to yard at specified railtrack with proper ordering.
        
        Args:
            train: Train object to place
            railtrack_id: Which railtrack to place it on
            
        Returns:
            bool: Success of placement
        """
        if railtrack_id >= self.n_railtracks:
            return False
        
        # Calculate train length
        train_length = self.train_head_length + len(train.wagons) * self.wagon_length
        
        # Get departure time for ordering
        departure_time = getattr(train, 'departure_time', datetime.max)
        
        # Find position on track based on departure order
        position_start = self._find_train_position_on_track(railtrack_id, train_length, departure_time)
        
        if position_start is None:
            return False
        
        # Mark positions as occupied
        position_end = position_start + train_length
        self.dynamic_rail_mask[railtrack_id, position_start:position_end] = False
        
        # Update tracking structures
        train_entry = (departure_time, (position_start, position_end), train)
        bisect.insort(self.trains_on_track[railtrack_id], train_entry)
        self.train_to_track[train.train_id] = railtrack_id
        
        # Update property arrays
        self.track_train_counts[railtrack_id] += 1
        container_count = sum(len(wagon.containers) for wagon in train.wagons)
        pickup_count = sum(len(wagon.pickup_container_ids) for wagon in train.wagons)
        self.track_container_counts[railtrack_id] += container_count
        self.track_pickup_counts[railtrack_id] += pickup_count
        
        # Update lookup tables
        self._update_train_lookups(train)
        
        if self.validate:
            print(f"Added train {train.train_id} to track {railtrack_id} at positions [{position_start}, {position_end})")
        
        return True

    def _find_train_position_on_track(self, railtrack_id: int, train_length: int, departure_time: datetime) -> Optional[int]:
        """Find optimal position for train based on departure order."""
        track_length = self.n_bays * self.split_factor
        trains = self.trains_on_track[railtrack_id]
        
        if not trains:
            # Empty track - place at start
            if train_length <= track_length:
                return 0
            return None
        
        # Find all occupied positions
        occupied_ranges = [(start, end) for _, (start, end), _ in trains]
        occupied_ranges.sort()
        
        # Try to find a gap that fits the train
        # Check space at the beginning
        if occupied_ranges[0][0] >= train_length:
            return 0
        
        # Check gaps between trains
        for i in range(len(occupied_ranges) - 1):
            gap_start = occupied_ranges[i][1]
            gap_end = occupied_ranges[i + 1][0]
            gap_size = gap_end - gap_start
            
            if gap_size >= train_length:
                return gap_start
        
        # Check space at the end
        last_end = occupied_ranges[-1][1]
        if track_length - last_end >= train_length:
            return last_end
        
        return None

    def remove_train(self, train_id: str) -> Optional[Train]:
        """Remove train from railtrack."""
        if train_id not in self.train_to_track:
            return None
        
        railtrack_id = self.train_to_track[train_id]
        trains = self.trains_on_track[railtrack_id]
        
        # Find and remove train
        for i, (dep_time, (start, end), train) in enumerate(trains):
            if train.train_id == train_id:
                # Free positions
                self.dynamic_rail_mask[railtrack_id, start:end] = True
                
                # Remove from lists
                trains.pop(i)
                del self.train_to_track[train_id]
                
                # Update property arrays
                self.track_train_counts[railtrack_id] -= 1
                container_count = sum(len(wagon.containers) for wagon in train.wagons)
                pickup_count = sum(len(wagon.pickup_container_ids) for wagon in train.wagons)
                self.track_container_counts[railtrack_id] -= container_count
                self.track_pickup_counts[railtrack_id] -= pickup_count
                
                # Clear lookup tables
                self._clear_train_lookups(train)
                
                return train
        
        return None

    def add_train_to_queue(self, train: Train):
        """Add train to train queue."""
        self.trains.add_vehicle(train)
        if self.validate:
            print(f"Train {train.train_id} added to queue. Queue size: {self.trains.size()}")

    def add_truck_to_queue(self, truck: Truck):
        """Add truck to truck queue."""
        self.trucks.add_vehicle(truck)
        if self.validate:
            print(f"Truck {truck.truck_id} added to queue. Queue size: {self.trucks.size()}")

    def add_truck_to_yard(self, truck: Truck, position: Tuple[int, int, int]) -> bool:
        """Add truck to parking position."""
        _, bay, split = position  # Ignore first element (should be 0 for parking)
        pos_idx = bay * self.split_factor + split
        
        if pos_idx >= self.n_bays * self.split_factor:
            return False
        
        if not self.dynamic_parking_mask[0, pos_idx]:
            return False
        
        # Mark position as occupied
        self.dynamic_parking_mask[0, pos_idx] = False
        
        # Update tracking
        self.active_trucks[position] = truck
        self.truck_positions[position] = truck.truck_id
        
        # Update property arrays
        self.truck_container_counts[pos_idx] = len(truck.containers)
        self.truck_pickup_counts[pos_idx] = len(getattr(truck, 'pickup_container_ids', set()))
        self.truck_weights[pos_idx] = sum(c.weight for c in truck.containers) if truck.containers else 0.0
        
        # Update lookup tables
        self._update_truck_lookups(truck, position)
        
        return True

    def remove_truck(self, position: Tuple[int, int, int]) -> Optional[Truck]:
        """Remove truck from parking position."""
        if position not in self.active_trucks:
            return None
        
        truck = self.active_trucks[position]
        _, bay, split = position
        pos_idx = bay * self.split_factor + split
        
        # Free position
        self.dynamic_parking_mask[0, pos_idx] = True
        
        # Clear tracking
        del self.active_trucks[position]
        if position in self.truck_positions:
            del self.truck_positions[position]
        
        # Clear property arrays
        self.truck_container_counts[pos_idx] = 0
        self.truck_pickup_counts[pos_idx] = 0
        self.truck_weights[pos_idx] = 0.0
        
        # Clear lookup tables
        self._clear_truck_lookups(position)
        
        return truck

    def process_current_trains(self) -> int:
        """Process trains in queue and assign them to tracks."""
        placed_count = 0
        
        while not self.trains.is_empty():
            train = self.trains.get_next_vehicle()
            if train is None:
                break
            
            # Try each railtrack
            placed = False
            for railtrack_id in range(self.n_railtracks):
                if self.add_train_to_yard(train, railtrack_id):
                    placed_count += 1
                    placed = True
                    break
            
            if not placed:
                # Return to queue
                self.trains.add_vehicle(train)
                break
        
        return placed_count

    def process_current_trucks(self) -> int:
        """Process trucks in queue."""
        placed_count = 0
        
        while not self.trucks.is_empty():
            truck = self.trucks.get_next_vehicle()
            if truck is None:
                break
            
            # Find optimal position
            optimal_pos = self._find_optimal_truck_position(truck)
            
            if optimal_pos is not None:
                if self.add_truck_to_yard(truck, optimal_pos):
                    placed_count += 1
                else:
                    self.trucks.add_vehicle(truck)
                    break
            else:
                self.trucks.add_vehicle(truck)
                break
        
        return placed_count

    def find_moves_optimized(self) -> Dict[str, Dict[str, Any]]:
        """Find all possible moves with optimized algorithms."""
        all_moves = {}
        move_counter = 0
        
        # Collect containers from all sources
        train_containers = []
        for track_trains in self.trains_on_track.values():
            for _, _, train in track_trains:
                for wagon_idx, wagon in enumerate(train.wagons):
                    for container in wagon.containers:
                        train_containers.append((container, train.train_id, wagon_idx))
        
        truck_containers = []
        for truck_pos, truck in self.active_trucks.items():
            for container in truck.containers:
                truck_containers.append((container, truck_pos))
        
        # Process pickup requests
        pickup_requests = set()
        pickup_requests.update(self.pickup_to_train.keys())
        pickup_requests.update(self.pickup_to_truck.keys())
        
        # Find yard containers with pickup requests
        yard_pickup_containers = pickup_requests & self.yard_container_set
        
        # 1. Train -> Truck moves
        for container, train_id, wagon_idx in train_containers:
            if container.container_id in self.pickup_to_truck:
                truck_pos = self.pickup_to_truck[container.container_id]
                move_id = f"move_{move_counter}"
                all_moves[move_id] = {
                    'container_id': container.container_id,
                    'source_type': 'train',
                    'source_pos': (train_id, wagon_idx),
                    'dest_type': 'truck',
                    'dest_pos': truck_pos,
                    'move_type': 'train_to_truck',
                    'priority': 10.0
                }
                move_counter += 1
        
        # 2. Truck -> Train moves
        for container, truck_pos in truck_containers:
            if container.container_id in self.pickup_to_train:
                train_id, wagon_idx = self.pickup_to_train[container.container_id]
                move_id = f"move_{move_counter}"
                all_moves[move_id] = {
                    'container_id': container.container_id,
                    'source_type': 'truck',
                    'source_pos': truck_pos,
                    'dest_type': 'train',
                    'dest_pos': (train_id, wagon_idx),
                    'move_type': 'truck_to_train',
                    'priority': 10.0
                }
                move_counter += 1
        
        # 3. Yard -> Vehicle moves
        for container_id in yard_pickup_containers:
            container_coords = self._get_yard_container_coords(container_id)
            if not container_coords:
                continue
            
            # Check train pickups
            if container_id in self.pickup_to_train:
                train_id, wagon_idx = self.pickup_to_train[container_id]
                move_id = f"move_{move_counter}"
                all_moves[move_id] = {
                    'container_id': container_id,
                    'source_type': 'yard',
                    'source_pos': container_coords,
                    'dest_type': 'train',
                    'dest_pos': (train_id, wagon_idx),
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
                    'source_pos': container_coords,
                    'dest_type': 'truck',
                    'dest_pos': truck_pos,
                    'move_type': 'from_yard',
                    'priority': 9.0
                }
                move_counter += 1
        
        # 4. Vehicle -> Yard moves
        import_containers = []
        
        for container, train_id, wagon_idx in train_containers:
            if hasattr(container, 'direction') and container.direction == 'Import':
                import_containers.append((container, 'train', (train_id, wagon_idx)))
        
        for container, truck_pos in truck_containers:
            if hasattr(container, 'direction') and container.direction == 'Import':
                import_containers.append((container, 'truck', truck_pos))
        
        # Process imports with caching
        for container, source_type, source_pos in import_containers:
            positions = self._get_yard_positions_for_container(container)
            
            for position in positions[:2]:  # Limit options
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
        
        # 5. Yard reshuffling
        yard_moves = self.yard.return_possible_yard_moves(max_proximity=3)
        
        for container_id, move_data in yard_moves.items():
            source_coords = move_data['source_coords']
            destinations = move_data['destinations']
            
            for dest in destinations[:3]:
                if dest != source_coords[0][:3]:
                    move_id = f"move_{move_counter}"
                    all_moves[move_id] = {
                        'container_id': container_id,
                        'source_type': 'yard',
                        'source_pos': source_coords,
                        'dest_type': 'yard',
                        'dest_pos': dest,
                        'move_type': 'yard_to_yard',
                        'priority': 5.0
                    }
                    move_counter += 1
        
        # 6. Yard -> Stack (terminal trucks)
        available_truck = self.get_available_terminal_truck()
        if available_truck is not None:
            sb_t_count = 0
            for container_id, pos in list(self.yard_container_index.items())[:20]:  # Limit scan
                if sb_t_count >= 5:
                    break
                
                container_coords = self._get_yard_container_coords(container_id)
                if not container_coords:
                    continue
                
                row, bay, tier, split = pos
                try:
                    container = self.yard.get_container_at(row, bay, tier, split)
                except IndexError:
                    continue
                
                if container and container.container_type in ['Swap Body', 'Trailer']:
                    if (container_id not in self.pickup_to_train and 
                        container_id not in self.pickup_to_truck):
                        
                        move_id = f"move_{move_counter}"
                        all_moves[move_id] = {
                            'container_id': container_id,
                            'source_type': 'yard',
                            'source_pos': container_coords,
                            'dest_type': 'stack',
                            'dest_pos': None,
                            'move_type': 'yard_to_stack',
                            'priority': 3.0
                        }
                        move_counter += 1
                        sb_t_count += 1
        
        return all_moves

    def reorganize_logistics(self) -> Tuple[np.ndarray, float]:
        """Reorganize logistics after moves with penalty system."""
        trains_departed_early = 0
        trains_departed_late = 0
        trucks_departed = 0
        total_penalty = 0.0
        
        current_time = datetime.now()
        
        # Check all trains across all tracks
        trains_to_remove = []
        for track_id, track_trains in self.trains_on_track.items():
            for dep_time, pos_range, train in track_trains:
                ready_to_depart = all(
                    len(wagon.pickup_container_ids) == 0 
                    for wagon in train.wagons
                )
                
                if dep_time and current_time >= dep_time:
                    # Forced departure
                    if ready_to_depart:
                        trains_departed_late += 1
                    else:
                        # Calculate penalty
                        unfulfilled = sum(len(w.pickup_container_ids) for w in train.wagons)
                        undelivered = sum(len(w.containers) for w in train.wagons)
                        penalty = unfulfilled * 5.0 + undelivered * 3.0 + 10.0
                        total_penalty -= penalty
                        trains_departed_late += 1
                        
                        # Salvage containers
                        self._salvage_train_containers(train)
                    
                    trains_to_remove.append(train.train_id)
                    
                elif ready_to_depart and dep_time and current_time <= dep_time:
                    trains_departed_early += 1
                    trains_to_remove.append(train.train_id)
        
        # Remove departed trains
        for train_id in trains_to_remove:
            self.remove_train(train_id)
        
        # Check trucks
        trucks_to_remove = []
        for truck_pos, truck in list(self.active_trucks.items()):
            pickup_complete = len(getattr(truck, 'pickup_container_ids', set())) == 0
            delivery_complete = len(truck.containers) == 0
            
            arrival_time = getattr(truck, 'arrival_time', current_time)
            time_in_terminal = (current_time - arrival_time).total_seconds() / 3600.0
            
            if time_in_terminal > 8.0 and not (pickup_complete and delivery_complete):
                # Penalty
                unfulfilled = len(getattr(truck, 'pickup_container_ids', set()))
                undelivered = len(truck.containers)
                penalty = unfulfilled * 3.0 + undelivered * 2.0 + 5.0
                total_penalty -= penalty
                
                self._salvage_truck_containers(truck_pos, truck)
                trucks_departed += 1
                trucks_to_remove.append(truck_pos)
                
            elif pickup_complete and delivery_complete:
                trucks_departed += 1
                trucks_to_remove.append(truck_pos)
        
        # Remove departed trucks
        for truck_pos in trucks_to_remove:
            self.remove_truck(truck_pos)
        
        # Update terminal trucks
        for truck in self.terminal_trucks:
            if truck['busy'] and truck['completion_time'] <= 0:
                truck['busy'] = False
            elif truck['busy']:
                truck['completion_time'] -= 1
        
        # Process queues
        self.process_current_trains()
        self.process_current_trucks()
        
        # Update move cache
        self.available_moves_cache = self.find_moves_optimized()
        
        return np.array([trains_departed_early, trains_departed_late, trucks_departed]), total_penalty

    # ==================== HELPER METHODS ====================

    def _update_train_lookups(self, train: Train):
        """Update lookup tables for train containers."""
        train_id = train.train_id
        for wagon_idx, wagon in enumerate(train.wagons):
            for container in wagon.containers:
                self.container_to_train[container.container_id] = (train_id, wagon_idx)
            for container_id in wagon.pickup_container_ids:
                self.pickup_to_train[container_id] = (train_id, wagon_idx)

    def _clear_train_lookups(self, train: Train):
        """Clear lookup tables for train."""
        train_id = train.train_id
        
        # Remove container mappings
        to_remove = [cid for cid, (tid, _) in self.container_to_train.items() if tid == train_id]
        for cid in to_remove:
            del self.container_to_train[cid]
        
        # Remove pickup mappings
        to_remove = [cid for cid, (tid, _) in self.pickup_to_train.items() if tid == train_id]
        for cid in to_remove:
            del self.pickup_to_train[cid]

    def _update_truck_lookups(self, truck: Truck, position: Tuple):
        """Update lookup tables for truck."""
        for container in truck.containers:
            self.container_to_truck[container.container_id] = position
        
        if hasattr(truck, 'pickup_container_ids'):
            for container_id in truck.pickup_container_ids:
                self.pickup_to_truck[container_id] = position

    def _clear_truck_lookups(self, position: Tuple):
        """Clear lookup tables for truck."""
        to_remove = [cid for cid, pos in self.container_to_truck.items() if pos == position]
        for cid in to_remove:
            del self.container_to_truck[cid]
        
        to_remove = [cid for cid, pos in self.pickup_to_truck.items() if pos == position]
        for cid in to_remove:
            del self.pickup_to_truck[cid]

    def _find_optimal_truck_position(self, truck: Truck) -> Optional[Tuple[int, int, int]]:
        """Find optimal parking position for truck."""
        if not hasattr(truck, 'pickup_container_ids') or not truck.pickup_container_ids:
            return self._find_first_available_parking_position()
        
        # Find target positions based on pickup requests
        target_bays = []
        for container_id in truck.pickup_container_ids:
            if container_id in self.pickup_to_train:
                train_id, wagon_idx = self.pickup_to_train[container_id]
                # Find train position
                railtrack_id = self.train_to_track.get(train_id)
                if railtrack_id is not None:
                    for _, (start, end), train in self.trains_on_track[railtrack_id]:
                        if train.train_id == train_id:
                            wagon_pos = start + self.train_head_length + wagon_idx * self.wagon_length
                            target_bay = wagon_pos // self.split_factor
                            target_bays.append(target_bay)
                            break
        
        if not target_bays:
            return self._find_first_available_parking_position()
        
        # Find best available position near targets
        for target_bay in target_bays:
            for offset in [0, -1, 1, -2, 2]:
                check_bay = target_bay + offset
                if 0 <= check_bay < self.n_bays:
                    for split in range(self.split_factor):
                        pos_idx = check_bay * self.split_factor + split
                        if self.dynamic_parking_mask[0, pos_idx]:
                            return (0, int(check_bay), int(split))  # Convert to Python int
        
        return self._find_first_available_parking_position()

    def _find_first_available_parking_position(self) -> Optional[Tuple[int, int, int]]:
        """Find first available parking position."""
        available = np.where(self.dynamic_parking_mask[0])[0]
        if len(available) > 0:
            pos_idx = int(available[0])  # Convert to Python int
            bay = int(pos_idx // self.split_factor)
            split = int(pos_idx % self.split_factor)
            return (0, bay, split)
        return None

    def _get_yard_container_coords(self, container_id: str) -> List[Tuple]:
        """Get full coordinates for container in yard."""
        if container_id not in self.yard_container_index:
            return []
        
        row, bay, tier, split = self.yard_container_index[container_id]
        
        # Find actual container and its full coordinates
        try:
            container = self.yard.get_container_at(row, bay, tier, split)
            if container and container.container_id == container_id:
                container_length = self.yard.container_lengths.get(container.container_type, 1)
                coords = []
                
                # Find actual start position
                for start_split in range(max(0, split - container_length + 1), 
                                       min(self.yard.split_factor, split + 1)):
                    valid = True
                    test_coords = []
                    
                    for i in range(container_length):
                        c_bay = bay + (start_split + i) // self.yard.split_factor
                        c_split = (start_split + i) % self.yard.split_factor
                        
                        if c_bay < self.yard.n_bays and c_split < self.yard.split_factor:
                            test_container = self.yard.get_container_at(row, c_bay, tier, c_split)
                            if test_container and test_container.container_id == container_id:
                                test_coords.append((row, c_bay, c_split, tier))
                            else:
                                valid = False
                                break
                        else:
                            valid = False
                            break
                    
                    if valid and len(test_coords) == container_length:
                        return test_coords
        except:
            pass
        
        return []

    def _get_yard_positions_for_container(self, container: Container) -> List[Tuple]:
        """Get available yard positions for container type."""
        goods_type = 'r' if container.goods_type == 'Reefer' else \
                    'dg' if container.goods_type == 'Dangerous' else \
                    'sb_t' if container.container_type in ['Trailer', 'Swap Body'] else 'reg'
        
        cache_key = (goods_type, container.container_type)
        
        if self.yard_cache_dirty or cache_key not in self.cached_yard_positions:
            center_bay = self.yard.n_bays // 2
            try:
                positions = self.yard.search_insertion_position(
                    center_bay, goods_type, container.container_type, max_proximity=5
                )
                self.cached_yard_positions[cache_key] = positions[:10]
            except:
                self.cached_yard_positions[cache_key] = []
        
        return self.cached_yard_positions[cache_key]

    def _salvage_train_containers(self, train: Train):
        """Salvage containers from departing train."""
        salvaged_count = 0
        for wagon in train.wagons:
            containers_to_salvage = list(wagon.containers)
            for container in containers_to_salvage:
                wagon.remove_container(container.container_id)
                if self._place_container_in_yard_from_vehicle(container):
                    salvaged_count += 1
                    if container.container_id in self.pickup_to_train:
                        del self.pickup_to_train[container.container_id]

    def _salvage_truck_containers(self, truck_pos: Tuple, truck: Truck):
        """Salvage containers from departing truck."""
        containers_to_salvage = list(truck.containers)
        for container in containers_to_salvage:
            truck.remove_container(container.container_id)
            if self._place_container_in_yard_from_vehicle(container):
                if container.container_id in self.pickup_to_truck:
                    del self.pickup_to_truck[container.container_id]

    def _place_container_in_yard_from_vehicle(self, container: Container) -> bool:
        """Place salvaged container in yard."""
        positions = self._get_yard_positions_for_container(container)
        
        for placement in positions[:5]:  # Try up to 5 positions
            coords = self.yard.get_container_coordinates_from_placement(
                placement, container.container_type
            )
            try:
                self.yard.add_container(container, coords)
                self._update_yard_container_index(container.container_id, coords[0])
                return True
            except:
                continue
        
        return False

    def _rebuild_yard_container_index(self):
        """Rebuild container index for yard."""
        self.yard_container_index.clear()
        self.yard_container_set.clear()
        
        indexed_containers = set()
        
        for row in range(self.yard.n_rows):
            for bay in range(self.yard.n_bays):
                for tier in range(self.yard.n_tiers):
                    for split in range(self.yard.split_factor):
                        container = self.yard.get_container_at(row, bay, tier, split)
                        if container and container.container_id not in indexed_containers:
                            self.yard_container_index[container.container_id] = (row, bay, tier, split)
                            self.yard_container_set.add(container.container_id)
                            indexed_containers.add(container.container_id)

    def _update_yard_container_index(self, container_id: str, position: Tuple = None):
        """Update yard container index."""
        if position is None:
            self.yard_container_index.pop(container_id, None)
            self.yard_container_set.discard(container_id)
        else:
            self.yard_container_index[container_id] = position
            self.yard_container_set.add(container_id)
        self.yard_cache_dirty = True

    def sync_yard_index(self):
        """Synchronize yard container index."""
        self._rebuild_yard_container_index()
        self.yard_cache_dirty = True

    def add_container_to_yard(self, container: Container, coordinates: List[Tuple]) -> bool:
        """Add container to yard."""
        try:
            self.yard.add_container(container, coordinates)
            if coordinates:
                self._update_yard_container_index(container.container_id, coordinates[0])
            return True
        except:
            return False

    def remove_container_from_yard(self, coordinates: List[Tuple]) -> Optional[Container]:
        """Remove container from yard."""
        try:
            container = self.yard.remove_container(coordinates)
            if container:
                self._update_yard_container_index(container.container_id, None)
            return container
        except:
            return None

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
            self.track_train_counts.astype(np.float32),
            self.track_container_counts.astype(np.float32),
            self.track_pickup_counts.astype(np.float32)
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
    
    def get_queue_state_tensor(self, as_tensor: bool = True) -> torch.Tensor:
        """Get queue states as tensor."""
        queue_info = np.array([
            self.trains.size(),
            self.trucks.size()
        ], dtype=np.float32)
        
        if as_tensor:
            return torch.from_numpy(queue_info).to(self.device)
        return queue_info

    def _print_validation_info(self):
        """Print validation information."""
        print("="*60)
        print("BOOLEAN LOGISTICS VALIDATION")
        print("="*60)
        print(f"Rows: {self.n_rows}")
        print(f"Bays: {self.n_bays}")
        print(f"Rail tracks: {self.n_railtracks}")
        print(f"Split factor: {self.split_factor}")
        print(f"Rail mask shape: {self.dynamic_rail_mask.shape}")
        print(f"Parking mask shape: {self.dynamic_parking_mask.shape}")
        print(f"Wagon length (positions): {self.wagon_length}")
        print(f"Train head length (positions): {self.train_head_length}")
        print("Initialization complete!")
        print("="*60)

if __name__ == '__main__':
    # Test the implementation
    from simulation.terminal_components.BooleanStorage import BooleanStorageYard
    from datetime import datetime, timedelta
    
    print("="*80)
    print("TESTING REFACTORED BOOLEAN LOGISTICS")
    print("="*80)
    
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
    
    # Create logistics manager with proper n_bays parameter
    logistics = BooleanLogistics(
        n_rows=15,
        n_bays=20,  # Now properly passed
        n_railtracks=6,
        split_factor=4,
        yard=test_yard,
        validate=True
    )
    
    # Print initial state
    print("\nInitial logistics state:")
    logistics.print_masks()
    
    # Test basic functionality
    print("\n" + "="*60)
    print("TESTING BASIC OPERATIONS")
    print("="*60)
    
    # CREATE CONTAINERS IN YARD THAT MATCH PICKUP REQUESTS
    print("\n1. Adding test containers to yard...")
    
    # Create containers that will be picked up
    pickup_container_1 = ContainerFactory.create_container("PICKUP_001", "TWEU", "Export", "Regular")
    pickup_container_2 = ContainerFactory.create_container("PICKUP_002", "FEU", "Export", "Regular")
    pickup_container_3 = ContainerFactory.create_container("PICKUP_003", "THEU", "Export", "Regular")
    
    # Add them to yard
    positions_1 = test_yard.search_insertion_position(5, 'reg', 'TWEU', 3)
    positions_2 = test_yard.search_insertion_position(8, 'reg', 'FEU', 3)
    positions_3 = test_yard.search_insertion_position(10, 'reg', 'THEU', 3)
    
    if positions_1:
        coords_1 = test_yard.get_container_coordinates_from_placement(positions_1[0], 'TWEU')
        test_yard.add_container(pickup_container_1, coords_1)
        print(f"  Added {pickup_container_1.container_id} to yard at {coords_1}")
    
    if positions_2:
        coords_2 = test_yard.get_container_coordinates_from_placement(positions_2[0], 'FEU')
        test_yard.add_container(pickup_container_2, coords_2)
        print(f"  Added {pickup_container_2.container_id} to yard at {coords_2}")
    
    if positions_3:
        coords_3 = test_yard.get_container_coordinates_from_placement(positions_3[0], 'THEU')
        test_yard.add_container(pickup_container_3, coords_3)
        print(f"  Added {pickup_container_3.container_id} to yard at {coords_3}")
    
    # CRITICAL: Sync the logistics yard index after manually adding containers
    logistics.sync_yard_index()
    print(f"\nSynced yard index: {len(logistics.yard_container_index)} containers indexed")
    
    # Test multiple trains with different departure times
    print("\n2. Creating test trains with departure priorities...")
    
    current_time = datetime.now()
    
    # Create test train 1 - departs in 2 hours
    test_train_1 = Train("TEST_TRAIN_001", num_wagons=3)
    test_train_1.departure_time = current_time + timedelta(hours=2)
    test_train_1.wagons[0].add_pickup_container("PICKUP_001")
    test_train_1.wagons[1].add_pickup_container("PICKUP_002")
    print(f"  Train 1: {test_train_1.train_id}, departs at {test_train_1.departure_time.strftime('%H:%M')}")
    
    # Create test train 2 - departs in 1 hour (should be placed first)
    test_train_2 = Train("TEST_TRAIN_002", num_wagons=2)
    test_train_2.departure_time = current_time + timedelta(hours=1)
    test_train_2.wagons[0].add_pickup_container("PICKUP_003")
    print(f"  Train 2: {test_train_2.train_id}, departs at {test_train_2.departure_time.strftime('%H:%M')}")
    
    # Create test train 3 - departs in 3 hours
    test_train_3 = Train("TEST_TRAIN_003", num_wagons=2)
    test_train_3.departure_time = current_time + timedelta(hours=3)
    # No pickup requests, just has containers to deliver
    import_container = ContainerFactory.create_container("IMPORT_001", "FEU", "Import", "Regular")
    test_train_3.wagons[0].add_container(import_container)
    print(f"  Train 3: {test_train_3.train_id}, departs at {test_train_3.departure_time.strftime('%H:%M')}")
    
    # Create test trucks
    print("\n3. Creating test trucks...")
    
    test_truck_1 = Truck("TEST_TRUCK_001")
    test_truck_1.add_pickup_container_id("PICKUP_001")  # Wants same container as train 1
    print(f"  Truck 1: {test_truck_1.truck_id}, wants {test_truck_1.pickup_container_ids}")
    
    test_truck_2 = Truck("TEST_TRUCK_002")
    # Truck 2 brings a container for delivery
    delivery_container = ContainerFactory.create_container("DELIVERY_001", "TWEU", "Import", "Regular")
    test_truck_2.add_container(delivery_container)
    print(f"  Truck 2: {test_truck_2.truck_id}, delivering {delivery_container.container_id}")
    
    # Add vehicles to queues
    print("\n4. Adding vehicles to queues...")
    logistics.add_train_to_queue(test_train_1)
    logistics.add_train_to_queue(test_train_2)
    logistics.add_train_to_queue(test_train_3)
    logistics.add_truck_to_queue(test_truck_1)
    logistics.add_truck_to_queue(test_truck_2)
    
    # Process queues
    print("\n5. Processing vehicle queues...")
    trains_placed = logistics.process_current_trains()
    trucks_placed = logistics.process_current_trucks()
    
    print(f"\nTrains placed: {trains_placed}")
    print(f"Trucks placed: {trucks_placed}")
    
    # Show current state
    print("\nCurrent logistics state after placement:")
    logistics.print_masks()
    
    # Find moves
    print("\n6. Finding possible moves...")
    moves = logistics.find_moves_optimized()
    print(f"Found {len(moves)} possible moves")
    
    # Categorize and show moves
    if moves:
        move_types = defaultdict(list)
        for move_id, move in moves.items():
            move_types[move['move_type']].append((move_id, move))
        
        print("\nMove breakdown by type:")
        for move_type, move_list in move_types.items():
            print(f"\n  {move_type}: {len(move_list)} moves")
            for move_id, move in move_list[:3]:  # Show first 3 of each type
                print(f"    {move_id}: {move['container_id']} "
                      f"({move['source_type']} -> {move['dest_type']})")
    
    # Test tensor conversion
    print("\n7. Testing tensor conversion...")
    rail_tensor = logistics.get_rail_state_tensor()
    parking_tensor = logistics.get_parking_state_tensor()
    train_props = logistics.get_train_properties_tensor()
    truck_props = logistics.get_truck_properties_tensor()
    queue_tensor = logistics.get_queue_state_tensor()
    
    print(f"\nTensor shapes:")
    print(f"  Rail tensor: {rail_tensor.shape} - dtype: {rail_tensor.dtype}")
    print(f"  Parking tensor: {parking_tensor.shape} - dtype: {parking_tensor.dtype}")
    print(f"  Train properties: {train_props.shape} - dtype: {train_props.dtype}")
    print(f"  Truck properties: {truck_props.shape} - dtype: {truck_props.dtype}")
    print(f"  Queue state: {queue_tensor.shape} - dtype: {queue_tensor.dtype}")
    
    # Test reorganization with departures
    print("\n8. Testing reorganization and departures...")
    
    # Simulate time passing - force a train departure
    test_train_2.departure_time = current_time - timedelta(minutes=1)  # Make it overdue
    
    summary, penalty = logistics.reorganize_logistics()
    early_departures, late_departures, truck_departures = summary
    
    print(f"\nReorganization results:")
    print(f"  Trains departed early: {early_departures}")
    print(f"  Trains departed late: {late_departures}")
    print(f"  Trucks departed: {truck_departures}")
    print(f"  Total penalty: {penalty}")
    
    # Show final state
    print("\nFinal logistics state:")
    logistics.print_masks()
    
    # Performance test
    print("\n9. Performance test - finding moves 100 times...")
    import time
    
    start_time = time.time()
    for _ in range(100):
        moves = logistics.find_moves_optimized()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    print(f"Average time to find moves: {avg_time*1000:.2f} ms")
    print(f"Moves per second: {1/avg_time:.1f}")
    
    print("\n" + "="*80)
    print("✓ All tests completed successfully!")
    print("="*80)