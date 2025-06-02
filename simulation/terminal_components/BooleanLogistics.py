from typing import Dict, Tuple, List, Optional, Set, Any
from simulation.terminal_components.Container import Container, ContainerFactory
from simulation.terminal_components.Train import Train
from simulation.terminal_components.Truck import Truck
import numpy as np
import torch
from collections import defaultdict, deque
import time

class BoolLogistics:
    """
    Optimized logistics manager for rails, parking, and vehicle queues.
    Focus on O(1) operations and memory-speed tradeoffs.
    """
    
    def __init__(self, 
                 n_rows: int,
                 split_factor: int, 
                 n_rail_tracks: int = 6,
                 n_parking_spots: int = 29,
                 device: str = 'cuda'):
        
        self.n_rows = n_rows
        self.split_factor = split_factor
        self.n_rail_tracks = n_rail_tracks
        self.n_parking_spots = n_parking_spots
        self.device = device
        
        # Rail and parking dimensions
        self.rail_length = n_rows * split_factor  # Each rail track spans all rows
        
        # OPTIMIZED: Direct masks for rails and parking
        # Rails: (rail_length, n_rail_tracks) - each track can hold vehicles
        self.dynamic_rails = np.ones((self.rail_length, n_rail_tracks), dtype=bool)
        
        # Parking: (rail_length, 1) - parallel to rails
        self.dynamic_parking = np.ones((self.rail_length, 1), dtype=bool)
        
        # Initialize vehicle storage arrays
        self._init_vehicle_arrays()
        
        # Container combination rules for 80-foot slots (2 bays * split_factor)
        self.valid_combinations = {
            'TWEU_4': ['TWEU', 'TWEU', 'TWEU', 'TWEU'],
            'TWEU_2_FEU': ['TWEU', 'TWEU', 'FEU'],
            'THEU_FFEU': ['THEU', 'FFEU']
        }
        
        # Pre-compute combination masks for O(1) validation
        self._precompute_combination_masks()
        
        # Queue management
        self.train_queue = deque()  # Waiting trains
        self.truck_queue = deque()  # Waiting trucks
        
        # Active vehicles on rails/parking
        self.active_trains = {}  # track_id -> Train object
        self.active_trucks = {}  # spot_id -> Truck object
        
        # ID lookup tables for O(1) container matching
        self._build_lookup_tables()
        
    def _init_vehicle_arrays(self):
        """Initialize arrays for vehicle properties and tensor conversion."""
        # Train arrays - each track can hold one train
        self.train_lengths = np.zeros(self.n_rail_tracks, dtype=np.int16)
        self.train_wagon_counts = np.zeros(self.n_rail_tracks, dtype=np.int16)
        self.train_container_counts = np.zeros(self.n_rail_tracks, dtype=np.int16)
        self.train_pickup_counts = np.zeros(self.n_rail_tracks, dtype=np.int16)
        self.train_occupied = np.zeros(self.n_rail_tracks, dtype=bool)
        
        # Truck arrays - each parking spot can hold one truck
        self.truck_container_counts = np.zeros(self.n_parking_spots, dtype=np.int16)
        self.truck_pickup_counts = np.zeros(self.n_parking_spots, dtype=np.int16)
        self.truck_weights = np.zeros(self.n_parking_spots, dtype=np.float32)
        self.truck_occupied = np.zeros(self.n_parking_spots, dtype=bool)
        
        # Queue arrays (fixed size for tensor conversion)
        self.max_queue_size = 50  # Reasonable maximum
        self.train_queue_lengths = np.zeros(self.max_queue_size, dtype=np.int16)
        self.train_queue_wagon_counts = np.zeros(self.max_queue_size, dtype=np.int16)
        self.train_queue_occupied = np.zeros(self.max_queue_size, dtype=bool)
        
        self.truck_queue_weights = np.zeros(self.max_queue_size, dtype=np.float32)
        self.truck_queue_pickup_counts = np.zeros(self.max_queue_size, dtype=np.int16)
        self.truck_queue_occupied = np.zeros(self.max_queue_size, dtype=bool)
    
    def _precompute_combination_masks(self):
        """Pre-compute valid container combination masks for O(1) validation."""
        self.combination_masks = {}
        
        for combo_name, container_types in self.valid_combinations.items():
            # Create a mask for this combination
            mask = np.zeros(len(container_types), dtype=np.int8)
            type_to_id = {'TWEU': 1, 'THEU': 2, 'FEU': 3, 'FFEU': 4}
            
            for i, ctype in enumerate(container_types):
                mask[i] = type_to_id.get(ctype, 0)
            
            self.combination_masks[combo_name] = mask
    
    def _build_lookup_tables(self):
        """Build lookup tables for O(1) container ID matching."""
        # Container ID -> Vehicle lookup
        self.container_to_train = {}  # container_id -> (track_id, wagon_idx)
        self.container_to_truck = {}  # container_id -> spot_id
        
        # Pickup request lookup
        self.pickup_to_train = {}  # container_id -> (track_id, wagon_idx)
        self.pickup_to_truck = {}  # container_id -> spot_id
        
        # Reverse lookup for fast removal
        self.train_containers = defaultdict(set)  # track_id -> set of container_ids
        self.truck_containers = defaultdict(set)  # spot_id -> set of container_ids
    
    def add_train_to_queue(self, train: Train):
        """Add train to waiting queue with property updates."""
        if len(self.train_queue) >= self.max_queue_size:
            return False  # Queue full
        
        queue_idx = len(self.train_queue)
        self.train_queue.append(train)
        
        # Update queue arrays
        train_length = len(train.wagons) + 1  # +1 for locomotive
        self.train_queue_lengths[queue_idx] = train_length
        self.train_queue_wagon_counts[queue_idx] = len(train.wagons)
        self.train_queue_occupied[queue_idx] = True
        
        return True
    
    def add_truck_to_queue(self, truck: Truck):
        """Add truck to waiting queue with property updates."""
        if len(self.truck_queue) >= self.max_queue_size:
            return False  # Queue full
        
        queue_idx = len(self.truck_queue)
        self.truck_queue.append(truck)
        
        # Update queue arrays
        total_weight = sum(c.weight for c in truck.containers) if truck.containers else 0
        self.truck_queue_weights[queue_idx] = total_weight
        self.truck_queue_pickup_counts[queue_idx] = len(getattr(truck, 'pickup_container_ids', set()))
        self.truck_queue_occupied[queue_idx] = True
        
        return True
    
    def try_place_train(self, track_id: int) -> bool:
        """
        Try to place next train from queue onto specified track.
        Returns True if successful.
        """
        if not self.train_queue or track_id >= self.n_rail_tracks:
            return False
        
        train = self.train_queue[0]  # FIFO
        train_length = len(train.wagons) + 1  # +1 for locomotive
        
        # Check if track has enough space
        if not self._check_rail_space(track_id, train_length):
            return False
        
        # Place train
        self._place_train_on_track(train, track_id)
        
        # Remove from queue and update arrays
        self.train_queue.popleft()
        self._shift_queue_arrays('train')
        
        return True
    
    def try_place_truck(self, spot_id: int) -> bool:
        """
        Try to place next truck from queue onto specified parking spot.
        Returns True if successful.
        """
        if not self.truck_queue or spot_id >= self.n_parking_spots:
            return False
        
        # Check if spot is available
        if self.truck_occupied[spot_id]:
            return False
        
        truck = self.truck_queue[0]  # FIFO
        
        # Place truck
        self._place_truck_on_spot(truck, spot_id)
        
        # Remove from queue and update arrays
        self.truck_queue.popleft()
        self._shift_queue_arrays('truck')
        
        return True
    
    def auto_place_trucks_from_queue(self) -> int:
        """
        Automatically place trucks from queue based on their pickup requirements.
        Places trucks parallel to wagons containing their pickup containers (±1 position allowed).
        
        Returns:
            Number of trucks successfully placed
        """
        placed_count = 0
        
        # Keep trying to place trucks until we can't place any more
        while self.truck_queue:
            truck = self.truck_queue[0]  # Peek at first truck
            
            # Try to find optimal placement for this truck
            optimal_spot = self._find_optimal_truck_spot(truck)
            
            if optimal_spot is not None:
                # Place the truck
                self._place_truck_on_spot(truck, optimal_spot)
                
                # Remove from queue and update arrays
                self.truck_queue.popleft()
                self._shift_queue_arrays('truck')
                
                placed_count += 1
            else:
                # Can't place this truck, stop trying (maintains FIFO order)
                break
        
        return placed_count
    
    def _find_optimal_truck_spot(self, truck: Truck) -> Optional[int]:
        """
        Find optimal parking spot for truck based on pickup requirements.
        
        Returns:
            Optimal spot_id or None if no suitable spot available
        """
        if not hasattr(truck, 'pickup_container_ids') or not truck.pickup_container_ids:
            # Truck has no specific pickup requirements, place in first available spot
            return self._find_first_available_spot()
        
        # Find wagons containing the pickup containers
        target_positions = []
        
        for container_id in truck.pickup_container_ids:
            if container_id in self.pickup_to_train:
                track_id, wagon_idx = self.pickup_to_train[container_id]
                # Convert wagon position to parking spot index
                # Each wagon spans 2 bays * split_factor positions
                wagon_start_pos = (wagon_idx + 1) * 2 * self.split_factor  # +1 for locomotive
                wagon_center_pos = wagon_start_pos + self.split_factor  # Center of wagon
                target_positions.append(wagon_center_pos)
        
        if not target_positions:
            # No matching wagons found, place in first available spot
            return self._find_first_available_spot()
        
        # Find best parking spot (parallel or ±1 to target positions)
        best_spot = None
        min_distance = float('inf')
        
        for target_pos in target_positions:
            # Check parallel position and ±1 positions
            candidate_spots = [
                target_pos - 1,  # One before
                target_pos,      # Parallel
                target_pos + 1   # One after
            ]
            
            for spot in candidate_spots:
                if 0 <= spot < self.n_parking_spots and not self.truck_occupied[spot]:
                    # Calculate distance (preference for exact parallel)
                    distance = abs(spot - target_pos)
                    if distance < min_distance:
                        min_distance = distance
                        best_spot = spot
        
        return best_spot
    
    def _find_first_available_spot(self) -> Optional[int]:
        """Find first available parking spot."""
        for spot_id in range(self.n_parking_spots):
            if not self.truck_occupied[spot_id]:
                return spot_id
        return None
    
    def _check_rail_space(self, track_id: int, required_length: int) -> bool:
        """Check if rail track has enough consecutive space."""
        if self.train_occupied[track_id]:
            return False  # Track already occupied
        
        # Each rail slot spans split_factor positions
        required_positions = required_length * self.split_factor
        
        # Check if we have enough consecutive free positions
        track_mask = self.dynamic_rails[:, track_id]
        
        # Find longest consecutive True sequence
        consecutive_count = 0
        max_consecutive = 0
        
        for occupied in track_mask:
            if occupied:
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                consecutive_count = 0
        
        return max_consecutive >= required_positions
    
    def _place_train_on_track(self, train: Train, track_id: int):
        """Place train on track and update all data structures."""
        train_length = len(train.wagons) + 1
        
        # Mark rail positions as occupied
        positions_needed = train_length * self.split_factor
        self.dynamic_rails[:positions_needed, track_id] = False
        
        # Update train arrays
        self.train_lengths[track_id] = train_length
        self.train_wagon_counts[track_id] = len(train.wagons)
        self.train_occupied[track_id] = True
        
        # Count containers
        container_count = sum(len(wagon.containers) for wagon in train.wagons)
        pickup_count = sum(len(wagon.pickup_container_ids) for wagon in train.wagons)
        
        self.train_container_counts[track_id] = container_count
        self.train_pickup_counts[track_id] = pickup_count
        
        # Update lookup tables
        self.active_trains[track_id] = train
        self._update_train_lookups(train, track_id)
    
    def _place_truck_on_spot(self, truck: Truck, spot_id: int):
        """Place truck on spot and update all data structures."""
        # Mark spot as occupied
        self.truck_occupied[spot_id] = True
        
        # Update truck arrays
        total_weight = sum(c.weight for c in truck.containers) if truck.containers else 0
        self.truck_weights[spot_id] = total_weight
        self.truck_container_counts[spot_id] = len(truck.containers)
        
        pickup_count = len(getattr(truck, 'pickup_container_ids', set()))
        self.truck_pickup_counts[spot_id] = pickup_count
        
        # Update lookup tables
        self.active_trucks[spot_id] = truck
        self._update_truck_lookups(truck, spot_id)
    
    def _update_train_lookups(self, train: Train, track_id: int):
        """Update lookup tables for train containers and pickup requests."""
        for wagon_idx, wagon in enumerate(train.wagons):
            # Containers currently on train
            for container in wagon.containers:
                self.container_to_train[container.container_id] = (track_id, wagon_idx)
                self.train_containers[track_id].add(container.container_id)
            
            # Pickup requests
            for container_id in wagon.pickup_container_ids:
                self.pickup_to_train[container_id] = (track_id, wagon_idx)
    
    def _update_truck_lookups(self, truck: Truck, spot_id: int):
        """Update lookup tables for truck containers and pickup requests."""
        # Containers currently on truck
        for container in truck.containers:
            self.container_to_truck[container.container_id] = spot_id
            self.truck_containers[spot_id].add(container.container_id)
        
        # Pickup requests
        if hasattr(truck, 'pickup_container_ids'):
            for container_id in truck.pickup_container_ids:
                self.pickup_to_truck[container_id] = spot_id
    
    def _shift_queue_arrays(self, vehicle_type: str):
        """Shift queue arrays after removing first element."""
        if vehicle_type == 'train':
            # Shift all elements one position left
            self.train_queue_lengths[:-1] = self.train_queue_lengths[1:]
            self.train_queue_wagon_counts[:-1] = self.train_queue_wagon_counts[1:]
            self.train_queue_occupied[:-1] = self.train_queue_occupied[1:]
            
            # Clear last position
            last_idx = len(self.train_queue)
            self.train_queue_lengths[last_idx] = 0
            self.train_queue_wagon_counts[last_idx] = 0
            self.train_queue_occupied[last_idx] = False
            
        elif vehicle_type == 'truck':
            # Shift all elements one position left
            self.truck_queue_weights[:-1] = self.truck_queue_weights[1:]
            self.truck_queue_pickup_counts[:-1] = self.truck_queue_pickup_counts[1:]
            self.truck_queue_occupied[:-1] = self.truck_queue_occupied[1:]
            
            # Clear last position
            last_idx = len(self.truck_queue)
            self.truck_queue_weights[last_idx] = 0.0
            self.truck_queue_pickup_counts[last_idx] = 0
            self.truck_queue_occupied[last_idx] = False
    
    def remove_train(self, track_id: int) -> Optional[Train]:
        """Remove train from track and update all data structures."""
        if track_id not in self.active_trains:
            return None
        
        train = self.active_trains[track_id]
        
        # Free rail positions
        self.dynamic_rails[:, track_id] = True
        
        # Clear train arrays
        self.train_lengths[track_id] = 0
        self.train_wagon_counts[track_id] = 0
        self.train_container_counts[track_id] = 0
        self.train_pickup_counts[track_id] = 0
        self.train_occupied[track_id] = False
        
        # Clear lookup tables
        self._clear_train_lookups(track_id)
        del self.active_trains[track_id]
        
        return train
    
    def remove_truck(self, spot_id: int) -> Optional[Truck]:
        """Remove truck from spot and update all data structures."""
        if spot_id not in self.active_trucks:
            return None
        
        truck = self.active_trucks[spot_id]
        
        # Free parking spot
        self.truck_occupied[spot_id] = False
        
        # Clear truck arrays
        self.truck_weights[spot_id] = 0.0
        self.truck_container_counts[spot_id] = 0
        self.truck_pickup_counts[spot_id] = 0
        
        # Clear lookup tables
        self._clear_truck_lookups(spot_id)
        del self.active_trucks[spot_id]
        
        return truck
    
    def _clear_train_lookups(self, track_id: int):
        """Clear lookup tables for removed train."""
        # Remove all container mappings for this track
        container_ids_to_remove = list(self.train_containers[track_id])
        
        for container_id in container_ids_to_remove:
            self.container_to_train.pop(container_id, None)
            self.pickup_to_train.pop(container_id, None)
        
        self.train_containers[track_id].clear()
    
    def _clear_truck_lookups(self, spot_id: int):
        """Clear lookup tables for removed truck."""
        # Remove all container mappings for this spot
        container_ids_to_remove = list(self.truck_containers[spot_id])
        
        for container_id in container_ids_to_remove:
            self.container_to_truck.pop(container_id, None)
            self.pickup_to_truck.pop(container_id, None)
        
        self.truck_containers[spot_id].clear()
    
    def find_container_moves(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Find all valid container moves between rails and parking.
        O(1) lookup using pre-built tables.
        
        Returns:
            Dict mapping container_id -> [(source, destination), ...]
        """
        moves = defaultdict(list)
        
        # Rail to parking moves (pickup requests)
        for container_id, spot_id in self.pickup_to_truck.items():
            if container_id in self.container_to_train:
                track_id, wagon_idx = self.container_to_train[container_id]
                source = f"rail_{track_id}_{wagon_idx}"
                destination = f"parking_{spot_id}"
                moves[container_id].append((source, destination))
        
        # Parking to rail moves (pickup requests)
        for container_id, (track_id, wagon_idx) in self.pickup_to_train.items():
            if container_id in self.container_to_truck:
                spot_id = self.container_to_truck[container_id]
                source = f"parking_{spot_id}"
                destination = f"rail_{track_id}_{wagon_idx}"
                moves[container_id].append((source, destination))
        
        return dict(moves)
    
    # ==================== TENSOR CONVERSION METHODS ====================
    
    def get_rail_state_tensor(self, as_tensor: bool = True) -> torch.Tensor:
        """Get rail occupancy state as tensor."""
        if as_tensor:
            return torch.from_numpy(self.dynamic_rails).to(self.device)
        return self.dynamic_rails
    
    def get_parking_state_tensor(self, as_tensor: bool = True) -> torch.Tensor:
        """Get parking occupancy state as tensor."""
        if as_tensor:
            return torch.from_numpy(self.dynamic_parking).to(self.device)
        return self.dynamic_parking
    
    def get_train_properties_tensor(self, as_tensor: bool = True) -> torch.Tensor:
        """
        Get train properties as tensor.
        
        Returns:
            Tensor of shape (n_rail_tracks, 5) with columns:
            [occupied, length, wagon_count, container_count, pickup_count]
        """
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
        """
        Get truck properties as tensor.
        
        Returns:
            Tensor of shape (n_parking_spots, 4) with columns:
            [occupied, weight, container_count, pickup_count]
        """
        properties = np.stack([
            self.truck_occupied.astype(np.float32),
            self.truck_weights,
            self.truck_container_counts.astype(np.float32),
            self.truck_pickup_counts.astype(np.float32)
        ], axis=1)
        
        if as_tensor:
            return torch.from_numpy(properties).to(self.device)
        return properties
    
    def get_queue_state_tensor(self, as_tensor: bool = True) -> Dict[str, torch.Tensor]:
        """Get queue states as tensors."""
        train_queue_tensor = np.stack([
            self.train_queue_occupied.astype(np.float32),
            self.train_queue_lengths.astype(np.float32),
            self.train_queue_wagon_counts.astype(np.float32)
        ], axis=1)
        
        truck_queue_tensor = np.stack([
            self.truck_queue_occupied.astype(np.float32),
            self.truck_queue_weights,
            self.truck_queue_pickup_counts.astype(np.float32)
        ], axis=1)
        
        if as_tensor:
            return {
                'train_queue': torch.from_numpy(train_queue_tensor).to(self.device),
                'truck_queue': torch.from_numpy(truck_queue_tensor).to(self.device)
            }
        
        return {
            'train_queue': train_queue_tensor,
            'truck_queue': truck_queue_tensor
        }