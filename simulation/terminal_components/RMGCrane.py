# simulation/terminal_components/RMGCrane.py (optimized version)
from typing import List, Dict, Tuple, Optional, Set, Any
import numpy as np
import re

class RMGCrane:
    """Rail Mounted Gantry (RMG) Crane for moving containers in the terminal."""
    
    def __init__(self, 
                 crane_id: str,
                 terminal: Any,
                 start_bay: int,
                 end_bay: int,
                 current_position: Tuple[int, int] = (0, 0)):
        """Initialize a new RMG crane."""
        self.crane_id = crane_id
        self.terminal = terminal
        self.start_bay = start_bay
        self.end_bay = end_bay
        self.current_position = current_position
        self.movement_time_cache = {}
        
        # Crane specifications
        self.trolley_speed = 70.0 / 60.0  # m/s (converted from m/min)
        self.hoisting_speed = 28.0 / 60.0  # m/s (converted from m/min)
        self.gantry_speed = 130.0 / 60.0  # m/s (converted from m/min)
        self.trolley_acceleration = 0.3  # m/s²
        self.hoisting_acceleration = 0.2  # m/s²
        self.gantry_acceleration = 0.1  # m/s²
        self.max_height = 20.0  # meters
        self.container_heights = {
            "TWEU": 2.59,
            "THEU": 2.59,
            "FEU": 2.59,
            "FFEU": 2.59,
            "default": 2.59
        }
        self.ground_vehicle_height = 1.5  # meters
        
        # Cache for position checks
        self._position_type_cache = {}
        self._valid_moves_cache = {}
        self._cache_valid = True
    
    def reset(self, position: Tuple[int, int] = None):
        """Reset the crane to its initial position."""
        if position is None:
            self.current_position = (self.start_bay, 0)
        else:
            self.current_position = position
        self.movement_time_cache = {}
        self._valid_moves_cache = {}
        self._cache_valid = False
    
    def get_valid_moves(self, 
                        storage_yard: Any, 
                        trucks_in_terminal: Dict[str, Any],
                        trains_in_terminal: Dict[str, Any]) -> Dict[Tuple[str, str], float]:
        """Get all valid moves the crane can make from its current position."""
        # Use cached valid moves if available and environment hasn't changed
        if self._cache_valid and self._valid_moves_cache:
            return self._valid_moves_cache
        
        # First, get all source positions where a container can be picked up
        source_positions = self._get_source_positions(storage_yard, trucks_in_terminal, trains_in_terminal)
        
        # Next, for each source, find all valid destinations
        valid_moves = {}
        
        for source_position in source_positions:
            container = self._get_container_at_position(source_position, storage_yard, 
                                                       trucks_in_terminal, trains_in_terminal)
            
            if container is None:
                continue
                
            destinations = self._get_destination_positions(source_position, container, storage_yard, 
                                                         trucks_in_terminal, trains_in_terminal)
            
            for dest_position in destinations:
                # Check if the move is within this crane's operational area
                if self._is_position_in_crane_area(source_position) and self._is_position_in_crane_area(dest_position):
                    # Calculate the estimated time for this move
                    cache_key = (source_position, dest_position, container.container_id if hasattr(container, 'container_id') else str(container))
                    
                    # Use cached time if available
                    if cache_key in self.movement_time_cache:
                        time_estimate = self.movement_time_cache[cache_key]
                    else:
                        time_estimate = self.estimate_movement_time(source_position, dest_position, container)
                        self.movement_time_cache[cache_key] = time_estimate
                        
                    valid_moves[(source_position, dest_position)] = time_estimate
        
        # Cache the result
        self._valid_moves_cache = valid_moves
        self._cache_valid = True
        
        return valid_moves
    
    def move_container(self, 
                      source_position: str, 
                      destination_position: str,
                      storage_yard: Any,
                      trucks_in_terminal: Dict[str, Any],
                      trains_in_terminal: Dict[str, Any]) -> Tuple[Optional[Any], float]:
        """Move a container from source to destination."""
        # Check if both positions are in this crane's area
        if not (self._is_position_in_crane_area(source_position) and 
                self._is_position_in_crane_area(destination_position)):
            return None, 0
        
        # Check if there's a container at the source
        container = self._get_container_at_position(source_position, storage_yard, 
                                                   trucks_in_terminal, trains_in_terminal)
        
        if container is None:
            return None, 0
        
        # Check if the container can be placed at the destination
        if not self._can_place_container(destination_position, container, storage_yard, 
                                       trucks_in_terminal, trains_in_terminal):
            return None, 0
        
        # Remove the container from its source
        removed_container = None
        
        if self._is_storage_position(source_position):
            removed_container = storage_yard.remove_container(source_position)
        elif self._is_truck_position(source_position):
            truck = self._get_truck_at_position(source_position, trucks_in_terminal)
            if truck:
                removed_container = truck.remove_container(container.container_id if hasattr(container, 'container_id') else None)
        elif self._is_train_position(source_position):
            removed_container = self._remove_container_from_train(source_position, container, trains_in_terminal)
        
        if removed_container is None:
            return None, 0
        
        # Track the source position in the container for rule enforcement
        if hasattr(removed_container, '__dict__'):
            removed_container._source_position = source_position

        # Calculate the time required for the move
        cache_key = (source_position, destination_position, container.container_id if hasattr(container, 'container_id') else str(container))
        
        if cache_key in self.movement_time_cache:
            time_taken = self.movement_time_cache[cache_key]
        else:
            time_taken = self.calculate_movement_time(source_position, destination_position, container)
            self.movement_time_cache[cache_key] = time_taken
        
        # Place the container at its destination
        success = False
        
        if self._is_storage_position(destination_position):
            success = storage_yard.add_container(destination_position, removed_container)
        elif self._is_truck_position(destination_position):
            truck = self._get_truck_at_position(destination_position, trucks_in_terminal)
            if truck:
                success = truck.add_container(removed_container)
                
                # Check if this container completes a pickup request
                if hasattr(truck, 'pickup_container_ids') and hasattr(removed_container, 'container_id'):
                    if removed_container.container_id in truck.pickup_container_ids:
                        truck.remove_pickup_container_id(removed_container.container_id)
        elif self._is_train_position(destination_position):
            success = self._add_container_to_train(destination_position, removed_container, trains_in_terminal)
            
            # Check if this container completes a pickup request
            for train in trains_in_terminal.values():
                for wagon in train.wagons:
                    if hasattr(removed_container, 'container_id') and removed_container.container_id in wagon.pickup_container_ids:
                        wagon.remove_pickup_container(removed_container.container_id)
                        break
        
        # Update the crane's position and caches
        if success:
            # Extract bay and row from position strings
            dest_bay, dest_row = self._position_to_bay_row(destination_position)
            self.current_position = (dest_bay, dest_row)
            
            # Invalidate caches
            self._cache_valid = False
            
            return removed_container, time_taken
        else:
            # If the placement failed, we need to put the container back
            if self._is_storage_position(source_position):
                storage_yard.add_container(source_position, removed_container)
            elif self._is_truck_position(source_position):
                truck = self._get_truck_at_position(source_position, trucks_in_terminal)
                if truck:
                    truck.add_container(removed_container)
            elif self._is_train_position(source_position):
                self._add_container_to_train(source_position, removed_container, trains_in_terminal)
                
            return None, 0
    
    def calculate_movement_time(self, source_position: str, destination_position: str, container: Any) -> float:
        """Calculate the time needed to move a container."""
        # Check the cache first
        cache_key = (source_position, destination_position, container.container_id if hasattr(container, 'container_id') else str(container))
        if cache_key in self.movement_time_cache:
            return self.movement_time_cache[cache_key]
        
        # Get source and destination positions
        if not hasattr(self.terminal, 'positions'):
            # Fallback to estimate if terminal positions not available
            return self.estimate_movement_time(source_position, destination_position, container)
            
        if source_position not in self.terminal.positions or destination_position not in self.terminal.positions:
            # Fallback to estimate if positions not found
            return self.estimate_movement_time(source_position, destination_position, container)
            
        src_pos = self.terminal.positions[source_position]
        dst_pos = self.terminal.positions[destination_position]
        
        # Determine movement components
        # 1. Gantry movement (movement of entire crane along the rails)
        gantry_distance = abs(src_pos[1] - dst_pos[1])
        
        # 2. Trolley movement (movement of trolley across the crane bridge)
        trolley_distance = abs(src_pos[0] - dst_pos[0])
        
        # Determine if significant gantry movement is needed
        significant_gantry_movement = gantry_distance > 1.0  # threshold in meters
        
        # Calculate gantry movement time if needed
        gantry_time = 0.0
        if significant_gantry_movement:
            gantry_time = self._calculate_travel_time(
                gantry_distance,
                self.gantry_speed,
                self.gantry_acceleration
            )
        
        # Calculate trolley movement time
        trolley_time = self._calculate_travel_time(
            trolley_distance,
            self.trolley_speed,
            self.trolley_acceleration
        )
        
        # Determine vertical distances based on source and destination types
        source_type = self._get_position_type(source_position)
        dest_type = self._get_position_type(destination_position)
        
        # Get container height
        container_height = self.container_heights.get(
            container.container_type if hasattr(container, 'container_type') else "default", 
            self.container_heights["default"]
        )
        
        # Calculate vertical distances
        vertical_distance_up = 0.0
        vertical_distance_down = 0.0
        
        # Determine vertical movement based on source and destination types
        if source_type == 'train' and dest_type == 'storage':
            # Train to storage
            vertical_distance_up = self.max_height - (self.ground_vehicle_height + container_height)
            vertical_distance_down = self.max_height - 0  # Assume empty storage spot
            
        elif source_type == 'storage' and dest_type == 'train':
            # Storage to train
            vertical_distance_up = self.max_height - 0  # Assume single container
            vertical_distance_down = self.max_height - (self.ground_vehicle_height + container_height)
            
        elif source_type == 'truck' and dest_type == 'storage':
            # Truck to storage
            vertical_distance_up = self.max_height - (self.ground_vehicle_height + container_height)
            vertical_distance_down = self.max_height - 0  # Assume empty storage spot
            
        elif source_type == 'storage' and dest_type == 'truck':
            # Storage to truck
            vertical_distance_up = self.max_height - 0  # Assume single container
            vertical_distance_down = self.max_height - (self.ground_vehicle_height + container_height)
            
        elif source_type == 'train' and dest_type == 'truck':
            # Train to truck (direct move)
            vertical_distance_up = self.max_height - (self.ground_vehicle_height + container_height)
            vertical_distance_down = self.max_height - (self.ground_vehicle_height + container_height)
            
        elif source_type == 'truck' and dest_type == 'train':
            # Truck to train (direct move)
            vertical_distance_up = self.max_height - (self.ground_vehicle_height + container_height)
            vertical_distance_down = self.max_height - (self.ground_vehicle_height + container_height)
            
        elif source_type == 'storage' and dest_type == 'storage':
            # Storage to storage (reshuffling)
            vertical_distance_up = self.max_height - 0  # Assume single container
            vertical_distance_down = self.max_height - 0  # Assume empty destination
        
        # Calculate vertical movement times
        vertical_up_time = self._calculate_travel_time(
            vertical_distance_up, 
            self.hoisting_speed, 
            self.hoisting_acceleration
        )
        
        vertical_down_time = self._calculate_travel_time(
            vertical_distance_down, 
            self.hoisting_speed, 
            self.hoisting_acceleration
        )
        
        # Calculate total time with proper sequencing
        horizontal_movement_time = gantry_time + trolley_time
        
        # Lifting can happen while trolley moves (after gantry movement)
        if significant_gantry_movement:
            # Gantry moves first, then trolley and lifting happen together
            time_after_gantry = max(vertical_up_time, trolley_time)
            total_time = gantry_time + time_after_gantry + vertical_down_time
        else:
            # No gantry movement, lifting and trolley can happen simultaneously
            total_time = max(vertical_up_time, trolley_time) + vertical_down_time
        
        # Add some fixed time for attaching/detaching the container
        attach_detach_time = 10.0  # seconds
        total_time += attach_detach_time
        
        # Add time for current position to source
        # This accounts for the crane's current position
        current_to_source_time = self._calculate_crane_movement_time(
            self.current_position, 
            self._position_to_bay_row(source_position)
        )
        
        total_time += current_to_source_time
        
        # Cache the result
        self.movement_time_cache[cache_key] = total_time
        
        return total_time
    
    def estimate_movement_time(self, source_position: str, destination_position: str, container: Any) -> float:
        """Estimate movement time without detailed calculations."""
        # Calculate time based on the current position of the crane
        source_bay, source_row = self._position_to_bay_row(source_position)
        dest_bay, dest_row = self._position_to_bay_row(destination_position)
        current_bay, current_row = self.current_position
        
        # Calculate distances
        current_to_source_distance = abs(current_bay - source_bay) + abs(current_row - source_row)
        source_to_dest_distance = abs(source_bay - dest_bay) + abs(source_row - dest_row)
        
        # Estimate time based on distances
        # These are rough estimates, adjust based on actual terminal dimensions
        meters_per_bay = 6  # Approximate meters per bay
        meters_per_row = 10  # Approximate meters per row
        
        current_to_source_meters = (
            abs(current_bay - source_bay) * meters_per_bay + 
            abs(current_row - source_row) * meters_per_row
        )
        
        source_to_dest_meters = (
            abs(source_bay - dest_bay) * meters_per_bay + 
            abs(source_row - dest_row) * meters_per_row
        )
        
        # Estimate travel times
        current_to_source_time = self._calculate_travel_time(
            current_to_source_meters,
            self.gantry_speed,  # Use gantry speed as approximation
            self.gantry_acceleration
        )
        
        source_to_dest_time = self._calculate_travel_time(
            source_to_dest_meters,
            self.gantry_speed,
            self.gantry_acceleration
        )
        
        # Add hoisting times (rough estimates)
        hoisting_time = 30  # seconds
        
        # Attach/detach time
        attach_detach_time = 10  # seconds
        
        total_time = current_to_source_time + source_to_dest_time + hoisting_time + attach_detach_time
        
        return total_time
    
    def _calculate_travel_time(self, distance: float, max_speed: float, acceleration: float) -> float:
        """Calculate travel time with acceleration and deceleration."""
        # No movement needed
        if distance <= 0:
            return 0.0
        
        # Calculate the distance needed to reach max speed
        accel_distance = 0.5 * max_speed**2 / acceleration
        
        # If we can't reach max speed (distance is too short)
        if distance <= 2 * accel_distance:
            # Time to accelerate and then immediately decelerate
            peak_speed = np.sqrt(acceleration * distance)
            time = 2 * peak_speed / acceleration
        else:
            # Time to accelerate + time at max speed + time to decelerate
            accel_time = max_speed / acceleration
            constant_speed_distance = distance - 2 * accel_distance
            constant_speed_time = constant_speed_distance / max_speed
            time = 2 * accel_time + constant_speed_time
            
        return time
    
    def _get_source_positions(self, 
                             storage_yard: Any, 
                             trucks_in_terminal: Dict[str, Any],
                             trains_in_terminal: Dict[str, Any]) -> List[str]:
        """Get all positions where a container can be picked up."""
        source_positions = []
        
        # Check storage yard
        for row in storage_yard.row_names:
            for bay in range(1, storage_yard.num_bays + 1):
                position = f"{row}{bay}"
                if self._is_position_in_crane_area(position):
                    container, _ = storage_yard.get_top_container(position)
                    if container is not None:
                        source_positions.append(position)
        
        # Check trucks
        for spot, truck in trucks_in_terminal.items():
            if self._is_position_in_crane_area(spot) and truck.has_containers():
                source_positions.append(spot)
        
        # Check trains
        for track, train in trains_in_terminal.items():
            for i, wagon in enumerate(train.wagons):
                if not wagon.is_empty():
                    slot = f"{track.lower()}_{i+1}"
                    if self._is_position_in_crane_area(slot):
                        source_positions.append(slot)
        
        return source_positions
    
    def _get_destination_positions(self, 
                                source_position: str,
                                container: Any,
                                storage_yard: Any,
                                trucks_in_terminal: Dict[str, Any],
                                trains_in_terminal: Dict[str, Any]) -> List[str]:
        """Get all positions where a container can be placed based on strict movement rules."""
        destinations = []
        source_type = self._get_position_type(source_position)
        
        # Rule 1: Container on train -> can only move to a truck with matching container ID
        if source_type == 'train':
            # Only allow moves to trucks with matching pickup container ID
            for spot, truck in trucks_in_terminal.items():
                if self._is_position_in_crane_area(spot) and hasattr(truck, 'pickup_container_ids'):
                    if container.container_id in truck.pickup_container_ids:
                        if not truck.is_full() and truck.has_space_for_container(container):
                            destinations.append(spot)
                            
            # If no matching truck is available, allow storage based on container type
            if not destinations:
                # For swap body/trailer - only allowed areas
                if container.container_type in ["Trailer", "Swap Body"]:
                    # Get the appropriate special area
                    area_type = 'trailer' if container.container_type == "Trailer" else 'swap_body'
                    
                    # Only allow storage in this area
                    for row in storage_yard.row_names:
                        for bay in range(1, storage_yard.num_bays + 1):
                            position = f"{row}{bay}"
                            if position != source_position and self._is_position_in_crane_area(position):
                                # Check if position is in the correct special area
                                for area_row, start_bay, end_bay in storage_yard.special_areas.get(area_type, []):
                                    if row == area_row and start_bay <= bay <= end_bay:
                                        if storage_yard.can_accept_container(position, container):
                                            destinations.append(position)
                                            break
                else:
                    # For regular/reefer/dangerous goods containers
                    for row in storage_yard.row_names:
                        for bay in range(1, storage_yard.num_bays + 1):
                            position = f"{row}{bay}"
                            if position != source_position and self._is_position_in_crane_area(position):
                                if storage_yard.can_accept_container(position, container):
                                    destinations.append(position)
        
        # Rule 2: Container on truck -> can only move to wagons looking
        # Continuing RMGCrane class implementation

        # Rule 2: Container on truck -> can only move to wagons looking for that container
        elif source_type == 'truck':
            # Only allow moves to wagons that have this container in their pickup list
            for track, train in trains_in_terminal.items():
                for i, wagon in enumerate(train.wagons):
                    slot = f"{track.lower()}_{i+1}"
                    if slot != source_position and self._is_position_in_crane_area(slot):
                        if container.container_id in wagon.pickup_container_ids:
                            destinations.append(slot)
                            
            # If no matching wagon, allow storage
            if not destinations:
                for row in storage_yard.row_names:
                    for bay in range(1, storage_yard.num_bays + 1):
                        position = f"{row}{bay}"
                        if position != source_position and self._is_position_in_crane_area(position):
                            if storage_yard.can_accept_container(position, container):
                                destinations.append(position)
        
        # Rule 3: Container in storage -> allowed to move to assigned truck/train or other storage
        elif source_type == 'storage':
            # Check for trucks looking for this container
            for spot, truck in trucks_in_terminal.items():
                if spot != source_position and self._is_position_in_crane_area(spot):
                    if hasattr(truck, 'pickup_container_ids') and container.container_id in truck.pickup_container_ids:
                        if not truck.is_full() and truck.has_space_for_container(container):
                            destinations.append(spot)
            
            # Check for trains looking for this container
            for track, train in trains_in_terminal.items():
                for i, wagon in enumerate(train.wagons):
                    slot = f"{track.lower()}_{i+1}"
                    if slot != source_position and self._is_position_in_crane_area(slot):
                        if container.container_id in wagon.pickup_container_ids:
                            destinations.append(slot)
            
            # Allow pre-marshalling within the 5-bay distance limit
            source_bay = int(re.findall(r'\d+', source_position)[0]) - 1  # Extract bay number
            for row in storage_yard.row_names:
                for bay in range(1, storage_yard.num_bays + 1):
                    position = f"{row}{bay}"
                    if position != source_position and self._is_position_in_crane_area(position):
                        # Check pre-marshalling distance constraint
                        dest_bay = bay - 1
                        if abs(source_bay - dest_bay) <= 5:  # Limit to 5 positions
                            if storage_yard.can_accept_container(position, container):
                                destinations.append(position)
        
        return destinations

    def _is_position_in_crane_area(self, position: str) -> bool:
        """Check if a position is within this crane's operational area."""
        bay, _ = self._position_to_bay_row(position)
        return self.start_bay <= bay <= self.end_bay
    def _get_position_type(self, position: str) -> str:
        """Determine the type of a position (train, truck, storage) with caching."""
        if position in self._position_type_cache:
            return self._position_type_cache[position]
            
        if self._is_train_position(position):
            position_type = 'train'
        elif self._is_truck_position(position):
            position_type = 'truck'
        else:
            position_type = 'storage'
            
        # Cache the result
        self._position_type_cache[position] = position_type
        return position_type
    def _position_to_bay_row(self, position: str) -> Tuple[int, int]:
        """Convert a position string to (bay, row) coordinates."""
        if position.startswith('t') and '_' in position:
            # Train position (e.g., 't1_2')
            parts = position.split('_')
            track = int(parts[0][1:])
            slot = int(parts[1])
            # Map to coordinate system (bay = slot, row = track)
            return slot - 1, track - 1
            
        elif position.startswith('p_'):
            # Truck position (e.g., 'p_3')
            spot = int(position.split('_')[1])
            # Map to coordinate system (bay = spot, row = parking row)
            return spot - 1, len(self.terminal.track_names) if hasattr(self.terminal, 'track_names') else 3
            
        else:
            # Storage position (e.g., 'A1')
            # Extract row letter and bay number
            row_letter = position[0]
            bay_number = int(position[1:])
            
            # Convert row letter to index (A=0, B=1, etc.)
            row_index = ord(row_letter) - ord('A')
            
            # Map to coordinate system
            # Storage is above train tracks and parking
            row_offset = len(self.terminal.track_names) + 1 if hasattr(self.terminal, 'track_names') else 4
            return bay_number - 1, row_index + row_offset
    def _is_storage_position(self, position: str) -> bool:
        """Check if a position is in the storage yard."""
        return bool(position and position[0].isalpha() and position[1:].isdigit())

    def _is_truck_position(self, position: str) -> bool:
        """Check if a position is a truck parking spot."""
        return bool(position and position.startswith('p_'))

    def _is_train_position(self, position: str) -> bool:
        """Check if a position is a train slot."""
        return bool(position and position.startswith('t') and '_' in position)
    def _calculate_crane_movement_time(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> float:
        """Calculate time for crane to move from current position to source position."""
        # These are rough estimates, adjust based on actual terminal dimensions
        meters_per_bay = 6  # Approximate meters per bay
        meters_per_row = 10  # Approximate meters per row
        
        # Calculate distances
        bay_distance = abs(from_pos[0] - to_pos[0]) * meters_per_bay
        row_distance = abs(from_pos[1] - to_pos[1]) * meters_per_row
        
        # Calculate movement times
        bay_time = self._calculate_travel_time(
            bay_distance,
            self.trolley_speed,
            self.trolley_acceleration
        )
        
        row_time = self._calculate_travel_time(
            row_distance,
            self.gantry_speed,
            self.gantry_acceleration
        )
        
        # Return the maximum (movements can happen in parallel)
        return max(bay_time, row_time)

    def _get_container_at_position(self, 
                                position: str,
                                storage_yard: Any,
                                trucks_in_terminal: Dict[str, Any],
                                trains_in_terminal: Dict[str, Any]) -> Optional[Any]:
        """Get the container at a position."""
        # Get position type
        position_type = self._get_position_type(position)
        
        if position_type == 'storage':
            # Get top container from storage
            container, _ = storage_yard.get_top_container(position)
            return container
            
        elif position_type == 'truck':
            # Get container from truck
            truck = self._get_truck_at_position(position, trucks_in_terminal)
            if truck and hasattr(truck, 'containers') and truck.containers:
                return truck.containers[0]  # Return the first container
            return None
            
        elif position_type == 'train':
            # Get container from train
            train_id, wagon_index = self._parse_train_position(position, trains_in_terminal)
            if train_id and wagon_index is not None:
                train = trains_in_terminal.get(train_id)
                if train and 0 <= wagon_index < len(train.wagons):
                    wagon = train.wagons[wagon_index]
                    if wagon.containers:
                        return wagon.containers[0]  # Return the first container
            return None
            
        return None

    def _can_place_container(self, 
                        position: str,
                        container: Any,
                        storage_yard: Any,
                        trucks_in_terminal: Dict[str, Any],
                        trains_in_terminal: Dict[str, Any]) -> bool:
        """Check if a container can be placed at a position based on strict rules."""
        # Get the type of the position
        position_type = self._get_position_type(position)
        
        if position_type == 'storage':
            # Check if container can be added to storage with all constraints
            return storage_yard.can_accept_container(position, container)
            
        elif position_type == 'truck':
            # Check if container can be added to truck
            truck = self._get_truck_at_position(position, trucks_in_terminal)
            if truck and hasattr(truck, 'add_container'):
                # Only allow if this truck is specifically looking for this container
                if hasattr(truck, 'pickup_container_ids') and hasattr(container, 'container_id'):
                    if container.container_id in truck.pickup_container_ids:
                        # Verify truck has space
                        return not truck.is_full() and truck.has_space_for_container(container)
                # Or it's a delivery truck with space (only for containers from storage)
                elif (hasattr(container, '_source_position') and 
                    self._is_storage_position(container._source_position) and
                    not truck.is_pickup_truck):
                    return not truck.is_full() and truck.has_space_for_container(container)
            return False
            
        elif position_type == 'train':
            # Check if container can be added to train
            train_id, wagon_index = self._parse_train_position(position, trains_in_terminal)
            if train_id and wagon_index is not None:
                train = trains_in_terminal.get(train_id)
                if train and 0 <= wagon_index < len(train.wagons):
                    wagon = train.wagons[wagon_index]
                    
                    # Only allow if wagon is looking for this specific container
                    if hasattr(container, 'container_id') and container.container_id in wagon.pickup_container_ids:
                        return wagon.get_available_length() >= container.length
            return False
            
        return False

    def _get_position_type(self, position: str) -> str:
        """Determine the type of a position (train, truck, storage) with caching."""
        if position in self._position_type_cache:
            return self._position_type_cache[position]
            
        if self._is_train_position(position):
            position_type = 'train'
        elif self._is_truck_position(position):
            position_type = 'truck'
        else:
            position_type = 'storage'
            
        # Cache the result
        self._position_type_cache[position] = position_type
        return position_type

    def _is_storage_position(self, position: str) -> bool:
        """Check if a position is in the storage yard."""
        # Storage positions typically start with a letter and are followed by a number
        return position and position[0].isalpha() and position[1:].isdigit()

    def _is_truck_position(self, position: str) -> bool:
        """Check if a position is a truck parking spot."""
        return position and position.startswith('p_')

    def _is_train_position(self, position: str) -> bool:
        """Check if a position is a train slot."""
        return position and position.startswith('t') and '_' in position

    def _get_truck_at_position(self, position: str, trucks_in_terminal: Dict[str, Any]) -> Optional[Any]:
        """Get the truck at a position."""
        return trucks_in_terminal.get(position)

    def _parse_train_position(self, position: str, trains_in_terminal: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
        """Parse a train position string to get train ID and wagon index."""
        if not self._is_train_position(position):
            return None, None
        
        parts = position.split('_')
        if len(parts) != 2:
            return None, None
        
        track_num = int(parts[0][1:])
        slot_num = int(parts[1])
        
        # Convert track number to track ID
        track_id = f"T{track_num}"
        
        # Check if there's a train on this track
        if track_id not in trains_in_terminal:
            return None, None
        
        # Slot number corresponds to wagon index (0-based)
        wagon_index = slot_num - 1
        
        return track_id, wagon_index

    def _remove_container_from_train(self, position: str, container: Any, trains_in_terminal: Dict[str, Any]) -> Optional[Any]:
        """Remove a container from a train."""
        train_id, wagon_index = self._parse_train_position(position, trains_in_terminal)
        if train_id and wagon_index is not None:
            train = trains_in_terminal.get(train_id)
            if train and 0 <= wagon_index < len(train.wagons):
                wagon = train.wagons[wagon_index]
                
                # Find the container in the wagon
                for i, container_in_wagon in enumerate(wagon.containers):
                    if (hasattr(container, 'container_id') and hasattr(container_in_wagon, 'container_id') and 
                        container.container_id == container_in_wagon.container_id):
                        return wagon.containers.pop(i)
        
        return None

    def _add_container_to_train(self, position: str, container: Any, trains_in_terminal: Dict[str, Any]) -> bool:
        """Add a container to a train."""
        train_id, wagon_index = self._parse_train_position(position, trains_in_terminal)
        if train_id and wagon_index is not None:
            train = trains_in_terminal.get(train_id)
            if train and 0 <= wagon_index < len(train.wagons):
                wagon = train.wagons[wagon_index]
                return wagon.add_container(container)
        
        return False