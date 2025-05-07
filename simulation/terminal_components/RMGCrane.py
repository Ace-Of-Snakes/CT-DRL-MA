from typing import List, Dict, Tuple, Optional, Set, Any
import numpy as np


class RMGCrane:
    """
    Rail Mounted Gantry (RMG) Crane for moving containers in the terminal.
    
    Attributes:
        crane_id: Unique identifier for the crane
        terminal: Reference to the terminal object for distance calculations
        start_bay: Starting bay of this crane's operational area
        end_bay: Ending bay of this crane's operational area
        current_position: Current position of the crane as (bay, row)
        movement_time_cache: Cache for movement time calculations
    """
    
    def __init__(self, 
                 crane_id: str,
                 terminal: Any,
                 start_bay: int,
                 end_bay: int,
                 current_position: Tuple[int, int] = (0, 0)):
        """
        Initialize a new RMG crane.
        
        Args:
            crane_id: Unique identifier for the crane
            terminal: Reference to the terminal object
            start_bay: Starting bay of operational area
            end_bay: Ending bay of operational area
            current_position: Initial position as (bay, row)
        """
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
    
    def reset(self, position: Tuple[int, int] = None):
        """
        Reset the crane to its initial position.
        
        Args:
            position: New position for the crane (if None, uses start_bay, 0)
        """
        if position is None:
            self.current_position = (self.start_bay, 0)
        else:
            self.current_position = position
        self.movement_time_cache = {}
    
    def get_valid_moves(self, 
                        storage_yard: Any, 
                        trucks_in_terminal: Dict[str, Any],
                        trains_in_terminal: Dict[str, Any]) -> Dict[Tuple[str, str], float]:
        """
        Get all valid moves the crane can make from its current position.
        
        Args:
            storage_yard: Reference to the storage yard
            trucks_in_terminal: Dictionary of trucks currently in the terminal
            trains_in_terminal: Dictionary of trains currently in the terminal
            
        Returns:
            Dictionary mapping (source, destination) tuples to estimated time
        """
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
                    time_estimate = self.estimate_movement_time(source_position, dest_position, container)
                    valid_moves[(source_position, dest_position)] = time_estimate
        
        return valid_moves
    
    def move_container(self, 
                      source_position: str, 
                      destination_position: str,
                      storage_yard: Any,
                      trucks_in_terminal: Dict[str, Any],
                      trains_in_terminal: Dict[str, Any]) -> Tuple[Optional[Any], float]:
        """
        Move a container from source to destination.
        
        Args:
            source_position: Source position string
            destination_position: Destination position string
            storage_yard: Reference to the storage yard
            trucks_in_terminal: Dictionary of trucks currently in the terminal
            trains_in_terminal: Dictionary of trains currently in the terminal
            
        Returns:
            Tuple of (container, time_taken) or (None, 0) if move failed
        """
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
        if self._is_storage_position(source_position):
            removed_container = storage_yard.remove_container(source_position)
        elif self._is_truck_position(source_position):
            truck = self._get_truck_at_position(source_position, trucks_in_terminal)
            removed_container = truck.remove_container(container.container_id if hasattr(container, 'container_id') else None)
        elif self._is_train_position(source_position):
            removed_container = self._remove_container_from_train(source_position, container, trains_in_terminal)
        else:
            return None, 0
        
        # Calculate the time required for the move
        time_taken = self.calculate_movement_time(source_position, destination_position, container)
        
        # Place the container at its destination
        if self._is_storage_position(destination_position):
            success = storage_yard.add_container(destination_position, container)
        elif self._is_truck_position(destination_position):
            truck = self._get_truck_at_position(destination_position, trucks_in_terminal)
            success = truck.add_container(container)
            
            # Check if this container completes a pickup request
            if hasattr(truck, 'pickup_container_ids') and container.container_id in truck.pickup_container_ids:
                truck.remove_pickup_container_id(container.container_id)
                
        elif self._is_train_position(destination_position):
            success = self._add_container_to_train(destination_position, container, trains_in_terminal)
            
            # Check if this container completes a pickup request
            for train in trains_in_terminal.values():
                for wagon in train.wagons:
                    if container.container_id in wagon.pickup_container_ids:
                        wagon.remove_pickup_container(container.container_id)
                        break
        else:
            success = False
        
        # Update the crane's position to the destination
        if success:
            # Extract bay and row from position strings
            dest_bay, dest_row = self._position_to_bay_row(destination_position)
            self.current_position = (dest_bay, dest_row)
            return container, time_taken
        else:
            # If the placement failed, we need to put the container back
            if self._is_storage_position(source_position):
                storage_yard.add_container(source_position, removed_container)
            elif self._is_truck_position(source_position):
                truck = self._get_truck_at_position(source_position, trucks_in_terminal)
                truck.add_container(removed_container)
            elif self._is_train_position(source_position):
                self._add_container_to_train(source_position, removed_container, trains_in_terminal)
                
            return None, 0
    
    def calculate_movement_time(self, source_position: str, destination_position: str, container: Any) -> float:
        """
        Calculate the time needed to move a container from source to destination.
        
        Args:
            source_position: Source position string
            destination_position: Destination position string
            container: Container to move
            
        Returns:
            Time in seconds needed for the movement
        """
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
        """
        Estimate movement time without detailed calculations.
        Used when terminal position data is not available.
        
        Args:
            source_position: Source position string
            destination_position: Destination position string
            container: Container to move
            
        Returns:
            Estimated time in seconds
        """
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
        """
        Calculate travel time with acceleration and deceleration.
        
        Args:
            distance: Distance to travel in meters
            max_speed: Maximum speed in meters per second
            acceleration: Acceleration/deceleration in meters per second squared
        
        Returns:
            Time in seconds needed for the travel
        """
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
        """
        Get all positions where a container can be picked up.
        
        Args:
            storage_yard: Reference to the storage yard
            trucks_in_terminal: Dictionary of trucks currently in the terminal
            trains_in_terminal: Dictionary of trains currently in the terminal
            
        Returns:
            List of position strings where containers can be picked up
        """
        source_positions = []
        
        # Check storage yard
        for row in storage_yard.row_names:
            for bay in range(storage_yard.num_bays):
                position = f"{row}{bay+1}"
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
        """
        Get all positions where a container can be placed.
        
        Args:
            source_position: Source position of the container
            container: Container to be moved
            storage_yard: Reference to the storage yard
            trucks_in_terminal: Dictionary of trucks currently in the terminal
            trains_in_terminal: Dictionary of trains currently in the terminal
            
        Returns:
            List of position strings where the container can be placed
        """
        destinations = []
        
        # Storage positions
        for row in storage_yard.row_names:
            for bay in range(storage_yard.num_bays):
                position = f"{row}{bay+1}"
                if position != source_position and self._is_position_in_crane_area(position):
                    if storage_yard.can_accept_container(position, container):
                        destinations.append(position)
        
        # Truck positions
        for spot, truck in trucks_in_terminal.items():
            if spot != source_position and self._is_position_in_crane_area(spot):
                # Check if this is a pickup truck looking for this container
                if hasattr(truck, 'pickup_container_ids') and container.container_id in truck.pickup_container_ids:
                    # Priority destination for pickup
                    if not truck.is_full() and truck.has_space_for_container(container):
                        destinations.append(spot)
                elif hasattr(truck, 'is_pickup_truck') and not truck.is_pickup_truck:
                    # Delivery truck can receive containers
                    if not truck.is_full() and truck.has_space_for_container(container):
                        destinations.append(spot)
        
        # Train positions
        for track, train in trains_in_terminal.items():
            for i, wagon in enumerate(train.wagons):
                slot = f"{track.lower()}_{i+1}"
                if slot != source_position and self._is_position_in_crane_area(slot):
                    # Check if this wagon has a pickup request for this container
                    if container.container_id in wagon.pickup_container_ids:
                        # Priority destination for pickup
                        if wagon.get_available_length() >= container.length:
                            destinations.append(slot)
                    elif source_position.startswith('p_'):
                        # Container is from a truck, check if train is accepting deliveries
                        if wagon.get_available_length() >= container.length:
                            destinations.append(slot)
        
        return destinations
    
    def _is_position_in_crane_area(self, position: str) -> bool:
        """
        Check if a position is within this crane's operational area.
        
        Args:
            position: Position string to check
            
        Returns:
            Boolean indicating if position is in this crane's area
        """
        bay, _ = self._position_to_bay_row(position)
        return self.start_bay <= bay <= self.end_bay
    
    def _position_to_bay_row(self, position: str) -> Tuple[int, int]:
        """
        Convert a position string to (bay, row) coordinates.
        
        Args:
            position: Position string (e.g., 'A1', 't1_2', 'p_3')
            
        Returns:
            Tuple of (bay, row) indices
        """
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
    
    def _calculate_crane_movement_time(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> float:
        """
        Calculate time for crane to move from current position to source position.
        
        Args:
            from_pos: Current position as (bay, row)
            to_pos: Target position as (bay, row)
            
        Returns:
            Time in seconds
        """
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
        """
        Get the container at a position.
        
        Args:
            position: Position string
            storage_yard: Reference to the storage yard
            trucks_in_terminal: Dictionary of trucks currently in the terminal
            trains_in_terminal: Dictionary of trains currently in the terminal
            
        Returns:
            Container object or None if no container is found
        """
        if self._is_storage_position(position):
            # Get top container from storage
            container, _ = storage_yard.get_top_container(position)
            return container
            
        elif self._is_truck_position(position):
            # Get container from truck
            truck = self._get_truck_at_position(position, trucks_in_terminal)
            if truck and hasattr(truck, 'containers') and truck.containers:
                return truck.containers[0]  # Return the first container
            return None
            
        elif self._is_train_position(position):
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
        """
        Check if a container can be placed at a position.
        
        Args:
            position: Position string
            container: Container to place
            storage_yard: Reference to the storage yard
            trucks_in_terminal: Dictionary of trucks currently in the terminal
            trains_in_terminal: Dictionary of trains currently in the terminal
            
        Returns:
            Boolean indicating if the container can be placed
        """
        if self._is_storage_position(position):
            # Check if container can be added to storage
            return storage_yard.can_accept_container(position, container)
            
        elif self._is_truck_position(position):
            # Check if container can be added to truck
            truck = self._get_truck_at_position(position, trucks_in_terminal)
            if truck and hasattr(truck, 'add_container'):
                # Check if this is a pickup truck looking for this container
                if hasattr(truck, 'pickup_container_ids') and hasattr(container, 'container_id'):
                    if container.container_id in truck.pickup_container_ids:
                        # Verify truck has space
                        return not truck.is_full() and truck.has_space_for_container(container)
                elif hasattr(truck, 'is_pickup_truck') and not truck.is_pickup_truck:
                    # Check if delivery truck has space
                    return not truck.is_full() and truck.has_space_for_container(container)
            return False
            
        elif self._is_train_position(position):
            # Check if container can be added to train
            train_id, wagon_index = self._parse_train_position(position, trains_in_terminal)
            if train_id and wagon_index is not None:
                train = trains_in_terminal.get(train_id)
                if train and 0 <= wagon_index < len(train.wagons):
                    wagon = train.wagons[wagon_index]
                    
                    # Check if wagon is looking for this container
                    if hasattr(container, 'container_id') and container.container_id in wagon.pickup_container_ids:
                        return True
                    
                    # Check if wagon has space
                    return wagon.get_available_length() >= container.length
            return False
            
        return False
    
    def _get_position_type(self, position: str) -> str:
        """
        Determine the type of a position (train, truck, storage).
        
        Args:
            position: Position string
            
        Returns:
            Type of the position ('train', 'truck', or 'storage')
        """
        if self._is_train_position(position):
            return 'train'
        elif self._is_truck_position(position):
            return 'truck'
        else:
            return 'storage'
    
    def _is_storage_position(self, position: str) -> bool:
        """Check if a position is in the storage yard."""
        # Storage positions typically start with a letter and are followed by a number
        return position[0].isalpha() and position[1:].isdigit()
    
    def _is_truck_position(self, position: str) -> bool:
        """Check if a position is a truck parking spot."""
        return position.startswith('p_')
    
    def _is_train_position(self, position: str) -> bool:
        """Check if a position is a train slot."""
        return position.startswith('t') and '_' in position
    
    def _get_truck_at_position(self, position: str, trucks_in_terminal: Dict[str, Any]) -> Optional[Any]:
        """Get the truck at a position."""
        return trucks_in_terminal.get(position)
    
    def _parse_train_position(self, position: str, trains_in_terminal: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
        """
        Parse a train position string to get train ID and wagon index.
        
        Args:
            position: Position string (e.g., 't1_2')
            trains_in_terminal: Dictionary of trains currently in the terminal
            
        Returns:
            Tuple of (train_id, wagon_index) or (None, None) if invalid
        """
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
        """
        Remove a container from a train.
        
        Args:
            position: Train position string
            container: Container to remove
            trains_in_terminal: Dictionary of trains currently in the terminal
            
        Returns:
            Removed container or None if removal failed
        """
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
        """
        Add a container to a train.
        
        Args:
            position: Train position string
            container: Container to add
            trains_in_terminal: Dictionary of trains currently in the terminal
            
        Returns:
            Boolean indicating success
        """
        train_id, wagon_index = self._parse_train_position(position, trains_in_terminal)
        if train_id and wagon_index is not None:
            train = trains_in_terminal.get(train_id)
            if train and 0 <= wagon_index < len(train.wagons):
                wagon = train.wagons[wagon_index]
                return wagon.add_container(container)
        
        return False


if __name__ == "__main__":
    # Mock classes for testing
    class MockTerminal:
        def __init__(self):
            self.track_names = ['T1', 'T2', 'T3']
            self.positions = {
                'A1': (0, 10),
                'A2': (1, 10),
                'B1': (0, 11),
                'B2': (1, 11),
                't1_1': (0, 0),
                't1_2': (1, 0),
                't2_1': (0, 1),
                'p_1': (0, 5),
                'p_2': (1, 5)
            }
    
    class MockContainer:
        def __init__(self, container_id, container_type, length=6.0, weight=20000, goods_type="Regular"):
            self.container_id = container_id
            self.container_type = container_type
            self.length = length
            self.weight = weight
            self.goods_type = goods_type
        
        def can_stack_with(self, other_container):
            return True
    
    class MockStorageYard:
        def __init__(self):
            self.row_names = ['A', 'B', 'C']
            self.num_bays = 5
            self.containers = {
                'A1': MockContainer("CONT1", "TWEU"),
                'B2': MockContainer("CONT2", "FEU")
            }
        
        def get_top_container(self, position):
            return self.containers.get(position), 1
        
        def can_accept_container(self, position, container):
            return position not in self.containers
        
        def remove_container(self, position):
            if position in self.containers:
                return self.containers.pop(position)
            return None
        
        def add_container(self, position, container):
            if position not in self.containers:
                self.containers[position] = container
                return True
            return False
    
    class MockTruck:
        def __init__(self, truck_id, containers=None):
            self.truck_id = truck_id
            self.containers = containers or []
            self.pickup_container_ids = set()
            self.is_pickup_truck = len(self.containers) == 0
            self.is_full = False
        
        def has_containers(self):
            return len(self.containers) > 0
        
        def remove_container(self, container_id=None):
            if self.containers:
                return self.containers.pop(0)
            return None
        
        def add_container(self, container):
            self.containers.append(container)
            return True
        
        def has_space_for_container(self, container):
            return not self.is_full
    
    class MockTrain:
        def __init__(self, train_id, num_wagons=3):
            self.train_id = train_id
            self.wagons = [MockWagon(f"{train_id}_W{i+1}") for i in range(num_wagons)]
    
    class MockWagon:
        def __init__(self, wagon_id, containers=None):
            self.wagon_id = wagon_id
            self.containers = containers or []
            self.pickup_container_ids = set()
        
        def add_container(self, container):
            self.containers.append(container)
            return True
        
        def get_available_length(self):
            return 20.0  # Always has space
    
    # Create test objects
    terminal = MockTerminal()
    storage_yard = MockStorageYard()
    
    trucks_in_terminal = {
        'p_1': MockTruck("T1", [MockContainer("CONT3", "TWEU")]),
        'p_2': MockTruck("T2")
    }
    
    # Add a container to pickup list
    trucks_in_terminal['p_2'].pickup_container_ids.add("CONT1")
    
    trains_in_terminal = {
        'T1': MockTrain("TR1", 3)
    }
    
    # Add containers to train
    trains_in_terminal['T1'].wagons[0].containers.append(MockContainer("CONT4", "FEU"))
    
    # Create crane and test functionality
    crane = RMGCrane("RMG1", terminal, 0, 10, (0, 0))
    
    print("Testing RMG Crane functionality:")
    print("Initial position:", crane.current_position)
    
    # Get valid moves
    valid_moves = crane.get_valid_moves(storage_yard, trucks_in_terminal, trains_in_terminal)
    print(f"Found {len(valid_moves)} valid moves:")
    for (source, dest), time in valid_moves.items():
        print(f"  {source} -> {dest}: {time:.2f} seconds")
    
    # Test a move
    if valid_moves:
        source, dest = list(valid_moves.keys())[0]
        container, time = crane.move_container(source, dest, storage_yard, trucks_in_terminal, trains_in_terminal)
        print(f"\nMoved container {container.container_id} from {source} to {dest}")
        print(f"Time taken: {time:.2f} seconds")
        print(f"New crane position: {crane.current_position}")