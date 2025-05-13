import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
import time


class TensorActionMaskGenerator:
    """
    Generates action masks for the terminal environment using tensor operations.
    This significantly speeds up the calculation of valid moves.
    """
    
    def __init__(self, 
                 environment,  # Reference to the TerminalEnvironment
                 device: str = None):
        """
        Initialize the action mask generator.
        
        Args:
            environment: TerminalEnvironment object
            device: Device to use for tensors ('cuda' for GPU if available)
        """
        self.env = environment
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device} for action mask generation")
        
        # Get dimensions from environment
        self.num_cranes = len(self.env.cranes)
        self.num_terminal_trucks = len(self.env.terminal_trucks)
        self.num_positions = len(self.env.position_to_idx)
        
        # Initialize tensors for position types (for fast lookup)
        self._initialize_position_tensors()
        
        # Cache for action masks
        self.mask_cache = {}
        self.last_update_time = -1

    def _initialize_position_tensors(self):
        """Initialize tensors for position types and mappings."""
        # Create position type tensors
        self.storage_positions = torch.zeros(self.num_positions, dtype=torch.bool, device=self.device)
        self.rail_positions = torch.zeros(self.num_positions, dtype=torch.bool, device=self.device)
        self.truck_positions = torch.zeros(self.num_positions, dtype=torch.bool, device=self.device)
        
        # Fill tensors based on position types
        for pos, idx in self.env.position_to_idx.items():
            pos_type = self.env._get_position_type(pos)
            if pos_type == 'storage':
                self.storage_positions[idx] = True
            elif pos_type == 'train':
                self.rail_positions[idx] = True
            elif pos_type == 'truck':
                self.truck_positions[idx] = True
        
        # Create position encoding tensors
        self.position_to_idx_tensor = {pos: torch.tensor([idx], device=self.device) 
                                      for pos, idx in self.env.position_to_idx.items()}
        
        # Create bay row mapping tensor for storage positions
        self.storage_row_bay_tensor = torch.zeros((self.num_positions, 2), dtype=torch.int, device=self.device)
        for pos in self.env.position_to_idx:
            idx = self.env.position_to_idx[pos]
            if pos[0].isalpha() and pos[1:].isdigit():
                row = pos[0]
                bay = int(pos[1:])
                row_idx = ord(row) - ord('A')
                self.storage_row_bay_tensor[idx, 0] = row_idx
                self.storage_row_bay_tensor[idx, 1] = bay - 1  # 0-based indexing
        
        # Create container compatibility tensors (to be filled during operation)
        self.container_type_codes = {'Regular': 0, 'Reefer': 1, 'Dangerous': 2, 'Trailer': 3, 'Swap Body': 4}

    def generate_masks(self) -> Dict[str, torch.Tensor]:
        """
        Generate all action masks for the environment.
        
        Returns:
            Dictionary of action masks for crane movement, truck parking, and terminal trucks
        """
        # Check if we can use cached masks
        current_time = self.env.current_simulation_time
        if current_time == self.last_update_time and self.mask_cache:
            return self.mask_cache
        
        # Initialize all masks
        crane_action_mask = torch.zeros((self.num_cranes, self.num_positions, self.num_positions), 
                                       dtype=torch.int8, device=self.device)
        truck_parking_mask = torch.zeros((10, len(self.env.parking_spots)), 
                                        dtype=torch.int8, device=self.device)
        terminal_truck_mask = torch.zeros((self.num_terminal_trucks, self.num_positions, self.num_positions), 
                                        dtype=torch.int8, device=self.device)
        
        # Generate each type of mask
        start_time = time.time()
        self._generate_crane_action_mask_tensor(crane_action_mask)
        crane_time = time.time() - start_time
        
        start_time = time.time()
        self._generate_truck_parking_mask_tensor(truck_parking_mask)
        truck_time = time.time() - start_time
        
        start_time = time.time()
        self._generate_terminal_truck_mask_tensor(terminal_truck_mask)
        terminal_time = time.time() - start_time
        
        # Create mask dictionary and cache it
        masks = {
            'crane_movement': crane_action_mask,
            'truck_parking': truck_parking_mask,
            'terminal_truck': terminal_truck_mask
        }
        
        self.mask_cache = masks
        self.last_update_time = current_time
        
        # Performance logging
        total_time = crane_time + truck_time + terminal_time
        if hasattr(self.env, 'log_performance') and self.env.log_performance:
            print(f"Mask generation times: Crane {crane_time:.3f}s, "
                  f"Truck {truck_time:.3f}s, Terminal truck {terminal_time:.3f}s, "
                  f"Total {total_time:.3f}s")
        
        return masks

    def _generate_crane_action_mask_tensor(self, crane_action_mask: torch.Tensor):
        """
        Generate action mask for crane movements using tensor operations.
        
        Args:
            crane_action_mask: Tensor to fill with crane action mask [cranes, sources, destinations]
        """
        # For each crane
        for i, crane in enumerate(self.env.cranes):
            # Skip if crane is not available yet
            if self.env.current_simulation_time < self.env.crane_available_times[i]:
                continue
            
            # Generate source positions tensor
            source_positions = self._get_source_positions_tensor(crane)
            
            # For each valid source position
            for source_idx in torch.nonzero(source_positions).squeeze(-1):
                source_position = self.env.idx_to_position[source_idx.item()]
                
                # Get the container at this position
                container = self.env._get_container_at_position(source_position)
                
                if container is None:
                    continue
                
                # Get valid destinations for this container
                destinations = self._get_destination_positions_tensor(
                    crane, source_position, container)
                
                # Update action mask for this source and all valid destinations
                for dest_idx in torch.nonzero(destinations).squeeze(-1):
                    # Get destination position
                    dest_position = self.env.idx_to_position[dest_idx.item()]
                    
                    # Apply operational constraints
                    if self._is_rail_position_tensor(source_idx) and self._is_rail_position_tensor(dest_idx):
                        # No rail slot to rail slot movements
                        continue
                    
                    if self._is_storage_position_tensor(source_idx) and self._is_storage_position_tensor(dest_idx):
                        # Extract bay numbers
                        source_bay = int(source_position[1:]) - 1
                        dest_bay = int(dest_position[1:]) - 1
                        
                        # Check pre-marshalling distance constraint
                        if abs(source_bay - dest_bay) > 5:
                            continue
                        
                        # Check for stacking compatibility
                        existing_container, _ = self.env.storage_yard.get_top_container(dest_position)
                        
                        # If there's a container at the destination and we're trying to stack on top of it
                        if existing_container is not None:
                            # Check if stacking is safe
                            if not container.can_be_stacked_on(existing_container):
                                continue
                    
                    # Swap body/trailer placement restrictions in storage
                    if (container and hasattr(container, 'container_type') and 
                        container.container_type in ["Trailer", "Swap Body"] and 
                        self._is_storage_position_tensor(dest_idx)):
                        # Must be placed in appropriate area
                        if container.container_type == "Trailer" and not self.env._is_in_special_area(dest_position, 'trailer'):
                            continue
                        elif container.container_type == "Swap Body" and not self.env._is_in_special_area(dest_position, 'swap_body'):
                            continue
                    
                    # Update the action mask
                    crane_action_mask[i, source_idx, dest_idx] = 1

    def _generate_truck_parking_mask_tensor(self, truck_parking_mask: torch.Tensor):
        """
        Generate action mask for truck parking using tensor operations.
        
        Args:
            truck_parking_mask: Tensor to fill with truck parking mask [trucks, parking spots]
        """
        # Get available parking spots as a tensor
        available_spots = torch.ones(len(self.env.parking_spots), dtype=torch.bool, device=self.device)
        for spot in self.env.trucks_in_terminal:
            if spot in self.env.parking_spots:
                spot_idx = self.env.parking_spots.index(spot)
                available_spots[spot_idx] = False
        
        # Get trucks from the queue
        trucks_in_queue = list(self.env.truck_queue.vehicles.queue)
        
        # For each truck in the queue, determine valid parking spots
        for truck_idx, truck in enumerate(trucks_in_queue):
            if truck_idx >= 10:  # Only consider first 10 trucks
                break
                
            # First check: Can only park in available spots
            truck_parking_mask[truck_idx, available_spots] = 1

    def _generate_terminal_truck_mask_tensor(self, terminal_truck_mask: torch.Tensor):
        """
        Generate action mask for terminal truck movements using tensor operations.
        
        Args:
            terminal_truck_mask: Tensor to fill with terminal truck mask [trucks, sources, destinations]
        """
        for truck_idx, truck in enumerate(self.env.terminal_trucks):
            # Skip if truck is not available yet
            if self.env.current_simulation_time < self.env.terminal_truck_available_times[truck_idx]:
                continue
            
            # Find all trailer and swap body source positions
            for pos in self.env.position_to_idx:
                if not pos[0].isalpha() or not pos[1:].isdigit():
                    continue  # Skip non-storage positions
                    
                container = self.env._get_container_at_position(pos)
                
                if container is None or not hasattr(container, 'container_type'):
                    continue
                    
                if container.container_type not in ["Trailer", "Swap Body"]:
                    continue
                
                # Source position found
                source_idx = self.env.position_to_idx[pos]
                
                # Find valid destinations for this container
                for dest_pos in self.env.position_to_idx:
                    if not dest_pos[0].isalpha() or not dest_pos[1:].isdigit():
                        continue  # Skip non-storage positions
                        
                    # Check if destination is appropriate for this container type
                    if container.container_type == "Trailer" and not self.env._is_in_special_area(dest_pos, 'trailer'):
                        continue
                    elif container.container_type == "Swap Body" and not self.env._is_in_special_area(dest_pos, 'swap_body'):
                        continue
                    
                    # Destination must be empty
                    if self.env.storage_yard.get_top_container(dest_pos)[0] is not None:
                        continue
                    
                    # Get destination index
                    dest_idx = self.env.position_to_idx[dest_pos]
                    
                    # Can't move to same position
                    if source_idx == dest_idx:
                        continue
                    
                    # Valid movement found
                    terminal_truck_mask[truck_idx, source_idx, dest_idx] = 1

    def _get_source_positions_tensor(self, crane) -> torch.Tensor:
        """
        Get all positions where a container can be picked up as a tensor mask.
        
        Args:
            crane: The RMG crane to use
            
        Returns:
            Boolean tensor of valid source positions [num_positions]
        """
        # Create empty tensor for all positions
        valid_sources = torch.zeros(self.num_positions, dtype=torch.bool, device=self.device)
        
        # Check storage yard
        for row in self.env.storage_yard.row_names:
            for bay in range(1, self.env.storage_yard.num_bays + 1):
                position = f"{row}{bay}"
                
                # Only consider positions in the crane's operational area
                if not crane._is_position_in_crane_area(position, self.env.storage_yard):
                    continue
                
                if position in self.env.position_to_idx:
                    # Check if there's a container at this position
                    container, _ = self.env.storage_yard.get_top_container(position)
                    if container is not None:
                        idx = self.env.position_to_idx[position]
                        valid_sources[idx] = True
        
        # Check trucks
        for spot, truck in self.env.trucks_in_terminal.items():
            if crane._is_position_in_crane_area(spot, self.env.storage_yard) and truck.has_containers():
                if spot in self.env.position_to_idx:
                    idx = self.env.position_to_idx[spot]
                    valid_sources[idx] = True
        
        # Check trains
        for track, train in self.env.trains_in_terminal.items():
            for i, wagon in enumerate(train.wagons):
                if not wagon.is_empty():
                    slot = f"{track.lower()}_{i+1}"
                    if crane._is_position_in_crane_area(slot, self.env.storage_yard) and slot in self.env.position_to_idx:
                        idx = self.env.position_to_idx[slot]
                        valid_sources[idx] = True
        
        return valid_sources

    def _get_destination_positions_tensor(self, 
                                      crane, 
                                      source_position: str, 
                                      container: Any) -> torch.Tensor:
        """
        Get all positions where a container can be placed as a tensor mask.
        
        Args:
            crane: The RMG crane to use
            source_position: Source position string
            container: Container object to move
            
        Returns:
            Boolean tensor of valid destination positions [num_positions]
        """
        valid_dests = torch.zeros(self.num_positions, dtype=torch.bool, device=self.device)
        source_type = self.env._get_position_type(source_position)
        
        # Handle different source types with appropriate destination rules
        if source_type == 'train':
            # Train to truck with matching pickup ID
            for spot, truck in self.env.trucks_in_terminal.items():
                if crane._is_position_in_crane_area(spot, self.env.storage_yard) and hasattr(truck, 'pickup_container_ids'):
                    if hasattr(container, 'container_id') and container.container_id in truck.pickup_container_ids:
                        if not truck.is_full() and spot in self.env.position_to_idx:
                            idx = self.env.position_to_idx[spot]
                            valid_dests[idx] = True
            
            # If no matching truck, allow storage based on container type
            if not torch.any(valid_dests):
                # For storage destinations, check placement constraints
                for row in self.env.storage_yard.row_names:
                    for bay in range(1, self.env.storage_yard.num_bays + 1):
                        position = f"{row}{bay}"
                        if position != source_position and crane._is_position_in_crane_area(position, self.env.storage_yard):
                            # Check container type constraints
                            if hasattr(container, 'container_type'):
                                if container.container_type in ["Trailer", "Swap Body"]:
                                    # Check appropriate special area
                                    area_type = 'trailer' if container.container_type == "Trailer" else 'swap_body'
                                    is_valid_area = self.env._is_in_special_area(position, area_type)
                                    
                                    if is_valid_area and self.env.storage_yard.can_accept_container(position, container):
                                        if position in self.env.position_to_idx:
                                            idx = self.env.position_to_idx[position]
                                            valid_dests[idx] = True
                                else:
                                    # Regular/reefer/dangerous containers
                                    if self.env.storage_yard.can_accept_container(position, container):
                                        if position in self.env.position_to_idx:
                                            idx = self.env.position_to_idx[position]
                                            valid_dests[idx] = True
        
        elif source_type == 'truck':
            # Truck to wagons with matching pickup
            for track, train in self.env.trains_in_terminal.items():
                for i, wagon in enumerate(train.wagons):
                    slot = f"{track.lower()}_{i+1}"
                    if slot != source_position and crane._is_position_in_crane_area(slot, self.env.storage_yard):
                        if hasattr(container, 'container_id') and container.container_id in wagon.pickup_container_ids:
                            if slot in self.env.position_to_idx:
                                idx = self.env.position_to_idx[slot]
                                valid_dests[idx] = True
            
            # If no matching wagon, allow storage
            if not torch.any(valid_dests):
                for row in self.env.storage_yard.row_names:
                    for bay in range(1, self.env.storage_yard.num_bays + 1):
                        position = f"{row}{bay}"
                        if position != source_position and crane._is_position_in_crane_area(position, self.env.storage_yard):
                            if self.env.storage_yard.can_accept_container(position, container):
                                if position in self.env.position_to_idx:
                                    idx = self.env.position_to_idx[position]
                                    valid_dests[idx] = True
        
        elif source_type == 'storage':
            # Check for trucks looking for this container
            for spot, truck in self.env.trucks_in_terminal.items():
                if spot != source_position and crane._is_position_in_crane_area(spot, self.env.storage_yard):
                    if hasattr(truck, 'pickup_container_ids') and hasattr(container, 'container_id'):
                        if container.container_id in truck.pickup_container_ids:
                            if not truck.is_full() and spot in self.env.position_to_idx:
                                idx = self.env.position_to_idx[spot]
                                valid_dests[idx] = True
            
            # Check for wagons looking for this container
            for track, train in self.env.trains_in_terminal.items():
                for i, wagon in enumerate(train.wagons):
                    slot = f"{track.lower()}_{i+1}"
                    if slot != source_position and crane._is_position_in_crane_area(slot):
                        if hasattr(container, 'container_id') and container.container_id in wagon.pickup_container_ids:
                            if slot in self.env.position_to_idx:
                                idx = self.env.position_to_idx[slot]
                                valid_dests[idx] = True
            
            # Allow pre-marshalling within 5-bay distance limit
            if source_position[0].isalpha() and source_position[1:].isdigit():
                source_bay = int(source_position[1:]) - 1  # 0-based index
                
                for row in self.env.storage_yard.row_names:
                    for bay in range(1, self.env.storage_yard.num_bays + 1):
                        position = f"{row}{bay}"
                        if position != source_position and crane._is_position_in_crane_area(position, self.env.storage_yard):
                            # Check pre-marshalling distance constraint
                            dest_bay = bay - 1
                            if abs(source_bay - dest_bay) <= 5:  # Limit to 5 positions
                                if self.env.storage_yard.can_accept_container(position, container):
                                    if position in self.env.position_to_idx:
                                        idx = self.env.position_to_idx[position]
                                        valid_dests[idx] = True
        
        return valid_dests

    def _is_storage_position_tensor(self, position_idx: torch.Tensor) -> bool:
        """Check if a position index is a storage position."""
        return self.storage_positions[position_idx]

    def _is_rail_position_tensor(self, position_idx: torch.Tensor) -> bool:
        """Check if a position index is a rail position."""
        return self.rail_positions[position_idx]

    def _is_truck_position_tensor(self, position_idx: torch.Tensor) -> bool:
        """Check if a position index is a truck position."""
        return self.truck_positions[position_idx]

    def to_cpu(self):
        """Move all tensors to CPU for saving."""
        self.storage_positions = self.storage_positions.cpu()
        self.rail_positions = self.rail_positions.cpu()
        self.truck_positions = self.truck_positions.cpu()
        self.storage_row_bay_tensor = self.storage_row_bay_tensor.cpu()
        
        # Clear cache to force regeneration
        self.mask_cache = {}
        self.device = 'cpu'

    def to_device(self, device: str):
        """Move all tensors to specified device."""
        self.device = device
        self.storage_positions = self.storage_positions.to(device)
        self.rail_positions = self.rail_positions.to(device)
        self.truck_positions = self.truck_positions.to(device)
        self.storage_row_bay_tensor = self.storage_row_bay_tensor.to(device)
        
        # Clear cache to force regeneration
        self.mask_cache = {}


# This is a helper function to be added to TerminalEnvironment
def _get_observation_tensor(self):
    """Get the current observation as tensors for faster processing."""
    # Create observation tensors with torch for GPU compatibility
    
    # Crane positions and availability
    crane_positions = torch.tensor([crane.current_position for crane in self.cranes], 
                                 dtype=torch.float32, device=self.action_mask_generator.device)
    crane_available_times = torch.tensor(self.crane_available_times, 
                                       dtype=torch.float32, device=self.action_mask_generator.device)
    terminal_truck_available_times = torch.tensor(self.terminal_truck_available_times, 
                                                dtype=torch.float32, device=self.action_mask_generator.device)
    current_time = torch.tensor([self.current_simulation_time], 
                              dtype=torch.float32, device=self.action_mask_generator.device)
    
    # Yard state - can be a tensor from TensorStorageYard
    if hasattr(self.storage_yard, 'get_state_representation') and callable(getattr(self.storage_yard, 'get_state_representation')):
        # Use tensor-based representation if available
        yard_state = self.storage_yard.get_state_representation()
    else:
        # Fallback to numpy array
        yard_state = torch.tensor(self.storage_yard.get_state_representation(), 
                                dtype=torch.int32, device=self.action_mask_generator.device)
    
    # Parking status - use binary tensor
    parking_status = torch.zeros(len(self.parking_spots), 
                               dtype=torch.int8, device=self.action_mask_generator.device)
    for i, spot in enumerate(self.parking_spots):
        if spot in self.trucks_in_terminal:
            parking_status[i] = 1
    
    # Rail status - use binary tensor
    rail_status = torch.zeros((len(self.terminal.track_names), self.terminal.num_railslots_per_track), 
                            dtype=torch.int8, device=self.action_mask_generator.device)
    for i, track in enumerate(self.terminal.track_names):
        if track in self.trains_in_terminal:
            rail_status[i, :] = 1
    
    # Queue sizes
    queue_sizes = torch.tensor([self.truck_queue.size(), self.train_queue.size()], 
                             dtype=torch.int32, device=self.action_mask_generator.device)
    
    # Action masks (using tensor-based generation)
    action_mask = self.action_mask_generator.generate_masks()
    
    # Return combined observation dict
    return {
        'crane_positions': crane_positions,
        'crane_available_times': crane_available_times,
        'terminal_truck_available_times': terminal_truck_available_times,
        'current_time': current_time,
        'yard_state': yard_state,
        'parking_status': parking_status,
        'rail_status': rail_status,
        'queue_sizes': queue_sizes,
        'action_mask': action_mask
    }