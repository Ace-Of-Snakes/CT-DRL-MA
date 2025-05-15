import torch
import time
from typing import List, Dict, Tuple, Optional, Any


class GPUActionMaskGenerator:
    """
    GPU-accelerated generator for action masks in the terminal environment.
    Uses PyTorch tensors for efficient computation on GPU.
    """
    
    def __init__(self, 
                 environment,  # Reference to the TerminalEnvironment
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the action mask generator.
        
        Args:
            environment: TerminalEnvironment object
            device: Device to use for tensors ('cuda' for GPU or 'cpu')
        """
        self.env = environment
        self.device = device
        print(f"Initializing GPU action mask generator on device: {self.device}")
        
        # Get dimensions from environment
        self.num_cranes = len(self.env.cranes)
        self.num_terminal_trucks = len(self.env.terminal_trucks)
        self.num_positions = len(self.env.position_to_idx)
        
        # Initialize tensors for position types (for fast lookup)
        self._initialize_position_tensors()
        
        # Pre-allocate mask tensors to avoid repeated allocation
        self._preallocate_masks()
        
        # Cache for action masks
        self.mask_cache = {}
        self.last_update_time = -1
        
        # Performance tracking
        self.mask_generation_times = []
    
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
        
        # Create bay-row mapping tensor for storage positions
        self.storage_row_bay_tensor = torch.zeros((self.num_positions, 2), dtype=torch.int, device=self.device)
        for pos in self.env.position_to_idx:
            idx = self.env.position_to_idx[pos]
            if pos[0].isalpha() and pos[1:].isdigit():
                row = pos[0]
                bay = int(pos[1:])
                row_idx = ord(row) - ord('A')
                self.storage_row_bay_tensor[idx, 0] = row_idx
                self.storage_row_bay_tensor[idx, 1] = bay - 1  # 0-based indexing
    
    def _preallocate_masks(self):
        """Pre-allocate tensor masks to avoid repeated memory allocation."""
        # Crane movement mask: [crane, source, destination]
        self.crane_mask = torch.zeros((self.num_cranes, self.num_positions, self.num_positions), 
                                     dtype=torch.int8, device=self.device)
        
        # Truck parking mask: [truck, parking spot]
        self.truck_mask = torch.zeros((10, len(self.env.parking_spots)), 
                                     dtype=torch.int8, device=self.device)
        
        # Terminal truck mask: [terminal truck, source, destination]
        self.terminal_truck_mask = torch.zeros((self.num_terminal_trucks, 
                                              self.num_positions, self.num_positions), 
                                             dtype=torch.int8, device=self.device)
    
    def generate_masks(self) -> Dict[str, torch.Tensor]:
        """
        Generate all action masks for the environment using GPU tensors.
        
        Returns:
            Dictionary of action masks for crane movement, truck parking, and terminal trucks
        """
        # Check if we can use cached masks
        current_time = self.env.current_simulation_time
        if current_time == self.last_update_time and self.mask_cache:
            return self.mask_cache
        
        # Track generation time
        start_time = time.time()
        
        # Reset all masks to zero
        self.crane_mask.zero_()
        self.truck_mask.zero_()
        self.terminal_truck_mask.zero_()
        
        # Generate each mask type
        self._generate_crane_movement_mask()
        self._generate_truck_parking_mask()
        self._generate_terminal_truck_mask()
        
        # Create masks dictionary
        masks = {
            'crane_movement': self.crane_mask,
            'truck_parking': self.truck_mask,
            'terminal_truck': self.terminal_truck_mask
        }
        
        # Update cache
        self.mask_cache = masks
        self.last_update_time = current_time
        
        # Track performance
        self.mask_generation_times.append(time.time() - start_time)
        
        return masks
    
    def _generate_crane_movement_mask(self):
        """Generate mask for valid crane movements."""
        # For each crane
        for i, crane in enumerate(self.env.cranes):
            # Skip if crane is not available yet
            if self.env.current_simulation_time < self.env.crane_available_times[i]:
                continue
            
            # Find valid source positions (where containers can be picked up)
            for src_idx in range(self.num_positions):
                # Get source position
                source_position = self.env.idx_to_position.get(src_idx)
                if source_position is None:
                    continue
                
                # Check if position is in crane's area
                if not crane._is_position_in_crane_area(source_position, self.env.storage_yard):
                    continue
                
                # Check if there's a container at this position
                container = self.env._get_container_at_position(source_position)
                if container is None:
                    continue
                
                # For each potential destination
                for dst_idx in range(self.num_positions):
                    # Get destination position
                    dest_position = self.env.idx_to_position.get(dst_idx)
                    if dest_position is None:
                        continue
                    
                    # Check if destination is in crane's area
                    if not crane._is_position_in_crane_area(dest_position, self.env.storage_yard):
                        continue
                    
                    # Skip rail-to-rail movements
                    if self.rail_positions[src_idx] and self.rail_positions[dst_idx]:
                        continue
                    
                    # Check storage-to-storage premarshalling constraints
                    if self.storage_positions[src_idx] and self.storage_positions[dst_idx]:
                        # Extract bay numbers for distance check
                        if source_position[0].isalpha() and source_position[1:].isdigit() and \
                           dest_position[0].isalpha() and dest_position[1:].isdigit():
                            source_bay = int(source_position[1:]) - 1
                            dest_bay = int(dest_position[1:]) - 1
                            
                            # Check pre-marshalling distance constraint
                            if abs(source_bay - dest_bay) > 5:
                                continue
                    
                    # Check if destination can accept container
                    if self.storage_positions[dst_idx]:
                        if not self.env.storage_yard.can_accept_container(dest_position, container):
                            continue
                    elif self.truck_positions[dst_idx]:
                        # Check if truck exists and can accept container
                        truck = self.env.trucks_in_terminal.get(dest_position)
                        if truck is None or truck.is_full():
                            continue
                        
                        # If it's a pickup truck, check if container matches
                        if hasattr(truck, 'pickup_container_ids') and \
                           hasattr(container, 'container_id') and \
                           truck.pickup_container_ids and \
                           container.container_id not in truck.pickup_container_ids:
                            continue
                    elif self.rail_positions[dst_idx]:
                        # Check if train exists and has matching pickup ID
                        if '_' in dest_position:
                            track_id = f"T{dest_position.split('_')[0][1:]}"
                            train = self.env.trains_in_terminal.get(track_id)
                            
                            if train is None:
                                continue
                            
                            # Check if container is in train's pickup list
                            has_pickup = False
                            if hasattr(container, 'container_id'):
                                for wagon in train.wagons:
                                    if container.container_id in wagon.pickup_container_ids:
                                        has_pickup = True
                                        break
                                
                                if not has_pickup:
                                    continue
                    
                    # Mark as valid action
                    self.crane_mask[i, src_idx, dst_idx] = 1
    
    def _generate_truck_parking_mask(self):
        """Generate mask for valid truck parking actions."""
        # Get available parking spots
        available_spots = torch.ones(len(self.env.parking_spots), dtype=torch.bool, device=self.device)
        for i, spot in enumerate(self.env.parking_spots):
            if spot in self.env.trucks_in_terminal:
                available_spots[i] = False
        
        # Get trucks from queue
        trucks_in_queue = list(self.env.truck_queue.vehicles.queue)
        
        # Mark available spots for each truck
        for truck_idx, truck in enumerate(trucks_in_queue):
            if truck_idx >= 10:  # Only consider first 10 trucks
                break
            
            # Mark all available spots as valid
            for spot_idx in range(len(self.env.parking_spots)):
                if available_spots[spot_idx]:
                    self.truck_mask[truck_idx, spot_idx] = 1
    
    def _generate_terminal_truck_mask(self):
        """Generate mask for valid terminal truck actions."""
        # For each terminal truck
        for truck_idx, truck in enumerate(self.env.terminal_trucks):
            # Skip if truck is not available yet
            if self.env.current_simulation_time < self.env.terminal_truck_available_times[truck_idx]:
                continue
            
            # Find all trailer and swap body source positions
            for src_idx in range(self.num_positions):
                # Only consider storage positions
                if not self.storage_positions[src_idx]:
                    continue
                
                source_position = self.env.idx_to_position.get(src_idx)
                if source_position is None:
                    continue
                
                # Check if there's a container at this position
                container = self.env._get_container_at_position(source_position)
                if container is None or not hasattr(container, 'container_type'):
                    continue
                
                # Only terminal trucks can move trailers and swap bodies
                if container.container_type not in ["Trailer", "Swap Body"]:
                    continue
                
                # Find valid destinations
                for dst_idx in range(self.num_positions):
                    # Only consider storage positions
                    if not self.storage_positions[dst_idx]:
                        continue
                    
                    dest_position = self.env.idx_to_position.get(dst_idx)
                    if dest_position is None:
                        continue
                    
                    # Check if destination is appropriate for this container type
                    if container.container_type == "Trailer" and not self.env._is_in_special_area(dest_position, 'trailer'):
                        continue
                    elif container.container_type == "Swap Body" and not self.env._is_in_special_area(dest_position, 'swap_body'):
                        continue
                    
                    # Check if destination is empty
                    if self.env.storage_yard.get_top_container(dest_position)[0] is not None:
                        continue
                    
                    # Can't move to same position
                    if src_idx == dst_idx:
                        continue
                    
                    # This is a valid movement
                    self.terminal_truck_mask[truck_idx, src_idx, dst_idx] = 1
    
    def to_cpu(self):
        """Move all tensors to CPU for serialization/saving."""
        self.device = 'cpu'
        self.storage_positions = self.storage_positions.cpu()
        self.rail_positions = self.rail_positions.cpu()
        self.truck_positions = self.truck_positions.cpu()
        self.storage_row_bay_tensor = self.storage_row_bay_tensor.cpu()
        self.crane_mask = self.crane_mask.cpu()
        self.truck_mask = self.truck_mask.cpu()
        self.terminal_truck_mask = self.terminal_truck_mask.cpu()
        
        # Update position tensor mapping
        self.position_to_idx_tensor = {pos: tensor.cpu() for pos, tensor in self.position_to_idx_tensor.items()}
        
        # Clear cache to force regeneration
        self.mask_cache = {}
    
    def to_device(self, device: str):
        """Move all tensors to the specified device."""
        prev_device = self.device
        self.device = device
        
        # Only transfer if device has actually changed
        if prev_device != device:
            self.storage_positions = self.storage_positions.to(device)
            self.rail_positions = self.rail_positions.to(device)
            self.truck_positions = self.truck_positions.to(device)
            self.storage_row_bay_tensor = self.storage_row_bay_tensor.to(device)
            self.crane_mask = self.crane_mask.to(device)
            self.truck_mask = self.truck_mask.to(device)
            self.terminal_truck_mask = self.terminal_truck_mask.to(device)
            
            # Update position tensor mapping
            self.position_to_idx_tensor = {pos: tensor.to(device) for pos, tensor in self.position_to_idx_tensor.items()}
            
            # Clear cache to force regeneration
            self.mask_cache = {}
    
    def print_performance_stats(self):
        """Print performance statistics for mask generation."""
        if not self.mask_generation_times:
            print("No performance data available yet.")
            return
            
        import numpy as np
        times = np.array(self.mask_generation_times)
        
        print("\nAction Mask Generation Performance:")
        print(f"  Device: {self.device}")
        print(f"  Average generation time: {times.mean()*1000:.2f}ms")
        print(f"  Min generation time: {times.min()*1000:.2f}ms")
        print(f"  Max generation time: {times.max()*1000:.2f}ms")