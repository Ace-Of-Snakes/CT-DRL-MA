# Define standalone kernels for the terminal environment

import warp as wp

# Kernel for generating crane movement mask
@wp.kernel
def kernel_generate_crane_mask(crane_positions: wp.array(dtype=wp.float32, ndim=2),
                             crane_properties: wp.array(dtype=wp.float32, ndim=2),
                             yard_container_indices: wp.array(dtype=wp.int32, ndim=3),
                             stack_heights: wp.array(dtype=wp.int32, ndim=2),
                             rail_track_vehicles: wp.array(dtype=wp.int32, ndim=1),
                             parking_vehicles: wp.array(dtype=wp.int32, ndim=1),
                             container_positions: wp.array(dtype=wp.int32, ndim=1),
                             current_time: float,
                             num_cranes: int,
                             num_positions: int,
                             crane_mask: wp.array(dtype=wp.int32, ndim=3)):
    """Kernel to generate action mask for crane movements."""
    # Get thread indices
    crane_idx = wp.tid(0)
    src_idx = wp.tid(1)
    dst_idx = wp.tid(2)
    
    # Check bounds
    if (crane_idx >= num_cranes or 
        src_idx >= num_positions or 
        dst_idx >= num_positions):
        return
    
    # Default: invalid action
    crane_mask[crane_idx, src_idx, dst_idx] = 0
    
    # Check if crane is available
    crane_available_time = crane_properties[crane_idx, 2]
    if crane_available_time > current_time:
        return
        
    # Check if source position has a container
    source_has_container = False
    for container_idx in range(container_positions.shape[0]):
        if container_positions[container_idx] == src_idx:
            source_has_container = True
            break
            
    if not source_has_container:
        return
        
    # More validity checks would go here
    # For now, we'll just mark the action as valid
    crane_mask[crane_idx, src_idx, dst_idx] = 1

# Kernel for generating truck parking mask
@wp.kernel
def kernel_generate_truck_parking_mask(parking_vehicles: wp.array(dtype=wp.int32, ndim=1),
                                     vehicle_properties: wp.array(dtype=wp.float32, ndim=2),
                                     num_vehicles: int,
                                     num_parking_spots: int,
                                     truck_parking_mask: wp.array(dtype=wp.int32, ndim=2)):
    """Kernel to generate action mask for truck parking."""
    # Get thread indices
    truck_idx = wp.tid(0)
    spot_idx = wp.tid(1)
    
    # Check bounds
    if truck_idx >= num_vehicles or spot_idx >= num_parking_spots:
        return
    
    # Default: invalid action
    truck_parking_mask[truck_idx, spot_idx] = 0
    
    # Check if truck is active
    if vehicle_properties[truck_idx, 6] <= 0:
        return
    
    # Check if truck is ready to park (status = WAITING)
    if vehicle_properties[truck_idx, 1] != 1:  # WAITING status
        return
    
    # Check if parking spot is available
    if parking_vehicles[spot_idx] >= 0:
        return
    
    # Mark as valid action
    truck_parking_mask[truck_idx, spot_idx] = 1

# Kernel for generating terminal truck mask
@wp.kernel
def kernel_generate_terminal_truck_mask(terminal_truck_positions: wp.array(dtype=wp.int32, ndim=1),
                                      container_positions: wp.array(dtype=wp.int32, ndim=1),
                                      num_terminal_trucks: int,
                                      num_positions: int,
                                      terminal_truck_mask: wp.array(dtype=wp.int32, ndim=3)):
    """Kernel to generate action mask for terminal truck movements."""
    # Get thread indices
    truck_idx = wp.tid(0)
    src_idx = wp.tid(1)
    dst_idx = wp.tid(2)
    
    # Check bounds
    if (truck_idx >= num_terminal_trucks or
        src_idx >= num_positions or
        dst_idx >= num_positions):
        return
    
    # Default: invalid action
    terminal_truck_mask[truck_idx, src_idx, dst_idx] = 0
    
    # For now, just allow all terminal truck movements (simplified)
    terminal_truck_mask[truck_idx, src_idx, dst_idx] = 1