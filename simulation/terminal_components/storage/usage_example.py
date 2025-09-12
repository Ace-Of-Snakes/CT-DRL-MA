# simulation/terminal_components/storage/yard_usage_example.py

import numpy as np
import time
from typing import List
from simulation.terminal_components.storage_units.Container import Container
from simulation.terminal_components.storage.StorageYard import OptimizedStorageYard
from simulation.terminal_components.storage.TensorConverter import YardTensorConverter
from simulation.terminal_components.storage.storage_constants import *

def create_test_container(container_id: str, container_type: str, goods_type: str) -> Container:
    """Create a test container with specified properties."""
    return Container(
        container_id=container_id,
        direction="Import",
        container_type=container_type,
        arrival_date=None,
        departure_date=None,
        goods_type=goods_type,
        length_m=BAY_LENGTH_M if container_type == "40ft" else BASE_SLOT_LENGTH_M
    )

def test_yard_operations():
    """Test and demonstrate yard operations."""
    
    print("=== Initializing Storage Yard ===")
    
    # Define special areas (1-based indexing as per spec)
    special_areas = [
        # Reefer areas at both ends
        (1, 1, "r"), (1, 2, "r"), (1, 3, "r"), (1, 4, "r"), (1, 5, "r"),
        (15, 1, "r"), (15, 2, "r"), (15, 3, "r"), (15, 4, "r"), (15, 5, "r"),
        
        # Ground-only areas for trailers/swap bodies (first row)
        (1, 1, "sb_t"), (2, 1, "sb_t"), (3, 1, "sb_t"), (4, 1, "sb_t"), (5, 1, "sb_t"),
        (6, 1, "sb_t"), (7, 1, "sb_t"), (8, 1, "sb_t"), (9, 1, "sb_t"), (10, 1, "sb_t"),
        
        # Dangerous goods area in middle
        (7, 3, "dg"), (8, 3, "dg"), (9, 3, "dg"),
        (7, 4, "dg"), (8, 4, "dg"), (9, 4, "dg"),
    ]
    
    # Initialize yard
    yard = OptimizedStorageYard(
        n_rows=5,
        n_bays=15,
        n_tiers=4,
        special_areas=special_areas
    )
    
    print(f"Yard dimensions: {yard.n_rows} rows x {yard.n_bays} bays x {yard.n_tiers} tiers")
    print(f"Subslots per bay: {YARD_SPLIT_FACTOR}")
    
    # Test container placement
    print("\n=== Testing Container Placement ===")
    
    # Test 20ft container
    container_20ft = create_test_container("C001", "20ft", GOODS_REGULAR)
    positions_20ft = yard.find_placement_positions(container_20ft, near_bay=7, max_distance=5)
    print(f"Found {len(positions_20ft)} positions for 20ft container")
    
    if positions_20ft:
        row, bay, tier, start_split = positions_20ft[0]
        success = yard.add_container(container_20ft, row, bay, tier, start_split)
        print(f"Added 20ft container at ({row}, {bay}, {tier}, {start_split}): {success}")
    
    # Test 40ft container
    container_40ft = create_test_container("C002", "40ft", GOODS_REGULAR)
    positions_40ft = yard.find_placement_positions(container_40ft, near_bay=7, max_distance=5)
    print(f"Found {len(positions_40ft)} positions for 40ft container")
    
    if positions_40ft:
        row, bay, tier, start_split = positions_40ft[0]
        success = yard.add_container(container_40ft, row, bay, tier, start_split)
        print(f"Added 40ft container at ({row}, {bay}, {tier}, {start_split}): {success}")
    
    # Test 45ft container (cross-bay)
    container_45ft = create_test_container("C003", "45ft", GOODS_REGULAR)
    positions_45ft = yard.find_placement_positions(container_45ft, near_bay=7, max_distance=5)
    print(f"Found {len(positions_45ft)} positions for 45ft container")
    
    if positions_45ft:
        row, bay, tier, start_split = positions_45ft[0]
        success = yard.add_container(container_45ft, row, bay, tier, start_split)
        print(f"Added 45ft container at ({row}, {bay}, {tier}, {start_split}): {success}")
    
    # Test stacking
    print("\n=== Testing Stacking Rules ===")
    container_stack = create_test_container("C004", "20ft", GOODS_REGULAR)
    
    # Try to stack on top of the first 20ft container
    if positions_20ft:
        row, bay, tier, start_split = positions_20ft[0]
        success = yard.add_container(container_stack, row, bay, tier + 1, start_split)
        print(f"Stacking 20ft on 20ft: {success}")
    
    # Test special containers
    print("\n=== Testing Special Container Types ===")
    
    # Reefer container
    reefer_container = create_test_container("C005", "40ft", GOODS_REEFER)
    reefer_positions = yard.find_placement_positions(reefer_container, near_bay=0, max_distance=2)
    print(f"Found {len(reefer_positions)} positions for reefer container")
    
    # Dangerous goods container
    dg_container = create_test_container("C006", "40ft", GOODS_DANGEROUS)
    dg_positions = yard.find_placement_positions(dg_container, near_bay=7, max_distance=3)
    print(f"Found {len(dg_positions)} positions for dangerous goods container")
    
    # Trailer (ground-only)
    trailer = create_test_container("C007", "Trailer", GOODS_REGULAR)
    trailer.container_type = "Trailer"  # Override type
    trailer_positions = yard.find_placement_positions(trailer, near_bay=5, max_distance=5)
    print(f"Found {len(trailer_positions)} positions for trailer (ground-only)")
    
    # Test accessibility and moves
    print("\n=== Testing Accessibility and Moves ===")
    moveable = yard.get_moveable_containers()
    print(f"Moveable containers: {moveable}")
    
    moves = yard.get_yard_moves(max_distance=2)
    print(f"Possible moves: {len(moves)} containers can be moved")
    for container_id, destinations in list(moves.items()):  # Show first 2
        print(f"  {container_id}: {len(destinations)} alternative positions")
    
    # Test removal
    print("\n=== Testing Container Removal ===")
    removed = yard.remove_container("C001")
    if removed:
        print(f"Successfully removed container {removed.container_id}")
    else:
        print("Could not remove container (blocked or not found)")
    
    # Test tensor conversion
    print("\n=== Testing Tensor Conversion ===")
    converter = YardTensorConverter(yard, device='cpu')
    
    start_time = time.time()
    full_tensor = converter.to_tensor(normalize=True, resolution='full')
    print(f"Full resolution tensor shape: {full_tensor.shape}")
    print(f"Conversion time: {(time.time() - start_time) * 1000:.2f}ms")
    
    bay_tensor = converter.to_tensor(normalize=True, resolution='bay')
    print(f"Bay resolution tensor shape: {bay_tensor.shape}")
    
    compact_state = converter.get_compact_state()
    print(f"Compact state shape: {compact_state.shape}")
    print(f"Compact state values: {compact_state.numpy()}")
    
    action_mask = converter.get_action_mask()
    print(f"Action mask shape: {action_mask.shape}")
    print(f"Valid actions: {action_mask.sum().item()} positions")
    
    # Performance test
    print("\n=== Performance Testing ===")
    
    # Add many containers
    start_time = time.time()
    added_count = 0
    for i in range(100):
        test_container = create_test_container(f"PERF{i:03d}", "20ft", GOODS_REGULAR)
        positions = yard.find_placement_positions(test_container, near_bay=7, max_distance=5)
        if positions:
            row, bay, tier, start_split = positions[0]
            if yard.add_container(test_container, row, bay, tier, start_split):
                added_count += 1
    
    elapsed = time.time() - start_time
    print(f"Added {added_count} containers in {elapsed:.3f}s")
    print(f"Average time per container: {(elapsed / added_count) * 1000:.2f}ms")
    
    # Test move finding with many containers
    start_time = time.time()
    all_moves = yard.get_yard_moves(max_distance=5)
    elapsed = time.time() - start_time
    print(f"Found moves for {len(all_moves)} containers in {elapsed:.3f}s")
    
    # Memory usage
    print("\n=== Memory Usage ===")
    import sys
    
    tensor_size = full_tensor.element_size() * full_tensor.nelement()
    print(f"Full tensor memory: {tensor_size / 1024 / 1024:.2f} MB")
    
    yard_size = sys.getsizeof(yard.occupancy) + sys.getsizeof(yard.containers)
    print(f"Yard data structures: {yard_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    test_yard_operations()