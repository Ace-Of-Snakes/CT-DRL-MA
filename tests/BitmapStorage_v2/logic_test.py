def main():
    """Test the slot-tier bitmap yard implementation with Container class from repository."""
    import sys
    import os
    import time
    import matplotlib.pyplot as plt
    import torch
    from datetime import datetime
    
    # Ensure the repository root is in the path for imports
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    
    # Import the Container class and ContainerFactory from the repository
    from simulation.terminal_components.Container import Container, ContainerFactory
    from simulation.terminal_components.BitmapYard2 import SlotTierBitmapYard
    
    # Determine compute device - use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("\n===== Testing SlotTierBitmapYard Implementation with Repository Containers =====")
    
    # 1. Create a slot-tier bitmap storage yard
    yard = SlotTierBitmapYard(
        num_rows=5,            # 5 rows (A-E)
        num_bays=58,           # 58 bays per row
        max_tier_height=5,     # Maximum 5 containers high
        special_areas={
            'reefer': [('A', 1, 1), ('B', 1, 1), ('C', 1, 1), ('D', 1, 1), ('E', 1, 1),
                       ('A', 58, 58), ('B', 58, 58), ('C', 58, 58), ('D', 58, 58), ('E', 58, 58)],
            'dangerous': [('A', 28, 36), ('B', 28, 36), ('C', 28, 36)],
            'trailer': [('E', 1, 58)],
            'swap_body': [('E', 1, 58)]
        },
        device=device
    )
    
    print(f"Created slot-tier bitmap yard with {yard.num_rows} rows and {yard.num_bays} bays")
    print(f"Total positions: {yard.total_slots} positions ({yard.num_rows} rows × {yard.num_bays} bays × {yard.slots_per_bay} slots × {yard.max_tier_height} tiers)")
    
    # 2. Create test containers using ContainerFactory
    print("\n----- Creating test containers with ContainerFactory -----")
    
    # Current date for container attributes
    current_date = datetime.now()
    
    containers = [
        # Regular containers of different sizes
        ContainerFactory.create_container("TWEU001", "TWEU", "Import", "Regular", weight=20000),
        ContainerFactory.create_container("THEU001", "THEU", "Import", "Regular", weight=18000),
        ContainerFactory.create_container("FEU001", "FEU", "Export", "Regular", weight=25000),
        ContainerFactory.create_container("FFEU001", "FFEU", "Export", "Regular", weight=30000),
        
        # Reefer containers
        ContainerFactory.create_container("REEF001", "TWEU", "Import", "Reefer", weight=22000),
        ContainerFactory.create_container("REEF002", "FEU", "Export", "Reefer", weight=24000),
        
        # Dangerous goods
        ContainerFactory.create_container("DG001", "TWEU", "Import", "Dangerous", weight=19000),
        ContainerFactory.create_container("DG002", "FEU", "Export", "Dangerous", weight=27000),
        
        # Special types
        ContainerFactory.create_container("TRL001", "Trailer", "Export", "Regular", weight=15000),
        ContainerFactory.create_container("SB001", "Swap Body", "Export", "Regular", weight=12000),
    ]
    
    # Print container details
    for i, container in enumerate(containers):
        print(f"{i+1}. Created {container.container_id}: {container.container_type}, {container.goods_type}, " +
              f"Stackable: {container.is_stackable}, Compatibility: {container.stack_compatibility}")
    
    # 3. Test adding containers to the yard according to their size requirements
    print("\n----- Testing container placement with different container sizes -----")
    
    # Add containers to appropriate areas with different container types
    placements = [
        # TWEU (20ft) - requires 2 consecutive slots (starting at slot 1 or 3)
        ('D10.1-T1', containers[0]),  # TWEU in slots 1-2
        ('D11.3-T1', containers[0]),  # TWEU in slots 3-4
        
        # THEU (30ft) - requires 3 consecutive slots (starting at slot 1)
        ('D12.1-T1', containers[1]),  # THEU in slots 1-3
        
        # FEU (40ft) - requires all 4 slots in a bay (full bay)
        ('D13.1-T1', containers[2]),  # FEU in slots 1-4
        
        # FFEU (45ft) - requires 5 slots (4 in current bay + 1 in next bay)
        ('D14.1-T1', containers[3]),  # FFEU in bay 14 slots 1-4 + bay 15 slot 1
        
        # Reefer containers in reefer areas
        ('A1.1-T1', containers[4]),  # TWEU Reefer in slots 1-2
        ('A58.1-T1', containers[5]),  # FEU Reefer in slots 1-4
        
        # Dangerous goods in dangerous area
        ('A28.1-T1', containers[6]),  # TWEU Dangerous in slots 1-2
        ('B30.1-T1', containers[7]),  # FEU Dangerous in slots 1-4
        
        # Special containers in special areas
        ('E5.1-T1', containers[8]),   # Trailer in E row
        ('E10.1-T1', containers[9])   # Swap body in E row
    ]
    
    for position, container in placements:
        success = yard.add_container(position, container)
        print(f"Adding {container.container_id} ({container.container_type}) to {position}: {'Success' if success else 'Failed'}")
    
    # 4. Test stacking (vertical tier placement)
    print("\n----- Testing stacking (vertical tiers) -----")
    
    stack_position = 'D16.1-T1'
    stack_containers = [
        ContainerFactory.create_container("STACK001", "TWEU", "Import", "Regular", weight=24000),
        ContainerFactory.create_container("STACK002", "TWEU", "Import", "Regular", weight=20000),
        ContainerFactory.create_container("STACK003", "TWEU", "Import", "Regular", weight=18000),
    ]
    
    # Add first container to base tier
    success = yard.add_container(stack_position, stack_containers[0])
    print(f"Adding {stack_containers[0].container_id} to {stack_position}: {'Success' if success else 'Failed'}")
    
    # Add second container to tier 2
    tier2_pos = stack_position.replace('T1', 'T2')
    success = yard.add_container(tier2_pos, stack_containers[1])
    print(f"Adding {stack_containers[1].container_id} to {tier2_pos}: {'Success' if success else 'Failed'}")
    
    # Add third container to tier 3
    tier3_pos = stack_position.replace('T1', 'T3')
    success = yard.add_container(tier3_pos, stack_containers[2])
    print(f"Adding {stack_containers[2].container_id} to {tier3_pos}: {'Success' if success else 'Failed'}")
    
    # 5. Test invalid placements
    print("\n----- Testing invalid placements -----")
    
    # Try to add a container at slot 2 (not a valid start position for any container)
    invalid_pos1 = 'D20.2-T1'
    invalid_container1 = ContainerFactory.create_container("INVALID01", "TWEU", "Import", "Regular")
    success = yard.add_container(invalid_pos1, invalid_container1)
    print(f"Adding TWEU to invalid slot position {invalid_pos1}: {'Success' if success else 'Failed (expected)'}")
    
    # Try to add a reefer container to a non-reefer area
    invalid_pos2 = 'D20.1-T1'
    reefer_container = ContainerFactory.create_container("INVALID02", "TWEU", "Import", "Reefer")
    success = yard.add_container(invalid_pos2, reefer_container)
    print(f"Adding reefer container to non-reefer area {invalid_pos2}: {'Success' if success else 'Failed (expected)'}")
    
    # Try to add a trailer outside trailer area
    invalid_pos3 = 'A30.1-T1'
    trailer_container = ContainerFactory.create_container("INVALID03", "Trailer", "Export", "Regular")
    success = yard.add_container(invalid_pos3, trailer_container)
    print(f"Adding trailer outside trailer area {invalid_pos3}: {'Success' if success else 'Failed (expected)'}")
    
    # Try to stack on a trailer
    invalid_pos4 = 'E5.1-T2'  # Tier 2 on trailer position
    regular_container = ContainerFactory.create_container("INVALID04", "TWEU", "Import", "Regular")
    success = yard.add_container(invalid_pos4, regular_container)
    print(f"Adding container on top of trailer at {invalid_pos4}: {'Success' if success else 'Failed (expected)'}")
    
    # Try to add THEU starting at slot 3 (must start at slot 1)
    invalid_pos5 = 'D25.3-T1'
    theu_container = ContainerFactory.create_container("INVALID05", "THEU", "Import", "Regular")
    success = yard.add_container(invalid_pos5, theu_container)
    print(f"Adding THEU starting at slot 3 {invalid_pos5}: {'Success' if success else 'Failed (expected)'}")
    
    # 6. Test container lookup and retrieval
    print("\n----- Testing container lookup and retrieval -----")
    
    # Find a container by ID
    container_id = "TWEU001"
    position = yard.find_container(container_id)
    if position:
        container = yard.get_container(position)
        print(f"Found container {container_id} at position {position}")
        print(f"Container details: {container.container_type}, {container.goods_type}")
    else:
        print(f"Container {container_id} not found")
    
    # Get container at a specific position
    test_position = 'D13.1-T1'  # FEU container
    container = yard.get_container(test_position)
    if container:
        print(f"Container at {test_position}: {container.container_id}, {container.container_type}")
    else:
        print(f"No container at {test_position}")
    
    # 7. Test container removal
    print("\n----- Testing container removal -----")
    
    # Remove a container
    position_to_remove = 'D10.1-T1'
    removed_container = yard.remove_container(position_to_remove)
    
    if removed_container:
        print(f"Removed container {removed_container.container_id} from {position_to_remove}")
    else:
        print(f"No container removed from {position_to_remove}")
    
    # Verify the container is gone
    container = yard.get_container(position_to_remove)
    print(f"Container at {position_to_remove} after removal: {container}")
    
    # 8. Test position validation and special area checks
    print("\n----- Testing position validation and special areas -----")
    
    # Check if positions are in special areas
    test_areas = {
        'A1.1-T1': ['reefer'],
        'B30.1-T1': ['dangerous'],
        'E5.1-T1': ['trailer', 'swap_body'],
        'D15.1-T1': []
    }
    
    for pos, expected_areas in test_areas.items():
        areas_found = []
        
        for area_type in ['reefer', 'dangerous', 'trailer', 'swap_body']:
            if yard.is_position_in_special_area(pos, area_type):
                areas_found.append(area_type)
        
        if areas_found:
            print(f"Position {pos} is in special areas: {', '.join(areas_found)}")
        else:
            print(f"Position {pos} is not in any special area")
    
    # 9. Test proximity calculation
    print("\n----- Testing proximity calculation -----")
    
    test_positions = ['D13.1-T1', 'A1.1-T1', 'A28.1-T1', 'E5.1-T1', 'E10.1-T1']
    for position in test_positions:
        container = yard.get_container(position)
        if container:
            print(f"\nCalculating proximity for {container.container_id} at {position}:")
            
            # Calculate valid moves for different distances
            for n in [3, 5, 10]:
                start_time = time.time()
                valid_moves = yard.calc_possible_moves(position, n)
                calc_time = (time.time() - start_time) * 1000  # ms
                
                print(f"Valid moves within {n} bays ({len(valid_moves)} found): Sample - {valid_moves[:5] if valid_moves else 'None'}")
                print(f"Calculated in {calc_time:.3f} ms")
    
    # 10. Test batch computation
    print("\n----- Batch computation with proximity -----")
    
    # Get all positions with containers
    positions_with_containers = list(yard.container_registry.keys())
    
    # Test batch operation with all positions
    start_time = time.time()
    all_moves = yard.batch_calc_possible_moves(positions_with_containers, n=5)
    end_time = time.time()
    total_time = end_time - start_time
    # This should work
    new_ffeu = ContainerFactory.create_container("FFEU002", "FFEU", "Import", "Regular", weight=28000)
    success = yard.add_container('D14.1-T2', new_ffeu)
    print(f"Adding new FFEU container to D14.1-T2: {'Success' if success else 'Failed'}")

    num_positions = len(positions_with_containers)
    num_moves = sum(len(moves) for moves in all_moves.values())
    
    print(f"Batch calculated moves for {num_positions} positions in {total_time:.6f} seconds")
    print(f"Found {num_moves} possible moves total, average of {num_moves/num_positions:.1f} moves per position")
    print(f"Processing speed: {num_positions/total_time:.1f} positions/second")
    
    # 11. Test visualization
    print("\n----- Testing visualization -----")
    
    # Visualize the yard - both bitmap and 3D visualization
    try:
        # Bitmap visualization
        print("Visualizing occupied bitmap...")
        fig, ax = yard.visualize_bitmap(yard.occupied_bitmap, "Occupied Positions")
        plt.savefig('slottier_yard_bitmap.png', dpi=150)
        plt.close()
        
        # Detailed visualization showing tiers and slots
        print("Generating detailed tier-by-tier visualization...")
        fig, axes = yard.visualize_detailed_bitmap(show_tiers=True, figsize=(20, 10))
        plt.savefig('slottier_yard_detailed.png', dpi=150)
        plt.close()

        # Summary visualization
        print("Generating summary stack height visualization...")
        fig, ax = yard.visualize_detailed_bitmap(show_tiers=False, figsize=(15, 8))
        plt.savefig('slottier_yard_summary.png', dpi=150)
        plt.close()
        # 3D visualization
        print("Generating 3D visualization...")
        fig, ax = yard.visualize_3d(show_container_types=True, figsize=(15, 10))
        plt.savefig('slottier_yard_3d.png', dpi=150)
        plt.close()
        
        print("Visualizations saved as PNG files.")
    except Exception as e:
        print(f"Visualization error: {e}")
    
    # 12. Final summary
    print("\n----- Final Yard State -----")
    print(yard)
    
    container_count = len(yard.container_registry)
    print(f"Total containers: {container_count}")
    
    # Count containers by type
    container_types = {}
    for position, container in yard.container_registry.items():
        container_type = container.container_type
        if container_type in container_types:
            container_types[container_type] += 1
        else:
            container_types[container_type] = 1
    
    print("Containers by type:")
    for container_type, count in container_types.items():
        print(f"  {container_type}: {count} containers")
    
    print("\nSlotTierBitmapYard testing complete!")

if __name__ == "__main__":
    main()