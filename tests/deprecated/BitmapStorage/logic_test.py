
def main():
    """Test the bitmap-based storage yard implementation with Container class from repository."""
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
    from simulation.deprecated_components.BitmapYard import BitmapStorageYard 
    
    # Determine compute device - use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("\n===== Testing BitmapStorageYard Implementation with Repository Containers =====")
    
    # 1. Create a bitmap storage yard
    yard = BitmapStorageYard(
        num_rows=5,            # 6 rows (A-F)
        num_bays=58,           # 40 bays per row
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
    
    print(f"Created bitmap storage yard with {yard.num_rows} rows and {yard.num_bays} bays")
    
    # 2. Create test containers using ContainerFactory
    print("\n----- Creating test containers with ContainerFactory -----")
    
    # Current date for container attributes
    current_date = datetime.now()
    
    containers = [
        # Regular containers
        ContainerFactory.create_container("REG001", "TWEU", "Import", "Regular", weight=20000),
        ContainerFactory.create_container("REG002", "FEU", "Export", "Regular", weight=25000),
        ContainerFactory.create_container("REG003", "THEU", "Import", "Regular", weight=18000),
        
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
    
    # 3. Test adding containers to the yard
    print("\n----- Testing container placement -----")
    
    # Add containers to appropriate areas
    placements = [
        # Regular containers in regular areas
        ('D10', containers[0]),
        ('D11', containers[1]),
        ('D12', containers[2]),
        # Reefer containers in reefer areas
        ('A3', containers[3]),
        ('F38', containers[4]),
        # Dangerous goods in dangerous area
        ('C27', containers[5]),
        ('C28', containers[6]),
        # Special containers in special areas
        ('A18', containers[7]),  # Trailer
        ('B35', containers[8])   # Swap body
    ]
    
    for position, container in placements:
        success = yard.add_container(position, container)
        print(f"Adding {container.container_id} to {position}: {'Success' if success else 'Failed'}")
    
    # 4. Test stacking
    print("\n----- Testing stacking -----")
    stack_position = 'D15'
    stack_containers = [
        ContainerFactory.create_container("STACK001", "TWEU", "Import", "Regular", weight=24000),
        ContainerFactory.create_container("STACK002", "TWEU", "Import", "Regular", weight=20000),
        ContainerFactory.create_container("STACK003", "TWEU", "Import", "Regular", weight=18000),
    ]
    
    for i, container in enumerate(stack_containers):
        tier = i + 1
        success = yard.add_container(stack_position, container, tier)
        print(f"Adding {container.container_id} to {stack_position} tier {tier}: {'Success' if success else 'Failed'}")
    
    # 5. Test invalid placements
    print("\n----- Testing invalid placements -----")
    
    # Try to add a reefer container to a non-reefer area
    reefer_container = ContainerFactory.create_container("INVALID01", "TWEU", "Import", "Reefer")
    success = yard.add_container('D20', reefer_container)
    print(f"Adding reefer container to non-reefer area: {'Success' if success else 'Failed (expected)'}")
    
    # Try to add a trailer outside trailer area
    trailer_container = ContainerFactory.create_container("INVALID02", "Trailer", "Export", "Regular")
    success = yard.add_container('E30', trailer_container)
    print(f"Adding trailer outside trailer area: {'Success' if success else 'Failed (expected)'}")
    
    # Try to stack on a trailer
    regular_container = ContainerFactory.create_container("INVALID03", "TWEU", "Import", "Regular")
    success = yard.add_container('A18', regular_container, tier=2)
    print(f"Adding container on top of trailer: {'Success' if success else 'Failed (expected)'}")
    
    # 6. Visualize yard state
    print("\n----- Visualizing yard state -----")
    
    # Get the combined occupied bitmap (combined from all tiers)
    print("Visualizing all occupied positions...")
    yard.visualize_bitmap(yard.occupied_bitmap, "All Occupied Positions")
    
    # 7. Test proximity calculation
    print("\n----- Testing proximity calculation -----")
    
    test_positions = ['D10', 'A3', 'C27', 'A18', stack_position]
    for position in test_positions:
        container, tier = yard.get_top_container(position)
        if container:
            print(f"\nCalculating proximity for {container.container_id} at {position}:")
            
            # Calculate proximity mask for different distances
            for n in [3, 5, 10]:
                # Visualize the proximity mask
                yard.visualize_proximity(position, n)
                
                # Calculate valid moves
                start_time = time.time()
                valid_moves = yard.calc_possible_moves(position, n)
                calc_time = (time.time() - start_time) * 1000  # ms
                
                print(f"Valid moves within {n} bays ({len(valid_moves)}): {valid_moves}")
                print(f"Calculated in {calc_time:.3f} ms")
    
    # 8. Test batch computation
    print("\n----- Batch computation with all containers -----")
    
    # Create a large batch of random containers for testing
    batch_size = 100
    batch_containers = []
    
    for i in range(batch_size):
        container_id = f"BATCH{i:03d}"
        # Create a mix of container types
        container_type = ["TWEU", "FEU", "THEU", "Trailer", "Swap Body"][i % 5]
        goods_type = ["Regular", "Reefer", "Dangerous"][i % 3]
        
        # Create container with ContainerFactory
        container = ContainerFactory.create_container(
            container_id=container_id,
            container_type=container_type,
            goods_type=goods_type
        )
        
        batch_containers.append(container)
    
    # Test find_all_moves performance
    start_time = time.time()
    all_moves = yard.find_all_moves()
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Find all moves for {len(all_moves)} containers: {total_time:.6f} seconds")
    
    # Print summary
    move_count = sum(len(moves) for moves in all_moves.values())
    print(f"Found {move_count} possible moves for {len(all_moves)} containers")
    print(f"Average of {move_count / len(all_moves):.2f} moves per container")
    
    # 9. Profile performance with different proximity ranges
    print("\n----- Performance testing with different proximity ranges -----")
    
    n_values = [3, 5, 10, 20]
    times = []
    
    for n in n_values:
        start_time = time.time()
        
        # Calculate valid moves for all test positions
        for position in test_positions:
            yard.calc_possible_moves(position, n)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / len(test_positions) * 1000  # ms
        times.append(avg_time)
        
        print(f"N={n}: Average {avg_time:.3f} ms per position")
    
    # Visualize performance
    plt.figure(figsize=(10, 5))
    plt.plot(n_values, times, marker='o')
    plt.title('Proximity Calculation Performance')
    plt.xlabel('Proximity Range (n)')
    plt.ylabel('Average Time (ms)')
    plt.grid(True)
    plt.show()
    
    # 10. Print final stats
    print("\n----- Final Yard State -----")
    print(yard)
    
    container_count = sum(len(tiers) for position, tiers in yard.container_registry.items())
    print(f"Total containers: {container_count}")
    print(f"Containers per row:")
    
    for row in yard.row_names:
        count = 0
        for position, tiers in yard.container_registry.items():
            if position[0] == row:
                count += len(tiers)
        print(f"  Row {row}: {count} containers")

if __name__ == "__main__":
    main()