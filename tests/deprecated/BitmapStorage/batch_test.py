def main():
    """Test the bitmap-based storage yard implementation with focus on batching performance."""
    import sys
    import os
    import time
    import torch
    import matplotlib.pyplot as plt
    from datetime import datetime
    import random
    
    # Ensure the repository root is in the path for imports
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    
    # Import the Container class and ContainerFactory
    from simulation.terminal_components.Container import Container, ContainerFactory
    from simulation.deprecated_components.BitmapYard import BitmapStorageYard 
    
    # Determine compute device - use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("\n===== Testing BitmapStorageYard Batching Performance =====")
    
    # Create a bitmap storage yard
    yard = BitmapStorageYard(
        num_rows=5,
        num_bays=58,
        max_tier_height=5,
        special_areas={
            'reefer': [('A', 1, 1), ('B', 1, 1), ('C', 1, 1), ('D', 1, 1), ('E', 1, 1),
                       ('A', 58, 58), ('B', 58, 58), ('C', 58, 58), ('D', 58, 58), ('E', 58, 58)],
            'dangerous': [('A', 33, 36), ('B', 33, 36), ('C', 33, 36)],
            'trailer': [('E', 1, 58)],
            'swap_body': [('E', 1, 58)]
        },
        device=device
    )
    
    print(f"Created bitmap storage yard with {yard.num_rows} rows and {yard.num_bays} bays")
    
    # ----------------------------------------------------------------------------------
    # PERFORMANCE TEST 1: BATCH SIZE COMPARISON
    # ----------------------------------------------------------------------------------
    print("\n----- Test 1: Batch Size Performance Comparison -----")
    
    # Create a large number of containers for testing
    NUM_CONTAINERS = 16000
    container_types = ["TWEU", "FEU", "THEU", "Trailer", "Swap Body"]
    goods_types = ["Regular", "Reefer", "Dangerous"]
    containers = []
    
    for i in range(NUM_CONTAINERS):
        container_id = f"PERF{i:04d}"
        # Create a mix of container types
        container_type = container_types[i % len(container_types)]
        goods_type = goods_types[i % len(goods_types)]
        
        # Create container
        container = ContainerFactory.create_container(
            container_id=container_id,
            container_type=container_type,
            goods_type=goods_type
        )
        containers.append(container)
    
    # Place containers in valid positions
    positions_with_containers = []
    
    # Find valid positions for each container type
    for container in containers:
        # Place in valid position
        row = random.choice(yard.row_names)
        bay = random.randint(1, yard.num_bays)
        position = f"{row}{bay}"
        
        # Try to find a valid position
        attempts = 0
        while attempts < 10 and not yard.can_accept_container(position, container):
            row = random.choice(yard.row_names)
            bay = random.randint(1, yard.num_bays)
            position = f"{row}{bay}"
            attempts += 1
        
        if yard.can_accept_container(position, container):
            yard.add_container(position, container)
            positions_with_containers.append(position)
    
    print(f"Successfully placed {len(positions_with_containers)} containers in the yard")
    
    # ----------------------------------------------------------------------------------
    # Test sequential vs batch computation with different batch sizes
    # ----------------------------------------------------------------------------------
    
    # Define the batch sizes to test
    batch_sizes = [10, 50, 100, 200, 500, 1000, 2000, 4000, 8000, 16000]
    
    # Prepare results storage
    batch_times = []
    containers_per_second = []
    
    # Proximity range to use
    n = 1
    
    for batch_size in batch_sizes:
        # Split positions into batches
        position_batches = [positions_with_containers[i:i+batch_size] 
                           for i in range(0, len(positions_with_containers), batch_size)]
        
        # Start timing
        start_time = time.time()
        
        # Process each batch
        all_moves = {}
        for batch in position_batches:
            # Process the batch
            batch_results = yard.batch_calc_possible_moves(batch, n)
            all_moves.update(batch_results)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Calculate performance metrics
        cps = len(positions_with_containers) / elapsed_time
        
        print(f"Batch size {batch_size}: Processed {len(positions_with_containers)} containers in {elapsed_time:.4f} seconds ({cps:.2f} containers/sec)")
        
        batch_times.append(elapsed_time)
        containers_per_second.append(cps)
    
    # ----------------------------------------------------------------------------------
    # Test original sequential approach for comparison
    # ----------------------------------------------------------------------------------
    
    # Time original sequential approach (batch size = 1, one at a time)
    start_time = time.time()
    
    sequential_moves = {}
    for position in positions_with_containers:
        valid_moves = yard.calc_possible_moves(position, n)
        sequential_moves[position] = valid_moves
    
    end_time = time.time()
    sequential_time = end_time - start_time
    sequential_cps = len(positions_with_containers) / sequential_time
    
    print(f"Sequential: Processed {len(positions_with_containers)} containers in {sequential_time:.4f} seconds ({sequential_cps:.2f} containers/sec)")
    
    # ----------------------------------------------------------------------------------
    # Test full GPU-accelerated approach
    # ----------------------------------------------------------------------------------
    
    # Implement the GPU-accelerated find_all_moves method
    def find_all_moves_gpu(yard):
        """GPU-accelerated version of find_all_moves."""
        start_time = time.time()
        
        # Process all positions in one large batch
        all_moves = yard.batch_calc_possible_moves(positions_with_containers, n=5)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        cps = len(positions_with_containers) / elapsed_time
        
        print(f"GPU-accelerated: Processed {len(positions_with_containers)} containers in {elapsed_time:.4f} seconds ({cps:.2f} containers/sec)")
        
        return all_moves, elapsed_time, cps
    
    # Run the GPU-accelerated method
    gpu_moves, gpu_time, gpu_cps = find_all_moves_gpu(yard)
    
    # ----------------------------------------------------------------------------------
    # Visualize results with matplotlib
    # ----------------------------------------------------------------------------------
    
    # Plot 1: Processing time vs batch size
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(batch_sizes, batch_times, 'o-', label='Batched')
    plt.axhline(y=sequential_time, color='r', linestyle='--', label='Sequential')
    plt.axhline(y=gpu_time, color='g', linestyle='--', label='GPU Accelerated')
    plt.xlabel('Batch Size')
    plt.ylabel('Processing Time (seconds)')
    plt.title('Processing Time vs Batch Size')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Containers processed per second
    plt.subplot(1, 2, 2)
    plt.plot(batch_sizes, containers_per_second, 'o-', label='Batched')
    plt.axhline(y=sequential_cps, color='r', linestyle='--', label='Sequential')
    plt.axhline(y=gpu_cps, color='g', linestyle='--', label='GPU Accelerated')
    plt.xlabel('Batch Size')
    plt.ylabel('Containers per Second')
    plt.title('Processing Speed vs Batch Size')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('batch_performance.png', dpi=300)
    plt.show()
    
    # ----------------------------------------------------------------------------------
    # PERFORMANCE TEST 2: SCALING WITH NUMBER OF CONTAINERS
    # ----------------------------------------------------------------------------------
    print("\n----- Test 2: Scaling with Number of Containers -----")
    
    # Clear the yard
    yard.clear()
    
    # Test with different numbers of containers
    container_counts = [50, 100, 250, 500, 1000]
    scaling_times = []
    optimal_batch_size = 100  # Use the best batch size from Test 1
    
    for num_containers in container_counts:
        # Create containers
        test_containers = []
        for i in range(num_containers):
            container_id = f"SCALE{i:04d}"
            container_type = container_types[i % len(container_types)]
            goods_type = goods_types[i % len(goods_types)]
            
            container = ContainerFactory.create_container(
                container_id=container_id,
                container_type=container_type,
                goods_type=goods_type
            )
            test_containers.append(container)
        
        # Place containers in valid positions
        test_positions = []
        for container in test_containers:
            row = random.choice(yard.row_names)
            bay = random.randint(1, yard.num_bays)
            position = f"{row}{bay}"
            
            # Try to find a valid position
            attempts = 0
            while attempts < 10 and not yard.can_accept_container(position, container):
                row = random.choice(yard.row_names)
                bay = random.randint(1, yard.num_bays)
                position = f"{row}{bay}"
                attempts += 1
            
            if yard.can_accept_container(position, container):
                yard.add_container(position, container)
                test_positions.append(position)
        
        # Process with optimal batch size
        start_time = time.time()
        
        # Split positions into batches
        position_batches = [test_positions[i:i+optimal_batch_size] 
                           for i in range(0, len(test_positions), optimal_batch_size)]
        
        # Process each batch
        all_moves = {}
        for batch in position_batches:
            batch_results = yard.batch_calc_possible_moves(batch, n)
            all_moves.update(batch_results)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"Containers: {len(test_positions)}, Time: {elapsed_time:.4f} seconds, Speed: {len(test_positions)/elapsed_time:.2f} containers/sec")
        scaling_times.append(elapsed_time)
        
        # Clear the yard for next test
        yard.clear()
    
    # Plot scaling performance
    plt.figure(figsize=(10, 6))
    plt.plot(container_counts, scaling_times, 'o-')
    plt.xlabel('Number of Containers')
    plt.ylabel('Processing Time (seconds)')
    plt.title(f'Scaling Performance (Batch Size = {optimal_batch_size})')
    plt.grid(True)
    
    # Add containers per second as annotations
    for i, count in enumerate(container_counts):
        cps = count / scaling_times[i]
        plt.annotate(f"{cps:.1f} cont/sec", 
                    (count, scaling_times[i]),
                    textcoords="offset points",
                    xytext=(0,10), 
                    ha='center')
    
    plt.tight_layout()
    plt.savefig('scaling_performance.png', dpi=300)
    plt.show()
    
    print("\nPerformance testing complete! Results saved to batch_performance.png and scaling_performance.png")

if __name__ == "__main__":
    main()