def main():
    """Demonstrate 3D visualization of the bitmap-based storage yard."""
    import sys
    import os
    import matplotlib.pyplot as plt
    from datetime import datetime
    import random
    import torch
    
    # Ensure the repository root is in the path for imports
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    
    # Import the Container class and ContainerFactory
    from simulation.terminal_components.Container import Container, ContainerFactory
    from simulation.terminal_components.BitmapYard import BitmapStorageYard

    # Determine compute device - use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("\n===== 3D Container Yard Visualization Demo =====")
    
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
    
    # Create various container types for the demonstration
    containers = []
    
    # Regular containers of different types
    containers.extend([
        ContainerFactory.create_container(f"REG{i:03d}", "TWEU", "Import", "Regular")
        for i in range(10)
    ])
    
    containers.extend([
        ContainerFactory.create_container(f"FEU{i:03d}", "FEU", "Export", "Regular")
        for i in range(8)
    ])
    
    # Reefer containers
    containers.extend([
        ContainerFactory.create_container(f"REF{i:03d}", "TWEU", "Import", "Reefer")
        for i in range(5)
    ])
    
    # Dangerous goods
    containers.extend([
        ContainerFactory.create_container(f"DG{i:03d}", "FEU", "Import", "Dangerous")
        for i in range(5)
    ])
    
    # Special types
    containers.extend([
        ContainerFactory.create_container(f"TRL{i:03d}", "Trailer", "Export", "Regular")
        for i in range(3)
    ])
    
    containers.extend([
        ContainerFactory.create_container(f"SB{i:03d}", "Swap Body", "Export", "Regular") 
        for i in range(3)
    ])
    
    print(f"Created {len(containers)} containers for demonstration")
    
    # Place containers in yard, creating multiple stacks
    # 1. First, place some single containers
    placements = [
        # Regular containers in regular areas
        ('D10', containers[0]),
        ('D11', containers[1]),
        ('D12', containers[2]),
        ('D13', containers[10]),
        ('D14', containers[11]),
        
        # Reefer containers in reefer areas
        ('A3', containers[20]),
        ('F38', containers[21]),
        
        # Dangerous goods in dangerous area
        ('C27', containers[25]),
        ('C28', containers[26]),
        
        # Special containers in special areas
        ('A18', containers[30]),  # Trailer
        ('B35', containers[33])   # Swap body
    ]
    
    for position, container in placements:
        success = yard.add_container(position, container)
        print(f"Adding {container.container_id} to {position}: {'Success' if success else 'Failed'}")
    
    # 2. Create some stacks to demonstrate tier visualization
    # Stack 1: Three regular containers
    yard.add_container('D15', containers[3], tier=1)
    yard.add_container('D15', containers[4], tier=2)
    yard.add_container('D15', containers[5], tier=3)
    print(f"Created stack at D15 with 3 containers (tiers 1-3)")
    
    # Stack 2: Two FEU containers
    yard.add_container('D16', containers[12], tier=1)
    yard.add_container('D16', containers[13], tier=2)
    print(f"Created stack at D16 with 2 containers (tiers 1-2)")
    
    # Stack 3: Three reefer containers in reefer area
    yard.add_container('A4', containers[22], tier=1)
    yard.add_container('A4', containers[23], tier=2)
    yard.add_container('A4', containers[24], tier=3)
    print(f"Created stack at A4 with 3 reefer containers (tiers 1-3)")
    
    # Stack 4: Two dangerous goods containers
    yard.add_container('C29', containers[27], tier=1)
    yard.add_container('C29', containers[28], tier=2)
    print(f"Created stack at C29 with 2 dangerous goods containers (tiers 1-2)")
    
    # 3. Visualize the 2D overall yard state first (for comparison)
    print("\nDisplaying 2D visualization of all occupied positions...")
    yard.visualize_bitmap(yard.occupied_bitmap, "All Occupied Positions")
    
    # 4. Create 3D visualization
    print("\nGenerating 3D visualization of the container yard...")
    fig, ax = yard.visualize_3d(show_container_types=True, figsize=(15, 10))
    plt.savefig('container_yard_3d.png', dpi=300, bbox_inches='tight')
    
    # 5. Show from different angles
    print("\nShowing yard from different angles...")
    # Front view
    ax.view_init(elev=20, azim=240)
    plt.savefig('container_yard_3d_front.png', dpi=300, bbox_inches='tight')
    
    # Side view
    ax.view_init(elev=20, azim=150)
    plt.savefig('container_yard_3d_side.png', dpi=300, bbox_inches='tight')
    
    # Top view
    ax.view_init(elev=60, azim=210)
    plt.savefig('container_yard_3d_top.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print("\n3D visualization demo complete!")
    print("Visualization saved as container_yard_3d.png")

if __name__ == "__main__":
    main()