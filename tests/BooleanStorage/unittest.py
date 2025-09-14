import unittest
import time
import numpy as np
from datetime import datetime
from typing import List, Tuple
from simulation.terminal_components.storage_units.Container import Container
from simulation.terminal_components.storage.BooleanStorage import BooleanStorageYard, PlacementResult
from simulation.terminal_components.storage.constants import CONTAINER_LENGTHS_FT


# ============= USAGE EXAMPLE =============
def usage_example():
    """Simple usage example demonstrating basic operations."""
    print("=== USAGE EXAMPLE ===\n")
    
    # Create a small yard
    yard = BooleanStorageYard(
        n_rows=3,
        n_bays=10,
        n_tiers=3,
        coordinates=[
            (1, 1, "r"),   # Bay 1, Row 1 for reefers
            (10, 1, "r"),  # Bay 10, Row 1 for reefers
            (5, 2, "dg"),  # Bay 5, Row 2 for dangerous goods
        ]
    )
    
    # Create containers
    container_20ft = Container(
        container_id="C001",
        direction="Import",
        container_type="20ft",
        arrival_date=datetime.now(),
        departure_date=datetime.now(),
        goods_type="Regular",
        length_ft=20
    )
    
    container_40ft = Container(
        container_id="C002",
        direction="Import",
        container_type="40ft",
        arrival_date=datetime.now(),
        departure_date=datetime.now(),
        goods_type="Reefer",
        length_ft=40
    )
    
    # Search for placement
    placements_20 = yard.search_placement_all_tiers(container_20ft, target_bay=3, max_proximity=7)
    print(f"Found {len(placements_20)} placements for 20ft container")
    if placements_20:
        best = placements_20[0]
        print(f"Best placement: Row {best.row}, Bay {best.bay}, Tier {best.tier}, Split {best.start_split}")
        print('other placements: ', placements_20)
        yard.add_container(container_20ft, best)
        print("Container added successfully\n")
    
    # Add reefer container
    placements_40 = yard.search_placement_all_tiers(container_40ft, target_bay=0, max_proximity=1)
    print(f"Found {len(placements_40)} placements for 40ft reefer")
    if placements_40:
        best = placements_40[0]
        print(f"Best placement: Row {best.row}, Bay {best.bay}, Tier {best.tier}")
        yard.add_container(container_40ft, best)
    
    # Find moveable containers
    moveable = yard.find_moveable_containers()
    print(f"\nMoveable containers: {list(moveable.keys())}")


# ============= UNIT TESTS =============
class TestBooleanStorageYard(unittest.TestCase):
    
    def setUp(self):
        """Set up test yard with standard configuration."""
        self.yard = BooleanStorageYard(
            n_rows=5,
            n_bays=15,
            n_tiers=4,
            coordinates=[
                (1, 1, "r"), (1, 2, "r"),  # Reefer positions
                (15, 1, "r"), (15, 2, "r"),
                (7, 3, "dg"), (8, 3, "dg"),  # Dangerous goods
                (1, 5, "sb_t"), (2, 5, "sb_t"),  # Swap bodies
            ]
        )
    
    def test_initialization(self):
        """Test yard initialization."""
        self.assertEqual(self.yard.n_rows, 5)
        self.assertEqual(self.yard.n_bays, 15)
        self.assertEqual(self.yard.n_tiers, 4)
        self.assertEqual(self.yard.split_factor, 20)
        
        # Check mask shapes
        self.assertEqual(self.yard.base_mask.shape, (4, 5, 15*20))  # 15 bays * 4 splits
        self.assertEqual(self.yard.reefer_mask.shape, (4, 5, 15*20))
    
    def test_20ft_container_placement(self):
        """Test placing 20ft containers (2 sub-bays)."""
        container = Container(
            container_id="TEST20",
            direction="Import",
            container_type="20ft",
            arrival_date=datetime.now(),
            departure_date=datetime.now(),
            goods_type="Regular",
            length_ft=20
        )
        
        placements = self.yard.search_placement_all_tiers(container, target_bay=5, max_proximity=5)
        
        # Should find placements
        self.assertGreater(len(placements), 0)
        
        # First placement should be ground tier
        first = placements[0]
        self.assertEqual(first.tier, 0)
        
        # Valid start positions for 20ft (2 splits) are 0 or 2
        self.assertIn(first.start_split, [0, 2])
        
        # Add container
        self.yard.add_container(container, first)
        
        # Verify container is placed
        self.assertIsNotNone(self.yard.containers[first.row, first.bay, first.tier, first.start_split])
    
    def test_40ft_container_placement(self):
        """Test placing 40ft containers (full bay)."""
        container = Container(
            container_id="TEST40",
            direction="Import",
            container_type="40ft",
            arrival_date=datetime.now(),
            departure_date=datetime.now(),
            goods_type="Regular",
            length_ft=40
        )
        
        placements = self.yard.search_placement_all_tiers(container, target_bay=7, max_proximity=5)
        
        self.assertGreater(len(placements), 0)
        
        first = placements[0]
        # 40ft takes full bay, so start_split must be 0
        self.assertEqual(first.start_split, 0)
        
        # Add and verify
        self.yard.add_container(container, first)
        
        # All 4 splits should have the same container
        for split in range(4):
            self.assertEqual(
                self.yard.containers[first.row, first.bay, first.tier, split],
                container
            )
    
    def test_45ft_cross_bay_placement(self):
        """Test placing 45ft containers that span 2 bays."""
        container = Container(
            container_id="TEST45",
            direction="Import",
            container_type="45ft",
            arrival_date=datetime.now(),
            departure_date=datetime.now(),
            goods_type="Regular",
            length_ft=45
        )
        
        placements = self.yard.search_placement_all_tiers(container, target_bay=5, max_proximity=5)
        
        if placements:  # 45ft requires 5 splits, spanning 2 bays
            first = placements[0]
            self.yard.add_container(container, first)
            
            # Verify spans 2 bays
            placed_bays = set()
            for row in range(self.yard.n_rows):
                for bay in range(self.yard.n_bays):
                    for split in range(4):
                        if self.yard.containers[row, bay, first.tier, split] == container:
                            placed_bays.add(bay)
            
            self.assertEqual(len(placed_bays), 2, "45ft container should span 2 bays")
    
    def test_reefer_placement(self):
        """Test reefer containers only go in reefer positions."""
        reefer = Container(
            container_id="REEF1",
            direction="Import",
            container_type="40ft",
            arrival_date=datetime.now(),
            departure_date=datetime.now(),
            goods_type="Reefer",
            length_ft=40
        )
        
        # Search near reefer area (bay 0)
        placements = self.yard.search_placement_all_tiers(reefer, target_bay=0, max_proximity=5)
        
        # Should find placements in reefer zones
        self.assertGreater(len(placements), 0)
        
        # Verify placement is in reefer area
        first = placements[0]
        self.assertIn(first.row, [0, 1])  # Reefer rows
        self.assertIn(first.bay, [0, 1])  # Near reefer bays
    
    def test_dangerous_goods_placement(self):
        """Test dangerous goods placement in designated areas."""
        dangerous = Container(
            container_id="DG001",
            direction="Import",
            container_type="40ft",
            arrival_date=datetime.now(),
            departure_date=datetime.now(),
            goods_type="DangerousGoods",
            length_ft=40
        )
        
        # Search near dangerous goods area
        placements = self.yard.search_placement_all_tiers(dangerous, target_bay=7, max_proximity=5)
        
        if placements:
            first = placements[0]
            self.assertEqual(first.row, 2)  # DG row
            self.assertIn(first.bay, [6, 7])  # DG bays
    
    def test_stacking_behavior(self):
        """Test that containers stack properly."""
        # Place ground container
        ground = Container(
            container_id="GROUND",
            direction="Import",
            container_type="40ft",
            arrival_date=datetime.now(),
            departure_date=datetime.now(),
            goods_type="Regular",
            length_ft=40
        )
        
        placements = self.yard.search_placement_all_tiers(ground, target_bay=5, max_proximity=0)
        ground_placement = next(p for p in placements if p.tier == 0)
        self.yard.add_container(ground, ground_placement)
        
        # Now place another container
        stacked = Container(
            container_id="STACKED",
            direction="Import",
            container_type="40ft",
            arrival_date=datetime.now(),
            departure_date=datetime.now(),
            goods_type="Regular",
            length_ft=40
        )
        
        placements2 = self.yard.search_placement_all_tiers(stacked, target_bay=5, max_proximity=0)
        
        # Should have tier 1 available now
        tier1_placements = [p for p in placements2 if p.tier == 1]
        self.assertGreater(len(tier1_placements), 0)
        
        # Verify same position is available in tier 1
        same_pos_tier1 = [p for p in tier1_placements 
                         if p.row == ground_placement.row and p.bay == ground_placement.bay]
        self.assertGreater(len(same_pos_tier1), 0)
    
    def test_container_removal(self):
        """Test removing containers."""
        container = Container(
            container_id="REMOVE",
            direction="Import",
            container_type="20ft",
            arrival_date=datetime.now(),
            departure_date=datetime.now(),
            goods_type="Regular",
            length_ft=20
        )
        
        placements = self.yard.search_placement_all_tiers(container, target_bay=5, max_proximity=1)
        placement = placements[0]
        
        # Add then remove
        self.yard.add_container(container, placement)
        removed = self.yard.remove_container(placement, container)
        
        self.assertEqual(removed.container_id, "REMOVE")
        
        # Verify position is empty
        self.assertIsNone(self.yard.containers[placement.row, placement.bay, placement.tier, placement.start_split])
    
    def test_moveable_containers(self):
        """Test finding moveable containers."""
        # Add some containers
        c1 = Container("MOVE1", "Import", "20ft", datetime.now(), datetime.now(), "Regular", length_ft=20)
        c2 = Container("MOVE2", "Import", "40ft", datetime.now(), datetime.now(), "Regular", length_ft=40)
        
        p1 = self.yard.search_placement_all_tiers(c1, 5, 2)[0]
        p2 = self.yard.search_placement_all_tiers(c2, 7, 2)[0]
        
        self.yard.add_container(c1, p1)
        self.yard.add_container(c2, p2)
        
        moveable = self.yard.find_moveable_containers()
        
        # Both should be moveable (on ground tier)
        self.assertIn("MOVE1", moveable)
        self.assertIn("MOVE2", moveable)
        
        # Each should have alternative positions
        self.assertGreater(len(moveable["MOVE1"]), 0)
        self.assertGreater(len(moveable["MOVE2"]), 0)


# ============= STRESS TEST =============
def stress_test():
    """Stress test with large yard configuration."""
    print("\n=== STRESS TEST ===")
    print("Creating large yard: 58 bays x 5 rows x 5 tiers")
    
    start_time = time.time()
    
    # Create large yard with realistic special positions
    coordinates = []
    # Reefer positions on ends
    for row in range(1, 6):
        coordinates.extend([(1, row, "r"), (2, row, "r"), (57, row, "r"), (58, row, "r")])
    
    # Dangerous goods in middle
    for bay in range(28, 31):
        coordinates.extend([(bay, 3, "dg")])
    
    # Swap bodies along one edge
    for bay in range(1, 59):
        coordinates.append((bay, 1, "sb_t"))
    
    yard = BooleanStorageYard(
        n_rows=5,
        n_bays=58,
        n_tiers=5,
        coordinates=coordinates
    )
    
    init_time = time.time() - start_time
    print(f"Initialization time: {init_time:.3f}s")
    
    # Create diverse containers
    containers = []
    for i, length in enumerate(CONTAINER_LENGTHS_FT * 10):  # 70 containers
        containers.append(Container(
            container_id=f"C{i:04d}",
            direction="Import",
            container_type=f"{length}ft",
            arrival_date=datetime.now(),
            departure_date=datetime.now(),
            goods_type=["Regular", "Reefer", "DangerousGoods"][i % 3],
            length_ft=length
        ))
    
    # Benchmark placement search
    search_times = []
    successful_placements = 0
    
    for container in containers:
        start = time.time()
        placements = yard.search_placement_all_tiers(container, target_bay=29, max_proximity=30)
        search_time = time.time() - start
        search_times.append(search_time)
        
        if placements:
            yard.add_container(container, placements[0])
            successful_placements += 1
    
    print(f"\nPlacement Results:")
    print(f"  Containers placed: {successful_placements}/{len(containers)}")
    print(f"  Avg search time: {np.mean(search_times)*1000:.2f}ms")
    print(f"  Max search time: {np.max(search_times)*1000:.2f}ms")
    print(f"  Min search time: {np.min(search_times)*1000:.2f}ms")
    
    # Benchmark move finding
    start = time.time()
    moveable = yard.find_moveable_containers(max_proximity=5)
    move_time = time.time() - start
    
    print(f"\nMove Finding:")
    print(f"  Moveable containers: {len(moveable)}")
    print(f"  Time to find all moves: {move_time:.3f}s")
    
    # Calculate yard utilization
    total_positions = 5 * 58 * 5 * 4  # rows * bays * tiers * splits
    used_positions = np.sum(yard.containers != None)
    utilization = (used_positions / total_positions) * 100
    
    print(f"\nYard Statistics:")
    print(f"  Total positions: {total_positions}")
    print(f"  Used positions: {used_positions}")
    print(f"  Utilization: {utilization:.1f}%")


if __name__ == "__main__":
    # Run usage example
    usage_example()
    
    # Run unit tests
    print("\n=== RUNNING UNIT TESTS ===")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run stress test
    stress_test()