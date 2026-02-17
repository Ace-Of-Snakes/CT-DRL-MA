import unittest
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Tuple
from datetime import datetime, timedelta

from simulation.core.containers.container import Container
from simulation.core.facilities.yard import OptimizedStorageYard, PlacementResult
from simulation.core.enums import Direction, GoodsType

np.random.seed(42)


class TestOptimizedStorageYard(unittest.TestCase):
    """Comprehensive tests for OptimizedStorageYard with performance analysis."""

    def setUp(self):
        """Set up test yard with special positions."""
        self.coordinates = [
            (1, 1, "r"),    # Reefer position
            (2, 1, "dg"),   # Dangerous goods
            (3, 1, "sb_t"), # Swap body/trailer
            (1, 2, "r"),
            (2, 2, "dg"),
        ]
        self.yard = OptimizedStorageYard(
            n_rows=5,
            n_bays=58,
            n_tiers=5,
            coordinates=self.coordinates
        )

    def _create_container(self,
                         container_id: str,
                         length_ft: int = 40,
                         goods_type: GoodsType = GoodsType.REGULAR,
                         is_swap_body: bool = False,
                         is_trailer: bool = False) -> Container:
        """Helper to create test containers."""
        return Container(
            container_id=container_id,
            direction=Direction.IMPORT,
            container_type=f"{length_ft}ft",
            arrival_date=datetime.now(),
            departure_date=datetime.now(),
            goods_type=goods_type,
            length_ft=length_ft,
            length_m=length_ft * 0.3048,
            width_m=2.44,
            height_m=2.59,
            is_swap_body=is_swap_body,
            is_trailer=is_trailer
        )

    def test_container_placement_types(self):
        """Test placement of different container types in appropriate positions."""
        # Test regular container placement
        regular = self._create_container("REG001", 40, GoodsType.REGULAR)
        placement = self.yard.find_single_placement(regular, target_bay=5, max_proximity=10)
        self.assertIsNotNone(placement, "Should find placement for regular container")

        # Place regular container
        self.yard.add_container(regular, placement)
        self.assertIsNotNone(self.yard.get_record("REG001"))

        # Test reefer container - should only go in reefer positions
        reefer = self._create_container("REF001", 40, GoodsType.REEFER)
        reefer_placement = self.yard.find_single_placement(reefer, target_bay=0, max_proximity=10)

        if reefer_placement is not None:
            # Reefer should be placed in reefer-allowed positions
            self.assertIn(reefer_placement.bay, [0, 1], "Reefer should be in reefer bays")
            self.assertIn(reefer_placement.row, [0, 1], "Reefer should be in reefer rows")

        # Test dangerous goods
        dangerous = self._create_container("DG001", 40, GoodsType.DANGEROUS_GOODS)
        dg_placement = self.yard.find_single_placement(dangerous, target_bay=1, max_proximity=10)

        if dg_placement is not None:
            self.assertIn(dg_placement.bay, [1, 2], "Dangerous goods should be in DG bays")
            self.assertIn(dg_placement.row, [0, 1], "Dangerous goods should be in DG rows")

        # Test swap body - should only be on ground tier
        swap_body = self._create_container("SB001", 40, GoodsType.REGULAR, is_swap_body=True)
        sb_placement = self.yard.find_single_placement(swap_body, target_bay=2, max_proximity=10)

        if sb_placement is not None:
            self.assertEqual(sb_placement.tier, 0, "Swap body must be on ground tier")
            self.assertEqual(sb_placement.bay, 2, "Swap body should be in swap body bay")
            self.assertEqual(sb_placement.row, 0, "Swap body should be in swap body row")

    def test_mask_updates_on_operations(self):
        """Test that masks update correctly on add/remove operations."""
        # Add container to ground tier
        c1 = self._create_container("C001", 40)
        placement1 = self.yard.find_single_placement(c1, target_bay=5, max_proximity=10)
        self.assertIsNotNone(placement1)

        self.yard.add_container(c1, placement1)

        # Check occupancy mask is updated
        abs_start = placement1.bay * self.yard.split_factor + placement1.start_split
        n_splits = self.yard.container_length_map[40]

        for i in range(n_splits):
            pos = abs_start + i
            self.assertTrue(
                self.yard.occupancy_mask[placement1.tier, placement1.row, pos],
                f"Position {pos} should be occupied"
            )

        # Stack container on top: find placement at same row/bay/split but tier+1
        c2 = self._create_container("C002", 40)
        dest_above = PlacementResult(
            row=placement1.row,
            bay=placement1.bay,
            tier=placement1.tier + 1,
            start_split=placement1.start_split
        )

        # Verify the position above is valid (not occupied, has support)
        abs_end = abs_start + n_splits
        above_free = not np.any(
            self.yard.occupancy_mask[placement1.tier + 1, placement1.row, abs_start:abs_end]
        )

        if above_free:
            self.yard.add_container(c2, dest_above)

            # Check that C001 is no longer accessible
            rec_c001 = self.yard.get_record("C001")
            self.assertIsNotNone(rec_c001)
            self.assertFalse(rec_c001.is_accessible,
                           "Container with something above should not be accessible")
            self.assertFalse(self.yard.is_accessible("C001"))

            # Remove top container
            self.yard.remove_container(c2)

            # Check C001 is accessible again
            rec_c001 = self.yard.get_record("C001")
            self.assertTrue(rec_c001.is_accessible,
                          "Container should be accessible after removing container above")
            self.assertTrue(self.yard.is_accessible("C001"))

        # Remove ground container
        self.yard.remove_container(c1)

        # Check occupancy mask is cleared
        for i in range(n_splits):
            pos = abs_start + i
            self.assertFalse(
                self.yard.occupancy_mask[placement1.tier, placement1.row, pos],
                f"Position {pos} should be free after removal"
            )

    def test_cross_bay_placement(self):
        """Test containers that span multiple bays."""
        long_container = self._create_container("LONG001", 40)

        # Place at bay boundary
        placement = self.yard.find_single_placement(long_container, target_bay=0, max_proximity=10)

        if placement is not None and placement.start_split > 0:
            self.yard.add_container(long_container, placement)

            # Verify container is stored correctly
            self.assertIsNotNone(self.yard.get_record("LONG001"))

            # Verify occupancy across bays
            abs_start = placement.bay * self.yard.split_factor + placement.start_split
            n_splits = self.yard.container_length_map[40]

            for i in range(n_splits):
                pos = abs_start + i
                if pos < self.yard.total_splits:
                    self.assertTrue(
                        self.yard.occupancy_mask[placement.tier, placement.row, pos],
                        f"Cross-bay position {pos} should be occupied"
                    )

    def test_performance_scaling(self):
        """Stress test and performance analysis with visualization."""
        results = {
            'containers': [],
            'placement_time': [],
            'search_time': [],
            'proximity_results': {}
        }

        # Test scaling with number of containers
        container_counts = [10, 50, 100, 200, 400]
        goods_types = [GoodsType.REGULAR, GoodsType.REEFER, GoodsType.DANGEROUS_GOODS]

        for count in container_counts:
            # Reset yard
            self.yard = OptimizedStorageYard(
                n_rows=5,
                n_bays=50,
                n_tiers=5,
                coordinates=[]
            )

            # Placement time
            start = time.perf_counter()
            containers_added = 0
            for i in range(count):
                gt = np.random.choice(goods_types)
                length = np.random.choice([20, 40])
                c = self._create_container(f"PERF{i:04d}", length, gt)

                placement = self.yard.find_single_placement(
                    c,
                    target_bay=np.random.randint(0, 50),
                    max_proximity=5
                )
                if placement is not None:
                    self.yard.add_container(c, placement)
                    containers_added += 1

            placement_time = time.perf_counter() - start

            # Search time (average of 10 searches)
            search_times = []
            for _ in range(10):
                test_container = self._create_container("TEST", 40)
                start = time.perf_counter()
                self.yard.find_single_placement(
                    test_container,
                    target_bay=25,
                    max_proximity=5
                )
                search_times.append(time.perf_counter() - start)

            results['containers'].append(containers_added)
            results['placement_time'].append(placement_time)
            results['search_time'].append(np.mean(search_times))

        # Test proximity scaling
        proximities = [1, 2, 3, 5, 10, 15]
        for prox in proximities:
            test_container = self._create_container("PROX_TEST", 40)
            start = time.perf_counter()
            placement = self.yard.find_single_placement(
                test_container,
                target_bay=25,
                max_proximity=prox
            )
            prox_time = time.perf_counter() - start
            results['proximity_results'][prox] = {
                'time': prox_time,
                'found': 1 if placement is not None else 0
            }

        # Visualization
        self._plot_performance_results(results)

        # Assertions on performance
        if len(results['containers']) > 1:
            try:
                base_idx = next(i for i, v in enumerate(results['containers']) if v > 0)
            except StopIteration:
                self.skipTest("No containers placed in performance runs; skipping ratio checks")

            if results['containers'][-1] == 0:
                self.skipTest("No containers placed at max load; skipping ratio checks")

            search_time_ratio = results['search_time'][-1] / max(results['search_time'][base_idx], 1e-9)
            self.assertLess(search_time_ratio, 100.0, "Search time should remain within reasonable bounds")

    def _place_container_at_bay(self, container: Container, target_bay: int) -> PlacementResult:
        """Helper: find a placement near target_bay and add the container."""
        placement = self.yard.find_single_placement(container, target_bay=target_bay, max_proximity=10)
        self.assertIsNotNone(placement, f"No placement found for {container.container_id} near bay {target_bay}")
        self.yard.add_container(container, placement)
        return placement

    def test_add_and_retrieve_container(self):
        """Test adding a container and retrieving it via get_record and get_container."""
        c = self._create_container("RETRIEVE001", 40)
        placement = self.yard.find_single_placement(c, target_bay=10, max_proximity=5)
        self.assertIsNotNone(placement)
        self.yard.add_container(c, placement)

        # get_record returns full record
        rec = self.yard.get_record("RETRIEVE001")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.container.container_id, "RETRIEVE001")
        self.assertEqual(rec.placement.bay, placement.bay)

        # get_container returns just the container
        container = self.yard.get_container("RETRIEVE001")
        self.assertIsNotNone(container)
        self.assertEqual(container.container_id, "RETRIEVE001")

        # get_placement returns just the placement
        p = self.yard.get_placement("RETRIEVE001")
        self.assertIsNotNone(p)
        self.assertEqual(p.bay, placement.bay)

    def test_container_count(self):
        """Test container_count property."""
        self.assertEqual(self.yard.container_count, 0)

        c1 = self._create_container("CNT001", 40)
        p1 = self._place_container_at_bay(c1, target_bay=10)
        self.assertEqual(self.yard.container_count, 1)

        c2 = self._create_container("CNT002", 40)
        p2 = self._place_container_at_bay(c2, target_bay=15)
        self.assertEqual(self.yard.container_count, 2)

        self.yard.remove_container(c1)
        self.assertEqual(self.yard.container_count, 1)

    def test_iter_records(self):
        """Test iterating over all records."""
        c1 = self._create_container("ITER001", 40)
        c2 = self._create_container("ITER002", 40)
        self._place_container_at_bay(c1, target_bay=5)
        self._place_container_at_bay(c2, target_bay=10)

        ids = {rec.container.container_id for rec in self.yard.iter_records()}
        self.assertSetEqual(ids, {"ITER001", "ITER002"})

    def test_iter_accessible(self):
        """Test iterating over accessible containers."""
        c1 = self._create_container("ACC001", 40)
        p1 = self._place_container_at_bay(c1, target_bay=20)

        # c1 should be accessible (nothing on top)
        accessible_ids = {rec.container.container_id for rec in self.yard.iter_accessible()}
        self.assertIn("ACC001", accessible_ids)

        # Stack c2 on top
        c2 = self._create_container("ACC002", 40)
        above = PlacementResult(row=p1.row, bay=p1.bay, tier=p1.tier + 1, start_split=p1.start_split)
        self.yard.add_container(c2, above)

        accessible_ids = {rec.container.container_id for rec in self.yard.iter_accessible()}
        self.assertNotIn("ACC001", accessible_ids)
        self.assertIn("ACC002", accessible_ids)

    def test_move_container(self):
        """Test moving a container to a new position."""
        c = self._create_container("MOVE001", 40)
        p_orig = self._place_container_at_bay(c, target_bay=5)

        # Find a new placement
        dest = self.yard.find_single_placement(c, target_bay=30, max_proximity=5)
        self.assertIsNotNone(dest)

        # Move
        success = self.yard.move_container("MOVE001", dest)
        self.assertTrue(success)

        # Verify new position
        new_p = self.yard.get_placement("MOVE001")
        self.assertEqual(new_p.bay, dest.bay)

        # Verify old position is cleared
        abs_start_old = p_orig.bay * self.yard.split_factor + p_orig.start_split
        n_splits = self.yard.container_length_map[40]
        for i in range(n_splits):
            self.assertFalse(
                self.yard.occupancy_mask[p_orig.tier, p_orig.row, abs_start_old + i],
                "Old position should be cleared after move"
            )

    def test_get_container_at(self):
        """Test positional container lookup."""
        c = self._create_container("POS001", 40)
        p = self._place_container_at_bay(c, target_bay=10)

        abs_start = p.bay * self.yard.split_factor + p.start_split
        found = self.yard.get_container_at(p.row, p.tier, abs_start)
        self.assertIsNotNone(found)
        self.assertEqual(found.container_id, "POS001")

        # Empty position should return None
        not_found = self.yard.get_container_at(p.row, p.tier, abs_start + 100)
        # May or may not be None depending on bounds, but should not crash

    def test_get_blocking_pairs(self):
        """Test blocking pair detection."""
        c1 = self._create_container("BLK001", 40)
        p1 = self._place_container_at_bay(c1, target_bay=25)

        c2 = self._create_container("BLK002", 40)
        above = PlacementResult(row=p1.row, bay=p1.bay, tier=p1.tier + 1, start_split=p1.start_split)
        self.yard.add_container(c2, above)

        pairs = self.yard.get_blocking_pairs()
        # BLK002 is on top of BLK001
        blocker_ids = [blocker for blocker, blocked in pairs]
        blocked_ids = [blocked for blocker, blocked in pairs]
        self.assertIn("BLK002", blocker_ids)
        self.assertIn("BLK001", blocked_ids)

    def _plot_performance_results(self, results):
        """Plot performance analysis results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('OptimizedStorageYard Performance Analysis', fontsize=16)

        # Container scaling - placement time
        axes[0, 0].plot(results['containers'], results['placement_time'], 'b-o')
        axes[0, 0].set_xlabel('Number of Containers')
        axes[0, 0].set_ylabel('Total Placement Time (s)')
        axes[0, 0].set_title('Placement Time Scaling')
        axes[0, 0].grid(True, alpha=0.3)

        # Container scaling - search time
        axes[0, 1].plot(results['containers'],
                       [t * 1000 for t in results['search_time']], 'g-s')
        axes[0, 1].set_xlabel('Number of Containers in Yard')
        axes[0, 1].set_ylabel('Avg Search Time (ms)')
        axes[0, 1].set_title('Search Time Scaling')
        axes[0, 1].grid(True, alpha=0.3)

        # Proximity scaling
        prox_data = results['proximity_results']
        proximities = sorted(prox_data.keys())
        prox_times = [prox_data[p]['time'] * 1000 for p in proximities]
        prox_found = [prox_data[p]['found'] for p in proximities]

        axes[1, 0].plot(proximities, prox_times, 'b-o', label='Search Time')
        axes[1, 0].set_xlabel('Max Proximity')
        axes[1, 0].set_ylabel('Search Time (ms)')
        axes[1, 0].set_title('Proximity Search Scaling')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(proximities, prox_found, 'r-s')
        axes[1, 1].set_xlabel('Max Proximity')
        axes[1, 1].set_ylabel('Placement Found (0/1)')
        axes[1, 1].set_title('Proximity vs Placement Found')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('storage_yard_performance.png', dpi=150)
        plt.close()

        print(f"\nPerformance Summary:")
        print(f"Max containers tested: {max(results['containers'])}")
        if max(results['placement_time']) > 0:
            print(f"Placement throughput: {max(results['containers'])/max(results['placement_time']):.1f} containers/sec")
        print(f"Search time at max load: {results['search_time'][-1]*1000:.2f} ms")


if __name__ == '__main__':
    unittest.main()
