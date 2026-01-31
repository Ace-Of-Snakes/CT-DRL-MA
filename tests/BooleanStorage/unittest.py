import unittest
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Tuple
from datetime import datetime
from simulation2.core.Container import Container
from simulation2.facilities.BooleanStorage import BooleanStorageYard, PlacementResult
from datetime import datetime, timedelta
np.random.seed(42)

class TestBooleanStorage(unittest.TestCase):
    """Comprehensive tests for BooleanStorageYard with performance analysis."""
    
    def setUp(self):
        """Set up test yard with special positions."""
        self.coordinates = [
            (1, 1, "r"),   # Reefer position
            (2, 1, "dg"),  # Dangerous goods
            (3, 1, "sb_t"), # Swap body/trailer
            (1, 2, "r"),
            (2, 2, "dg"),
        ]
        self.yard = BooleanStorageYard(
            n_rows=5, 
            n_bays=58, 
            n_tiers=5,
            coordinates=self.coordinates
        )
    
    def _create_container(self, 
                         container_id: str, 
                         length_ft: int = 40,
                         goods_type: str = "Regular",
                         is_swap_body: bool = False,
                         is_trailer: bool = False) -> Container:
        """Helper to create test containers."""
        return Container(
            container_id=container_id,
            direction="Import",
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
        regular = self._create_container("REG001", 40, "Regular")
        placements = self.yard.search_placement_all_tiers(regular, target_bay=5, max_proximity=10)
        self.assertGreater(len(placements), 0, "Should find placements for regular container")
        
        # Place regular container
        self.yard.add_container(regular, placements[0])
        self.assertIn("REG001", self.yard.containers)
        
        # Test reefer container - should only go in reefer positions
        reefer = self._create_container("REF001", 40, "Reefer")
        reefer_placements = self.yard.search_placement_all_tiers(reefer, target_bay=0, max_proximity=10)
        
        for p in reefer_placements[:5]:  # Check first few placements
            # Bay 0 and 1 have reefer positions in rows 0 and 1
            self.assertIn(p.bay, [0, 1], "Reefer should be in reefer bays")
            self.assertIn(p.row, [0, 1], "Reefer should be in reefer rows")
        
        # Test dangerous goods
        dangerous = self._create_container("DG001", 40, "DangerousGoods")
        dg_placements = self.yard.search_placement_all_tiers(dangerous, target_bay=1, max_proximity=10)
        
        for p in dg_placements[:5]:
            self.assertIn(p.bay, [1, 2], "Dangerous goods should be in DG bays")
            self.assertIn(p.row, [0, 1], "Dangerous goods should be in DG rows")
        
        # Test swap body - should only be on ground tier
        swap_body = self._create_container("SB001", 40, "Regular", is_swap_body=True)
        sb_placements = self.yard.search_placement_all_tiers(swap_body, target_bay=2, max_proximity=10)
        
        for p in sb_placements[:5]:
            self.assertEqual(p.tier, 0, "Swap body must be on ground tier")
            self.assertEqual(p.bay, 2, "Swap body should be in swap body bay")
            self.assertEqual(p.row, 0, "Swap body should be in swap body row")
    
    def test_mask_updates_on_operations(self):
        """Test that masks update correctly on add/remove operations."""
        # Add container to ground tier
        c1 = self._create_container("C001", 40)
        placements = self.yard.search_placement_all_tiers(c1, target_bay=5, max_proximity=10)
        self.assertGreater(len(placements), 0)
        
        placement1 = placements[0]
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
        
        # Try to place another container at same position - should not be possible
        c2 = self._create_container("C002", 40)
        placements2 = self.yard.search_placement_all_tiers(c2, target_bay=placement1.bay, max_proximity=0)
        
        # Filter to same row and tier
        same_spot = [p for p in placements2 
                    if p.row == placement1.row and p.tier == placement1.tier 
                    and p.bay == placement1.bay and p.start_split == placement1.start_split]
        self.assertEqual(len(same_spot), 0, "Should not be able to place at occupied position")
        
        # Stack container on top
        placements_above = [
            p for p in placements2
            if p.row == placement1.row
            and p.tier == placement1.tier + 1
            and p.bay == placement1.bay
            and p.start_split == placement1.start_split  # ensure aligned start
        ]
        self.assertGreater(len(placements_above), 0, "Should be able to stack on top with aligned start")
        
        if placements_above:
            self.yard.add_container(c2, placements_above[0])
            
            # Check that C001 is no longer accessible
            self.assertFalse(self.yard.containers["C001"].is_accessible,
                           "Container with something above should not be accessible")
            self.assertNotIn("C001", self.yard.accessible_containers)
            
            # Remove top container
            self.yard.remove_container(c2)
            
            # Check C001 is accessible again
            self.assertTrue(self.yard.containers["C001"].is_accessible,
                          "Container should be accessible after removing container above")
            self.assertIn("C001", self.yard.accessible_containers)
        
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
        # Create a long container (if such exists in the system)
        long_container = self._create_container("LONG001", 40)
        
        # Place at bay boundary
        placements = self.yard.search_placement_all_tiers(long_container, target_bay=0, max_proximity=10)
        
        # Find placements that span bays (start_split > 0)
        cross_bay = [p for p in placements if p.start_split > 0]
        
        if cross_bay:
            self.yard.add_container(long_container, cross_bay[0])
            
            # Verify container is stored correctly
            self.assertIn("LONG001", self.yard.containers)
            
            # Verify occupancy across bays
            placement = cross_bay[0]
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
            'moveable_time': [],
            'proximity_results': {}
        }
        
        # Test scaling with number of containers
        container_counts = [10, 50, 100, 200, 400]
        
        for count in container_counts:
            # Reset yard
            self.yard = BooleanStorageYard(
                n_rows=5, 
                n_bays=50, 
                n_tiers=5,
                coordinates=[]
            )
            
            # Placement time
            start = time.perf_counter()
            containers_added = 0
            for i in range(count):
                c = self._create_container(f"PERF{i:04d}", 
                                          np.random.choice([20, 40]),
                                          np.random.choice(["Regular", "Reefer", "DangerousGoods"]))
                
                placements = self.yard.search_placement_all_tiers(c, 
                                                                 target_bay=np.random.randint(0, 50),
                                                                 max_proximity=5)
                if placements:
                    self.yard.add_container(c, placements[0])
                    containers_added += 1
            
            placement_time = time.perf_counter() - start
            
            # Search time (average of 10 searches)
            search_times = []
            for _ in range(10):
                test_container = self._create_container("TEST", 40)
                start = time.perf_counter()
                self.yard.search_placement_all_tiers(test_container, 
                                                    target_bay=25,
                                                    max_proximity=5)
                search_times.append(time.perf_counter() - start)
            
            # Moveable containers time
            start = time.perf_counter()
            moveable = self.yard.find_moveable_containers(max_proximity=5)
            moveable_time = time.perf_counter() - start
            
            results['containers'].append(containers_added)
            results['placement_time'].append(placement_time)
            results['search_time'].append(np.mean(search_times))
            results['moveable_time'].append(moveable_time)
        
        # Test proximity scaling
        proximities = [1, 2, 3, 5, 10, 15]
        for prox in proximities:
            test_container = self._create_container("PROX_TEST", 40)
            start = time.perf_counter()
            placements = self.yard.search_placement_all_tiers(test_container,
                                                             target_bay=25,
                                                             max_proximity=prox)
            prox_time = time.perf_counter() - start
            results['proximity_results'][prox] = {
                'time': prox_time,
                'found': len(placements)
            }
        
        # Visualization
        self._plot_performance_results(results)
        
        # Assertions on performance
        # In test_performance_scaling, before computing ratios:
        if len(results['containers']) > 1:
            # find first non-zero baseline
            try:
                base_idx = next(i for i, v in enumerate(results['containers']) if v > 0)
            except StopIteration:
                self.skipTest("No containers placed in performance runs; skipping ratio checks")

            # also require last to be > 0
            if results['containers'][-1] == 0:
                self.skipTest("No containers placed at max load; skipping ratio checks")

            search_time_ratio = results['search_time'][-1] / max(results['search_time'][base_idx], 1e-9)
            moveable_ratio = results['moveable_time'][-1] / max(results['moveable_time'][base_idx], 1e-9)
            container_ratio = results['containers'][-1] / results['containers'][base_idx]

            self.assertLess(search_time_ratio, 2.0, "Search time should not double with 40x containers")
            self.assertLess(moveable_ratio, container_ratio * 1.5, "Moveable finding should scale linearly or better")
            
    def _place_container_at_bay(self, container: Container, target_bay: int) -> PlacementResult:
            """Helper: find a placement near target_bay and add the container."""
            placements = self.yard.search_placement_all_tiers(container, target_bay=target_bay, max_proximity=10)
            self.assertGreater(len(placements), 0, f"No placements found for {container.container_id} near bay {target_bay}")
            chosen = placements[0]
            self.yard.add_container(container, chosen)
            return chosen

    def test_departures_listing_use_estimated_vs_actual(self):
        """Verify listing by day using estimated vs actual departure dates."""
        current_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        # Create containers with various departure/estimated dates
        c_today_a = self._create_container("TODAY_REG_A", 40, "Regular")
        c_today_a.departure_date = current_day.replace(hour=14)

        c_today_b = self._create_container("TODAY_REG_B", 40, "Regular")
        c_today_b.departure_date = current_day.replace(hour=9)

        c_tomorrow = self._create_container("TOMORROW_REG", 40, "Regular")
        c_tomorrow.departure_date = current_day + timedelta(days=1)

        # Estimated today (even though actual is in 2 days)
        c_est_today = self._create_container("EST_TODAY", 40, "Regular")
        c_est_today.departure_date = current_day + timedelta(days=2)
        c_est_today.estimated_departure = current_day.replace(hour=13)

        # Estimated tomorrow (actual is today)
        c_est_tomorrow = self._create_container("EST_TOMORROW", 40, "Regular")
        c_est_tomorrow.departure_date = current_day.replace(hour=11)
        c_est_tomorrow.estimated_departure = current_day + timedelta(days=1)

        # Place them in known bays
        p_today_a = self._place_container_at_bay(c_today_a, target_bay=10)
        p_today_b = self._place_container_at_bay(c_today_b, target_bay=10)
        p_tomorrow = self._place_container_at_bay(c_tomorrow, target_bay=12)
        p_est_today = self._place_container_at_bay(c_est_today, target_bay=5)
        p_est_tomorrow = self._place_container_at_bay(c_est_tomorrow, target_bay=20)

        # Expected bay map (1-based indexing for comparison)
        expected_bays_1b = {
            "TODAY_REG_A": p_today_a.bay + 1,
            "TODAY_REG_B": p_today_b.bay + 1,
            "TOMORROW_REG": p_tomorrow.bay + 1,
            "EST_TODAY": p_est_today.bay + 1,
            "EST_TOMORROW": p_est_tomorrow.bay + 1,
        }

        # Use estimated = True: include TODAY_REG_A, TODAY_REG_B, EST_TODAY
        res_est = self.yard.get_containers_departing_on(current_day, use_estimated=True, one_based_bay=True)
        got_ids_est = {cid for cid, _ in res_est}
        exp_ids_est = {"TODAY_REG_A", "TODAY_REG_B", "EST_TODAY"}
        self.assertSetEqual(got_ids_est, exp_ids_est, "Estimated-based listing mismatch")

        # Validate bays for estimated case
        for cid, bay in res_est:
            self.assertEqual(bay, expected_bays_1b[cid], f"Wrong bay for {cid} (estimated case)")

        # Use estimated = False: include TODAY_REG_A, TODAY_REG_B, EST_TOMORROW
        res_act = self.yard.get_containers_departing_on(current_day, use_estimated=False, one_based_bay=True)
        got_ids_act = {cid for cid, _ in res_act}
        exp_ids_act = {"TODAY_REG_A", "TODAY_REG_B", "EST_TOMORROW"}
        self.assertSetEqual(got_ids_act, exp_ids_act, "Actual-based listing mismatch")

        # Validate bays for actual case
        for cid, bay in res_act:
            self.assertEqual(bay, expected_bays_1b[cid], f"Wrong bay for {cid} (actual case)")

    def test_departures_grouped_by_bay(self):
        """Verify grouped-by-bay output using estimated departures."""
        current_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        # Two containers due today in same bay and one in a different bay
        a = self._create_container("DUE_A", 40, "Regular")
        b = self._create_container("DUE_B", 40, "Regular")
        c = self._create_container("DUE_C", 40, "Regular")

        for cont in (a, b, c):
            cont.departure_date = current_day.replace(hour=8)

        pa = self._place_container_at_bay(a, target_bay=7)
        pb = self._place_container_at_bay(b, target_bay=7)
        pc = self._place_container_at_bay(c, target_bay=15)

        grouped = self.yard.get_containers_departing_on_by_bay(current_day, use_estimated=True, one_based_bay=True)

        bay7 = pa.bay + 1
        bay15 = pc.bay + 1

        self.assertIn(bay7, grouped, "Expected bay 7 group missing")
        self.assertIn(bay15, grouped, "Expected bay 15 group missing")
        self.assertCountEqual(grouped[bay7], ["DUE_A", "DUE_B"], "Bay 7 group mismatch")
        self.assertEqual(grouped[bay15], ["DUE_C"], "Bay 15 group mismatch")

    def test_departures_zero_based_bay_indexing(self):
        """Verify optional zero-based bay indexing."""
        current_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        x = self._create_container("ZERO_A", 40, "Regular")
        x.departure_date = current_day.replace(hour=10)
        px = self._place_container_at_bay(x, target_bay=22)

        res0 = self.yard.get_containers_departing_on(current_day, use_estimated=True, one_based_bay=False)
        self.assertTrue(any(cid == "ZERO_A" for cid, _ in res0), "ZERO_A missing in zero-based output")
        for cid, bay in res0:
            if cid == "ZERO_A":
                self.assertEqual(bay, px.bay, "Zero-based bay should match placement.bay")

    def _plot_performance_results(self, results):
        """Plot performance analysis results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('BooleanStorageYard Performance Analysis', fontsize=16)
        
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
        
        # Container scaling - moveable finding
        axes[1, 0].plot(results['containers'], 
                       [t * 1000 for t in results['moveable_time']], 'r-^')
        axes[1, 0].set_xlabel('Number of Containers')
        axes[1, 0].set_ylabel('Find Moveable Time (ms)')
        axes[1, 0].set_title('Moveable Container Finding')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Proximity scaling
        prox_data = results['proximity_results']
        proximities = sorted(prox_data.keys())
        prox_times = [prox_data[p]['time'] * 1000 for p in proximities]
        prox_found = [prox_data[p]['found'] for p in proximities]
        
        ax = axes[1, 1]
        ax2 = ax.twinx()
        
        line1 = ax.plot(proximities, prox_times, 'b-o', label='Search Time')
        ax.set_xlabel('Max Proximity')
        ax.set_ylabel('Search Time (ms)', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        
        line2 = ax2.plot(proximities, prox_found, 'r-s', label='Placements Found')
        ax2.set_ylabel('Placements Found', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        ax.set_title('Proximity Search Scaling')
        ax.grid(True, alpha=0.3)
        
        # Legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        
        plt.tight_layout()
        plt.savefig('storage_yard_performance.png', dpi=150)
        plt.show()
        
        print(f"\nPerformance Summary:")
        print(f"Max containers tested: {max(results['containers'])}")
        print(f"Placement throughput: {max(results['containers'])/max(results['placement_time']):.1f} containers/sec")
        print(f"Search time at max load: {results['search_time'][-1]*1000:.2f} ms")
        print(f"Moveable finding at max load: {results['moveable_time'][-1]*1000:.2f} ms")


if __name__ == '__main__':
    unittest.main()