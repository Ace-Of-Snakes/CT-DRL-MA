# tests/test_TerminalGate.py

import unittest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch, call
from typing import List, Dict

# Import the classes we're testing
from simulation.terminal_components.systems.TerminalGate import (
    TerminalGate, Order, SPECIAL_CONTAINER_TYPES, TRUCK_MAX_LENGTH,
    SHORT_DWELL_THRESHOLD, SHORT_DWELL_EARLY_ARRIVAL_HOURS,
    CONTAINER_BATCH_SIZE
)
from simulation.terminal_components.storage_units.Container import Container
from simulation.terminal_components.vehicles.Truck import Truck


class TestTerminalGate(unittest.TestCase):
    """Comprehensive unit tests for TerminalGate class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Test date - MUST BE FIRST
        self.test_date = datetime(2024, 1, 15, 0, 0, 0)
        self.day_of_week = "Monday"
        
        # Mock factories
        self.mock_container_factory = Mock()
        self.mock_truck_factory = Mock()
        
        # Create test containers with different types
        self.regular_container = self._create_container("C001", "FEU", 12.19)
        self.trailer_container = self._create_container("C002", "Trailer", 13.72)
        self.swap_body_container = self._create_container("C003", "Swap Body", 13.72)
        self.small_container = self._create_container("C004", "TEU", 6.1)
        
        # Set up mock operator statistics
        self.mock_operator_stats = {
            "METRANS_Export": 2.5,  # Short dwell
            "BOX_Export": 7.0,      # Normal dwell
            "COSCO_Export": 15.0,   # Long dwell
            "METRANS_Import": 3.0,
            "BOX_Import": 8.0,
        }
        
        # Initialize TerminalGate with mocks
        self.gate = TerminalGate(
            container_factory=self.mock_container_factory,
            truck_factory=self.mock_truck_factory,
            operator_dwell_stats=self.mock_operator_stats
        )
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'gate'):
            self.gate.cleanup()
    
    def _create_container(self, container_id: str, container_type: str, length: float) -> Container:
        """Helper to create test containers."""
        return Container(
            container_id=container_id,
            direction="Import",
            container_type=container_type,
            arrival_date=self.test_date,
            departure_date=self.test_date + timedelta(days=5),
            length_m=length
        )
    
    def _create_truck(self, truck_id: str, is_pickup: bool = False) -> Truck:
        """Helper to create test trucks."""
        truck = Mock(spec=Truck)
        truck.truck_id = truck_id
        truck.arrival_time = self.test_date + timedelta(hours=8)
        truck.is_pickup_truck = is_pickup
        truck.is_delivery_truck = not is_pickup
        truck.containers = []
        truck.pickup_container_ids = set()
        return truck
    
    # ==================== Test Initialization ====================
    
    def test_initialization(self):
        """Test proper initialization of TerminalGate."""
        self.assertIsNotNone(self.gate.container_factory)
        self.assertIsNotNone(self.gate.truck_factory)
        self.assertIsNotNone(self.gate.time_encoder)
        self.assertIsNotNone(self.gate.executor)
        self.assertEqual(self.gate.operator_dwell_stats, self.mock_operator_stats)
    
    def test_short_dwell_operator_identification(self):
        """Test identification of short-dwell operators."""
        # METRANS should be identified as short-dwell (2.5 days)
        self.assertIn("METRANS", self.gate.short_dwell_operators)
        # COSCO should not be short-dwell (15 days)
        self.assertNotIn("COSCO", self.gate.short_dwell_operators)
    
    def test_initialization_without_stats(self):
        """Test initialization when operator stats need to be computed."""
        # Mock container factory methods
        self.mock_container_factory.get_available_operators.return_value = ["TEST_OP"]
        self.mock_container_factory.create_containers.return_value = [
            self._create_container(f"C{i}", "FEU", 12.19) for i in range(100)
        ]
        
        gate = TerminalGate(
            container_factory=self.mock_container_factory,
            truck_factory=self.mock_truck_factory,
            operator_dwell_stats=None  # Force computation
        )
        
        # Should have called get_available_operators
        self.mock_container_factory.get_available_operators.assert_called()
        gate.cleanup()
    
    # ==================== Test Import Processing ====================
    
    def test_process_imports_with_special_containers(self):
        """Test processing imports with trailers and swap bodies."""
        # Create mix of containers
        containers = [
            self.trailer_container,
            self.swap_body_container,
            self.regular_container,
            self.small_container
        ]
        
        # Mock truck factory to return trucks
        mock_trucks = [self._create_truck(f"TRK{i}", is_pickup=True) for i in range(4)]
        self.mock_truck_factory._generate_pickup_trucks.return_value = [mock_trucks[0]]
        
        trucks = self.gate._process_imports(containers, self.test_date, self.day_of_week)
        
        # Should create separate trucks for special containers
        # Each special container gets its own truck
        self.assertEqual(self.mock_truck_factory._generate_pickup_trucks.call_count, 3)
        
        # Verify special containers were processed separately
        calls = self.mock_truck_factory._generate_pickup_trucks.call_args_list
        
        # First two calls should be for individual special containers
        special_calls = calls[:2]
        for call_obj in special_calls:
            containers_arg = call_obj[1]['containers']
            self.assertEqual(len(containers_arg), 1)
            self.assertIn(containers_arg[0].container_type, SPECIAL_CONTAINER_TYPES)
    
    def test_process_imports_bundling_regular_containers(self):
        """Test bundling of regular containers into trucks."""
        # Create containers that should fit in one truck
        containers = [
            self._create_container(f"C{i}", "FEU", 12.19) for i in range(2)  # 24.38m total
        ]
        
        mock_truck = self._create_truck("TRK001", is_pickup=True)
        self.mock_truck_factory._generate_pickup_trucks.return_value = [mock_truck]
        
        trucks = self.gate._process_imports(containers, self.test_date, self.day_of_week)
        
        # Should create one truck for both containers (they fit in 24.4m)
        self.assertEqual(len(trucks), 1)
    
    def test_process_imports_bundling_overflow(self):
        """Test bundling when containers exceed truck capacity."""
        # Create containers that exceed one truck capacity
        containers = [
            self._create_container(f"C{i}", "FEU", 12.19) for i in range(3)  # 36.57m total
        ]
        
        mock_trucks = [self._create_truck(f"TRK{i}", is_pickup=True) for i in range(2)]
        self.mock_truck_factory._generate_pickup_trucks.side_effect = [
            [mock_trucks[0]], [mock_trucks[1]]
        ]
        
        trucks = self.gate._process_imports(containers, self.test_date, self.day_of_week)
        
        # Should create two trucks (containers don't fit in one)
        self.assertEqual(len(trucks), 2)
    
    def test_process_imports_empty_list(self):
        """Test processing empty import list."""
        trucks = self.gate._process_imports([], self.test_date, self.day_of_week)
        self.assertEqual(trucks, [])
        self.mock_truck_factory._generate_pickup_trucks.assert_not_called()
    
    def test_bundle_containers_vectorized(self):
        """Test the vectorized container bundling algorithm."""
        # Create containers with specific lengths
        containers = [
            self._create_container("C1", "FEU", 12.19),    # 12.19m
            self._create_container("C2", "TEU", 6.1),      # 6.1m
            self._create_container("C3", "TEU", 6.1),      # 6.1m
            self._create_container("C4", "HC", 12.19),     # 12.19m
        ]
        
        bundles = self.gate._bundle_containers_vectorized(containers)
        
        # Should create 2 bundles: [C1+C2+C3]=24.39m, [C4]=12.19m
        # or [C1+C2]=18.29m, [C4+C3]=18.29m depending on greedy packing
        self.assertGreaterEqual(len(bundles), 2)
        
        # Verify no bundle exceeds truck capacity
        for bundle in bundles:
            total_length = sum(c.length_m for c in bundle)
            self.assertLessEqual(total_length, TRUCK_MAX_LENGTH)
    
    # ==================== Test Export Processing ====================
    
    def test_process_exports_short_dwell_operator(self):
        """Test export processing for short-dwell operators."""
        export_operators = {
            "METRANS": {
                "num_containers": 10,
                "arrival_time": {"angle": 1.5}
            }
        }
        
        # Mock container creation
        mock_containers = [self._create_container(f"C{i}", "FEU", 12.19) for i in range(10)]
        self.mock_container_factory.create_containers.return_value = mock_containers
        
        # Mock truck creation
        mock_trucks = [self._create_truck(f"TRK{i}") for i in range(5)]
        self.mock_truck_factory._generate_delivery_trucks.return_value = mock_trucks
        
        trucks = self.gate._process_exports(export_operators, self.test_date, self.day_of_week)
        
        # Should create containers for export
        self.mock_container_factory.create_containers.assert_called_with(
            operator="METRANS",
            direction="Export",
            n_containers=10,
            base_arrival_date=unittest.mock.ANY,
            current_date=self.test_date
        )
        
        # Should generate delivery trucks
        self.assertEqual(len(trucks), 5)
    
    def test_process_exports_normal_dwell_operator(self):
        """Test export processing for normal-dwell operators."""
        export_operators = {
            "COSCO": {
                "num_containers": 20,
                "arrival_time": {"angle": 2.0}
            }
        }
        
        mock_containers = [self._create_container(f"C{i}", "FEU", 12.19) for i in range(20)]
        self.mock_container_factory.create_containers.return_value = mock_containers
        
        mock_trucks = [self._create_truck(f"TRK{i}") for i in range(10)]
        self.mock_truck_factory._generate_delivery_trucks.return_value = mock_trucks
        
        trucks = self.gate._process_exports(export_operators, self.test_date, self.day_of_week)
        
        self.assertEqual(len(trucks), 10)
    
    def test_process_exports_large_batch(self):
        """Test export processing with large container batches."""
        export_operators = {
            "BOX": {
                "num_containers": 2500,  # Larger than CONTAINER_BATCH_SIZE
                "arrival_time": {"angle": 1.0}
            }
        }
        
        # Mock container creation to return different batches
        batch1 = [self._create_container(f"C1_{i}", "FEU", 12.19) for i in range(1000)]
        batch2 = [self._create_container(f"C2_{i}", "FEU", 12.19) for i in range(1000)]
        batch3 = [self._create_container(f"C3_{i}", "FEU", 12.19) for i in range(500)]
        
        self.mock_container_factory.create_containers.side_effect = [batch1, batch2, batch3]
        
        mock_trucks = [self._create_truck(f"TRK{i}") for i in range(100)]
        self.mock_truck_factory._generate_delivery_trucks.return_value = mock_trucks
        
        trucks = self.gate._process_exports(export_operators, self.test_date, self.day_of_week)
        
        # Should have called create_containers 3 times (2500/1000 = 3 batches)
        self.assertEqual(self.mock_container_factory.create_containers.call_count, 3)
    
    def test_process_exports_empty(self):
        """Test processing empty export operators."""
        trucks = self.gate._process_exports({}, self.test_date, self.day_of_week)
        self.assertEqual(trucks, [])
    
    # ==================== Test Order Processing ====================
    
    def test_process_order_complete(self):
        """Test processing a complete order with imports and exports."""
        # Create order
        import_containers = [
            self.regular_container,
            self.trailer_container
        ]
        
        export_operators = {
            "METRANS": {"num_containers": 5, "arrival_time": {"angle": 1.0}},
            "BOX": {"num_containers": 10, "arrival_time": {"angle": 2.0}}
        }
        
        order = Order(
            import_containers=import_containers,
            export_operators=export_operators
        )
        
        # Mock the processing methods
        with patch.object(self.gate, '_process_imports') as mock_imports:
            with patch.object(self.gate, '_process_exports') as mock_exports:
                mock_imports.return_value = [self._create_truck("IMP1", is_pickup=True)]
                mock_exports.return_value = [self._create_truck("EXP1")]
                
                trucks = self.gate.process_order(order, self.test_date, self.day_of_week)
                
                # Should process both imports and exports
                mock_imports.assert_called_once()
                mock_exports.assert_called_once()
                
                # Should return combined trucks
                self.assertEqual(len(trucks), 2)
    
    def test_process_order_imports_only(self):
        """Test processing order with only imports."""
        order = Order(
            import_containers=[self.regular_container],
            export_operators={}
        )
        
        with patch.object(self.gate, '_process_imports') as mock_imports:
            mock_imports.return_value = [self._create_truck("IMP1", is_pickup=True)]
            
            trucks = self.gate.process_order(order, self.test_date, self.day_of_week)
            
            mock_imports.assert_called_once()
            self.assertEqual(len(trucks), 1)
    
    def test_process_order_exports_only(self):
        """Test processing order with only exports."""
        order = Order(
            import_containers=[],
            export_operators={"BOX": {"num_containers": 5, "arrival_time": {"angle": 1.0}}}
        )
        
        with patch.object(self.gate, '_process_exports') as mock_exports:
            mock_exports.return_value = [self._create_truck("EXP1")]
            
            trucks = self.gate.process_order(order, self.test_date, self.day_of_week)
            
            mock_exports.assert_called_once()
            self.assertEqual(len(trucks), 1)
    
    def test_process_order_error_handling(self):
        """Test error handling in order processing."""
        order = Order(
            import_containers=[self.regular_container],
            export_operators={"BOX": {"num_containers": 5, "arrival_time": {"angle": 1.0}}}
        )
        
        with patch.object(self.gate, '_process_imports') as mock_imports:
            with patch.object(self.gate, '_process_exports') as mock_exports:
                # Simulate error in import processing
                mock_imports.side_effect = Exception("Import error")
                mock_exports.return_value = [self._create_truck("EXP1")]
                
                trucks = self.gate.process_order(order, self.test_date, self.day_of_week)
                
                # Should still return export trucks despite import error
                self.assertEqual(len(trucks), 1)
                self.assertEqual(trucks[0].truck_id, "EXP1")
    
    # ==================== Test Truck Arrival Filtering ====================
    
    def test_get_arrived_trucks(self):
        """Test filtering of arrived trucks."""
        current_time = self.test_date + timedelta(hours=10)
        
        trucks = [
            Mock(truck_id="T1", arrival_time=self.test_date + timedelta(hours=8)),   # Arrived
            Mock(truck_id="T2", arrival_time=self.test_date + timedelta(hours=9)),   # Arrived
            Mock(truck_id="T3", arrival_time=self.test_date + timedelta(hours=11)),  # Not arrived
            Mock(truck_id="T4", arrival_time=self.test_date + timedelta(hours=12)),  # Not arrived
        ]
        
        arrived = self.gate.get_arrived_trucks(trucks, current_time)
        
        self.assertEqual(len(arrived), 2)
        self.assertEqual(arrived[0].truck_id, "T1")
        self.assertEqual(arrived[1].truck_id, "T2")
    
    def test_get_arrived_trucks_empty(self):
        """Test filtering with empty truck list."""
        current_time = self.test_date + timedelta(hours=10)
        arrived = self.gate.get_arrived_trucks([], current_time)
        self.assertEqual(arrived, [])
    
    def test_get_arrived_trucks_none_arrived(self):
        """Test filtering when no trucks have arrived."""
        current_time = self.test_date + timedelta(hours=6)
        
        trucks = [
            Mock(truck_id="T1", arrival_time=self.test_date + timedelta(hours=8)),
            Mock(truck_id="T2", arrival_time=self.test_date + timedelta(hours=9)),
        ]
        
        arrived = self.gate.get_arrived_trucks(trucks, current_time)
        self.assertEqual(len(arrived), 0)
    
    # ==================== Test Edge Cases ====================
    
    def test_single_container_exact_fit(self):
        """Test container that exactly fits truck capacity."""
        container = self._create_container("C1", "Special", TRUCK_MAX_LENGTH)
        
        bundles = self.gate._bundle_containers_vectorized([container])
        
        self.assertEqual(len(bundles), 1)
        self.assertEqual(len(bundles[0]), 1)
        self.assertEqual(bundles[0][0].container_id, "C1")
    
    def test_container_too_large(self):
        """Test container larger than truck capacity."""
        # This shouldn't happen in practice, but test defensive programming
        container = self._create_container("C1", "Oversized", TRUCK_MAX_LENGTH + 1)
        
        bundles = self.gate._bundle_containers_vectorized([container])
        
        # Should still create a bundle, even if oversized
        self.assertEqual(len(bundles), 1)
    
    def test_many_small_containers(self):
        """Test bundling many small containers."""
        # Create 20 small containers (1m each)
        containers = [self._create_container(f"C{i}", "Small", 1.0) for i in range(20)]
        
        bundles = self.gate._bundle_containers_vectorized(containers)
        
        # Should fit all 20 in one truck (20m < 24.4m)
        self.assertEqual(len(bundles), 1)
        self.assertEqual(len(bundles[0]), 20)
    
    def test_mixed_container_sizes(self):
        """Test bundling with highly varied container sizes."""
        containers = [
            self._create_container("C1", "Large", 20.0),
            self._create_container("C2", "Medium", 10.0),
            self._create_container("C3", "Small", 4.0),
            self._create_container("C4", "Tiny", 2.0),
            self._create_container("C5", "Tiny", 2.0),
        ]
        
        bundles = self.gate._bundle_containers_vectorized(containers)
        
        # Verify efficient packing
        for bundle in bundles:
            total_length = sum(c.length_m for c in bundle)
            self.assertLessEqual(total_length, TRUCK_MAX_LENGTH)
        
        # Should pack efficiently (likely 2 trucks: [20+4] and [10+2+2])
        self.assertLessEqual(len(bundles), 3)
    
    # ==================== Test Resource Management ====================
    
    def test_cleanup(self):
        """Test resource cleanup."""
        gate = TerminalGate(
            container_factory=self.mock_container_factory,
            truck_factory=self.mock_truck_factory,
            operator_dwell_stats=self.mock_operator_stats
        )
        
        # Mock the executor
        gate.executor = Mock()
        gate.cleanup()
        
        # Should call shutdown on executor
        gate.executor.shutdown.assert_called_once_with(wait=True)
    
    def test_destructor(self):
        """Test that destructor calls cleanup."""
        gate = TerminalGate(
            container_factory=self.mock_container_factory,
            truck_factory=self.mock_truck_factory,
            operator_dwell_stats=self.mock_operator_stats
        )
        
        with patch.object(gate, 'cleanup') as mock_cleanup:
            gate.__del__()
            mock_cleanup.assert_called_once()
    
    # ==================== Test Integration Scenarios ====================
    
    def test_realistic_large_order(self):
        """Test processing a realistic large order."""
        # Create a large mixed order
        import_containers = []
        
        # Add various container types
        for i in range(100):
            if i % 10 == 0:
                # 10% trailers
                import_containers.append(self._create_container(f"T{i}", "Trailer", 13.72))
            elif i % 10 == 1:
                # 10% swap bodies
                import_containers.append(self._create_container(f"S{i}", "Swap Body", 13.72))
            elif i % 3 == 0:
                # 30% TEU
                import_containers.append(self._create_container(f"TEU{i}", "TEU", 6.1))
            else:
                # 50% FEU
                import_containers.append(self._create_container(f"FEU{i}", "FEU", 12.19))
        
        export_operators = {
            "METRANS": {"num_containers": 50, "arrival_time": {"angle": 1.0}},
            "BOX": {"num_containers": 75, "arrival_time": {"angle": 2.0}},
            "COSCO": {"num_containers": 100, "arrival_time": {"angle": 3.0}},
        }
        
        order = Order(
            import_containers=import_containers,
            export_operators=export_operators
        )
        
        # Mock returns
        self.mock_truck_factory._generate_pickup_trucks.return_value = [self._create_truck("P1", True)]
        self.mock_truck_factory._generate_delivery_trucks.return_value = [self._create_truck("D1", False)]
        self.mock_container_factory.create_containers.return_value = [
            self._create_container(f"E{i}", "FEU", 12.19) for i in range(10)
        ]
        
        # Process the order
        trucks = self.gate.process_order(order, self.test_date, self.day_of_week)
        
        # Should process successfully
        self.assertIsInstance(trucks, list)
        self.assertGreater(len(trucks), 0)
    
    def test_concurrent_order_processing(self):
        """Test that concurrent processing works correctly."""
        import concurrent.futures
        
        orders = []
        for i in range(5):
            order = Order(
                import_containers=[self._create_container(f"C{i}", "FEU", 12.19)],
                export_operators={f"OP{i}": {"num_containers": 5, "arrival_time": {"angle": 1.0}}}
            )
            orders.append(order)
        
        # Mock returns
        self.mock_truck_factory._generate_pickup_trucks.return_value = [self._create_truck("P1", True)]
        self.mock_truck_factory._generate_delivery_trucks.return_value = [self._create_truck("D1", False)]
        self.mock_container_factory.create_containers.return_value = [
            self._create_container("E1", "FEU", 12.19)
        ]
        
        # Process orders concurrently
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(self.gate.process_order, order, self.test_date, self.day_of_week)
                for order in orders
            ]
            
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        
        # All orders should be processed
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIsInstance(result, list)


class TestOrderDataclass(unittest.TestCase):
    """Test the Order dataclass."""
    
    def test_order_creation(self):
        """Test creating an Order instance."""
        containers = [Mock(spec=Container)]
        operators = {"OP1": {"num_containers": 10}}
        
        order = Order(
            import_containers=containers,
            export_operators=operators
        )
        
        self.assertEqual(order.import_containers, containers)
        self.assertEqual(order.export_operators, operators)
    
    def test_order_empty(self):
        """Test creating an empty order."""
        order = Order(
            import_containers=[],
            export_operators={}
        )
        
        self.assertEqual(len(order.import_containers), 0)
        self.assertEqual(len(order.export_operators), 0)


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)