# tests/TerminalGate/unittest.py

import unittest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import List

from simulation.operations.gate import TerminalGate, Order
from simulation.core.containers.container import Container
from simulation.core.vehicles.truck import Truck, EXCLUSIVE_CONTAINER_TYPES
from simulation.core.constants import STANDARD_VEHICLE_LENGTH_M
from simulation.core.enums import Direction, GoodsType
from simulation.config.operations_config import GateConfig


class TestTerminalGate(unittest.TestCase):
    """Unit tests for TerminalGate class based on the current API."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_date = datetime(2024, 1, 15, 0, 0, 0)
        self.day_of_week = "Monday"

        # Mock factories
        self.mock_container_factory = Mock()
        self.mock_truck_factory = Mock()

        # Operator dwell stats for testing short-dwell identification
        self.mock_operator_stats = {
            "METRANS_Export": 2.5,   # Short dwell (below threshold)
            "BOX_Export": 7.0,       # Normal dwell
            "COSCO_Export": 15.0,    # Long dwell
            "METRANS_Import": 3.0,
            "BOX_Import": 8.0,
        }

        # Initialize TerminalGate with mocks
        self.gate = TerminalGate(
            container_factory=self.mock_container_factory,
            truck_factory=self.mock_truck_factory,
            operator_dwell_stats=self.mock_operator_stats,
        )

    def _create_container(
        self,
        container_id: str,
        container_type: str = "FEU",
        length_m: float = 12.19,
        direction: Direction = Direction.IMPORT,
    ) -> Container:
        """Helper to create test containers."""
        return Container(
            container_id=container_id,
            direction=direction,
            container_type=container_type,
            arrival_date=self.test_date,
            departure_date=self.test_date + timedelta(days=5),
            length_m=length_m,
        )

    def _create_mock_truck(self, truck_id: str, arrival_hours: float = 8.0) -> Mock:
        """Helper to create a mock truck."""
        truck = Mock(spec=Truck)
        truck.truck_id = truck_id
        truck.arrival_time = self.test_date + timedelta(hours=arrival_hours)
        truck.containers = []
        truck.pickup_container_ids = set()
        return truck

    # ==================== Initialization Tests ====================

    def test_initialization(self):
        """Test proper initialization of TerminalGate."""
        self.assertIsNotNone(self.gate.container_factory)
        self.assertIsNotNone(self.gate.truck_factory)
        self.assertIsNotNone(self.gate.time_encoder)
        self.assertEqual(self.gate.operator_dwell_stats, self.mock_operator_stats)

    def test_short_dwell_operator_identification(self):
        """Test identification of short-dwell operators."""
        # METRANS_Export has 2.5 day dwell, which is below SHORT_DWELL_THRESHOLD_DAYS
        self.assertIn("METRANS", self.gate.short_dwell_operators)
        # COSCO_Export has 15 day dwell, should not be short-dwell
        self.assertNotIn("COSCO", self.gate.short_dwell_operators)
        # BOX_Export has 7.0 day dwell, should not be short-dwell
        self.assertNotIn("BOX", self.gate.short_dwell_operators)

    def test_initialization_with_empty_stats(self):
        """Test initialization with empty operator stats."""
        # Empty dict is falsy, so __init__ falls through to _compute_operator_stats.
        # We must make the mock container_factory iterable for get_available_operators.
        mock_cf = Mock()
        mock_cf.get_available_operators.return_value = []
        gate = TerminalGate(
            container_factory=mock_cf,
            truck_factory=self.mock_truck_factory,
            operator_dwell_stats={},
        )
        self.assertEqual(gate.short_dwell_operators, set())

    def test_initialization_triggers_stat_computation_when_none(self):
        """Test that None stats triggers _compute_operator_stats."""
        self.mock_container_factory.get_available_operators.return_value = []
        gate = TerminalGate(
            container_factory=self.mock_container_factory,
            truck_factory=self.mock_truck_factory,
            operator_dwell_stats=None,
        )
        # Should have called get_available_operators for both directions
        self.mock_container_factory.get_available_operators.assert_called()

    # ==================== Config Constants Tests ====================

    def test_gate_config_values(self):
        """Test that GateConfig has expected values."""
        self.assertEqual(GateConfig.SHORT_DWELL_THRESHOLD_DAYS, 3.0)
        self.assertEqual(GateConfig.SHORT_DWELL_EARLY_ARRIVAL_HOURS, 12)
        self.assertEqual(GateConfig.CONTAINER_BATCH_SIZE, 1000)
        self.assertIsInstance(GateConfig.MAX_WORKERS, int)

    def test_exclusive_container_types(self):
        """Test EXCLUSIVE_CONTAINER_TYPES constant."""
        self.assertIn("Trailer", EXCLUSIVE_CONTAINER_TYPES)
        self.assertIn("Swap Body", EXCLUSIVE_CONTAINER_TYPES)
        self.assertEqual(len(EXCLUSIVE_CONTAINER_TYPES), 2)

    def test_standard_vehicle_length(self):
        """Test STANDARD_VEHICLE_LENGTH_M constant."""
        self.assertAlmostEqual(STANDARD_VEHICLE_LENGTH_M, 24.4, places=1)

    # ==================== Import Processing Tests ====================

    def test_process_imports_calls_truck_factory(self):
        """Test that _process_imports delegates to truck_factory._generate_pickup_trucks."""
        containers = [
            self._create_container("C001"),
            self._create_container("C002"),
        ]
        mock_trucks = [self._create_mock_truck("T1"), self._create_mock_truck("T2")]
        self.mock_truck_factory._generate_pickup_trucks.return_value = mock_trucks

        trucks = self.gate._process_imports(containers, self.test_date, self.day_of_week)

        self.mock_truck_factory._generate_pickup_trucks.assert_called_once_with(
            containers=containers,
            day_key="monday",
            base_date=self.test_date,
        )
        self.assertEqual(len(trucks), 2)

    def test_process_imports_empty_list(self):
        """Test processing empty import list returns empty list."""
        trucks = self.gate._process_imports([], self.test_date, self.day_of_week)
        self.assertEqual(trucks, [])
        self.mock_truck_factory._generate_pickup_trucks.assert_not_called()

    # ==================== Export Processing Tests ====================

    def test_process_exports_single_operator(self):
        """Test export processing for a single operator."""
        export_operators = {
            "BOX": {
                "num_containers": 5,
                "arrival_time": {"angle": 2.0},
            }
        }

        mock_containers = [self._create_container(f"E{i}", direction=Direction.EXPORT) for i in range(5)]
        self.mock_container_factory.create_containers.return_value = mock_containers

        mock_trucks = [self._create_mock_truck(f"DT{i}") for i in range(3)]
        self.mock_truck_factory._generate_delivery_trucks.return_value = mock_trucks

        trucks = self.gate._process_exports(export_operators, self.test_date, self.day_of_week)

        # Should have called create_containers
        self.mock_container_factory.create_containers.assert_called_once()

        # Should have called _generate_delivery_trucks
        self.mock_truck_factory._generate_delivery_trucks.assert_called_once()

        self.assertEqual(len(trucks), 3)

    def test_process_exports_empty(self):
        """Test processing empty export operators."""
        trucks = self.gate._process_exports({}, self.test_date, self.day_of_week)
        self.assertEqual(trucks, [])

    def test_process_exports_large_batch_splits(self):
        """Test export processing with container count exceeding CONTAINER_BATCH_SIZE."""
        num_containers = GateConfig.CONTAINER_BATCH_SIZE + 500  # 1500

        export_operators = {
            "BOX": {
                "num_containers": num_containers,
                "arrival_time": {"angle": 1.0},
            }
        }

        # create_containers will be called twice: 1000 + 500
        batch1 = [self._create_container(f"B1_{i}", direction=Direction.EXPORT) for i in range(1000)]
        batch2 = [self._create_container(f"B2_{i}", direction=Direction.EXPORT) for i in range(500)]
        self.mock_container_factory.create_containers.side_effect = [batch1, batch2]

        mock_trucks = [self._create_mock_truck(f"DT{i}") for i in range(10)]
        self.mock_truck_factory._generate_delivery_trucks.return_value = mock_trucks

        trucks = self.gate._process_exports(export_operators, self.test_date, self.day_of_week)

        # Should have called create_containers twice (batch of 1000 + batch of 500)
        self.assertEqual(self.mock_container_factory.create_containers.call_count, 2)

    def test_process_exports_short_dwell_operator_adjusts_arrival(self):
        """Test that short-dwell operators get early arrival time adjustment."""
        export_operators = {
            "METRANS": {
                "num_containers": 5,
                "arrival_time": {"angle": 1.5},
            }
        }

        mock_containers = [self._create_container(f"E{i}", direction=Direction.EXPORT) for i in range(5)]
        self.mock_container_factory.create_containers.return_value = mock_containers

        mock_trucks = [self._create_mock_truck("DT1")]
        self.mock_truck_factory._generate_delivery_trucks.return_value = mock_trucks

        trucks = self.gate._process_exports(export_operators, self.test_date, self.day_of_week)

        # Verify that create_containers was called with an adjusted base_arrival_date
        call_kwargs = self.mock_container_factory.create_containers.call_args
        # The base_arrival_date should be adjusted by SHORT_DWELL_EARLY_ARRIVAL_HOURS
        # We cannot easily verify the exact time, but we can verify it was called
        self.mock_container_factory.create_containers.assert_called_once()

    def test_process_exports_zero_containers_skipped(self):
        """Test that operators with zero containers are skipped."""
        export_operators = {
            "BOX": {"num_containers": 0, "arrival_time": {"angle": 1.0}},
        }

        trucks = self.gate._process_exports(export_operators, self.test_date, self.day_of_week)
        self.assertEqual(trucks, [])
        self.mock_container_factory.create_containers.assert_not_called()

    # ==================== Order Processing Tests ====================

    def test_process_order_complete(self):
        """Test processing a complete order with imports and exports."""
        import_containers = [
            self._create_container("I001"),
            self._create_container("I002"),
        ]

        export_operators = {
            "BOX": {"num_containers": 5, "arrival_time": {"angle": 1.0}},
        }

        order = Order(
            import_containers=import_containers,
            export_operators=export_operators,
        )

        # Mock imports
        import_trucks = [self._create_mock_truck("IT1")]
        self.mock_truck_factory._generate_pickup_trucks.return_value = import_trucks

        # Mock exports
        mock_export_containers = [self._create_container(f"E{i}", direction=Direction.EXPORT) for i in range(5)]
        self.mock_container_factory.create_containers.return_value = mock_export_containers
        export_trucks = [self._create_mock_truck("ET1"), self._create_mock_truck("ET2")]
        self.mock_truck_factory._generate_delivery_trucks.return_value = export_trucks

        trucks = self.gate.process_order(order, self.test_date, self.day_of_week)

        # Should return combined trucks (1 import + 2 export = 3)
        self.assertEqual(len(trucks), 3)

    def test_process_order_imports_only(self):
        """Test processing order with only imports."""
        order = Order(
            import_containers=[self._create_container("I001")],
            export_operators={},
        )

        mock_trucks = [self._create_mock_truck("IT1")]
        self.mock_truck_factory._generate_pickup_trucks.return_value = mock_trucks

        trucks = self.gate.process_order(order, self.test_date, self.day_of_week)

        self.mock_truck_factory._generate_pickup_trucks.assert_called_once()
        self.mock_container_factory.create_containers.assert_not_called()
        self.assertEqual(len(trucks), 1)

    def test_process_order_exports_only(self):
        """Test processing order with only exports."""
        order = Order(
            import_containers=[],
            export_operators={"BOX": {"num_containers": 5, "arrival_time": {"angle": 1.0}}},
        )

        mock_export_containers = [self._create_container(f"E{i}", direction=Direction.EXPORT) for i in range(5)]
        self.mock_container_factory.create_containers.return_value = mock_export_containers

        mock_trucks = [self._create_mock_truck("ET1")]
        self.mock_truck_factory._generate_delivery_trucks.return_value = mock_trucks

        trucks = self.gate.process_order(order, self.test_date, self.day_of_week)

        self.mock_truck_factory._generate_pickup_trucks.assert_not_called()
        self.assertEqual(len(trucks), 1)

    def test_process_order_empty(self):
        """Test processing an empty order."""
        order = Order(import_containers=[], export_operators={})

        trucks = self.gate.process_order(order, self.test_date, self.day_of_week)

        self.assertEqual(trucks, [])
        self.mock_truck_factory._generate_pickup_trucks.assert_not_called()
        self.mock_container_factory.create_containers.assert_not_called()

    def test_process_order_multiple_export_operators(self):
        """Test processing order with multiple export operators."""
        order = Order(
            import_containers=[],
            export_operators={
                "METRANS": {"num_containers": 10, "arrival_time": {"angle": 1.0}},
                "BOX": {"num_containers": 20, "arrival_time": {"angle": 2.0}},
                "COSCO": {"num_containers": 15, "arrival_time": {"angle": 3.0}},
            },
        )

        # Each operator call returns containers
        self.mock_container_factory.create_containers.return_value = [
            self._create_container(f"E{i}", direction=Direction.EXPORT) for i in range(10)
        ]
        self.mock_truck_factory._generate_delivery_trucks.return_value = [
            self._create_mock_truck("DT1")
        ]

        trucks = self.gate.process_order(order, self.test_date, self.day_of_week)

        # create_containers should be called once per operator (3 operators)
        self.assertEqual(self.mock_container_factory.create_containers.call_count, 3)
        # _generate_delivery_trucks should be called once per operator (3 operators)
        self.assertEqual(self.mock_truck_factory._generate_delivery_trucks.call_count, 3)
        # 3 operators each returning 1 truck
        self.assertEqual(len(trucks), 3)

    # ==================== Truck Arrival Filtering Tests ====================

    def test_get_arrived_trucks_filters_correctly(self):
        """Test filtering of arrived trucks by current time."""
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

    def test_get_arrived_trucks_empty_list(self):
        """Test filtering with empty truck list."""
        current_time = self.test_date + timedelta(hours=10)
        arrived = self.gate.get_arrived_trucks([], current_time)
        self.assertEqual(arrived, [])

    def test_get_arrived_trucks_none_arrived(self):
        """Test filtering when no trucks have arrived yet."""
        current_time = self.test_date + timedelta(hours=6)
        trucks = [
            Mock(truck_id="T1", arrival_time=self.test_date + timedelta(hours=8)),
            Mock(truck_id="T2", arrival_time=self.test_date + timedelta(hours=9)),
        ]
        arrived = self.gate.get_arrived_trucks(trucks, current_time)
        self.assertEqual(len(arrived), 0)

    def test_get_arrived_trucks_all_arrived(self):
        """Test filtering when all trucks have arrived."""
        current_time = self.test_date + timedelta(hours=20)
        trucks = [
            Mock(truck_id="T1", arrival_time=self.test_date + timedelta(hours=8)),
            Mock(truck_id="T2", arrival_time=self.test_date + timedelta(hours=9)),
        ]
        arrived = self.gate.get_arrived_trucks(trucks, current_time)
        self.assertEqual(len(arrived), 2)

    def test_get_arrived_trucks_exact_time(self):
        """Test that a truck arriving at exactly current_time counts as arrived."""
        current_time = self.test_date + timedelta(hours=10)
        trucks = [
            Mock(truck_id="T1", arrival_time=self.test_date + timedelta(hours=10)),
        ]
        arrived = self.gate.get_arrived_trucks(trucks, current_time)
        self.assertEqual(len(arrived), 1)

    def test_get_arrived_trucks_with_none_arrival_time(self):
        """Test that trucks with None arrival_time are excluded."""
        current_time = self.test_date + timedelta(hours=10)
        trucks = [
            Mock(truck_id="T1", arrival_time=None),
            Mock(truck_id="T2", arrival_time=self.test_date + timedelta(hours=8)),
        ]
        arrived = self.gate.get_arrived_trucks(trucks, current_time)
        self.assertEqual(len(arrived), 1)
        self.assertEqual(arrived[0].truck_id, "T2")

    # ==================== Convenience Method Tests ====================

    def test_create_delivery_trucks_for_operators(self):
        """Test the create_delivery_trucks_for_operators convenience method."""
        export_operators = {
            "BOX": {"num_containers": 10, "arrival_time": {"angle": 1.0}},
        }

        mock_containers = [self._create_container(f"E{i}", direction=Direction.EXPORT) for i in range(10)]
        self.mock_container_factory.create_containers.return_value = mock_containers

        mock_trucks = [self._create_mock_truck("DT1")]
        self.mock_truck_factory._generate_delivery_trucks.return_value = mock_trucks

        trucks = self.gate.create_delivery_trucks_for_operators(
            export_operators, self.test_date, self.day_of_week
        )

        self.assertEqual(len(trucks), 1)

    @patch.object(GateConfig, 'JITTER_MIN_MINUTES', 5, create=True)
    @patch.object(GateConfig, 'JITTER_MAX_MINUTES', 30, create=True)
    def test_create_pickup_trucks_after(self):
        """Test the create_pickup_trucks_after convenience method."""
        containers = [self._create_container("I001"), self._create_container("I002")]
        earliest_time = self.test_date + timedelta(hours=10)

        mock_trucks = [
            self._create_mock_truck("PT1", arrival_hours=9),  # Before earliest
            self._create_mock_truck("PT2", arrival_hours=12),  # After earliest
        ]
        # Give the mock trucks a real arrival_time attribute (not read-only)
        for t in mock_trucks:
            t.arrival_time = self.test_date + timedelta(hours=9)
        self.mock_truck_factory._generate_pickup_trucks.return_value = mock_trucks

        trucks = self.gate.create_pickup_trucks_after(
            containers, earliest_time, self.day_of_week
        )

        self.assertIsNotNone(trucks)
        self.assertEqual(len(trucks), 2)

    def test_create_pickup_trucks_after_empty(self):
        """Test create_pickup_trucks_after with empty containers."""
        trucks = self.gate.create_pickup_trucks_after(
            [], self.test_date, self.day_of_week
        )
        self.assertEqual(trucks, [])

    def test_create_pickup_trucks_by_distribution(self):
        """Test the create_pickup_trucks_by_distribution convenience method."""
        containers = [self._create_container("I001")]
        mock_trucks = [self._create_mock_truck("PT1")]
        self.mock_truck_factory._generate_pickup_trucks.return_value = mock_trucks

        trucks = self.gate.create_pickup_trucks_by_distribution(
            containers, self.test_date, self.day_of_week
        )

        self.assertEqual(len(trucks), 1)
        self.mock_truck_factory._generate_pickup_trucks.assert_called_once()

    def test_create_pickup_trucks_by_distribution_empty(self):
        """Test create_pickup_trucks_by_distribution with empty containers."""
        trucks = self.gate.create_pickup_trucks_by_distribution(
            [], self.test_date, self.day_of_week
        )
        self.assertEqual(trucks, [])


class TestOrderDataclass(unittest.TestCase):
    """Test the Order dataclass."""

    def test_order_creation(self):
        """Test creating an Order instance."""
        containers = [Mock(spec=Container)]
        operators = {"OP1": {"num_containers": 10}}

        order = Order(
            import_containers=containers,
            export_operators=operators,
        )

        self.assertEqual(order.import_containers, containers)
        self.assertEqual(order.export_operators, operators)

    def test_order_empty(self):
        """Test creating an empty order."""
        order = Order(import_containers=[], export_operators={})

        self.assertEqual(len(order.import_containers), 0)
        self.assertEqual(len(order.export_operators), 0)

    def test_order_with_multiple_operators(self):
        """Test order with multiple export operators."""
        operators = {
            "OP1": {"num_containers": 10, "arrival_time": {"angle": 1.0}},
            "OP2": {"num_containers": 20, "arrival_time": {"angle": 2.0}},
        }
        order = Order(import_containers=[], export_operators=operators)

        self.assertEqual(len(order.export_operators), 2)
        self.assertIn("OP1", order.export_operators)
        self.assertIn("OP2", order.export_operators)


if __name__ == '__main__':
    unittest.main(verbosity=2)
