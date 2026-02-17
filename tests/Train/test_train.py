import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from simulation.core.containers.container import Container
from simulation.core.vehicles.train import Train
from simulation.core.vehicles.wagon import Wagon
from simulation.core.enums import TrainStatus
from simulation.core.constants import STANDARD_VEHICLE_LENGTH_M


class TestTrain(unittest.TestCase):
    """Unit tests for the optimized Train class."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_time = datetime(2025, 1, 1, 10, 0, 0)
        self.arrival_time = self.base_time
        self.departure_time = self.base_time + timedelta(hours=4)

        # Create mock containers
        self.container_regular = Mock(spec=Container)
        self.container_regular.container_id = "C001"
        self.container_regular.container_type = "FEU"
        self.container_regular.length = 12.19
        self.container_regular.length_m = 12.19
        self.container_regular.goods_type = "Regular"

        self.container_trailer = Mock(spec=Container)
        self.container_trailer.container_id = "T001"
        self.container_trailer.container_type = "Trailer"
        self.container_trailer.length = 12.19
        self.container_trailer.length_m = 12.19
        self.container_trailer.goods_type = "Regular"

        self.container_small = Mock(spec=Container)
        self.container_small.container_id = "C002"
        self.container_small.container_type = "TWEU"
        self.container_small.length = 6.06
        self.container_small.length_m = 6.06
        self.container_small.goods_type = "Regular"

    def test_initialization_with_defaults(self):
        """Test train initialization with default parameters."""
        train = Train()

        self.assertIsNotNone(train.train_id)
        self.assertTrue(train.train_id.startswith("TRN"))
        self.assertEqual(len(train.wagons), 10)
        self.assertEqual(train.status, TrainStatus.ARRIVING)
        self.assertEqual(len(train.container_locations), 0)
        self.assertEqual(len(train.wagon_by_id), 10)
        self.assertEqual(len(train.wagons_with_space), 10)
        self.assertEqual(len(train.empty_wagons), 10)

    def test_initialization_with_custom_parameters(self):
        """Test train initialization with custom parameters."""
        train = Train(
            train_id="TRN12345",
            num_wagons=5,
            wagon_length=20.0,
            arrival_time=self.arrival_time,
            departure_time=self.departure_time,
            rail_track="T1"
        )

        self.assertEqual(train.train_id, "TRN12345")
        self.assertEqual(len(train.wagons), 5)
        self.assertEqual(train.wagons[0].length, 20.0)
        self.assertEqual(train.arrival_time, self.arrival_time)
        self.assertEqual(train.departure_time, self.departure_time)
        self.assertEqual(train.rail_track, "T1")

    def test_initialization_with_invalid_parameters(self):
        """Test train initialization with invalid parameters."""
        with self.assertRaises(ValueError):
            Train(num_wagons=0)

        with self.assertRaises(ValueError):
            Train(wagon_length=-10)

    def test_add_container_to_specific_wagon(self):
        """Test adding a container to a specific wagon - O(1) operation."""
        train = Train(num_wagons=3)

        result = train.add_container(self.container_regular, wagon_index=1)

        self.assertTrue(result)
        self.assertIn("C001", train.container_locations)
        self.assertEqual(train.container_locations["C001"].wagon_index, 1)
        self.assertEqual(train._total_containers, 1)
        self.assertIn("C001", train.wagons[1].containers)

    def test_add_container_automatic_placement(self):
        """Test adding a container without specifying wagon."""
        train = Train(num_wagons=3)

        result = train.add_container(self.container_regular)

        self.assertTrue(result)
        self.assertIn("C001", train.container_locations)
        self.assertEqual(train._total_containers, 1)

    def test_add_exclusive_container(self):
        """Test adding an exclusive container (trailer/swap body)."""
        train = Train(num_wagons=3)

        # Add regular container first
        train.add_container(self.container_regular, wagon_index=0)

        # Add trailer - exclusive container logic is currently disabled,
        # so trailers are placed like regular containers via wagons_with_space.
        result = train.add_container(self.container_trailer)

        self.assertTrue(result)
        self.assertIn("T001", train.container_locations)

    def test_add_duplicate_container(self):
        """Test adding duplicate container ID."""
        train = Train()

        train.add_container(self.container_regular)
        result = train.add_container(self.container_regular)

        self.assertFalse(result)
        self.assertEqual(train._total_containers, 1)

    def test_remove_container_o1_operation(self):
        """Test removing a container - O(1) operation."""
        train = Train(num_wagons=2)

        # Add container
        train.add_container(self.container_regular, wagon_index=1)

        # Remove container - O(1) operation
        removed = train.remove_container("C001")

        self.assertEqual(removed, self.container_regular)
        self.assertNotIn("C001", train.container_locations)
        self.assertEqual(train._total_containers, 0)
        self.assertNotIn("C001", train.wagons[1].containers)
        self.assertIn(1, train.empty_wagons)

    def test_remove_nonexistent_container(self):
        """Test removing a non-existent container."""
        train = Train()

        removed = train.remove_container("C999")
        self.assertIsNone(removed)

    def test_find_container_o1_operation(self):
        """Test finding a container - O(1) operation."""
        train = Train(num_wagons=3)

        # Add containers to different wagons
        train.add_container(self.container_regular, wagon_index=1)
        train.add_container(self.container_small, wagon_index=2)

        # Find container - O(1) lookup
        wagon, position = train.find_container("C001")

        self.assertEqual(wagon.wagon_id, train.wagons[1].wagon_id)
        self.assertEqual(position, 0)

        wagon, position = train.find_container("C002")
        self.assertEqual(wagon.wagon_id, train.wagons[2].wagon_id)

    def test_has_container_o1_operation(self):
        """Test checking if train has container - O(1) operation."""
        train = Train()

        train.add_container(self.container_regular)

        self.assertTrue(train.has_container("C001"))
        self.assertFalse(train.has_container("C999"))

    def test_get_all_container_ids_o1_operation(self):
        """Test getting all container IDs - O(1) operation."""
        train = Train(num_wagons=2)

        train.add_container(self.container_regular, wagon_index=0)
        train.add_container(self.container_small, wagon_index=1)

        container_ids = train.get_all_container_ids()

        self.assertEqual(len(container_ids), 2)
        self.assertIn("C001", container_ids)
        self.assertIn("C002", container_ids)

    def test_get_container_count_o1_operation(self):
        """Test getting container count - O(1) operation."""
        train = Train()

        self.assertEqual(train.get_container_count(), 0)

        train.add_container(self.container_regular)
        self.assertEqual(train.get_container_count(), 1)

        train.add_container(self.container_small)
        self.assertEqual(train.get_container_count(), 2)

        train.remove_container("C001")
        self.assertEqual(train.get_container_count(), 1)

    def test_space_tracking(self):
        """Test wagon space tracking for optimization."""
        train = Train(num_wagons=2, wagon_length=25.0)

        # Initially all wagons have space and are empty
        self.assertEqual(len(train.wagons_with_space), 2)
        self.assertEqual(len(train.empty_wagons), 2)

        # Add container to wagon 0 (12.19m into 25m wagon -> 12.81m remaining > 6.1m MIN)
        train.add_container(self.container_regular, wagon_index=0)
        self.assertIn(0, train.wagons_with_space)  # Still has some space
        self.assertNotIn(0, train.empty_wagons)  # Not empty anymore

        # Fill wagon 0 further so remaining < MIN_CONTAINER_LENGTH_M (6.1m)
        # 25.0 - 12.19 = 12.81m remaining, need to add > 6.71m to make remaining < 6.1m
        filler = Mock(spec=Container)
        filler.container_id = "C003"
        filler.container_type = "FEU"
        filler.length = 12.5
        filler.length_m = 12.5

        train.add_container(filler, wagon_index=0)
        # 25.0 - 12.19 - 12.5 = 0.31m remaining < 6.1m MIN => wagon is full
        self.assertNotIn(0, train.wagons_with_space)  # No space left
        self.assertNotIn(0, train.empty_wagons)

    def test_add_pickup_container(self):
        """Test adding pickup container IDs."""
        train = Train(num_wagons=2)

        result = train.add_pickup_container("C100", wagon_index=1)
        self.assertTrue(result)
        self.assertIn("C100", train.wagons[1].pickup_container_ids)
        self.assertEqual(train._total_pickup_ids, 1)

        result = train.add_pickup_container("C101")
        self.assertTrue(result)
        self.assertIn("C101", train.wagons[0].pickup_container_ids)
        self.assertEqual(train._total_pickup_ids, 2)

    def test_remove_pickup_container(self):
        """Test removing pickup container IDs."""
        train = Train(num_wagons=2)

        train.add_pickup_container("C100", wagon_index=1)
        train.add_pickup_container("C101", wagon_index=0)

        result = train.remove_pickup_container("C100")
        self.assertTrue(result)
        self.assertNotIn("C100", train.wagons[1].pickup_container_ids)
        self.assertEqual(train._total_pickup_ids, 1)

        result = train.remove_pickup_container("C999")
        self.assertFalse(result)

    def test_is_fully_loaded_o1_operation(self):
        """Test checking if fully loaded - O(1) operation."""
        train = Train()

        self.assertTrue(train.is_fully_loaded())

        train.add_pickup_container("C100")
        self.assertFalse(train.is_fully_loaded())

        train.remove_pickup_container("C100")
        self.assertTrue(train.is_fully_loaded())

    def test_has_space_for_container(self):
        """Test checking space availability."""
        train = Train(num_wagons=2, wagon_length=30.0)

        # Regular container
        self.assertTrue(train.has_space_for_container(self.container_regular))

        # Fill first wagon partially
        train.add_container(self.container_regular, wagon_index=0)
        self.assertTrue(train.has_space_for_container(self.container_small))

        # Trailer should also fit (exclusive logic is disabled)
        self.assertTrue(train.has_space_for_container(self.container_trailer))

        # Add trailer to wagon 1
        train.add_container(self.container_trailer, wagon_index=1)

        # Both wagons still have space, so another container can fit
        another_trailer = Mock(spec=Container)
        another_trailer.container_type = "Trailer"
        another_trailer.length = 12.0
        another_trailer.length_m = 12.0
        self.assertTrue(train.has_space_for_container(another_trailer))

        # Fill both wagons so remaining < MIN_CONTAINER_LENGTH_M (6.1m)
        # Wagon 0: 30.0 - 12.19 = 17.81m free -> add 12.0m -> 5.81m free (full)
        filler1 = Mock(spec=Container)
        filler1.container_id = "FILL1"
        filler1.container_type = "FEU"
        filler1.length = 12.0
        filler1.length_m = 12.0
        train.add_container(filler1, wagon_index=0)

        # Wagon 1: 30.0 - 12.19 = 17.81m free -> add 12.0m -> 5.81m free (full)
        filler2 = Mock(spec=Container)
        filler2.container_id = "FILL2"
        filler2.container_type = "FEU"
        filler2.length = 12.0
        filler2.length_m = 12.0
        train.add_container(filler2, wagon_index=1)

        # Now each wagon has ~5.81m free which is < 6.1m MIN,
        # and another_trailer needs 12.0m -> no wagon can fit it
        self.assertFalse(train.has_space_for_container(another_trailer))

    def test_loading_operations(self):
        """Test loading time operations."""
        train = Train()
        current_time = self.base_time

        train.start_loading(current_time)
        self.assertEqual(train.loading_start_time, current_time)
        self.assertEqual(train.status, TrainStatus.LOADING)

        end_time = current_time + timedelta(hours=1)
        train.complete_loading(end_time)
        self.assertEqual(train.loading_complete_time, end_time)
        self.assertEqual(train.status, TrainStatus.DEPARTING)

    def test_depart(self):
        """Test train departure."""
        train = Train()
        departure_time = self.base_time + timedelta(hours=4)

        train.depart(departure_time)
        self.assertEqual(train.status, TrainStatus.DEPARTED)
        self.assertEqual(train.realised_departure_time, departure_time)

    def test_get_stats(self):
        """Test getting train statistics."""
        train = Train(
            train_id="TRN123",
            num_wagons=2,
            rail_track="T1"
        )

        train.add_container(self.container_regular)
        train.add_pickup_container("C100")

        stats = train.get_stats()

        self.assertEqual(stats['train_id'], "TRN123")
        self.assertEqual(stats['num_wagons'], 2)
        self.assertEqual(stats['total_containers'], 1)
        self.assertEqual(stats['pickup_containers'], 1)
        self.assertEqual(stats['status'], TrainStatus.ARRIVING.value)
        self.assertEqual(stats['rail_track'], "T1")
        self.assertIn('total_capacity', stats)
        self.assertIn('used_capacity', stats)
        self.assertIn('utilization_rate', stats)

    def test_performance_with_many_containers(self):
        """Test O(1) performance with many containers."""
        train = Train(num_wagons=50, wagon_length=25.0)

        # Add 100 containers
        containers = []
        for i in range(100):
            cont = Mock(spec=Container)
            cont.container_id = f"C{i:04d}"
            cont.container_type = "TWEU"
            cont.length = 6.0
            containers.append(cont)
            train.add_container(cont)

        # All these operations should be O(1)
        self.assertTrue(train.has_container("C0050"))
        self.assertEqual(train.get_container_count(), 100)

        wagon, pos = train.find_container("C0075")
        self.assertIsNotNone(wagon)

        removed = train.remove_container("C0025")
        self.assertIsNotNone(removed)
        self.assertEqual(train.get_container_count(), 99)

        ids = train.get_all_container_ids()
        self.assertEqual(len(ids), 99)


if __name__ == '__main__':
    unittest.main()
