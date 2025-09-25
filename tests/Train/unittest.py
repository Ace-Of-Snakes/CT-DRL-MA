import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from simulation2.core.Container import Container
from simulation2.core.Train import (
    Train,
    Wagon,
    TRAIN_STATUS_ARRIVING,
    TRAIN_STATUS_LOADING,
    TRAIN_STATUS_DEPARTING,
    TRAIN_STATUS_DEPARTED,
    DEFAULT_NUM_WAGONS,
    DEFAULT_WAGON_LENGTH,
    EXCLUSIVE_CONTAINER_TYPES
)


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
        self.container_regular.goods_type = "Regular"
        
        self.container_trailer = Mock(spec=Container)
        self.container_trailer.container_id = "T001"
        self.container_trailer.container_type = "Trailer"
        self.container_trailer.length = 12.19
        self.container_trailer.goods_type = "Regular"
        
        self.container_small = Mock(spec=Container)
        self.container_small.container_id = "C002"
        self.container_small.container_type = "TWEU"
        self.container_small.length = 6.06
        self.container_small.goods_type = "Regular"
    
    def test_initialization_with_defaults(self):
        """Test train initialization with default parameters."""
        train = Train()
        
        self.assertIsNotNone(train.train_id)
        self.assertTrue(train.train_id.startswith("TRN"))
        self.assertEqual(len(train.wagons), DEFAULT_NUM_WAGONS)
        self.assertEqual(train.status, TRAIN_STATUS_ARRIVING)
        self.assertEqual(len(train.container_locations), 0)
        self.assertEqual(len(train.wagon_by_id), DEFAULT_NUM_WAGONS)
        self.assertEqual(len(train.wagons_with_space), DEFAULT_NUM_WAGONS)
        self.assertEqual(len(train.empty_wagons), DEFAULT_NUM_WAGONS)
    
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
        
        # Add trailer - should go to empty wagon
        result = train.add_container(self.container_trailer)
        
        self.assertTrue(result)
        self.assertIn("T001", train.container_locations)
        location = train.container_locations["T001"]
        self.assertNotEqual(location.wagon_index, 0)  # Not in wagon with regular container
    
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
        train = Train(num_wagons=2, wagon_length=15.0)
        
        # Initially all wagons have space and are empty
        self.assertEqual(len(train.wagons_with_space), 2)
        self.assertEqual(len(train.empty_wagons), 2)
        
        # Add container to wagon 0
        train.add_container(self.container_regular, wagon_index=0)
        self.assertIn(0, train.wagons_with_space)  # Still has some space
        self.assertNotIn(0, train.empty_wagons)  # Not empty anymore
        
        # Fill wagon 0
        small_container = Mock(spec=Container)
        small_container.container_id = "C003"
        small_container.container_type = "TWEU"
        small_container.length = 2.8  # Will fill the wagon
        
        train.add_container(small_container, wagon_index=0)
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
        train = Train(num_wagons=2, wagon_length=15.0)
        
        # Regular container
        self.assertTrue(train.has_space_for_container(self.container_regular))
        
        # Fill first wagon
        train.add_container(self.container_regular, wagon_index=0)
        self.assertTrue(train.has_space_for_container(self.container_small))
        
        # Exclusive container needs empty wagon
        self.assertTrue(train.has_space_for_container(self.container_trailer))
        
        # Add trailer to wagon 1
        train.add_container(self.container_trailer, wagon_index=1)
        
        # No empty wagons left for another trailer
        another_trailer = Mock(spec=Container)
        another_trailer.container_type = "Trailer"
        another_trailer.length = 12.0
        self.assertFalse(train.has_space_for_container(another_trailer))
    
    def test_loading_operations(self):
        """Test loading time operations."""
        train = Train()
        current_time = self.base_time
        
        train.start_loading(current_time)
        self.assertEqual(train.loading_start_time, current_time)
        self.assertEqual(train.status, TRAIN_STATUS_LOADING)
        
        end_time = current_time + timedelta(hours=1)
        train.complete_loading(end_time)
        self.assertEqual(train.loading_complete_time, end_time)
        self.assertEqual(train.status, TRAIN_STATUS_DEPARTING)
    
    def test_depart(self):
        """Test train departure."""
        train = Train()
        departure_time = self.base_time + timedelta(hours=4)
        
        train.depart(departure_time)
        self.assertEqual(train.status, TRAIN_STATUS_DEPARTED)
        self.assertEqual(train.departure_time, departure_time)
    
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
        self.assertEqual(stats['status'], TRAIN_STATUS_ARRIVING)
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