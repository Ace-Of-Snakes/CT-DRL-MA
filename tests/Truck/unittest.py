# test_truck.py

import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from simulation.terminal_components.vehicles.Truck import (
    Truck,
    TRUCK_MAX_LENGTH_STANDARD,
    TRUCK_MIN_CONTAINER_LENGTH,
    TRUCK_STATUS_ARRIVING,
    TRUCK_STATUS_WAITING,
    TRUCK_STATUS_LOADING,
    TRUCK_STATUS_DEPARTING,
    TRUCK_STATUS_DEPARTED,
    EXCLUSIVE_CONTAINER_TYPES,
    TRUCK_ID_PREFIX
)
from simulation.terminal_components.storage_units.Container import Container


class TestTruckInitialization(unittest.TestCase):
    """Test truck initialization and basic properties."""
    
    def test_default_initialization(self):
        """Test truck initialization with default values."""
        truck = Truck()
        
        self.assertIsNotNone(truck.truck_id)
        self.assertTrue(truck.truck_id.startswith(TRUCK_ID_PREFIX))
        self.assertEqual(truck.max_length, TRUCK_MAX_LENGTH_STANDARD)
        self.assertEqual(truck.containers, [])
        self.assertIsNone(truck.arrival_time)
        self.assertIsNone(truck.parking_spot)
        self.assertEqual(truck.status, TRUCK_STATUS_ARRIVING)

        # A truck is only delivery or pickup if it has containers or IDs stored
        self.assertFalse(truck.is_pickup_truck)
        self.assertFalse(truck.is_delivery_truck)
    
    def test_custom_initialization(self):
        """Test truck initialization with custom values."""
        arrival = datetime(2025, 1, 15, 10, 0)
        truck = Truck(
            truck_id="CUSTOM_001",
            max_length=TRUCK_MAX_LENGTH_STANDARD,
            arrival_time=arrival,
            parking_spot="P15"
        )
        
        self.assertEqual(truck.truck_id, "CUSTOM_001")
        self.assertEqual(truck.max_length, TRUCK_MAX_LENGTH_STANDARD)
        self.assertEqual(truck.arrival_time, arrival)
        self.assertEqual(truck.parking_spot, "P15")
    
    def test_initialization_with_containers(self):
        """Test initialization with containers."""
        container1 = Mock(spec=Container)
        container1.container_id = "C001"
        container1.length = 6.0
        container1.container_type = "FEU"
        
        container2 = Mock(spec=Container)
        container2.container_id = "C002"
        container2.length = 6.0
        container2.container_type = "FEU"
        
        # Test with list of containers
        truck1 = Truck(containers=[container1, container2])
        self.assertEqual(len(truck1.containers), 2)
        self.assertFalse(truck1.is_pickup_truck)
        self.assertTrue(truck1.is_delivery_truck)
        
        # Test with single container
        truck2 = Truck(containers=container1)
        self.assertEqual(len(truck2.containers), 1)
        self.assertFalse(truck2.is_pickup_truck)
        self.assertTrue(truck2.is_delivery_truck)
    
    def test_invalid_max_length(self):
        """Test that invalid max_length raises error."""
        with self.assertRaises(ValueError):
            Truck(max_length=0)
        
        with self.assertRaises(ValueError):
            Truck(max_length=-10)


class TestContainerManagement(unittest.TestCase):
    """Test container add/remove operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.truck = Truck()
        
        # Create mock containers
        self.container_small = Mock(spec=Container)
        self.container_small.container_id = "SMALL_001"
        self.container_small.length = 6.06  # TWEU
        self.container_small.container_type = "TWEU"
        
        self.container_large = Mock(spec=Container)
        self.container_large.container_id = "LARGE_001"
        self.container_large.length = 12.19  # FEU
        self.container_large.container_type = "FEU"
        
        self.container_trailer = Mock(spec=Container)
        self.container_trailer.container_id = "TRAILER_001"
        self.container_trailer.length = 13.0
        self.container_trailer.container_type = "Trailer"
        
        self.container_swap = Mock(spec=Container)
        self.container_swap.container_id = "SWAP_001"
        self.container_swap.length = 7.45
        self.container_swap.container_type = "Swap Body"
    
    def test_add_regular_container(self):
        """Test adding regular containers."""
        self.assertTrue(self.truck.add_container(self.container_small))
        self.assertEqual(len(self.truck.containers), 1)
        
        # Add another that fits
        self.assertTrue(self.truck.add_container(self.container_small))
        self.assertEqual(len(self.truck.containers), 2)
    
    def test_add_container_exceeds_length(self):
        """Test that containers exceeding length are rejected."""
        self.assertTrue(self.truck.add_container(self.container_large))
        self.assertTrue(self.truck.add_container(self.container_large))
        
        # Try to add another large container (would exceed 13.6m)
        self.assertFalse(self.truck.add_container(self.container_large))
        self.assertEqual(len(self.truck.containers), 2)
    
    def test_add_exclusive_container(self):
        """Test adding exclusive container types."""
        # Add trailer to empty truck
        self.assertTrue(self.truck.add_container(self.container_trailer))
        self.assertEqual(len(self.truck.containers), 1)
        
        # Cannot add another container
        self.assertFalse(self.truck.add_container(self.container_small))
        self.assertEqual(len(self.truck.containers), 1)
    
    def test_cannot_add_exclusive_to_occupied_truck(self):
        """Test that exclusive containers cannot be added to occupied truck."""
        self.truck.add_container(self.container_small)
        
        self.assertFalse(self.truck.add_container(self.container_trailer))
        self.assertFalse(self.truck.add_container(self.container_swap))
        self.assertEqual(len(self.truck.containers), 1)
    
    def test_remove_container_by_id(self):
        """Test removing container by ID."""
        self.truck.add_container(self.container_small)
        self.truck.add_container(self.container_large)
        
        removed = self.truck.remove_container("SMALL_001")
        self.assertIsNotNone(removed)
        self.assertEqual(removed.container_id, "SMALL_001")
        self.assertEqual(len(self.truck.containers), 1)
    
    def test_remove_container_no_id(self):
        """Test removing first container when no ID specified."""
        self.truck.add_container(self.container_small)
        self.truck.add_container(self.container_large)
        
        removed = self.truck.remove_container()
        self.assertEqual(removed.container_id, "SMALL_001")
        self.assertEqual(len(self.truck.containers), 1)
    
    def test_remove_nonexistent_container(self):
        """Test removing container that doesn't exist."""
        self.truck.add_container(self.container_small)
        
        removed = self.truck.remove_container("NONEXISTENT")
        self.assertIsNone(removed)
        self.assertEqual(len(self.truck.containers), 1)
    
    def test_can_accommodate_container(self):
        """Test checking if container can be accommodated."""
        # Empty truck can accommodate regular container
        self.assertTrue(self.truck.can_accommodate_container(self.container_small))
        self.assertTrue(self.truck.can_accommodate_container(self.container_trailer))
        
        # Add a container
        self.truck.add_container(self.container_small)
        
        # Can still accommodate small container
        self.assertTrue(self.truck.can_accommodate_container(self.container_small))
        
        # Cannot accommodate exclusive types
        self.assertFalse(self.truck.can_accommodate_container(self.container_trailer))
        
        # Check None handling
        self.assertFalse(self.truck.can_accommodate_container(None))


class TestTruckOperations(unittest.TestCase):
    """Test truck operational methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.truck = Truck()
        self.base_time = datetime(2025, 1, 15, 8, 0)
    
    def test_start_loading(self):
        """Test starting loading operation."""
        self.truck.start_loading(self.base_time)
        
        self.assertEqual(self.truck.loading_start_time, self.base_time)
        self.assertEqual(self.truck.status, TRUCK_STATUS_LOADING)
    
    def test_start_loading_without_time(self):
        """Test that start_loading requires time."""
        with self.assertRaises(ValueError):
            self.truck.start_loading(None)
    
    def test_complete_loading(self):
        """Test completing loading operation."""
        complete_time = self.base_time + timedelta(minutes=30)
        self.truck.complete_loading(complete_time)
        
        self.assertEqual(self.truck.loading_complete_time, complete_time)
        self.assertEqual(self.truck.status, TRUCK_STATUS_DEPARTING)
    
    def test_complete_loading_without_time(self):
        """Test that complete_loading requires time."""
        with self.assertRaises(ValueError):
            self.truck.complete_loading(None)
    
    def test_depart(self):
        """Test truck departure."""
        depart_time = self.base_time + timedelta(hours=1)
        self.truck.depart(depart_time)
        
        self.assertEqual(self.truck.departure_time, depart_time)
        self.assertEqual(self.truck.status, TRUCK_STATUS_DEPARTED)
    
    def test_depart_without_time(self):
        """Test that depart requires time."""
        with self.assertRaises(ValueError):
            self.truck.depart(None)


class TestPickupManagement(unittest.TestCase):
    """Test pickup container ID management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.truck = Truck()
    
    def test_add_pickup_container_id(self):
        """Test adding pickup container IDs."""
        self.truck.add_pickup_container_id("PICKUP_001")
        
        self.assertIn("PICKUP_001", self.truck.pickup_container_ids)
        self.assertTrue(self.truck.is_pickup_truck)
    
    def test_add_empty_pickup_id(self):
        """Test that empty pickup ID raises error."""
        with self.assertRaises(ValueError):
            self.truck.add_pickup_container_id("")
        
        with self.assertRaises(ValueError):
            self.truck.add_pickup_container_id(None)
    
    def test_remove_pickup_container_id(self):
        """Test removing pickup container IDs."""
        self.truck.add_pickup_container_id("PICKUP_001")
        self.truck.add_pickup_container_id("PICKUP_002")
        
        result = self.truck.remove_pickup_container_id("PICKUP_001")
        self.assertTrue(result)
        self.assertNotIn("PICKUP_001", self.truck.pickup_container_ids)
        self.assertIn("PICKUP_002", self.truck.pickup_container_ids)
    
    def test_remove_nonexistent_pickup_id(self):
        """Test removing nonexistent pickup ID."""
        result = self.truck.remove_pickup_container_id("NONEXISTENT")
        self.assertFalse(result)


class TestTruckStatus(unittest.TestCase):
    """Test truck status methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.truck = Truck()
        
        self.container = Mock(spec=Container)
        self.container.container_id = "C001"
        self.container.length = 6.0
        self.container.container_type = "FEU"
    
    def test_get_available_length(self):
        """Test calculating available length."""
        self.assertEqual(self.truck.get_available_length(), TRUCK_MAX_LENGTH_STANDARD)
        
        self.truck.add_container(self.container)
        self.assertAlmostEqual(
            self.truck.get_available_length(),
            TRUCK_MAX_LENGTH_STANDARD - 6.0
        )
    
    def test_is_full(self):
        """Test checking if truck is full."""
        self.assertFalse(self.truck.is_full())
        
        # Add containers to nearly fill truck
        large_container = Mock(spec=Container)
        large_container.length = TRUCK_MAX_LENGTH_STANDARD - 1.0
        large_container.container_type = "FEU"
        
        self.truck.add_container(large_container)
        self.assertTrue(self.truck.is_full())
    
    def test_is_ready_to_depart_pickup_truck(self):
        """Test departure readiness for pickup truck."""
        self.truck.is_pickup_truck = True
        self.truck.add_pickup_container_id("PICKUP_001")
        
        self.assertFalse(self.truck.is_ready_to_depart())
        
        self.truck.remove_pickup_container_id("PICKUP_001")
        self.assertTrue(self.truck.is_ready_to_depart())
    
    def test_is_ready_to_depart_delivery_truck(self):
        """Test departure readiness for delivery truck."""
        self.truck.is_delivery_truck = True
        self.truck.add_container(self.container)
        
        self.assertFalse(self.truck.is_ready_to_depart())
        
        self.truck.remove_container()
        self.assertTrue(self.truck.is_ready_to_depart())


class TestTruckStats(unittest.TestCase):
    """Test truck statistics generation."""
    
    def test_get_stats_basic(self):
        """Test basic stats generation."""
        arrival = datetime(2025, 1, 15, 8, 0)
        truck = Truck(
            truck_id="TEST_001",
            arrival_time=arrival,
            parking_spot="P10"
        )
        
        stats = truck.get_stats()
        
        self.assertEqual(stats['parking_spot'], "P10")
        self.assertEqual(stats['arrival_time'], arrival)

class TestStringRepresentations(unittest.TestCase):
    """Test string representations."""
    
    def test_str_empty_truck(self):
        """Test string representation of empty truck."""
        truck = Truck(truck_id="TEST_001")
        expected = "Truck TEST_001: Empty, status: arriving"
        self.assertEqual(str(truck), expected)
    
    def test_str_single_container(self):
        """Test string representation with one container."""
        truck = Truck(truck_id="TEST_001")
        container = Mock(spec=Container)
        container.container_id = "C001"
        container.length = 6.0
        container.container_type = "FEU"
        
        truck.add_container(container)
        expected = "Truck TEST_001: Carrying C001, status: arriving"
        self.assertEqual(str(truck), expected)
    
    def test_str_multiple_containers(self):
        """Test string representation with multiple containers."""
        truck = Truck(truck_id="TEST_001")
        for i in range(2):
            container = Mock(spec=Container)
            container.container_id = f"C00{i}"
            container.length = 6.0
            container.container_type = "FEU"
            truck.add_container(container)
        
        expected = "Truck TEST_001: Carrying 2 containers, status: arriving"
        self.assertEqual(str(truck), expected)
    
    def test_repr(self):
        """Test developer representation."""
        truck = Truck(truck_id="TEST_001", parking_spot="P5")
        truck.status = TRUCK_STATUS_LOADING
        
        expected = "Truck(id=TEST_001, containers=0, status=loading, parking=P5)"
        self.assertEqual(repr(truck), expected)


if __name__ == '__main__':
    unittest.main()