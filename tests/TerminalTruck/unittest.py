# test_terminal_truck.py

import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock
from simulation2.core.TerminalTruck import (
    TerminalTruck,
    TERMINAL_TRUCK_MAX_LENGTH,
    TERMINAL_TRUCK_STATUS_IDLE,
    TERMINAL_TRUCK_STATUS_BUSY,
    TERMINAL_TRUCK_ID_PREFIX,
    TERMINAL_TRUCK_ALLOWED_TYPES,
    TERMINAL_TRUCK_MIN_TASK_TIME,
    TERMINAL_TRUCK_MAX_TASK_TIME
)
from simulation2.core.Container import Container


class TestTerminalTruckInitialization(unittest.TestCase):
    """Test terminal truck initialization and basic properties."""
    
    def test_default_initialization(self):
        """Test terminal truck initialization with defaults."""
        truck = TerminalTruck()
        
        self.assertIsNotNone(truck.truck_id)
        self.assertTrue(truck.truck_id.startswith(TERMINAL_TRUCK_ID_PREFIX))
        self.assertEqual(truck.max_length, TERMINAL_TRUCK_MAX_LENGTH)
        self.assertEqual(truck.status, TERMINAL_TRUCK_STATUS_IDLE)
        self.assertIsNone(truck.current_source)
        self.assertIsNone(truck.current_destination)
        self.assertIsNone(truck.task_start_time)
        self.assertIsNone(truck.task_completion_time)
        self.assertTrue(truck.is_terminal_truck)
        self.assertFalse(truck.is_pickup_truck)
        self.assertFalse(truck.is_delivery_truck)
        self.assertIsNone(truck.parking_spot)  # Terminal trucks don't use parking
    
    def test_custom_initialization(self):
        """Test terminal truck with custom values."""
        arrival = datetime(2025, 1, 15, 10, 0)
        truck = TerminalTruck(
            truck_id="CUSTOM_TTR_001",
            max_length=30.0,
            arrival_time=arrival
        )
        
        self.assertEqual(truck.truck_id, "CUSTOM_TTR_001")
        self.assertEqual(truck.max_length, 30.0)
        self.assertEqual(truck.arrival_time, arrival)
    
    def test_inheritance_from_truck(self):
        """Test that TerminalTruck properly inherits from Truck."""
        truck = TerminalTruck()
        
        # Should have parent class methods
        self.assertTrue(hasattr(truck, 'add_container'))
        self.assertTrue(hasattr(truck, 'remove_container'))
        self.assertTrue(hasattr(truck, 'get_available_length'))
        self.assertTrue(hasattr(truck, 'has_containers'))
        self.assertTrue(hasattr(truck, 'is_full'))
        self.assertTrue(hasattr(truck, 'get_stats'))


class TestContainerManagement(unittest.TestCase):
    """Test container operations specific to terminal trucks."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.truck = TerminalTruck()
        
        # Create mock containers
        self.trailer = Mock(spec=Container)
        self.trailer.container_id = "TRAILER_001"
        self.trailer.length = 12.19
        self.trailer.container_type = "Trailer"
        
        self.swap_body = Mock(spec=Container)
        self.swap_body.container_id = "SWAP_001"
        self.swap_body.length = 7.45
        self.swap_body.container_type = "Swap Body"
        
        self.regular_container = Mock(spec=Container)
        self.regular_container.container_id = "FEU_001"
        self.regular_container.length = 12.19
        self.regular_container.container_type = "FEU"
    
    def test_add_trailer(self):
        """Test adding a trailer."""
        self.assertTrue(self.truck.add_container(self.trailer))
        self.assertEqual(len(self.truck.containers), 1)
        self.assertEqual(self.truck.containers[0].container_id, "TRAILER_001")
    
    def test_add_swap_body(self):
        """Test adding a swap body."""
        self.assertTrue(self.truck.add_container(self.swap_body))
        self.assertEqual(len(self.truck.containers), 1)
        self.assertEqual(self.truck.containers[0].container_id, "SWAP_001")
    
    def test_reject_regular_container(self):
        """Test that regular containers are rejected."""
        self.assertFalse(self.truck.add_container(self.regular_container))
        self.assertEqual(len(self.truck.containers), 0)
    
    def test_single_container_limit(self):
        """Test that only one container is allowed at a time."""
        self.assertTrue(self.truck.add_container(self.trailer))
        
        # Try to add another
        self.assertFalse(self.truck.add_container(self.swap_body))
        self.assertEqual(len(self.truck.containers), 1)
    
    def test_add_none_container(self):
        """Test that None container is rejected."""
        with self.assertRaises(AssertionError):
            self.truck.add_container(None)
        self.assertEqual(len(self.truck.containers), 0)

    def test_remove_container(self):
        """Test removing a container."""
        self.truck.add_container(self.trailer)
        
        removed = self.truck.remove_container()
        self.assertIsNotNone(removed)
        self.assertEqual(removed.container_id, "TRAILER_001")
        self.assertEqual(len(self.truck.containers), 0)


class TestTaskManagement(unittest.TestCase):
    """Test task assignment and management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.truck = TerminalTruck()
        self.base_time = datetime(2025, 1, 15, 10, 0)
    
    def test_assign_task_normal(self):
        """Test assigning a normal task."""
        self.truck.assign_task(
            source="YARD_A",
            destination="YARD_B",
            task_time=180.0,  # 3 minutes
            current_time=self.base_time
        )
        
        self.assertEqual(self.truck.status, TERMINAL_TRUCK_STATUS_BUSY)
        self.assertEqual(self.truck.current_source, "YARD_A")
        self.assertEqual(self.truck.current_destination, "YARD_B")
        self.assertEqual(self.truck.task_start_time, self.base_time)
        expected_completion = self.base_time + timedelta(seconds=180)
        self.assertEqual(self.truck.task_completion_time, expected_completion)
    
    def test_assign_task_no_time_uses_max(self):
        """Test that not providing task_time uses maximum."""
        self.truck.assign_task(
            source="YARD_A",
            destination="YARD_B",
            task_time=None,  # Should default to max
            current_time=self.base_time
        )
        
        expected_completion = self.base_time + timedelta(seconds=TERMINAL_TRUCK_MAX_TASK_TIME)
        self.assertEqual(self.truck.task_completion_time, expected_completion)
    
    def test_assign_task_validation(self):
        """Test task assignment validation."""
        # Missing source
        with self.assertRaises(ValueError) as context:
            self.truck.assign_task("", "YARD_B", 180.0, self.base_time)
        self.assertIn("Source position must be provided", str(context.exception))
        
        # Missing destination
        with self.assertRaises(ValueError) as context:
            self.truck.assign_task("YARD_A", "", 180.0, self.base_time)
        self.assertIn("Destination position must be provided", str(context.exception))
        
        # Missing current time
        with self.assertRaises(ValueError) as context:
            self.truck.assign_task("YARD_A", "YARD_B", 180.0, None)
        self.assertIn("Current time must be provided", str(context.exception))
        
        # Negative task time
        with self.assertRaises(ValueError) as context:
            self.truck.assign_task("YARD_A", "YARD_B", -10.0, self.base_time)
        self.assertIn("Task time must be non-negative", str(context.exception))
        
        # Excessive task time
        with self.assertRaises(ValueError) as context:
            self.truck.assign_task(
                "YARD_A", "YARD_B", 
                TERMINAL_TRUCK_MAX_TASK_TIME + 1, 
                self.base_time
            )
        self.assertIn("Task time exceeds maximum", str(context.exception))
    
    def test_complete_task(self):
        """Test completing a task."""
        # Assign a task first
        self.truck.assign_task("YARD_A", "YARD_B", 180.0, self.base_time)
        
        # Add a container for the task
        container = Mock(spec=Container)
        container.container_type = "Trailer"
        container.length = 12.0
        container.container_id = "TEST_001"
        self.truck.add_container(container)
        
        # Complete the task
        complete_time = self.base_time + timedelta(seconds=180)
        self.truck.complete_task(complete_time)
        
        self.assertEqual(self.truck.status, TERMINAL_TRUCK_STATUS_IDLE)
        self.assertIsNone(self.truck.current_source)
        self.assertIsNone(self.truck.current_destination)
        self.assertIsNone(self.truck.task_start_time)
        self.assertIsNone(self.truck.task_completion_time)
        self.assertEqual(len(self.truck.containers), 0)  # Container removed
        self.assertEqual(self.truck.loading_complete_time, complete_time)
    
    def test_complete_task_requires_time(self):
        """Test that complete_task requires current time."""
        with self.assertRaises(ValueError) as context:
            self.truck.complete_task(None)
        self.assertIn("Current time must be provided", str(context.exception))


class TestAvailabilityChecks(unittest.TestCase):
    """Test availability and status checking methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.truck = TerminalTruck()
        self.base_time = datetime(2025, 1, 15, 10, 0)
    
    def test_is_available_initial(self):
        """Test that truck is initially available."""
        self.assertTrue(self.truck.is_available())
        self.assertFalse(self.truck.is_busy())
    
    def test_is_busy_during_task(self):
        """Test busy status during task."""
        self.truck.assign_task("YARD_A", "YARD_B", 180.0, self.base_time)
        
        self.assertFalse(self.truck.is_available())
        self.assertTrue(self.truck.is_busy())
    
    def test_is_available_after_task(self):
        """Test availability after completing task."""
        self.truck.assign_task("YARD_A", "YARD_B", 180.0, self.base_time)
        self.truck.complete_task(self.base_time + timedelta(seconds=180))
        
        self.assertTrue(self.truck.is_available())
        self.assertFalse(self.truck.is_busy())
    
    def test_not_available_with_container(self):
        """Test that truck with container is not available."""
        container = Mock(spec=Container)
        container.container_type = "Trailer"
        container.length = 12.0
        container.container_id = "TEST_001"
        self.truck.add_container(container)
        
        self.assertFalse(self.truck.is_available())

class TestStringRepresentations(unittest.TestCase):
    """Test string representations."""
    
    def test_str_idle_empty(self):
        """Test string representation of idle empty truck."""
        truck = TerminalTruck(truck_id="TERM_001")
        expected = "Terminal Truck TERM_001: Empty, status: idle"
        self.assertEqual(str(truck), expected)
    
    def test_str_with_container(self):
        """Test string representation with container."""
        truck = TerminalTruck(truck_id="TERM_002")
        container = Mock(spec=Container)
        container.container_type = "Trailer"
        container.container_id = "TRAILER_001"
        container.length = 12.0
        truck.add_container(container)
        
        expected = "Terminal Truck TERM_002: Carrying: TRAILER_001, status: idle"
        self.assertEqual(str(truck), expected)
    
    def test_str_busy_with_route(self):
        """Test string representation when busy with route."""
        truck = TerminalTruck(truck_id="TERM_003")
        base_time = datetime(2025, 1, 15, 10, 0)
        truck.assign_task("YARD_A", "YARD_B", 180.0, base_time)
        
        expected = "Terminal Truck TERM_003: Empty, status: busy (YARD_A → YARD_B)"
        self.assertEqual(str(truck), expected)
    
    def test_repr(self):
        """Test developer representation."""
        truck = TerminalTruck(truck_id="TERM_004")
        truck.status = TERMINAL_TRUCK_STATUS_BUSY
        
        container = Mock(spec=Container)
        container.container_type = "Swap Body"
        container.container_id = "SWAP_001"
        container.length = 7.45
        truck.add_container(container)
        
        expected = "TerminalTruck(id=TERM_004, containers=1, status=busy)"
        self.assertEqual(repr(truck), expected)


class TestIntegrationScenarios(unittest.TestCase):
    """Test complete workflow scenarios."""
    
    def test_complete_transport_workflow(self):
        """Test a complete transport workflow."""
        truck = TerminalTruck()
        base_time = datetime(2025, 1, 15, 10, 0)
        
        # Initially available
        self.assertTrue(truck.is_available())
        
        # Load a trailer
        trailer = Mock(spec=Container)
        trailer.container_type = "Trailer"
        trailer.container_id = "TRAILER_001"
        trailer.length = 12.19
        self.assertTrue(truck.add_container(trailer))
        
        # Not available with container
        self.assertFalse(truck.is_available())
        
        # Assign transport task
        truck.assign_task(
            source="STORAGE_A",
            destination="STORAGE_B",
            task_time=240.0,
            current_time=base_time
        )
        
        # Check busy status
        self.assertTrue(truck.is_busy())
        self.assertFalse(truck.is_available())
        
        # Complete task
        complete_time = base_time + timedelta(seconds=240)
        truck.complete_task(complete_time)
        
        # Check final state
        self.assertEqual(truck.status, TERMINAL_TRUCK_STATUS_IDLE)
        self.assertTrue(truck.is_available())
        self.assertFalse(truck.is_busy())
        self.assertEqual(len(truck.containers), 0)
        self.assertIsNone(truck.current_source)
        self.assertIsNone(truck.current_destination)
    
    def test_cannot_transport_regular_containers(self):
        """Test that terminal trucks cannot transport regular containers."""
        truck = TerminalTruck()
        
        # Try various regular container types
        container_types = ["TWEU", "THEU", "FEU", "FFEU"]
        
        for c_type in container_types:
            container = Mock(spec=Container)
            container.container_type = c_type
            container.container_id = f"{c_type}_001"
            container.length = 12.0
            
            self.assertFalse(
                truck.add_container(container),
                f"Should reject {c_type} container"
            )
        
        # Truck should still be empty
        self.assertEqual(len(truck.containers), 0)
        self.assertTrue(truck.is_available())


if __name__ == '__main__':
    unittest.main()