import unittest
from unittest.mock import Mock, MagicMock
from datetime import datetime
from simulation2.core.Container import Container
from simulation2.core.Wagon import (
    Wagon,
    WAGON_LENGTH_STANDARD,
    MIN_CONTAINER_LENGTH
) 

class TestWagon(unittest.TestCase):
    """Unit tests for the Wagon class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.wagon = Wagon("TEST_W001")
        
        # Create mock containers
        self.small_container = Mock(spec=Container)
        self.small_container.container_id = "SMALL_001"
        self.small_container.container_type = "TWEU"
        self.small_container.length = 6.06
        
        self.medium_container = Mock(spec=Container)
        self.medium_container.container_id = "MEDIUM_001"
        self.medium_container.container_type = "FEU"
        self.medium_container.length = 12.19
        
        self.large_container = Mock(spec=Container)
        self.large_container.container_id = "LARGE_001"
        self.large_container.container_type = "FEU"
        self.large_container.length = 12.19
        
        self.trailer = Mock(spec=Container)
        self.trailer.container_id = "TRAILER_001"
        self.trailer.container_type = "Trailer"
        self.trailer.length = 13.5
        
        self.swap_body = Mock(spec=Container)
        self.swap_body.container_id = "SWAP_001"
        self.swap_body.container_type = "Swap Body"
        self.swap_body.length = 7.45
    
    def test_initialization(self):
        """Test wagon initialization with default and custom values."""
        # Default initialization
        wagon1 = Wagon("W001")
        self.assertEqual(wagon1.wagon_id, "W001")
        self.assertEqual(wagon1.length, WAGON_LENGTH_STANDARD)
        self.assertEqual(len(wagon1.containers), 0)
        self.assertEqual(len(wagon1.pickup_container_ids), 0)
        
        # Custom length
        wagon2 = Wagon("W002", length= WAGON_LENGTH_STANDARD)
        self.assertEqual(wagon2.length, WAGON_LENGTH_STANDARD)
    
    def test_add_standard_container(self):
        """Test adding standard containers to wagon."""
        # Add first container
        self.assertTrue(self.wagon.add_container(self.small_container))
        self.assertEqual(len(self.wagon.containers), 1)
        
        # Add second container that fits
        self.assertTrue(self.wagon.add_container(self.medium_container))
        self.assertEqual(len(self.wagon.containers), 2)
        
        # Try to add container that doesn't fit
        self.assertFalse(self.wagon.add_container(self.large_container))
        self.assertEqual(len(self.wagon.containers), 2)
    
    def test_add_exclusive_container(self):
        """Test adding exclusive containers (trailer/swap body)."""
        # Add trailer to empty wagon
        self.assertTrue(self.wagon.add_container(self.trailer))
        self.assertEqual(len(self.wagon.containers), 1)
        
        # Can add another container to wagon with trailer
        self.assertTrue(self.wagon.add_container(self.small_container))
        self.assertEqual(len(self.wagon.containers), 2)
    
    def test_remove_container(self):
        """Test removing containers from wagon."""
        # Add containers
        self.wagon.add_container(self.small_container)
        self.wagon.add_container(self.medium_container)
        
        # Remove existing container
        removed = self.wagon.remove_container("SMALL_001")
        self.assertIsNotNone(removed)
        self.assertEqual(removed.container_id, "SMALL_001")
        self.assertEqual(len(self.wagon.containers), 1)
        
        # Try to remove non-existent container
        removed = self.wagon.remove_container("NONEXISTENT")
        self.assertIsNone(removed)
        self.assertEqual(len(self.wagon.containers), 1)
    
    def test_pickup_container_management(self):
        """Test managing pickup container IDs."""
        # Add pickup IDs
        self.wagon.add_pickup_container("PICKUP_001")
        self.wagon.add_pickup_container("PICKUP_002")
        self.assertEqual(len(self.wagon.pickup_container_ids), 2)
        self.assertIn("PICKUP_001", self.wagon.pickup_container_ids)
        
        # Add duplicate (should be ignored due to set)
        self.wagon.add_pickup_container("PICKUP_001")
        self.assertEqual(len(self.wagon.pickup_container_ids), 2)
        
        # Remove pickup ID
        self.wagon.remove_pickup_container("PICKUP_001")
        self.assertEqual(len(self.wagon.pickup_container_ids), 1)
        self.assertNotIn("PICKUP_001", self.wagon.pickup_container_ids)
        
        # Remove non-existent (should not raise error)
        self.wagon.remove_pickup_container("NONEXISTENT")
        self.assertEqual(len(self.wagon.pickup_container_ids), 1)
    
    def test_get_available_length(self):
        """Test calculating available length."""
        # Empty wagon
        self.assertEqual(self.wagon.get_available_length(), WAGON_LENGTH_STANDARD)
        
        # Add container
        self.wagon.add_container(self.small_container)
        expected = WAGON_LENGTH_STANDARD - 6.06
        self.assertAlmostEqual(self.wagon.get_available_length(), expected, places=2)
        
        # Add another container
        self.wagon.add_container(self.medium_container)
        expected = WAGON_LENGTH_STANDARD - 6.06 - 12.19
        self.assertAlmostEqual(self.wagon.get_available_length(), expected, places=2)
    
    def test_is_empty(self):
        """Test checking if wagon is empty."""
        self.assertTrue(self.wagon.is_empty())
        
        self.wagon.add_container(self.small_container)
        self.assertFalse(self.wagon.is_empty())
        
        self.wagon.remove_container("SMALL_001")
        self.assertTrue(self.wagon.is_empty())
    
    def test_is_full(self):
        """Test checking if wagon is full."""
        # Empty wagon is not full
        self.assertFalse(self.wagon.is_full())
        
        # Add containers to nearly fill wagon
        self.wagon.add_container(self.small_container)
        self.wagon.add_container(self.medium_container)
        
        # Should be full if less than MIN_CONTAINER_LENGTH available
        available = self.wagon.get_available_length()
        if available < MIN_CONTAINER_LENGTH:
            self.assertTrue(self.wagon.is_full())
        else:
            self.assertFalse(self.wagon.is_full())
    
    def test_has_exclusive_container(self):
        """Test checking for exclusive containers."""
        self.assertFalse(self.wagon.has_exclusive_container())
        
        # Add standard container
        self.wagon.add_container(self.small_container)
        self.assertFalse(self.wagon.has_exclusive_container())
        
        # Clear and add exclusive container
        self.wagon.containers.clear()
        self.wagon.add_container(self.trailer)
        self.assertTrue(self.wagon.has_exclusive_container())
    
    def test_string_representations(self):
        """Test string representations."""
        self.wagon.add_container(self.small_container)
        
        str_repr = str(self.wagon)
        self.assertIn("TEST_W001", str_repr)
        self.assertIn("1 containers", str_repr)
        self.assertIn("m available", str_repr)
        
        repr_repr = repr(self.wagon)
        self.assertIn("TEST_W001", repr_repr)
        self.assertIn("containers=1", repr_repr)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Wagon with zero length (edge case)
        zero_wagon = Wagon("ZERO", length=0)
        self.assertFalse(zero_wagon.add_container(self.small_container))
        self.assertTrue(zero_wagon.is_full())
        
        # Container exactly fitting
        exact_container = Mock(spec=Container)
        exact_container.container_id = "EXACT"
        exact_container.container_type = "FEU"
        exact_container.length = WAGON_LENGTH_STANDARD
        
        self.assertTrue(self.wagon.add_container(exact_container))
        self.assertAlmostEqual(self.wagon.get_available_length(), 0.0)
        self.assertTrue(self.wagon.is_full())


if __name__ == '__main__':
    unittest.main()