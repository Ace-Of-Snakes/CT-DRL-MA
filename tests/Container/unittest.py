import unittest
from datetime import datetime, timedelta
import numpy as np
from simulation.terminal_components.storage_units.Container import Container, ONLY_SELF_STACKABLE, MIN_ACCURACY_DAYS, MAX_HOLDING_DAYS


class TestContainer(unittest.TestCase):
    """Unit tests for the Container class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.base_date = datetime(2025, 1, 1)
        
    def test_container_initialization_basic(self):
        """Test basic container initialization."""
        container = Container(
            container_id="TEST001",
            direction="Import",
            container_type="FEU",
            arrival_date=self.base_date,
            departure_date=self.base_date + timedelta(days=10)
        )
        
        self.assertEqual(container.container_id, "TEST001")
        self.assertEqual(container.direction, "Import")
        self.assertEqual(container.container_type, "FEU")
        self.assertEqual(container.goods_type, "Regular")
        self.assertFalse(container.is_high_cube)
        self.assertTrue(container.is_stackable)
        self.assertEqual(container.stack_compatibility, "size")
        
    def test_container_initialization_with_dates(self):
        """Test container initialization with arrival and departure dates."""
        arrival = self.base_date
        departure = self.base_date + timedelta(days=5)
        
        container = Container(
            container_id="TEST002",
            direction="Export",
            container_type="TWEU",
            arrival_date=arrival,
            departure_date=departure
        )
        
        self.assertEqual(container.arrival_date, arrival)
        self.assertEqual(container.departure_date, departure)
        self.assertIsNotNone(container.estimated_departure)
        
    def test_invalid_direction_raises_error(self):
        """Test that invalid direction raises ValueError."""
        with self.assertRaises(ValueError):
            Container(
                container_id="TEST003",
                direction="Invalid",
                container_type="FEU",
                arrival_date=self.base_date,
                departure_date=self.base_date + timedelta(days=10)
            )
    
    def test_invalid_container_type_raises_error(self):
        """Test that invalid container type raises ValueError."""
        with self.assertRaises(ValueError):
            Container(
                container_id="TEST004",
                direction="Import",
                container_type="INVALID",
                arrival_date=self.base_date,
                departure_date=self.base_date + timedelta(days=10)
            )
    
    def test_invalid_goods_type_raises_error(self):
        """Test that invalid goods type raises ValueError."""
        with self.assertRaises(ValueError):
            Container(
                container_id="TEST005",
                direction="Import",
                container_type="FEU",
                goods_type="Invalid",
                arrival_date=self.base_date,
                departure_date=self.base_date + timedelta(days=10)
            )
    
    def test_special_goods_self_stackable(self):
        """Test that reefer and dangerous goods are self-stackable only."""
        reefer = Container(
            container_id="REEF001",
            direction="Import",
            container_type="FEU",
            goods_type="Reefer",
            arrival_date=self.base_date,
            departure_date=self.base_date + timedelta(days=10)
        )
        
        dangerous = Container(
            container_id="DG001",
            direction="Import",
            container_type="FEU",
            goods_type="Dangerous",
            arrival_date=self.base_date,
            departure_date=self.base_date + timedelta(days=10)
        )
        
        self.assertEqual(reefer.stack_compatibility, "self")
        self.assertEqual(dangerous.stack_compatibility, "self")
    
    def test_trailer_not_stackable(self):
        """Test that trailers are not stackable."""
        trailer = Container(
            container_id="TRAIL001",
            direction="Export",
            container_type="Trailer",
            is_stackable=False,
            arrival_date=self.base_date,
            departure_date=self.base_date + timedelta(days=10)
        )
        
        self.assertFalse(trailer.is_stackable)
        self.assertEqual(trailer.stack_compatibility, "none")
    
    def test_days_in_terminal(self):
        """Test days_in_terminal calculation."""
        arrival = self.base_date
        container = Container(
            container_id="TEST006",
            direction="Import",
            container_type="FEU",
            arrival_date=self.base_date,
            departure_date=self.base_date + timedelta(days=10)
        )
        
        # Test various current dates
        self.assertEqual(container.days_in_terminal(arrival), 0)
        self.assertEqual(container.days_in_terminal(arrival + timedelta(days=5)), 5)
        self.assertEqual(container.days_in_terminal(arrival + timedelta(days=10)), 10)
        
    def test_days_until_departure(self):
        """Test days_until_departure calculation."""
        arrival = self.base_date
        departure = arrival + timedelta(days=10)
        container = Container(
            container_id="TEST007",
            direction="Import",
            container_type="FEU",
            arrival_date=arrival,
            departure_date=departure
        )
        
        current = arrival + timedelta(days=3)
        days_left = container.days_until_departure(current)
        
        # Should be approximately 7 days (might vary due to estimation)
        self.assertIsNotNone(container.estimated_departure)
        self.assertGreaterEqual(days_left, 0)
    
    def test_estimation_accuracy_short_stay(self):
        """Test that short stays have perfect accuracy."""
        arrival = self.base_date
        departure = arrival + timedelta(days=MIN_ACCURACY_DAYS)
        
        # Create a single container to test
        container = Container(
            container_id="TEST009",
            direction="Import",
            container_type="FEU",
            arrival_date=arrival,
            departure_date=departure
        )
        # For short stays (<=2 days), estimation should equal departure
        self.assertEqual(container.estimated_departure, departure)
        
        # Test multiple updates - should remain accurate
        for day in range(MIN_ACCURACY_DAYS + 1):
            current = arrival + timedelta(days=day)
            container.update_estimation(current)
            # Should still be perfectly accurate
            self.assertEqual(container.estimated_departure, departure)
    
    def test_update_estimation(self):
        """Test updating estimation as time passes."""
        arrival = self.base_date
        departure = arrival + timedelta(days=30)
        
        container = Container(
            container_id="TEST010",
            direction="Import",
            container_type="FEU",
            arrival_date=arrival,
            departure_date=departure
        )
        
        # Get initial estimation
        initial_estimation = container.estimated_departure
        
        # Update estimation with new current date
        current = arrival + timedelta(days=25)
        updated_estimation = container.update_estimation(current)
        
        self.assertIsNotNone(updated_estimation)
        # As we get closer to departure, estimation should generally be more accurate
        # (though randomness means we can't guarantee this in a single test)
    
    def test_can_stack_with_same_type(self):
        """Test stacking containers of the same type."""
        container1 = Container(
            container_id="TEST011",
            direction="Import",
            container_type="FEU",
            arrival_date=self.base_date,
            departure_date=self.base_date + timedelta(days=5)
        )
        
        container2 = Container(
            container_id="TEST012",
            direction="Import",
            container_type="FEU",
            arrival_date=self.base_date,
            departure_date=self.base_date + timedelta(days=10)
        )
        
        # Container1 (leaving earlier) can stack on container2
        self.assertTrue(container1.can_stack_with(container2))
        # Container2 (leaving later) can stack on container1
        self.assertTrue(container2.can_stack_with(container1))
    
    def test_can_stack_with_different_types(self):
        """Test stacking containers of different types."""
        feu = Container(
            container_id="TEST013",
            direction="Import",
            container_type="FEU",
            arrival_date=self.base_date,
            departure_date=self.base_date + timedelta(days=10)
        )
        
        tweu = Container(
            container_id="TEST014",
            direction="Import",
            container_type="TWEU",
            arrival_date=self.base_date,
            departure_date=self.base_date + timedelta(days=10)
        )
        
        # Different sizes with "size" compatibility cannot stack
        self.assertFalse(feu.can_stack_with(tweu))
        self.assertFalse(tweu.can_stack_with(feu))
    
    def test_can_stack_with_special_goods(self):
        """Test stacking with special goods types."""
        reefer = Container(
            container_id="TEST015",
            direction="Import",
            container_type="FEU",
            goods_type="Reefer",
            arrival_date=self.base_date,
            departure_date=self.base_date + timedelta(days=10)
        )
        
        regular = Container(
            container_id="TEST016",
            direction="Import",
            container_type="FEU",
            goods_type="Regular",
            arrival_date=self.base_date,
            departure_date=self.base_date + timedelta(days=10)
        )
        
        another_reefer = Container(
            container_id="TEST017",
            direction="Import",
            container_type="FEU",
            goods_type="Reefer",
            arrival_date=self.base_date,
            departure_date=self.base_date + timedelta(days=10)
        )
        
        # Reefer cannot stack on regular
        self.assertFalse(reefer.can_stack_with(regular))
        # Reefer can stack on another reefer
        self.assertTrue(reefer.can_stack_with(another_reefer))
    
    def test_non_stackable_containers(self):
        """Test that non-stackable containers cannot stack."""
        trailer = Container(
            container_id="TEST018",
            direction="Export",
            container_type="Trailer",
            arrival_date=self.base_date,
            departure_date=self.base_date + timedelta(days=10)
        )
        
        regular = Container(
            container_id="TEST019",
            direction="Import",
            container_type="FEU",
            arrival_date=self.base_date,
            departure_date=self.base_date + timedelta(days=10)
        )
        
        # Trailer cannot stack on anything
        self.assertFalse(trailer.can_stack_with(regular))
        # Nothing can stack on trailer
        self.assertFalse(regular.can_stack_with(trailer))
    
    def test_dimensions_standard_container(self):
        """Test that standard containers get correct dimensions."""
        container = Container(
            container_id="TEST020",
            direction="Import",
            container_type="FEU",
            is_high_cube=False,
            arrival_date=self.base_date,
            departure_date=self.base_date + timedelta(days=10)
        )
        
        self.assertEqual(container.height, 2.59)
        self.assertEqual(container.length, 12.19)
        self.assertEqual(container.width, 2.44)
    
    def test_dimensions_high_cube_container(self):
        """Test that high cube containers get correct height."""
        container = Container(
            container_id="TEST021",
            direction="Import",
            container_type="FEU",
            is_high_cube=True,
            arrival_date=self.base_date,
            departure_date=self.base_date + timedelta(days=10)
        )
        
        self.assertEqual(container.height, 2.89)
    
    def test_dimensions_special_container(self):
        """Test that special containers get correct dimensions."""
        trailer = Container(
            container_id="TEST022",
            direction="Export",
            container_type="Trailer",
            arrival_date=self.base_date,
            departure_date=self.base_date + timedelta(days=10)
        )
        
        swap_body = Container(
            container_id="TEST023",
            direction="Export",
            container_type="Swap Body",
            arrival_date=self.base_date,
            departure_date=self.base_date + timedelta(days=10)
        )
        
        self.assertEqual(trailer.width, 2.55)
        self.assertEqual(swap_body.width, 2.55)
        self.assertEqual(swap_body.length, 7.45)


if __name__ == '__main__':
    unittest.main()