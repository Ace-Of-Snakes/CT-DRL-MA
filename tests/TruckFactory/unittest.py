import unittest
import tempfile
import pickle
import os
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import gaussian_kde
from simulation.terminal_components.vehicles.TruckFactory import TruckFactory, TRUCK_LENGTH
from simulation.terminal_components.storage_units.Container import Container


class TestTruckFactory(unittest.TestCase):
    """Unit tests for TruckFactory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test KDE models
        self.temp_dir = tempfile.mkdtemp()
        self.delivery_dir = os.path.join(self.temp_dir, "delivery")
        self.pickup_dir = os.path.join(self.temp_dir, "pickup")
        os.makedirs(self.delivery_dir)
        os.makedirs(self.pickup_dir)
        
        # Create mock KDE models
        self._create_mock_kde_models()
        
        # Initialize factory with test data
        self.factory = TruckFactory(kde_folder=self.temp_dir)
        
        # Create test containers
        self.test_containers = self._create_test_containers()
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_mock_kde_models(self):
        """Create mock KDE models for testing."""
        # Generate sample data for KDE (arrival times in hours)
        np.random.seed(42)
        
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            # Delivery times peak around 9am
            delivery_data = np.random.normal(9, 2, 100)
            delivery_kde = gaussian_kde(delivery_data)
            
            with open(os.path.join(self.delivery_dir, f"{day}_kde.pkl"), 'wb') as f:
                pickle.dump(delivery_kde, f)
            
            # Pickup times peak around 3pm
            pickup_data = np.random.normal(15, 2, 100)
            pickup_kde = gaussian_kde(pickup_data)
            
            with open(os.path.join(self.pickup_dir, f"{day}_kde.pkl"), 'wb') as f:
                pickle.dump(pickup_kde, f)
    
    def _create_test_containers(self, n=10):
        """Create test containers."""
        containers = []
        base_date = datetime(2024, 1, 15)
        
        for i in range(n):
            container = Container(
                container_id=f"TEST{i:04d}",
                direction="Import" if i % 2 == 0 else "Export",
                container_type="FEU",
                arrival_date=base_date,
                departure_date=base_date + timedelta(days=3),
                length_m=12.19 if i % 3 == 0 else 6.1  # Mix of 40ft and 20ft
            )
            containers.append(container)
        
        return containers
    
    def test_initialization(self):
        """Test factory initialization."""
        self.assertIsNotNone(self.factory)
        self.assertIsInstance(self.factory.delivery_kde_models, dict)
        self.assertIsInstance(self.factory.pickup_kde_models, dict)
        self.assertEqual(self.factory._id_counter, 0)
    
    def test_kde_loading(self):
        """Test KDE model loading."""
        # Check that models were loaded
        self.assertEqual(len(self.factory.delivery_kde_models), 5)
        self.assertEqual(len(self.factory.pickup_kde_models), 5)
        
        # Check specific days
        self.assertIn('monday', self.factory.delivery_kde_models)
        self.assertIn('friday', self.factory.pickup_kde_models)
        
        # Check model types
        self.assertIsInstance(
            self.factory.delivery_kde_models['monday'], 
            gaussian_kde
        )
    
    def test_truck_length_fixed(self):
        """Test that all trucks are 80 feet (24.4m)."""
        trucks = self.factory.generate_trucks(
            day_of_week="Monday",
            delivery_containers=self.test_containers[:5],
            base_date=datetime(2024, 1, 15)
        )
        
        for truck in trucks:
            self.assertEqual(truck.max_length, TRUCK_LENGTH)
            self.assertEqual(truck.max_length, 24.4)
    
    def test_delivery_truck_generation(self):
        """Test generation of delivery trucks."""
        delivery_containers = self.test_containers[:5]
        
        trucks = self.factory.generate_trucks(
            day_of_week="Monday",
            delivery_containers=delivery_containers,
            base_date=datetime(2024, 1, 15)
        )
        
        self.assertGreater(len(trucks), 0)
        
        for truck in trucks:
            # Check truck properties
            self.assertTrue(truck.is_delivery_truck)
            self.assertFalse(truck.is_pickup_truck)
            self.assertIsNotNone(truck.arrival_time)
            self.assertIsNotNone(truck.parking_spot)
            
            # Check containers are loaded
            self.assertGreater(len(truck.containers), 0)
            
            # Check truck capacity not exceeded
            total_length = sum(c.length_m for c in truck.containers)
            self.assertLessEqual(total_length, TRUCK_LENGTH)
    
    def test_pickup_truck_generation(self):
        """Test generation of pickup trucks."""
        pickup_containers = self.test_containers[5:]
        
        trucks = self.factory.generate_trucks(
            day_of_week="Tuesday",
            pickup_containers=pickup_containers,
            base_date=datetime(2024, 1, 16)
        )
        
        self.assertGreater(len(trucks), 0)
        
        for truck in trucks:
            # Check truck properties
            self.assertTrue(truck.is_pickup_truck)
            self.assertFalse(truck.is_delivery_truck)
            self.assertIsNotNone(truck.arrival_time)
            
            # Check pickup IDs are set, not actual containers
            self.assertEqual(len(truck.containers), 0)
            self.assertGreater(len(truck.pickup_container_ids), 0)
    
    def test_container_packing_efficiency(self):
        """Test that container packing is efficient."""
        # Create containers that should fit in one truck
        small_containers = [
            Container(
                container_id=f"SMALL{i:03d}",
                direction="Export",
                container_type="TEU",
                arrival_date=datetime(2024, 1, 15),
                departure_date=datetime(2024, 1, 18),
                length_m=6.1  # 20ft containers
            )
            for i in range(3)  # 3 * 6.1 = 18.3m < 24.4m
        ]
        
        trucks = self.factory.generate_trucks(
            day_of_week="Wednesday",
            delivery_containers=small_containers
        )
        
        # Should fit in one truck
        self.assertEqual(len(trucks), 1)
        self.assertEqual(len(trucks[0].containers), 3)
    
    def test_container_packing_overflow(self):
        """Test that containers overflow to new trucks when needed."""
        # Create containers that won't fit in one truck
        large_containers = [
            Container(
                container_id=f"LARGE{i:03d}",
                direction="Export",
                container_type="FEU",
                arrival_date=datetime(2024, 1, 15),
                departure_date=datetime(2024, 1, 18),
                length_m=12.19  # 40ft containers
            )
            for i in range(3)  # 3 * 12.19 = 36.57m > 24.4m
        ]
        
        trucks = self.factory.generate_trucks(
            day_of_week="Thursday",
            delivery_containers=large_containers
        )
        
        # Should need at least 2 trucks
        self.assertGreaterEqual(len(trucks), 2)
        
        # All containers should be loaded
        total_loaded = sum(len(t.containers) for t in trucks)
        self.assertEqual(total_loaded, 3)
    
    def test_arrival_time_sampling(self):
        """Test that arrival times are sampled correctly."""
        base_date = datetime(2024, 1, 15, 0, 0)  # Monday midnight
        
        trucks = self.factory.generate_trucks(
            day_of_week="Monday",
            delivery_containers=self.test_containers[:5],
            base_date=base_date
        )
        
        for truck in trucks:
            # Check arrival is on the same day
            self.assertEqual(truck.arrival_time.date(), base_date.date())
            
            # Check arrival is within business hours (6am - 10pm)
            hour = truck.arrival_time.hour
            self.assertGreaterEqual(hour, 6)
            self.assertLessEqual(hour, 22)
    
    def test_no_kde_fallback(self):
        """Test fallback when no KDE model exists."""
        # Saturday has no KDE model in our test setup
        trucks = self.factory.generate_trucks(
            day_of_week="Saturday",
            delivery_containers=self.test_containers[:2],
            base_date=datetime(2024, 1, 20)
        )
        
        self.assertGreater(len(trucks), 0)
        
        # Should still have valid arrival times
        for truck in trucks:
            self.assertIsNotNone(truck.arrival_time)
            hour = truck.arrival_time.hour
            self.assertGreaterEqual(hour, 6)
            self.assertLessEqual(hour, 22)
    
    def test_unique_truck_ids(self):
        """Test that truck IDs are unique."""
        trucks1 = self.factory.generate_trucks(
            day_of_week="Monday",
            delivery_containers=self.test_containers[:3]
        )
        
        trucks2 = self.factory.generate_trucks(
            day_of_week="Monday",
            delivery_containers=self.test_containers[3:6]
        )
        
        all_ids = [t.truck_id for t in trucks1 + trucks2]
        self.assertEqual(len(all_ids), len(set(all_ids)))  # All unique
    
    def test_mixed_delivery_pickup(self):
        """Test generating both delivery and pickup trucks."""
        delivery = self.test_containers[:3]
        pickup = self.test_containers[3:6]
        
        trucks = self.factory.generate_trucks(
            day_of_week="Friday",
            delivery_containers=delivery,
            pickup_containers=pickup,
            base_date=datetime(2024, 1, 19)
        )
        
        delivery_trucks = [t for t in trucks if t.is_delivery_truck]
        pickup_trucks = [t for t in trucks if t.is_pickup_truck]
        
        self.assertGreater(len(delivery_trucks), 0)
        self.assertGreater(len(pickup_trucks), 0)
        
        # Check no truck is both delivery and pickup
        for truck in trucks:
            self.assertNotEqual(truck.is_delivery_truck, truck.is_pickup_truck)
    
    def test_parking_spot_assignment(self):
        """Test that parking spots are assigned."""
        trucks = self.factory.generate_trucks(
            day_of_week="Monday",
            delivery_containers=self.test_containers[:3],
            parking_spot_prefix="TEST"
        )
        
        for truck in trucks:
            self.assertIsNotNone(truck.parking_spot)
            self.assertTrue(truck.parking_spot.startswith("TEST"))
    
    def test_kde_summary(self):
        """Test KDE summary method."""
        summary = self.factory.get_kde_summary()
        
        self.assertIn('delivery_models', summary)
        self.assertIn('pickup_models', summary)
        
        self.assertEqual(summary['delivery_models']['count'], 5)
        self.assertEqual(summary['pickup_models']['count'], 5)
        
        self.assertIn('monday', summary['delivery_models']['available_days'])
        self.assertIn('friday', summary['pickup_models']['available_days'])
    
    def test_empty_container_lists(self):
        """Test handling of empty container lists."""
        trucks = self.factory.generate_trucks(
            day_of_week="Monday",
            delivery_containers=[],
            pickup_containers=[]
        )
        
        self.assertEqual(len(trucks), 0)
    
    def test_none_container_lists(self):
        """Test handling of None container lists."""
        trucks = self.factory.generate_trucks(
            day_of_week="Monday",
            delivery_containers=None,
            pickup_containers=None
        )
        
        self.assertEqual(len(trucks), 0)


if __name__ == '__main__':
    unittest.main()