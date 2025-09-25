import unittest
import json
import tempfile
from simulation2.core.Train import Train
from simulation.terminal_components.systems.tools.TimeEncoder import WeeklyTimeEncoder
from simulation.terminal_components.systems.tools.DPParser import DrivingPlanParser

class TestDrivingPlanParser(unittest.TestCase):
    
    def setUp(self):
        """Create minimal test JSON file."""
        self.test_data = {
            "driving_plan": {
                "details": {},
                "trains": {
                    "TEST1": {
                        "operator": "TestOp",
                        "plan": {
                            "0": {
                                "arrival": ["Monday", "10:00", "Monday", "10:30"],
                                "departure": ["Tuesday", "14:00", "Tuesday", "14:30"],
                                "mirrored_on": ["Wednesday"]
                            }
                        }
                    },
                    "TEST2": {
                        "operator": "TestOp2",
                        "plan": {
                            "0": {
                                "arrival": ["Friday", "22:00", "Friday", "22:30"],
                                "departure": ["Saturday", "06:00", "Saturday", "06:30"],
                                "mirrored_on": []
                            }
                        }
                    }
                }
            }
        }
        
        # Create temp file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.test_data, self.temp_file)
        self.temp_file.close()
        
        self.parser = DrivingPlanParser(self.temp_file.name)
    
    def tearDown(self):
        """Clean up temp file."""
        import os
        os.unlink(self.temp_file.name)
    
    def test_load_json(self):
        """Test JSON loading."""
        trains_data = self.parser._load_json()
        self.assertEqual(len(trains_data), 2)
        self.assertIn("TEST1", trains_data)
        self.assertIn("TEST2", trains_data)
    
    def test_parse_time(self):
        """Test time parsing to encoded values."""
        result = self.parser._parse_time("Monday", "10:30")
        self.assertIn('angle', result)
        self.assertIn('sin', result)
        self.assertIn('cos', result)
        self.assertIn('seconds', result)
        self.assertEqual(result['seconds'], 37800)  # Monday 10:30 in seconds
    
    def test_generate_train_id(self):
        """Test train ID generation."""
        id1 = self.parser._generate_train_id("TEST", "Op", 0)
        self.assertEqual(id1, "TEST_Op")
        
        id2 = self.parser._generate_train_id("TEST", "Op", 1)
        self.assertEqual(id2, "TEST_Op_1")
    
    def test_create_trains(self):
        """Test train creation including mirrored trains."""
        trains = self.parser.create_trains()
        
        # Should have 3 trains: 2 base + 1 mirrored
        self.assertEqual(len(trains), 3)
        
        # Check all are Train instances
        for train in trains:
            self.assertIsInstance(train, Train)
            self.assertEqual(train.wagons.__len__(), 29)
    
    def test_train_schedule_encoded(self):
        """Test that schedule_encoded is properly set."""
        trains = self.parser.create_trains()
        
        for train in trains:
            self.assertTrue(hasattr(train, 'schedule_encoded'))
            schedule = train.schedule_encoded
            
            # Check required fields
            self.assertIn('arrival', schedule)
            self.assertIn('departure', schedule)
            self.assertIn('operator', schedule)
            self.assertIn('stay_duration', schedule)
            
            # Check stay_duration structure
            self.assertIn('hours', schedule['stay_duration'])
            self.assertIn('seconds', schedule['stay_duration'])
            self.assertIsInstance(schedule['stay_duration']['hours'], float)
    
    def test_mirrored_train_offset(self):
        """Test that mirrored trains have correct day offset."""
        trains = self.parser.create_trains()
        
        # Find base and mirrored train
        base_train = next(t for t in trains if "TEST1_TestOp" in t.train_id and "_1" not in t.train_id)
        mirror_train = next(t for t in trains if "TEST1_TestOp_1" in t.train_id)
        
        # Base arrives Monday, mirror should arrive Wednesday
        encoder = WeeklyTimeEncoder()
        base_angle = base_train.schedule_encoded['arrival']['angle']
        mirror_angle = mirror_train.schedule_encoded['arrival']['angle']
        
        # Angle difference should be ~2 days worth
        angle_diff = mirror_angle - base_angle
        days_diff = (angle_diff / (2 * 3.14159)) * 7  # Convert radians to days
        self.assertAlmostEqual(days_diff, 2.0, places=1)
    
    def test_stay_duration_calculation(self):
        """Test stay duration is correctly calculated."""
        trains = self.parser.create_trains()
        test_train = trains[0]
        
        stay = test_train.schedule_encoded['stay_duration']
        
        # For TEST1: Monday 10:00 to Tuesday 14:00 = 28 hours
        if "TEST1" in test_train.train_id:
            self.assertAlmostEqual(stay['hours'], 28.0, places=1)
        # For TEST2: Friday 22:00 to Saturday 06:00 = 8 hours  
        elif "TEST2" in test_train.train_id:
            self.assertAlmostEqual(stay['hours'], 8.0, places=1)
    
    def test_invalid_day_handling(self):
        """Test handling of invalid/typo days in mirrored_on."""
        # Add train with typo
        self.test_data['driving_plan']['trains']['TEST3'] = {
            "operator": "TestOp3",
            "plan": {
                "0": {
                    "arrival": ["Monday", "10:00", "Monday", "10:30"],
                    "departure": ["Monday", "14:00", "Monday", "14:30"],
                    "mirrored_on": ["Tueday"]  # Typo
                }
            }
        }
        
        # Should handle typo (Tueday -> Tuesday) without crashing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_data, f)
            temp_path = f.name
        
        parser = DrivingPlanParser(temp_path)
        trains = parser.create_trains()
        
        # Should create trains despite typo
        self.assertGreater(len(trains), 0)
        
        import os
        os.unlink(temp_path)

if __name__ == "__main__":
    unittest.main()