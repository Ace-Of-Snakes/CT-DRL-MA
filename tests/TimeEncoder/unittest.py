import unittest
from simulation.terminal_components.systems.tools.TimeEncoder import WeeklyTimeEncoder

class TestWeeklyTimeEncoder(unittest.TestCase):
    def setUp(self):
        self.enc = WeeklyTimeEncoder()
    
    def test_encode_decode_roundtrip(self):
        """Test encoding and decoding returns original values."""
        result = self.enc.encode('wednesday', 14, 30)
        day, hour, minute = self.enc.decode(result['angle'])
        self.assertEqual((day, hour, minute), ('wednesday', 14, 30))
    
    def test_sincos_decode(self):
        """Test decoding from sin/cos values."""
        result = self.enc.encode('monday', 0, 0)
        day, hour, minute = self.enc.decode_from_sincos(result['sin'], result['cos'])
        self.assertEqual((day, hour, minute), ('monday', 0, 0))
    
    def test_subtract_forward(self):
        """Test subtracting timestamps in same week."""
        diff = self.enc.subtract('monday-09-30', 'wednesday-14-45')
        self.assertAlmostEqual(diff['hours'], 53.25, places=2)
    
    def test_subtract_wraparound(self):
        """Test subtracting with weekly wraparound."""
        diff = self.enc.subtract('friday-20-00', 'monday-08-00')
        self.assertAlmostEqual(diff['hours'], 60.0, places=1)
    
    def test_invalid_day(self):
        """Test invalid day raises error."""
        with self.assertRaises(ValueError):
            self.enc.encode('notaday', 10, 30)
    
    def test_invalid_timestamp_format(self):
        """Test invalid timestamp format raises error."""
        with self.assertRaises(ValueError):
            self.enc.subtract('monday-9:30', 'wednesday-14-45')
    
    def test_case_insensitive(self):
        """Test day names are case insensitive."""
        r1 = self.enc.encode('MONDAY', 10, 0)
        r2 = self.enc.encode('monday', 10, 0)
        self.assertEqual(r1['angle'], r2['angle'])

unittest.main()