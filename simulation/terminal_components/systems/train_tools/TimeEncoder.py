import numpy as np
from typing import Tuple, Dict

class WeeklyTimeEncoder:
    """Efficient weekly time encoder using sine/cosine circular encoding."""
    
    # Day mapping for fast lookup
    DAY_MAP = {
        'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
        'friday': 4, 'saturday': 5, 'sunday': 6
    }
    DAY_NAMES = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    
    def __init__(self):
        """Initialize with seconds in a week."""
        self.week_seconds = 7 * 24 * 60 * 60  # 604,800 seconds
        self.day_seconds = 24 * 60 * 60  # 86,400 seconds
        
    def encode(self, day_of_week: str, hour: int, minute: int) -> Dict[str, float]:
        """
        Encode time to sine/cosine values and angle.
        
        Args:
            day_of_week: Day name (case-insensitive)
            hour: Hour (0-23)
            minute: Minute (0-59)
            
        Returns:
            Dict with 'angle' (radians), 'sin', 'cos' values
        """
        # Get day index
        day_idx = self.DAY_MAP.get(day_of_week.lower())
        if day_idx is None:
            raise ValueError(f"Invalid day: {day_of_week}")
        
        # Calculate total seconds from start of week
        seconds = day_idx * self.day_seconds + hour * 3600 + minute * 60
        
        # Convert to angle (0 to 2π for full week)
        angle = (seconds / self.week_seconds) * 2 * np.pi
        
        return {
            'angle': angle,
            'sin': np.sin(angle),
            'cos': np.cos(angle),
            'seconds': seconds
        }
    
    def decode(self, angle: float) -> Tuple[str, int, int]:
        """
        Decode angle back to day, hour, minute.
        
        Args:
            angle: Angle in radians (0 to 2π)
            
        Returns:
            Tuple of (day_of_week, hour, minute)
        """
        # Normalize angle to [0, 2π]
        angle = angle % (2 * np.pi)
        
        # Convert to seconds
        seconds = int((angle / (2 * np.pi)) * self.week_seconds)
        
        # Extract components
        day_idx = seconds // self.day_seconds
        remaining = seconds % self.day_seconds
        
        hour = remaining // 3600
        minute = (remaining % 3600) // 60
        
        return (self.DAY_NAMES[day_idx], hour, minute)
    
    def decode_from_sincos(self, sin_val: float, cos_val: float) -> Tuple[str, int, int]:
        """
        Decode from sine/cosine values.
        
        Args:
            sin_val: Sine value
            cos_val: Cosine value
            
        Returns:
            Tuple of (day_of_week, hour, minute)
        """
        angle = np.arctan2(sin_val, cos_val)
        # Adjust to [0, 2π]
        if angle < 0:
            angle += 2 * np.pi
        return self.decode(angle)
    
    def subtract(self, timestamp1: str, timestamp2: str) -> Dict[str, float]:
        """
        Subtract timestamp1 from timestamp2 (timestamp2 - timestamp1).
        
        Args:
            timestamp1: String in format 'weekday-hour-minute' (e.g., 'monday-09-30')
            timestamp2: String in format 'weekday-hour-minute' (e.g., 'wednesday-14-45')
            
        Returns:
            Dict with difference in seconds, hours, days, and as angle
        """
        # Parse timestamp1
        parts1 = timestamp1.lower().split('-')
        if len(parts1) != 3:
            raise ValueError(f"Invalid timestamp format: {timestamp1}")
        day1, hour1, minute1 = parts1[0], int(parts1[1]), int(parts1[2])
        
        # Parse timestamp2
        parts2 = timestamp2.lower().split('-')
        if len(parts2) != 3:
            raise ValueError(f"Invalid timestamp format: {timestamp2}")
        day2, hour2, minute2 = parts2[0], int(parts2[1]), int(parts2[2])
        
        # Get seconds for each timestamp
        enc1 = self.encode(day1, hour1, minute1)
        enc2 = self.encode(day2, hour2, minute2)
        
        # Calculate difference
        diff_seconds = enc2['seconds'] - enc1['seconds']
        
        # Handle negative differences (wrap around week)
        if diff_seconds < 0:
            diff_seconds += self.week_seconds
        
        return {
            'seconds': diff_seconds,
            'minutes': diff_seconds / 60,
            'hours': diff_seconds / 3600,
            'days': diff_seconds / self.day_seconds,
            'angle_diff': enc2['angle'] - enc1['angle']
        }