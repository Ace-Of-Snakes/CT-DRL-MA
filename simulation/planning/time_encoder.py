# simulation/planning/time_encoder.py
"""Efficient weekly time encoder using sine/cosine circular encoding."""
import numpy as np
from typing import Tuple, Dict

from simulation.core.constants import (
    SECONDS_PER_DAY, SECONDS_PER_WEEK,
    SECONDS_PER_HOUR, SECONDS_PER_MINUTE,
)

TWO_PI: float = 2.0 * np.pi


class WeeklyTimeEncoder:
    """Encodes day+time into continuous angular representation for scheduling."""

    DAY_MAP: Dict[str, int] = {
        'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
        'friday': 4, 'saturday': 5, 'sunday': 6,
    }
    DAY_NAMES: Tuple[str, ...] = (
        'monday', 'tuesday', 'wednesday', 'thursday',
        'friday', 'saturday', 'sunday',
    )

    def encode(self, day_of_week: str, hour: int, minute: int) -> Dict[str, float]:
        """Encode day+time to angle/sin/cos/seconds dict."""
        day_idx = self.DAY_MAP.get(day_of_week.lower())
        if day_idx is None:
            raise ValueError(f"Invalid day: {day_of_week}")

        seconds = (
            day_idx * SECONDS_PER_DAY
            + hour * SECONDS_PER_HOUR
            + minute * SECONDS_PER_MINUTE
        )
        angle = (seconds / SECONDS_PER_WEEK) * TWO_PI

        return {
            'angle': angle,
            'sin': float(np.sin(angle)),
            'cos': float(np.cos(angle)),
            'seconds': seconds,
        }

    def decode(self, angle: float) -> Tuple[str, int, int]:
        """Decode angle (radians) back to (day, hour, minute)."""
        angle = angle % TWO_PI
        seconds = int((angle / TWO_PI) * SECONDS_PER_WEEK)

        day_idx = seconds // SECONDS_PER_DAY
        remaining = seconds % SECONDS_PER_DAY
        hour = remaining // SECONDS_PER_HOUR
        minute = (remaining % SECONDS_PER_HOUR) // SECONDS_PER_MINUTE

        return (self.DAY_NAMES[day_idx], hour, minute)

    def decode_from_sincos(self, sin_val: float, cos_val: float) -> Tuple[str, int, int]:
        """Decode from sine/cosine values."""
        angle = np.arctan2(sin_val, cos_val)
        if angle < 0:
            angle += TWO_PI
        return self.decode(angle)

    def subtract(self, timestamp1: str, timestamp2: str) -> Dict[str, float]:
        """Time difference (ts2 - ts1). Format: 'weekday-hour-minute'."""
        def _parse(ts: str):
            parts = ts.lower().split('-')
            if len(parts) != 3:
                raise ValueError(f"Invalid timestamp format: {ts}")
            return parts[0], int(parts[1]), int(parts[2])

        day1, h1, m1 = _parse(timestamp1)
        day2, h2, m2 = _parse(timestamp2)

        s1 = self.encode(day1, h1, m1)['seconds']
        s2 = self.encode(day2, h2, m2)['seconds']

        diff = s2 - s1
        if diff < 0:
            diff += SECONDS_PER_WEEK

        return {
            'seconds': diff,
            'minutes': diff / 60,
            'hours': diff / 3600,
            'days': diff / SECONDS_PER_DAY,
            'angle_diff': (s2 / SECONDS_PER_WEEK - s1 / SECONDS_PER_WEEK) * TWO_PI,
        }