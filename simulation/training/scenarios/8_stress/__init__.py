# simulation/training/scenarios/8_stress/__init__.py
"""Tier 8 -- Stress test scenarios."""
from .crowded_yard_pickup import CrowdedYardPickup
from .full_mini_day import FullMiniDay

SCENARIOS = [CrowdedYardPickup, FullMiniDay]
