# simulation/training/scenarios/2_restack/__init__.py
"""Tier 2 -- Restack + load scenarios."""
from .unbury_for_truck import UnburyForTruck
from .unbury_for_train import UnburyForTrain

SCENARIOS = [UnburyForTruck, UnburyForTrain]
