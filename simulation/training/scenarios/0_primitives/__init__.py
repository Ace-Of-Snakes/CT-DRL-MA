# simulation/training/scenarios/0_primitives/__init__.py
"""Tier 0 -- Primitive single-action scenarios."""
from .park_truck import ParkTruck
from .train_import import TrainImport
from .yard_to_truck import YardToTruck
from .yard_to_train import YardToTrain

SCENARIOS = [ParkTruck, TrainImport, YardToTruck, YardToTrain]
