# simulation/training/scenarios/6_multi_vehicle/__init__.py
"""Tier 6 -- Multi-vehicle integration scenarios."""
from .multi_delivery_park import MultiDeliveryPark
from .delivery_to_train_pipeline import DeliveryToTrainPipeline

SCENARIOS = [MultiDeliveryPark, DeliveryToTrainPipeline]
