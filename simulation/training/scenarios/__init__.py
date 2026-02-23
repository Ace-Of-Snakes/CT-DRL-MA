# simulation/training/scenarios/__init__.py
"""Tutorial scenario package.

Re-exports the main public API so callers can do::

    from simulation.training.scenarios import ALL_SCENARIOS, SCENARIO_BY_ID
"""
from simulation.training.scenarios._base import (
    ANCHOR_BAY,
    CONTAINER_SIZES,
    CONTAINER_SPECS,
    DEFAULT_DEPARTURE,
    MoveRecord,
    StepEvent,
    SWAP_BODY_SPECS,
    TUTORIAL_TIME,
    TutorialResult,
    TutorialScenario,
)
from simulation.training.scenarios._registry import (
    ALL_SCENARIO_CLASSES,
    ALL_SCENARIOS,
    ONE_SHOT_SCENARIOS,
    REPEATABLE_SCENARIOS,
    SCENARIO_BY_ID,
    TIER_DEFS,
)
from simulation.training.scenarios._visualization import visualize_scenario

__all__ = [
    # Base
    "TutorialScenario",
    "TutorialResult",
    "MoveRecord",
    "StepEvent",
    # Constants
    "TUTORIAL_TIME",
    "DEFAULT_DEPARTURE",
    "ANCHOR_BAY",
    "CONTAINER_SPECS",
    "SWAP_BODY_SPECS",
    "CONTAINER_SIZES",
    # Registry
    "ALL_SCENARIO_CLASSES",
    "ALL_SCENARIOS",
    "SCENARIO_BY_ID",
    "ONE_SHOT_SCENARIOS",
    "REPEATABLE_SCENARIOS",
    "TIER_DEFS",
    # Visualization
    "visualize_scenario",
]
