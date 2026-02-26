# simulation/training/scenarios/_registry.py
"""Central registry of all tutorial scenarios.

Imports every tier's SCENARIOS list and builds the flat registries
(ALL_SCENARIOS, SCENARIO_BY_ID) plus tier definitions that match the
ordering in tutorial_runner._TIER_BOUNDARIES.
"""
from __future__ import annotations

import importlib
from typing import Dict, List, Tuple, Type

from simulation.training.scenarios._base import TutorialScenario

# ── Import tier modules ─────────────────────────────────────────────────
# Directory names start with digits, so we use importlib.
_tier_0 = importlib.import_module("simulation.training.scenarios.0_primitives")
_tier_1 = importlib.import_module("simulation.training.scenarios.1_chains")
_tier_2 = importlib.import_module("simulation.training.scenarios.2_restack")
_tier_3 = importlib.import_module("simulation.training.scenarios.3_generalization")
_tier_4 = importlib.import_module("simulation.training.scenarios.4_multi_step")
_tier_5 = importlib.import_module("simulation.training.scenarios.5_terminal_truck")
_tier_6 = importlib.import_module("simulation.training.scenarios.6_multi_vehicle")
_tier_7 = importlib.import_module("simulation.training.scenarios.7_bidirectional")
_tier_8 = importlib.import_module("simulation.training.scenarios.8_stress")

TIER_0: List[Type[TutorialScenario]] = _tier_0.SCENARIOS
TIER_1: List[Type[TutorialScenario]] = _tier_1.SCENARIOS
TIER_2: List[Type[TutorialScenario]] = _tier_2.SCENARIOS
TIER_3: List[Type[TutorialScenario]] = _tier_3.SCENARIOS
TIER_4: List[Type[TutorialScenario]] = _tier_4.SCENARIOS
TIER_5: List[Type[TutorialScenario]] = _tier_5.SCENARIOS
TIER_6: List[Type[TutorialScenario]] = _tier_6.SCENARIOS
TIER_7: List[Type[TutorialScenario]] = _tier_7.SCENARIOS
TIER_8: List[Type[TutorialScenario]] = _tier_8.SCENARIOS

# ── Flat list of all scenario classes ────────────────────────────────────
ALL_SCENARIO_CLASSES: List[Type[TutorialScenario]] = (
    TIER_0 + TIER_1 + TIER_2 + TIER_3 + TIER_4
    + TIER_5 + TIER_6 + TIER_7 + TIER_8
)

# ── Instantiated registries ─────────────────────────────────────────────
ALL_SCENARIOS: List[TutorialScenario] = [cls() for cls in ALL_SCENARIO_CLASSES]
SCENARIO_BY_ID: Dict[int, TutorialScenario] = {s.id: s for s in ALL_SCENARIOS}

# ── Repeatable / one-shot partitions ─────────────────────────────────────
ONE_SHOT_SCENARIOS: List[TutorialScenario] = [
    s for s in ALL_SCENARIOS if not s.repeatable
]
REPEATABLE_SCENARIOS: List[TutorialScenario] = [
    s for s in ALL_SCENARIOS if s.repeatable
]

# ── Tier definitions ─────────────────────────────────────────────────────
# Matches the 11-tier ordering from tutorial_runner._TIER_BOUNDARIES.
TIER_DEFS: List[Tuple[str, List[int]]] = [
    ("Primitives",          [1, 2, 3, 4]),
    ("Two-action chains",   [5, 6, 26, 27]),
    ("Full chains",         [7, 8]),
    ("Restack + load",      [9, 10]),
    ("Random generalize",   [11, 12]),
    ("Multi-step hard",     [13, 14]),
    ("Terminal truck",      [25, 22, 23, 24]),
    ("Multi-vehicle",       [15, 16]),
    ("Bidirectional train", [17]),
    ("Concurrent ops",      [18, 19]),
    ("Full complexity",     [20, 21]),
]
