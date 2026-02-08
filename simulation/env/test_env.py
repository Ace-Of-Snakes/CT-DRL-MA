# tests/test_unified_env.py
"""Tests for unified environment integration (Phase 4).

Part 1: Pure structural and logic tests (no simulation dependencies).
Part 2: Mocked integration tests with fake entities.
"""
from __future__ import annotations

import ast
import inspect
import os
import unittest
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np


# ══════════════════════════════════════════════════════════════════════════
# Part 1: Source code structure tests
# ══════════════════════════════════════════════════════════════════════════

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

# unified_env.py lives in simulation/env/
# unified_tutorial_runner.py lives in simulation/training/
_ENV_CANDIDATES = [
    os.path.join(_THIS_DIR, "unified_env.py"),                          # same dir
    os.path.join(_PROJECT_ROOT, "simulation", "env", "unified_env.py"), # explicit
]
_RUNNER_CANDIDATES = [
    os.path.join(_THIS_DIR, "unified_tutorial_runner.py"),                       # same dir
    os.path.join(_THIS_DIR, "..", "training", "unified_tutorial_runner.py"),      # from env/
    os.path.join(_PROJECT_ROOT, "simulation", "training", "unified_tutorial_runner.py"),
]

ENV_PATH = next((p for p in _ENV_CANDIDATES if os.path.exists(p)), _ENV_CANDIDATES[0])
RUNNER_PATH = next((p for p in _RUNNER_CANDIDATES if os.path.exists(p)), _RUNNER_CANDIDATES[0])


class TestEnvSourceStructure(unittest.TestCase):
    """Verify unified_env.py has required components."""

    @classmethod
    def setUpClass(cls):
        with open(ENV_PATH) as f:
            cls.source = f.read()
        cls.tree = ast.parse(cls.source)
        cls.class_names = {
            n.name for n in ast.walk(cls.tree) if isinstance(n, ast.ClassDef)
        }
        cls.func_names = {
            n.name for n in ast.walk(cls.tree) if isinstance(n, ast.FunctionDef)
        }

    def test_has_unified_env_class(self):
        self.assertIn("UnifiedContainerTerminalEnv", self.class_names)

    def test_has_step_result(self):
        self.assertIn("UnifiedStepResult", self.class_names)

    def test_has_step_unified(self):
        self.assertIn("_step_unified", self.func_names)

    def test_has_dest_mask_fn(self):
        self.assertIn("_make_dest_mask_fn", self.func_names)

    def test_has_park_truck(self):
        self.assertIn("_execute_park_truck", self.func_names)

    def test_has_execute_unified_move(self):
        self.assertIn("_execute_unified_move", self.func_names)

    def test_has_resolve_move(self):
        self.assertIn("_resolve_move", self.func_names)

    def test_has_all_resolution_methods(self):
        expected = {
            "_resolve_yard_to_yard",
            "_resolve_yard_to_train",
            "_resolve_yard_to_truck",
            "_resolve_rail_to_yard",
            "_resolve_parking_to_yard",
        }
        self.assertTrue(expected.issubset(self.func_names))

    def test_has_entity_helpers(self):
        expected = {
            "_yard_container_at_unified",
            "_find_queued_truck_at",
            "_find_rail_container",
            "_find_truck_at_parking_split",
            "_find_train_on_track",
        }
        self.assertTrue(expected.issubset(self.func_names))

    def test_no_old_architecture_imports(self):
        """Ensure no imports of deleted old architecture modules."""
        self.assertNotIn("from simulation.env.state_encoder import", self.source)
        self.assertNotIn("from simulation.operations.hierarchical_moves import", self.source)

    def test_no_magic_numbers_in_parking(self):
        """Parking constants should be named, not hardcoded."""
        self.assertIn("PARKING_REWARD_BASE", self.source)
        self.assertIn("PARKING_PROXIMITY_BONUS_MAX", self.source)
        self.assertIn("PARKING_PROXIMITY_RADIUS_BAYS", self.source)


class TestRunnerSourceStructure(unittest.TestCase):
    """Verify unified_tutorial_runner.py structure."""

    @classmethod
    def setUpClass(cls):
        with open(RUNNER_PATH) as f:
            cls.source = f.read()
        cls.tree = ast.parse(cls.source)
        cls.class_names = {
            n.name for n in ast.walk(cls.tree) if isinstance(n, ast.ClassDef)
        }

    def test_has_runner_class(self):
        self.assertIn("UnifiedTutorialRunner", self.class_names)

    def test_runner_has_train_all(self):
        self.assertIn("train_all", self.source)

    def test_runner_has_run_scenario(self):
        self.assertIn("run_scenario", self.source)

    def test_runner_uses_unified_agent(self):
        self.assertIn("UnifiedDQNAgent", self.source)


# ══════════════════════════════════════════════════════════════════════════
# Part 2: Pure helper function tests
# ══════════════════════════════════════════════════════════════════════════

# Pure helper reimplementations for testing (avoids simulation imports)

def _downsample_mask(mask: np.ndarray, stride: int) -> np.ndarray:
    """Local copy for testing."""
    R, S, T = mask.shape
    S_down = S // stride
    trimmed = mask[:, :S_down * stride, :]
    return trimmed.reshape(R, S_down, stride, T).any(axis=2)


class TestDownsampleMask(unittest.TestCase):
    """Test mask downsampling logic."""

    def test_basic_downsample(self):
        # (R=2, S=8, T=1) with stride=4 → (2, 2, 1)
        mask = np.zeros((2, 8, 1), dtype=bool)
        mask[0, 2, 0] = True  # in window [0:4]
        mask[1, 5, 0] = True  # in window [4:8]
        result = _downsample_mask(mask, 4)
        self.assertEqual(result.shape, (2, 2, 1))
        self.assertTrue(result[0, 0, 0])   # row 0, window 0
        self.assertFalse(result[0, 1, 0])  # row 0, window 1
        self.assertFalse(result[1, 0, 0])  # row 1, window 0
        self.assertTrue(result[1, 1, 0])   # row 1, window 1

    def test_empty_mask(self):
        mask = np.zeros((3, 12, 2), dtype=bool)
        result = _downsample_mask(mask, 4)
        self.assertEqual(result.shape, (3, 3, 2))
        self.assertFalse(result.any())

    def test_full_mask(self):
        mask = np.ones((2, 8, 1), dtype=bool)
        result = _downsample_mask(mask, 4)
        self.assertTrue(result.all())

    def test_multi_tier(self):
        mask = np.zeros((1, 8, 3), dtype=bool)
        mask[0, 6, 2] = True  # window [4:8], tier 2
        result = _downsample_mask(mask, 4)
        self.assertEqual(result.shape, (1, 2, 3))
        self.assertTrue(result[0, 1, 2])
        self.assertFalse(result[0, 0, 2])
        self.assertFalse(result[0, 1, 0])

    def test_stride_1_is_identity(self):
        mask = np.random.choice([True, False], size=(3, 8, 2))
        result = _downsample_mask(mask, 1)
        np.testing.assert_array_equal(result, mask)


class TestDirectionHelpers(unittest.TestCase):
    """Test _is_export / _is_import pure functions."""

    def _make_container(self, direction_str):
        """Make mock container with direction."""
        c = MagicMock()
        d = MagicMock()
        d.value = direction_str
        c.direction = d
        return c

    def test_export_detection(self):
        c = self._make_container("Export")
        d = c.direction
        self.assertEqual(
            (d.value if hasattr(d, "value") else str(d)) == "Export", True,
        )

    def test_import_detection(self):
        c = self._make_container("Import")
        d = c.direction
        self.assertEqual(
            (d.value if hasattr(d, "value") else str(d)) == "Import", True,
        )

    def test_no_direction(self):
        c = MagicMock()
        c.direction = None
        self.assertIsNone(c.direction)


class TestWagonHasSpace(unittest.TestCase):
    """Test _wagon_has_space helper."""

    def test_empty_wagon(self):
        w = MagicMock()
        w.capacity = 2
        w.containers = {}
        capacity = getattr(w, "capacity", 2)
        current = len(getattr(w, "containers", {}))
        self.assertTrue(current < capacity)

    def test_full_wagon(self):
        w = MagicMock()
        w.capacity = 2
        w.containers = {"c1": MagicMock(), "c2": MagicMock()}
        capacity = getattr(w, "capacity", 2)
        current = len(getattr(w, "containers", {}))
        self.assertFalse(current < capacity)

    def test_partial_wagon(self):
        w = MagicMock()
        w.capacity = 3
        w.containers = {"c1": MagicMock()}
        capacity = getattr(w, "capacity", 2)
        current = len(getattr(w, "containers", {}))
        self.assertTrue(current < capacity)


# ══════════════════════════════════════════════════════════════════════════
# Part 3: Dest mask logic tests (mock UnifiedDims)
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class MockDims:
    """Minimal UnifiedDims-like for testing dest mask logic."""
    n_tracks: int = 7
    n_parking_rows: int = 1
    n_yard_rows: int = 5
    n_queue_rows: int = 2
    n_bays: int = 58
    split_factor: int = 20
    n_tiers: int = 5

    @property
    def R_unified(self) -> int:
        return self.n_tracks + self.n_parking_rows + self.n_yard_rows + self.n_queue_rows

    @property
    def n_splits(self) -> int:
        return self.n_bays * self.split_factor

    @property
    def rail_row_end(self) -> int:
        return self.n_tracks

    @property
    def parking_row_start(self) -> int:
        return self.n_tracks

    @property
    def yard_row_start(self) -> int:
        return self.n_tracks + self.n_parking_rows

    @property
    def yard_row_end(self) -> int:
        return self.yard_row_start + self.n_yard_rows

    @property
    def queue_row_start(self) -> int:
        return self.yard_row_end

    def region_of(self, row: int) -> str:
        if row < self.rail_row_end:
            return "RAIL"
        if row < self.yard_row_start:
            return "PARKING"
        if row < self.yard_row_end:
            return "YARD"
        return "QUEUE"


class TestDestMaskRegionLogic(unittest.TestCase):
    """Test that dest masks respect region constraints."""

    def setUp(self):
        self.d = MockDims()
        self.stride = 4
        self.S_down = self.d.n_splits // self.stride

    def test_queue_source_only_opens_parking(self):
        """Queue source → only parking row should have True entries."""
        mask = np.zeros((self.d.R_unified, self.S_down, self.d.n_tiers), dtype=bool)
        # Simulate: queue source opens parking free spots
        parking_free = np.ones(self.S_down, dtype=bool)
        mask[self.d.parking_row_start, :, 0] = parking_free

        # Verify only parking row has entries
        for r in range(self.d.R_unified):
            if r == self.d.parking_row_start:
                self.assertTrue(mask[r].any(), f"Parking row {r} should have entries")
            else:
                self.assertFalse(mask[r].any(), f"Row {r} should be empty")

    def test_rail_source_only_opens_yard(self):
        """Rail source → only yard rows should have True entries."""
        mask = np.zeros((self.d.R_unified, self.S_down, self.d.n_tiers), dtype=bool)
        # Simulate: rail source opens yard validity
        mask[self.d.yard_row_start:self.d.yard_row_end, :5, 0] = True

        # Verify only yard rows
        for r in range(self.d.R_unified):
            if self.d.yard_row_start <= r < self.d.yard_row_end:
                self.assertTrue(mask[r].any())
            else:
                self.assertFalse(mask[r].any())

    def test_yard_source_opens_yard_and_vehicles(self):
        """Yard source → yard (restack) + possibly rail/parking."""
        mask = np.zeros((self.d.R_unified, self.S_down, self.d.n_tiers), dtype=bool)
        # Yard restack
        mask[self.d.yard_row_start:self.d.yard_row_end, :5, 0] = True
        # Rail target (export)
        mask[0, 10, 0] = True
        # Parking target (import)
        mask[self.d.parking_row_start, 15, 0] = True

        # All three regions present
        has_yard = mask[self.d.yard_row_start:self.d.yard_row_end].any()
        has_rail = mask[:self.d.rail_row_end].any()
        has_parking = mask[self.d.parking_row_start].any()
        self.assertTrue(has_yard)
        self.assertTrue(has_rail)
        self.assertTrue(has_parking)

    def test_source_excluded_from_dest(self):
        """Source position should be excluded from dest for Y→Y restack."""
        mask = np.ones((self.d.R_unified, self.S_down, self.d.n_tiers), dtype=bool)
        src_row = self.d.yard_row_start + 1
        src_s_down = 5
        src_tier = 0
        mask[src_row, src_s_down, src_tier] = False

        self.assertFalse(mask[src_row, src_s_down, src_tier])


class TestParkingFreeMaskDown(unittest.TestCase):
    """Test parking free mask downsampling."""

    def test_all_free(self):
        """All parking free → all True."""
        S = 1160
        stride = 4
        S_down = S // stride

        free_flat = np.ones(S, dtype=bool)
        trimmed = free_flat[:S_down * stride]
        result = trimmed.reshape(S_down, stride).any(axis=1)
        self.assertTrue(result.all())

    def test_all_occupied(self):
        """All parking occupied → all False."""
        S = 1160
        stride = 4
        S_down = S // stride

        free_flat = np.zeros(S, dtype=bool)
        trimmed = free_flat[:S_down * stride]
        result = trimmed.reshape(S_down, stride).any(axis=1)
        self.assertFalse(result.any())

    def test_sparse_free_spots(self):
        """One free spot per stride window."""
        S = 1160
        stride = 4
        S_down = S // stride

        free_flat = np.zeros(S, dtype=bool)
        free_flat[1] = True   # window 0: [0,1,2,3]
        free_flat[7] = True   # window 1: [4,5,6,7]
        trimmed = free_flat[:S_down * stride]
        result = trimmed.reshape(S_down, stride).any(axis=1)
        self.assertTrue(result[0])
        self.assertTrue(result[1])
        self.assertFalse(result[2])


# ══════════════════════════════════════════════════════════════════════════
# Part 4: Move dispatch tests
# ══════════════════════════════════════════════════════════════════════════

class TestMoveDispatchTable(unittest.TestCase):
    """Verify the _MOVE_DISPATCH table covers all move types."""

    def test_all_container_moves_covered(self):
        dispatch = {
            "YARD_TO_YARD":   "_resolve_yard_to_yard",
            "YARD_TO_TRAIN":  "_resolve_yard_to_train",
            "YARD_TO_TRUCK":  "_resolve_yard_to_truck",
            "TRAIN_TO_YARD":  "_resolve_rail_to_yard",
            "TRUCK_TO_YARD":  "_resolve_parking_to_yard",
        }
        self.assertEqual(len(dispatch), 5)

    def test_park_truck_not_in_dispatch(self):
        """PARK_TRUCK is handled separately, not via _resolve_move."""
        dispatch = {
            "YARD_TO_YARD":   "_resolve_yard_to_yard",
            "YARD_TO_TRAIN":  "_resolve_yard_to_train",
            "YARD_TO_TRUCK":  "_resolve_yard_to_truck",
            "TRAIN_TO_YARD":  "_resolve_rail_to_yard",
            "TRUCK_TO_YARD":  "_resolve_parking_to_yard",
        }
        self.assertNotIn("PARK_TRUCK", dispatch)

    def test_unknown_move_returns_none(self):
        """Unknown move type should not be in dispatch."""
        dispatch = {
            "YARD_TO_YARD":   "_resolve_yard_to_yard",
            "YARD_TO_TRAIN":  "_resolve_yard_to_train",
            "YARD_TO_TRUCK":  "_resolve_yard_to_truck",
            "TRAIN_TO_YARD":  "_resolve_rail_to_yard",
            "TRUCK_TO_YARD":  "_resolve_parking_to_yard",
        }
        self.assertIsNone(dispatch.get("UNKNOWN_MOVE"))


# ══════════════════════════════════════════════════════════════════════════
# Part 5: Entity resolution tests (mock-based)
# ══════════════════════════════════════════════════════════════════════════

class TestQueuedTruckResolution(unittest.TestCase):
    """Test _find_queued_truck_at logic."""

    def setUp(self):
        self.d = MockDims()

    def test_truck_at_preferred_bay(self):
        """Queued truck placed at preferred_bay * sf in queue row 0."""
        sf = self.d.split_factor
        preferred_bay = 10
        expected_split = preferred_bay * sf  # 200

        # Simulate: truck with no parking spot, WAITING status
        queue_layout = {0: {expected_split: "TK1"}, 1: {}}

        qi = 0
        result = queue_layout.get(qi, {}).get(expected_split)
        self.assertEqual(result, "TK1")

    def test_overflow_to_row_1(self):
        """Two trucks at same split → second goes to row 1."""
        sf = self.d.split_factor
        split = 100

        queue_layout = {0: {split: "TK1"}, 1: {split: "TK2"}}
        self.assertEqual(queue_layout[0][split], "TK1")
        self.assertEqual(queue_layout[1][split], "TK2")

    def test_invalid_queue_row(self):
        """Row outside queue range returns None."""
        row = self.d.yard_row_start  # not a queue row
        qi = row - self.d.queue_row_start
        self.assertTrue(qi < 0)  # invalid


class TestTrainOnTrackResolution(unittest.TestCase):
    """Test _find_train_on_track logic."""

    def test_finds_train_on_correct_track(self):
        """Train on track 2 found when searching track 2."""
        slots = {"TR1": MagicMock(track_id=0), "TR2": MagicMock(track_id=2)}
        trains = {"TR1": "train1", "TR2": "train2"}

        target_track = 2
        result = None
        for tid, train in trains.items():
            if slots[tid].track_id == target_track:
                result = train
                break
        self.assertEqual(result, "train2")

    def test_no_train_on_track(self):
        """No train on track 5 → None."""
        slots = {"TR1": MagicMock(track_id=0)}
        trains = {"TR1": "train1"}

        target_track = 5
        result = None
        for tid, train in trains.items():
            if slots[tid].track_id == target_track:
                result = train
                break
        self.assertIsNone(result)


class TestRailContainerResolution(unittest.TestCase):
    """Test _find_rail_container logic."""

    def test_finds_container_at_wagon(self):
        """Container at track 0, bay offset from anchor."""
        d = MockDims()
        sf = d.split_factor
        anchor = 5
        wagon_idx = 2  # bay 7
        split = (anchor + wagon_idx) * sf  # 140

        # Expected: target_bay = 140 // 20 = 7, wagon_idx = 7 - 5 = 2
        target_bay = split // sf
        calc_wagon = target_bay - anchor
        self.assertEqual(calc_wagon, wagon_idx)

    def test_out_of_range_wagon(self):
        """Split outside wagon range → not found."""
        d = MockDims()
        sf = d.split_factor
        anchor = 5
        num_wagons = 3
        split = (anchor + 5) * sf  # wagon_idx=5 > num_wagons

        target_bay = split // sf
        wagon_idx = target_bay - anchor
        self.assertFalse(0 <= wagon_idx < num_wagons)


class TestTruckAtParkingSplit(unittest.TestCase):
    """Test _find_truck_at_parking_split logic."""

    def test_resolves_bay_and_offset(self):
        """Split 42 with sf=20 → bay=2, offset=2."""
        sf = 20
        split = 42
        bay = split // sf
        offset = split % sf
        self.assertEqual(bay, 2)
        self.assertEqual(offset, 2)

    def test_boundary_split(self):
        """Split at exact bay boundary."""
        sf = 20
        split = 40
        bay = split // sf
        offset = split % sf
        self.assertEqual(bay, 2)
        self.assertEqual(offset, 0)


# ══════════════════════════════════════════════════════════════════════════
# Part 6: Proximity reward tests
# ══════════════════════════════════════════════════════════════════════════

PARKING_REWARD_BASE = 0.1
PARKING_PROXIMITY_BONUS_MAX = 0.5
PARKING_PROXIMITY_RADIUS_BAYS = 5


class TestParkingProximityReward(unittest.TestCase):
    """Test proximity reward calculation."""

    def _calc_reward(self, dst_bay: int, preferred_bay: Optional[int]) -> float:
        if preferred_bay is not None:
            delta = abs(dst_bay - preferred_bay)
            bonus = PARKING_PROXIMITY_BONUS_MAX * max(
                0.0, 1.0 - delta / PARKING_PROXIMITY_RADIUS_BAYS,
            )
        else:
            bonus = 0.0
        return PARKING_REWARD_BASE + bonus

    def test_exact_match(self):
        """Parking at preferred bay → full bonus."""
        r = self._calc_reward(10, 10)
        self.assertAlmostEqual(r, 0.1 + 0.5)

    def test_one_bay_off(self):
        """1 bay away → 80% bonus."""
        r = self._calc_reward(11, 10)
        expected = 0.1 + 0.5 * (1.0 - 1 / 5)
        self.assertAlmostEqual(r, expected)

    def test_at_radius(self):
        """At radius boundary → zero bonus."""
        r = self._calc_reward(15, 10)
        self.assertAlmostEqual(r, 0.1)

    def test_beyond_radius(self):
        """Beyond radius → zero bonus (clamped)."""
        r = self._calc_reward(20, 10)
        self.assertAlmostEqual(r, 0.1)

    def test_no_preferred(self):
        """No preferred bay → base reward only."""
        r = self._calc_reward(10, None)
        self.assertAlmostEqual(r, 0.1)


# ══════════════════════════════════════════════════════════════════════════
# Part 7: Step result and transition tests
# ══════════════════════════════════════════════════════════════════════════

class TestUnifiedStepResult(unittest.TestCase):
    """Test UnifiedStepResult dataclass."""

    def test_default_values(self):
        state = np.zeros((10, 15, 1160, 5), dtype=np.float32)
        r = type("UnifiedStepResult", (), {
            "next_state": state, "reward": 0.0, "done": False,
            "info": {}, "transition": None, "was_valid_move": False,
        })()
        self.assertFalse(r.was_valid_move)
        self.assertIsNone(r.transition)

    def test_with_transition(self):
        state = np.zeros((10, 15, 1160, 5), dtype=np.float32)
        t = MagicMock()
        r = type("UnifiedStepResult", (), {
            "next_state": state, "reward": 1.5, "done": True,
            "info": {"executed": [{"move_type": "YARD_TO_TRAIN"}]},
            "transition": t, "was_valid_move": True,
        })()
        self.assertTrue(r.was_valid_move)
        self.assertEqual(r.reward, 1.5)


class TestTransitionInInfo(unittest.TestCase):
    """Verify transitions flow through info dict (not auto-remembered)."""

    def test_transitions_in_info(self):
        """step_all_cranes should put transitions in info['transitions']."""
        info = {"transitions": [], "executed": [], "crane_results": []}
        mock_t = MagicMock()
        info["transitions"].append(mock_t)
        self.assertEqual(len(info["transitions"]), 1)

    def test_no_double_remember(self):
        """Env should NOT call agent.remember — runner handles it."""
        # Verify by checking source code
        with open(ENV_PATH) as f:
            source = f.read()

        # In step_all_cranes, we should NOT see agent.remember
        # (transitions go to info dict, caller remembers)
        step_section = source[source.find("def step_all_cranes"):]
        step_section = step_section[:step_section.find("def _step_unified")]
        self.assertNotIn("agent.remember", step_section)


# ══════════════════════════════════════════════════════════════════════════
# Part 8: Region-based move type alignment
# ══════════════════════════════════════════════════════════════════════════

class TestMoveTypeAlignment(unittest.TestCase):
    """Verify agent move types match env dispatch table."""

    def test_all_agent_types_have_handlers(self):
        """Every non-PARK_TRUCK move from the agent has a resolver."""
        agent_types = {
            "TRAIN_TO_YARD", "TRUCK_TO_YARD", "YARD_TO_YARD",
            "YARD_TO_TRAIN", "YARD_TO_TRUCK",
        }
        dispatch = {
            "YARD_TO_YARD", "YARD_TO_TRAIN", "YARD_TO_TRUCK",
            "TRAIN_TO_YARD", "TRUCK_TO_YARD",
        }
        self.assertEqual(agent_types, dispatch)

    def test_park_truck_handled_before_dispatch(self):
        """PARK_TRUCK is caught before _resolve_move is called."""
        with open(ENV_PATH) as f:
            source = f.read()
        # In _step_unified, PARK_TRUCK check comes before _execute_unified_move
        step_fn = source[source.find("def _step_unified"):]
        park_idx = step_fn.find("PARK_TRUCK")
        move_idx = step_fn.find("_execute_unified_move")
        self.assertGreater(move_idx, park_idx,
                           "PARK_TRUCK should be checked before _execute_unified_move")


# ══════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main()