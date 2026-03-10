# tests/EvalScenario/test_eval_scenario.py
"""Unit tests for the parameterized evaluation scenario system."""
import unittest
from datetime import timedelta

from simulation.training.eval.timing import (
    WorkloadEstimate,
    _axis_time,
    _move_time,
    estimate_feasible_time,
)
from simulation.training.eval.scenario import (
    EvalParams,
    ScalableEvalScenario,
    _CONTAINERS_PER_WAGON,
    _MAX_WAGONS_PER_TRACK,
)
from simulation.training.eval.metrics import SubTaskProgress, EvalRunResult
from simulation.training.eval.harness import default_eval_grid, default_eval_grid_with_trucks
from simulation.config.crane_config import CranePerformance


# ================================================================
# Timing tests
# ================================================================

class TestAxisTime(unittest.TestCase):
    """Test the trapezoidal motion profile."""

    def test_zero_distance(self):
        self.assertAlmostEqual(_axis_time(0.0, 2.0, 0.5), 0.0)

    def test_short_distance_triangular(self):
        """Very short distance → triangular profile."""
        t = _axis_time(0.1, 10.0, 1.0)
        self.assertGreater(t, 0.0)
        self.assertLess(t, 5.0)  # much less than trapezoidal

    def test_long_distance_trapezoidal(self):
        """Long distance → trapezoidal profile."""
        t = _axis_time(100.0, 2.0, 0.1)
        self.assertGreater(t, 20.0)  # substantial time for 100m

    def test_monotonic(self):
        """Longer distance → longer time."""
        t1 = _axis_time(10.0, 2.0, 0.5)
        t2 = _axis_time(50.0, 2.0, 0.5)
        t3 = _axis_time(100.0, 2.0, 0.5)
        self.assertLess(t1, t2)
        self.assertLess(t2, t3)


class TestMoveTime(unittest.TestCase):
    """Test the full move time computation."""

    def test_same_position(self):
        """Zero X/Y movement still has hoist + handling."""
        perf = CranePerformance()
        t = _move_time(0, 0, 0, 0, 0, 0, perf)
        # Even at same position: hoist down + up + down + handling
        self.assertGreater(t, perf.handling_time_s)

    def test_yard_to_train_nearby(self):
        """Typical YARD_TO_TRAIN time is 2-6 minutes."""
        perf = CranePerformance()
        # yard (row 2) → train (track 0), same bay
        t = _move_time(
            20 * 12.2, 36.0, 0.0,   # yard bay 20, row 0 at Y=36
            20 * 12.2, 0.0, 1.5,    # train track 0 at Y=0
            perf,
        )
        self.assertGreater(t, 120)  # > 2 min
        self.assertLess(t, 400)     # < 6.5 min


class TestEstimateFeasibleTime(unittest.TestCase):
    """Test the workload → time estimation."""

    def test_empty_workload(self):
        w = WorkloadEstimate()
        t = estimate_feasible_time(w)
        self.assertAlmostEqual(t, 0.0)

    def test_single_move(self):
        w = WorkloadEstimate(n_yard_to_train=1)
        t = estimate_feasible_time(w, n_cranes=1, safety_factor=1.0)
        self.assertGreater(t, 100)  # at least a couple minutes

    def test_parallelism(self):
        """2 cranes should be faster than 1."""
        w = WorkloadEstimate(n_train_to_yard=10, n_yard_to_train=10)
        t1 = estimate_feasible_time(w, n_cranes=1, safety_factor=1.0)
        t2 = estimate_feasible_time(w, n_cranes=2, safety_factor=1.0)
        self.assertAlmostEqual(t2, t1 / 2, places=0)

    def test_safety_factor(self):
        """Safety factor scales linearly."""
        w = WorkloadEstimate(n_train_to_yard=10)
        t1 = estimate_feasible_time(w, safety_factor=1.0)
        t2 = estimate_feasible_time(w, safety_factor=2.0)
        self.assertAlmostEqual(t2, t1 * 2.0, places=0)

    def test_realistic_80_containers(self):
        """80 containers (40 imp + 40 exp) should take 2-8 hours with safety=2."""
        w = WorkloadEstimate(
            n_train_to_yard=40,
            n_yard_to_train=40,
            n_yard_to_yard=12,
            avg_bay_spread=15.0,
        )
        t = estimate_feasible_time(w, n_cranes=2, safety_factor=2.0)
        hours = t / 3600
        self.assertGreater(hours, 1.5)   # at least 1.5 hours
        self.assertLess(hours, 10.0)     # at most 10 hours


# ================================================================
# Train planning tests
# ================================================================

class TestTrainPlanning(unittest.TestCase):
    """Test train/wagon count computation."""

    def test_zero_rail(self):
        p = EvalParams(n_imports=0, n_exports=0, n_delivery_trucks=5)
        sc = ScalableEvalScenario(p)
        self.assertEqual(sc._n_trains, 0)
        self.assertEqual(sc._wagons_per_train, [])

    def test_small_scenario(self):
        p = EvalParams(n_imports=5, n_exports=5)
        sc = ScalableEvalScenario(p)
        self.assertEqual(sc._n_trains, 1)
        self.assertEqual(len(sc._wagons_per_train), 1)
        # 10 containers / 1.5 per wagon = ~7 wagons needed
        self.assertGreaterEqual(sc._wagons_per_train[0], 7)

    def test_large_scenario_needs_multiple_trains(self):
        p = EvalParams(n_imports=40, n_exports=40)
        sc = ScalableEvalScenario(p)
        # 80 containers / 43 per train ≈ 2 trains
        self.assertGreaterEqual(sc._n_trains, 2)

    def test_max_trains_capped_at_7(self):
        p = EvalParams(n_imports=200, n_exports=200)
        sc = ScalableEvalScenario(p)
        self.assertLessEqual(sc._n_trains, 7)


# ================================================================
# Scenario construction tests
# ================================================================

class TestScenarioConstruction(unittest.TestCase):
    """Test ScalableEvalScenario properties."""

    def test_max_steps_positive(self):
        for n in [10, 20, 40, 80]:
            p = EvalParams(n_imports=n // 2, n_exports=n // 2)
            sc = ScalableEvalScenario(p)
            self.assertGreaterEqual(sc.max_steps, 30)

    def test_feasible_seconds_positive(self):
        p = EvalParams(n_imports=20, n_exports=20)
        sc = ScalableEvalScenario(p)
        self.assertGreater(sc._feasible_seconds, 0)

    def test_feasible_seconds_minimum_10min(self):
        p = EvalParams(n_imports=1, n_exports=1)
        sc = ScalableEvalScenario(p)
        self.assertGreaterEqual(sc._feasible_seconds, 600.0)

    def test_id_in_high_range(self):
        p = EvalParams(seed=42)
        sc = ScalableEvalScenario(p)
        self.assertGreaterEqual(sc.id, 9000)

    def test_label_format(self):
        p = EvalParams(n_imports=10, n_exports=10, yard_fill_pct=0.25)
        self.assertIn("imp=10", p.label)
        self.assertIn("exp=10", p.label)
        self.assertIn("fill=25%", p.label)

    def test_trucks_only_scenario(self):
        """Scenario with no trains, only trucks."""
        p = EvalParams(n_imports=0, n_exports=0,
                       n_delivery_trucks=5, n_pickup_trucks=5)
        sc = ScalableEvalScenario(p)
        self.assertEqual(sc._n_trains, 0)
        self.assertGreater(sc.max_steps, 0)


# ================================================================
# Setup + check_progress integration tests
# ================================================================

class TestScenarioSetup(unittest.TestCase):
    """Test setup() on a real environment.

    These tests require the full environment stack.
    """

    @classmethod
    def _make_env(cls):
        """Create a minimal TerminalEnv for testing."""
        from simulation.training.curriculum_trainer import create_env_factory
        factory = create_env_factory()
        return factory()

    def _run_setup(self, params: EvalParams):
        """Setup a scenario and return (env, scenario)."""
        from simulation.training.tutorial_runner import (
            _clear_yard, _clear_parking, _clear_rail,
        )
        env = self._make_env()
        scenario = ScalableEvalScenario(params)

        # Clear like TutorialRunner does
        from simulation.training.scenarios._base import TUTORIAL_TIME
        env.current_time = TUTORIAL_TIME
        env.trains.clear()
        env.trucks.clear()
        _clear_yard(env.yard)
        _clear_parking(env.parking)
        _clear_rail(env.rail)

        scenario.setup(env)
        return env, scenario

    def test_small_setup(self):
        """10 containers: 5 imports + 5 exports."""
        env, sc = self._run_setup(EvalParams(n_imports=5, n_exports=5, seed=100))

        # 5 exports should be in yard
        for i in range(5):
            self.assertIsNotNone(
                env.yard.get_container(f"EVAL100_EXP{i}"),
                f"Export {i} not found in yard",
            )

        # 1 train with 5 imports
        self.assertEqual(len(env.trains), 1)
        train = list(env.trains.values())[0]
        self.assertEqual(train.get_container_count(), 5)

        # Progress should show 0% on everything
        progress = sc.check_progress(env)
        self.assertEqual(len(progress), 2)  # imports + exports
        for st in progress:
            self.assertEqual(st.completed, 0, f"{st.name} should be 0 at start")

        # Success should be False
        self.assertFalse(sc.check_success(env))

    def test_medium_setup_with_trucks(self):
        """20 containers: 8 imp + 8 exp + 2 delivery + 2 pickup."""
        env, sc = self._run_setup(EvalParams(
            n_imports=8, n_exports=8,
            n_delivery_trucks=2, n_pickup_trucks=2,
            seed=200,
        ))

        # 8 exports + 2 pickup targets in yard
        yard_count = sum(
            1 for i in range(8)
            if env.yard.get_container(f"EVAL200_EXP{i}") is not None
        )
        self.assertEqual(yard_count, 8)

        # 2 delivery trucks + 2 pickup trucks = 4 in env.trucks
        self.assertEqual(len(env.trucks), 4)

        # Progress: 4 sub-tasks
        progress = sc.check_progress(env)
        self.assertEqual(len(progress), 4)

    def test_large_setup_multiple_trains(self):
        """80 containers needs 2 trains."""
        env, sc = self._run_setup(EvalParams(n_imports=40, n_exports=40, seed=300))

        self.assertGreaterEqual(len(env.trains), 2)

        # All 40 exports in yard
        exports_in_yard = sum(
            1 for i in range(40)
            if env.yard.get_container(f"EVAL300_EXP{i}") is not None
        )
        self.assertEqual(exports_in_yard, 40)

    def test_yard_fill_places_distractors(self):
        """50% yard fill should add ~145 distractor containers."""
        env, sc = self._run_setup(EvalParams(
            n_imports=5, n_exports=5, yard_fill_pct=0.50, seed=400,
        ))

        # Count total containers in yard
        total = sum(1 for _ in env.yard.iter_records())
        # 5 exports + many distractors
        self.assertGreater(total, 50)  # at least 50 (exports + distractors)

    def test_zero_fill(self):
        """0% fill should only have active containers."""
        env, sc = self._run_setup(EvalParams(
            n_imports=5, n_exports=5, yard_fill_pct=0.0, seed=500,
        ))

        total = sum(1 for _ in env.yard.iter_records())
        self.assertEqual(total, 5)  # just the 5 exports


# ================================================================
# Grid tests
# ================================================================

class TestDefaultGrids(unittest.TestCase):
    """Test the default evaluation grids."""

    def test_primary_grid(self):
        grid = default_eval_grid()
        self.assertEqual(len(grid), 15)  # 5 sizes x 3 fills

        # Check all have valid params
        for p in grid:
            self.assertGreaterEqual(p.n_imports, 0)
            self.assertGreaterEqual(p.n_exports, 0)
            self.assertGreaterEqual(p.yard_fill_pct, 0.0)
            self.assertLessEqual(p.yard_fill_pct, 1.0)

    def test_truck_grid(self):
        grid = default_eval_grid_with_trucks()
        self.assertEqual(len(grid), 12)  # 4 sizes x 3 fills

        # All entries should have trucks
        for p in grid:
            self.assertGreater(
                p.n_delivery_trucks + p.n_pickup_trucks, 0,
                f"Grid entry {p.label} has no trucks",
            )

    def test_all_grid_entries_feasible(self):
        """Every grid entry should produce a valid scenario."""
        grid = default_eval_grid() + default_eval_grid_with_trucks()
        for p in grid:
            sc = ScalableEvalScenario(p)
            self.assertGreater(sc._feasible_seconds, 0, f"Failed for {p.label}")
            self.assertGreater(sc.max_steps, 0, f"Failed for {p.label}")
            self.assertGreaterEqual(sc._n_trains, 0)


# ================================================================
# Metrics tests
# ================================================================

class TestMetrics(unittest.TestCase):
    """Test the metrics dataclasses."""

    def test_subtask_progress(self):
        st = SubTaskProgress("test", 3, 10)
        self.assertAlmostEqual(st.pct, 0.3)
        self.assertFalse(st.done)

        st2 = SubTaskProgress("test", 10, 10)
        self.assertTrue(st2.done)

    def test_eval_run_result_properties(self):
        p = EvalParams(n_imports=5, n_exports=5)
        r = EvalRunResult(
            params=p,
            agent_moves=20,
            move_type_counts={"YARD_TO_TRAIN": 5, "TRAIN_TO_YARD": 5, "YARD_TO_YARD": 10},
        )
        self.assertAlmostEqual(r.reshuffle_ratio, 0.5)
        self.assertAlmostEqual(r.productive_move_ratio, 0.5)
        self.assertEqual(r.total_containers, 10)
        self.assertAlmostEqual(r.moves_per_container, 2.0)


if __name__ == "__main__":
    unittest.main()
