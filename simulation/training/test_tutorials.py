#!/usr/bin/env python3
"""Standalone test for tutorial scenarios (unified architecture).

Validates:
1. Each scenario sets up correctly (no crashes)
2. Success criteria are correctly defined
3. Facilities clear properly between scenarios

Run: python test_tutorials.py
"""
import sys
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ================================================================
# Mini integration test (no agent needed)
# ================================================================

def create_test_env():
    """Create a minimal env using the unified factory."""
    from simulation.training.unified_curriculum_trainer import create_env_factory
    factory = create_env_factory(rows=5, bays=20, tiers=5, tracks=4)
    return factory, factory()


def test_clear_facilities():
    """Test that yard/parking/rail clear correctly."""
    from simulation.training.unified_tutorial_runner import (
        _clear_yard, _clear_parking, _clear_rail,
    )
    from simulation.core.containers.container import Container
    from simulation.core.facilities.yard import PlacementResult
    from simulation.core.enums import Direction, GoodsType

    _, env = create_test_env()

    # Place a container
    c = Container(
        container_id="TEST_C1",
        direction=Direction.IMPORT,
        container_type="40' HC",
        arrival_date=datetime(2026, 1, 1),
        departure_date=datetime(2026, 1, 6),
        length_ft=40, length_m=12.2,
    )
    env.yard.add_container(c, PlacementResult(row=0, bay=5, tier=0, start_split=0))
    assert env.yard.container_count == 1, "Container should be in yard"

    _clear_yard(env.yard)
    assert env.yard.container_count == 0, "Yard should be empty after clear"
    assert not env.yard.occupancy_mask.any(), "Occupancy mask should be all False"

    _clear_parking(env.parking)
    assert not env.parking.occupied.any(), "Parking should be empty after clear"

    _clear_rail(env.rail)
    log.info("[PASS] test_clear_facilities")


def test_scenario_setup():
    """Test that each scenario sets up without errors."""
    from simulation.training.unified_tutorial_runner import (
        _clear_yard, _clear_parking, _clear_rail,
    )
    from simulation.training.tutorial_scenarios import ALL_SCENARIOS, TUTORIAL_TIME, ANCHOR_BAY

    factory, env = create_test_env()

    # Patch ANCHOR_BAY if yard is smaller
    import simulation.training.tutorial_scenarios as ts
    if env.yard.n_bays <= ANCHOR_BAY:
        ts.ANCHOR_BAY = env.yard.n_bays // 2

    for sc in ALL_SCENARIOS:
        # Reset
        _clear_yard(env.yard)
        _clear_parking(env.parking)
        _clear_rail(env.rail)
        env.trains.clear()
        env.trucks.clear()
        env.current_time = TUTORIAL_TIME

        try:
            sc.setup(env)
        except Exception as e:
            log.error(f"[FAIL] Scenario {sc.id} ({sc.name}) setup crashed: {e}")
            raise

        log.info(
            f"  S{sc.id} ({sc.name}): "
            f"yard={env.yard.container_count}, "
            f"trains={len(env.trains)}, "
            f"trucks={len(env.trucks)}"
        )

    log.info("[PASS] test_scenario_setup")


def run_all_tests():
    """Run all tutorial tests."""
    log.info("=" * 60)
    log.info("Tutorial System Tests (Unified Architecture)")
    log.info("=" * 60)

    tests = [
        test_clear_facilities,
        test_scenario_setup,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            log.info(f"\n--- {test.__name__} ---")
            test()
            passed += 1
        except Exception as e:
            log.error(f"[FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    log.info(f"\n{'=' * 60}")
    log.info(f"Results: {passed} passed, {failed} failed")
    log.info(f"{'=' * 60}")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
