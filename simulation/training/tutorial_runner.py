# simulation/training/tutorial_runner.py
"""Runs tutorial scenarios and collects training transitions.

Usage in curriculum:
    runner = TutorialRunner(env_factory, agent_or_config)
    results = runner.train_all(epochs=50)  # repeat until mastery
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Union
from simulation.rl.multihead_dqn.config import MultiHeadDQNConfig
from simulation.rl.multihead_dqn.agent import MultiHeadDQNAgent
from simulation.training.tutorial_scenarios import (
    ALL_SCENARIOS,
    TUTORIAL_TIME,
    TutorialResult,
    TutorialScenario,
)

logger = logging.getLogger(__name__)

# ================================================================
# Constants
# ================================================================

TUTORIAL_END_HOUR = 23
TUTORIAL_END_MINUTE = 59
TUTORIAL_REWARD_TIMEOUT = -5.0   # penalty if max_steps exhausted
TUTORIAL_REWARD_SUCCESS = 5.0    # bonus on pass


# ================================================================
# Runner
# ================================================================

class TutorialRunner:
    """Execute tutorial scenarios, collect transitions, report results."""

    def __init__(
        self,
        env_factory: Callable,
        agent_or_config: Union["MultiHeadDQNAgent", "MultiHeadDQNConfig"],
        verbose: bool = True,
    ):
        """Initialize tutorial runner.
        
        Args:
            env_factory: Callable that creates ContainerTerminalEnv
            agent_or_config: Either an existing MultiHeadDQNAgent or a 
                            MultiHeadDQNConfig (agent will be created)
            verbose: Whether to log progress
        """
        self.env = env_factory()
        self.verbose = verbose
        
        # Handle both agent instance and config
        if hasattr(agent_or_config, 'act'):
            # It's an agent
            self.agent = agent_or_config
        else:
            # It's a config - create agent
            from simulation.rl.multihead_dqn.agent import MultiHeadDQNAgent
            self.agent = MultiHeadDQNAgent(agent_or_config)
            if self.verbose:
                logger.info("TutorialRunner created new agent from config")

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def run_scenario(self, scenario: TutorialScenario) -> TutorialResult:
        """Run one scenario, return result."""
        env = self.env
        self._reset_tutorial(env)
        scenario.setup(env)

        # Re-sync components that depend on yard/rail state
        env._update_train_heat()
        env.rmgc.set_layout(yard=env.yard, rail=env.rail, num_tracks=env.num_tracks)

        total_reward = 0.0
        agent_moves = 0
        auto_moves = 0
        move_log: List[Dict] = []

        for step in range(scenario.max_steps):
            state, reward, done, info = env.step_all_cranes(self.agent)
            total_reward += reward

            for ex in info.get("executed", []):
                move_log.append(ex)
                if ex.get("auto_transfer"):
                    auto_moves += 1
                else:
                    agent_moves += 1

            # Store transitions for training
            for t in info.get("transitions", []):
                self.agent.remember(t)

            # Check scenario success
            if scenario.check_success(env):
                total_reward += TUTORIAL_REWARD_SUCCESS
                passed = scenario.check_pass(env, agent_moves, auto_moves)
                return TutorialResult(
                    scenario_id=scenario.id,
                    name=scenario.name,
                    passed=passed,
                    steps=step + 1,
                    agent_moves=agent_moves,
                    auto_moves=auto_moves,
                    total_reward=total_reward,
                    move_log=move_log,
                )

            if done:
                break

        # Max steps exhausted -> fail
        total_reward += TUTORIAL_REWARD_TIMEOUT
        return TutorialResult(
            scenario_id=scenario.id,
            name=scenario.name,
            passed=False,
            steps=scenario.max_steps,
            agent_moves=agent_moves,
            auto_moves=auto_moves,
            total_reward=total_reward,
            move_log=move_log,
        )

    def run_all(self) -> List[TutorialResult]:
        """Run all scenarios once, return results."""
        results = []
        for sc in ALL_SCENARIOS:
            result = self.run_scenario(sc)
            results.append(result)
        return results

    def train_all(
        self,
        epochs: int = 500,
        learn_every: int = 4,
        mastery_threshold: float = 0.9,
        window_size: int = 20,
        min_epochs: int = 10,
        log_every: int = 5,
    ) -> Dict[str, object]:
        """Train agent on tutorials until mastery on ALL scenarios.

        Mastery = rolling pass rate >= threshold for ALL scenarios.
        
        Args:
            epochs: Maximum epochs to run
            learn_every: Steps between optimize() calls
            mastery_threshold: Required pass rate (0.9 = 90%)
            window_size: Rolling window for pass rate calculation
            min_epochs: Minimum epochs before checking mastery
            log_every: Print progress every N epochs
            
        Returns:
            Dict with pass_rates per scenario, epochs completed, mastered flag
        """
        from collections import deque
        
        # Track pass/fail history per scenario (rolling window)
        history: Dict[int, deque] = {
            sc.id: deque(maxlen=window_size) for sc in ALL_SCENARIOS
        }
        scenario_names = {sc.id: sc.name for sc in ALL_SCENARIOS}
        
        def get_pass_rates() -> Dict[int, float]:
            """Calculate rolling pass rate for each scenario."""
            rates = {}
            for sid, hist in history.items():
                if len(hist) == 0:
                    rates[sid] = 0.0
                else:
                    rates[sid] = sum(hist) / len(hist)
            return rates
        
        def all_mastered(rates: Dict[int, float]) -> bool:
            """Check if all scenarios meet mastery threshold."""
            return all(r >= mastery_threshold for r in rates.values())
        
        def print_progress(epoch: int, rates: Dict[int, float], force: bool = False):
            """Print detailed progress."""
            if not self.verbose:
                return
            if not force and epoch % log_every != 0:
                return
                
            print(f"\n  Epoch {epoch}/{epochs} - Tutorial Progress:", flush=True)
            for sc in ALL_SCENARIOS:
                rate = rates.get(sc.id, 0.0)
                bar_len = 20
                filled = int(rate * bar_len)
                bar = "=" * filled + "-" * (bar_len - filled)
                status = "OK" if rate >= mastery_threshold else "  "
                print(f"    [{bar}] {rate:5.1%} {status} S{sc.id}: {sc.name}", flush=True)
            
            avg_rate = sum(rates.values()) / len(rates) if rates else 0
            mastered_count = sum(1 for r in rates.values() if r >= mastery_threshold)
            print(f"    Average: {avg_rate:.1%} | Mastered: {mastered_count}/{len(ALL_SCENARIOS)}", flush=True)

        mastered = False
        epoch = 0
        
        for epoch in range(epochs):
            steps_this_epoch = 0
            epoch_passed = 0
            epoch_reward = 0.0

            for sc in ALL_SCENARIOS:
                result = self.run_scenario(sc)
                
                # Record pass/fail in rolling window
                history[sc.id].append(1 if result.passed else 0)
                
                if result.passed:
                    epoch_passed += 1
                epoch_reward += result.total_reward
                steps_this_epoch += result.steps

                # Periodic learning
                if steps_this_epoch % learn_every == 0:
                    self.agent.optimize()

            # Calculate current pass rates
            rates = get_pass_rates()
            
            # Log progress
            print_progress(epoch + 1, rates)
            
            # Check mastery (only after minimum epochs and full window)
            if epoch >= min_epochs and epoch >= window_size:
                if all_mastered(rates):
                    mastered = True
                    if self.verbose:
                        print(f"\n  MASTERY ACHIEVED at epoch {epoch + 1}!", flush=True)
                        print_progress(epoch + 1, rates, force=True)
                    break

        # Final summary
        final_rates = get_pass_rates()
        
        if self.verbose and not mastered:
            print(f"\n  Tutorial training ended after {epoch + 1} epochs", flush=True)
            print_progress(epoch + 1, final_rates, force=True)
            
            # Show which scenarios need work
            struggling = [
                (sc.id, sc.name, final_rates.get(sc.id, 0))
                for sc in ALL_SCENARIOS
                if final_rates.get(sc.id, 0) < mastery_threshold
            ]
            if struggling:
                print(f"\n  Scenarios below {mastery_threshold:.0%} threshold:", flush=True)
                for sid, name, rate in struggling:
                    print(f"    S{sid} {name}: {rate:.1%}", flush=True)

        return {
            "epochs_completed": epoch + 1,
            "mastered": mastered,
            "pass_rates": {sc.id: final_rates.get(sc.id, 0) for sc in ALL_SCENARIOS},
            "scenario_names": scenario_names,
        }

    # ----------------------------------------------------------------
    # Internal
    # ----------------------------------------------------------------

    def _reset_tutorial(self, env) -> None:
        """Clear env state for a tutorial scenario (no logistics/day plan)."""
        env.current_time = TUTORIAL_TIME
        env.day_index = 0

        # Create a minimal stub day_plan so _check_day_end doesn't return True
        env.day_plan = _StubDayPlan(TUTORIAL_TIME)
        env._scheduled_trains = []

        # Clear vehicles
        env.trains.clear()
        env.trucks.clear()
        env.terminal_trucks.clear()
        env._tt_busy_until.clear()
        env._admitted_truck_ids.clear()
        env._departed_cache.clear()
        env.reward_engine.reset_train_tracking()

        # Clear facilities
        _clear_yard(env.yard)
        _clear_parking(env.parking)
        _clear_rail(env.rail)

        # Reset cranes
        from simulation.env.env import CraneState
        from simulation.config.crane_config import CraneDefaults
        env.cranes = [CraneState(i, None) for i in range(env.num_cranes)]
        env.crane_zones = env._make_crane_zones(
            overlap_bays=CraneDefaults.ZONE_OVERLAP_BAYS,
        )

        # Sync RMGC
        env.rmgc.set_layout(yard=env.yard, rail=env.rail, num_tracks=env.num_tracks)
        env._train_heat_bays = set()


# ================================================================
# Facility clear helpers
# ================================================================

def _clear_yard(yard) -> None:
    """Remove all containers from an OptimizedStorageYard."""
    from simulation.core.facilities.yard import EMPTY_SLOT
    yard.occupancy_mask[:] = False
    yard.position_grid[:] = EMPTY_SLOT
    for i in range(len(yard._records)):
        yard._records[i] = None
    yard._id_to_idx.clear()
    yard._free_indices = list(range(len(yard._records) - 1, -1, -1))
    yard._accessible_mask[:] = False
    yard._tier_counts[:] = 0
    yard._support_cache.clear()


def _clear_parking(parking) -> None:
    """Remove all trucks from parking."""
    if parking is None:
        return
    parking.occupied[:] = False
    parking.truck_ids.fill(None)
    parking._truck_spots.clear()


def _clear_rail(rail) -> None:
    """Remove all trains from rail yard."""
    rail._train_to_slot.clear()
    rail._track_trains.clear()


# ================================================================
# Stub day plan for tutorials
# ================================================================

class _StubDayPlan:
    """Minimal day plan that keeps _check_day_end happy.

    Sets end-of-day far in the future so tutorials don't time-expire.
    """

    def __init__(self, base_time: datetime):
        self.date = base_time.replace(hour=0, minute=0, second=0)
        self.todays_trains = []
        self.trucks_today = []