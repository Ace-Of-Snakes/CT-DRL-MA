# simulation/training/unified_tutorial_runner.py
"""Tiered tutorial runner for the unified spatial agent.

The agent must master each tier before advancing to the next.
Previously mastered tiers are replayed every `maintenance_interval`
epochs to prevent catastrophic forgetting.

Tier layout (auto-built, with fallback for unknown scenario IDs):
  Tier 0  — Primitives:           S1-S4   (single actions)
  Tier 1  — Two-action chains:    S5-S6
  Tier 2  — Full chains:          S7-S8   (3 actions)
  Tier 3  — Restack + load:       S9-S10  (with distractors)
  Tier 4  — Random generalize:    S11-S12
  Tier 5  — Multi-step hard:      S13-S14
  Tier 6  — Multi-vehicle:        S15-S16
  Tier 7  — Bidirectional train:  S17
  Tier 8  — Concurrent ops:       S18-S19
  Tier 9  — Full complexity:      S20-S21
  Tier 10 — Terminal truck:       S22-S25
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

from simulation.rl.multihead_dqn.config import MultiHeadDQNConfig
from simulation.rl.multihead_dqn.unified_agent import UnifiedDQNAgent
from simulation.rl.multihead_dqn.unified_replay_buffer import UnifiedTransition
from simulation.training.tutorial_scenarios import (
    ALL_SCENARIOS,
    TUTORIAL_TIME,
    TutorialResult,
    TutorialScenario,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────

TUTORIAL_REWARD_TIMEOUT: float = -5.0
TUTORIAL_REWARD_SUCCESS: float = 5.0

# Tier boundaries: (name, scenario_ids).
# Unlisted scenarios are auto-grouped into a final "Advanced" tier.
_TIER_BOUNDARIES: List[Tuple[str, List[int]]] = [
    ("Primitives",         [1, 2, 3, 4]),
    ("Two-action chains",  [5, 6]),
    ("Full chains",        [7, 8]),
    ("Restack + load",     [9, 10]),
    ("Random generalize",  [11, 12]),
    ("Multi-step hard",    [13, 14]),
    ("Terminal truck",     [25, 22, 23, 24]),
    ("Multi-vehicle",      [15, 16]),
    ("Bidirectional train", [17]),
    ("Concurrent ops",     [18, 19]),
    ("Full complexity",    [20, 21]),
]

DEFAULT_MAINTENANCE_INTERVAL: int = 3


# ── Tier dataclass ───────────────────────────────────────────────────────

@dataclass
class TutorialTier:
    """A group of scenarios at the same difficulty level."""
    index: int
    name: str
    scenarios: List[TutorialScenario]

    @property
    def scenario_ids(self) -> List[int]:
        return [s.id for s in self.scenarios]


def _build_tiers() -> List[TutorialTier]:
    """Build tiers from ALL_SCENARIOS using _TIER_BOUNDARIES."""
    by_id = {sc.id: sc for sc in ALL_SCENARIOS}
    assigned: set = set()
    tiers: List[TutorialTier] = []

    for idx, (name, ids) in enumerate(_TIER_BOUNDARIES):
        scenarios = [by_id[sid] for sid in ids if sid in by_id]
        if scenarios:
            tiers.append(TutorialTier(index=idx, name=name, scenarios=scenarios))
            assigned.update(sc.id for sc in scenarios)

    remaining = [sc for sc in ALL_SCENARIOS if sc.id not in assigned]
    if remaining:
        tiers.append(TutorialTier(
            index=len(tiers), name="Advanced", scenarios=remaining,
        ))

    return tiers


TUTORIAL_TIERS: List[TutorialTier] = _build_tiers()


# ══════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════

class UnifiedTutorialRunner:
    """Execute tutorials with tiered curriculum progression."""

    def __init__(
        self,
        env_factory: Callable,
        agent_or_config: Union[UnifiedDQNAgent, MultiHeadDQNConfig],
        verbose: bool = True,
    ):
        self.env = env_factory()
        self.verbose = verbose

        if hasattr(agent_or_config, "act"):
            self.agent = agent_or_config
        else:
            self.agent = UnifiedDQNAgent(agent_or_config)
            if self.verbose:
                logger.info("UnifiedTutorialRunner created agent from config")

    # ── Public API ────────────────────────────────────────────────────

    def run_scenario(self, scenario: TutorialScenario) -> TutorialResult:
        """Run one scenario, return result."""
        env = self.env
        self._reset_tutorial(env)
        scenario.setup(env)

        env._update_train_heat()
        env.rmgc.set_layout(
            yard=env.yard, rail=env.rail, num_tracks=env.num_tracks,
        )

        total_reward = 0.0
        agent_moves = 0
        move_log: List[Dict] = []
        last_transition: Optional[UnifiedTransition] = None

        for step in range(scenario.max_steps):
            state, reward, done, info = env.step_all_cranes(self.agent)
            total_reward += reward

            for ex in info.get("executed", []):
                move_log.append(ex)
                agent_moves += 1

            for t in info.get("transitions", []):
                self.agent.remember(t)
                last_transition = t

            if scenario.check_success(env):
                total_reward += TUTORIAL_REWARD_SUCCESS
                if last_transition is not None:
                    last_transition.done = True
                    last_transition.reward += TUTORIAL_REWARD_SUCCESS
                passed = scenario.check_pass(env, agent_moves)
                return TutorialResult(
                    scenario_id=scenario.id,
                    name=scenario.name,
                    passed=passed,
                    steps=step + 1,
                    agent_moves=agent_moves,
                    total_reward=total_reward,
                    move_log=move_log,
                )

            if done:
                break

        total_reward += TUTORIAL_REWARD_TIMEOUT
        if last_transition is not None:
            last_transition.done = True
            last_transition.reward += TUTORIAL_REWARD_TIMEOUT
        return TutorialResult(
            scenario_id=scenario.id,
            name=scenario.name,
            passed=False,
            steps=scenario.max_steps,
            agent_moves=agent_moves,
            total_reward=total_reward,
            move_log=move_log,
        )

    def run_all(self) -> List[TutorialResult]:
        """Run all scenarios once (ignores tier structure)."""
        return [self.run_scenario(sc) for sc in ALL_SCENARIOS]

    def train_all(
        self,
        epochs: int = 500,
        learn_every: int = 4,
        mastery_threshold: float = 0.9,
        window_size: int = 20,
        min_epochs: int = 10,
        log_every: int = 5,
        maintenance_interval: int = DEFAULT_MAINTENANCE_INTERVAL,
    ) -> Dict[str, object]:
        """Train with tiered progression — master each tier before advancing."""
        tiers = TUTORIAL_TIERS
        n_tiers = len(tiers)
        all_sids = [sc.id for sc in ALL_SCENARIOS]
        scenario_names = {sc.id: sc.name for sc in ALL_SCENARIOS}

        history: Dict[int, deque] = {
            sid: deque(maxlen=window_size) for sid in all_sids
        }

        current_tier = 0
        tier_start_epoch: Dict[int, int] = {0: 0}
        mastered = False

        if self.verbose:
            self._print_tier_plan(tiers)

        for epoch in range(epochs):
            self.agent.set_tutorial_epsilon(epoch)
            steps_this_epoch = 0

            # ── Run current tier (every epoch) ───────────────────────
            steps_this_epoch += self._run_tier(
                tiers[current_tier], history, learn_every,
            )

            # ── Maintenance: replay mastered tiers periodically ──────
            if current_tier > 0 and epoch % maintenance_interval == 0:
                for ti in range(current_tier):
                    steps_this_epoch += self._run_tier(
                        tiers[ti], history, learn_every,
                    )

            # Extra optimization passes
            for _ in range(min(10, steps_this_epoch // 4)):
                self.agent.optimize()

            # ── Check tier mastery ───────────────────────────────────
            rates = self._get_pass_rates(history)
            self._log_progress(epoch + 1, epochs, rates, tiers, current_tier,
                               mastery_threshold, log_every)

            epochs_in_tier = epoch - tier_start_epoch.get(current_tier, 0)
            if epochs_in_tier >= min_epochs and epochs_in_tier >= window_size:
                if self._tier_mastered(tiers[current_tier], rates, mastery_threshold):
                    if self.verbose:
                        print(f"\n  ✓ TIER {current_tier} MASTERED "
                              f"({tiers[current_tier].name}) at epoch {epoch + 1}",
                              flush=True)

                    current_tier += 1
                    if current_tier >= n_tiers:
                        mastered = True
                        if self.verbose:
                            print(f"\n  ALL {n_tiers} TIERS MASTERED "
                                  f"at epoch {epoch + 1}!", flush=True)
                            self._log_progress(
                                epoch + 1, epochs, rates, tiers,
                                n_tiers - 1, mastery_threshold,
                                log_every=1, force=True,
                            )
                        break

                    tier_start_epoch[current_tier] = epoch + 1
                    if self.verbose:
                        print(f"  → Advancing to Tier {current_tier}: "
                              f"{tiers[current_tier].name} "
                              f"({len(tiers[current_tier].scenarios)} scenarios)",
                              flush=True)

        final_rates = self._get_pass_rates(history)
        self.agent.clear_epsilon_override()

        if self.verbose and not mastered:
            print(f"\n  Training ended after {epoch + 1} epochs "
                  f"(Tier {current_tier}/{n_tiers - 1})", flush=True)
            self._log_progress(
                epoch + 1, epochs, final_rates, tiers,
                current_tier, mastery_threshold,
                log_every=1, force=True,
            )

        return {
            "epochs_completed": epoch + 1,
            "mastered": mastered,
            "current_tier": current_tier,
            "n_tiers": n_tiers,
            "pass_rates": {sid: final_rates.get(sid, 0) for sid in all_sids},
            "scenario_names": scenario_names,
        }

    # ── Training helpers ──────────────────────────────────────────────

    def _run_tier(
        self,
        tier: TutorialTier,
        history: Dict[int, deque],
        learn_every: int,
    ) -> int:
        """Run all scenarios in a tier, update history. Returns total steps."""
        total_steps = 0
        for sc in tier.scenarios:
            result = self.run_scenario(sc)
            history[sc.id].append(1 if result.passed else 0)
            total_steps += result.steps
            if total_steps % learn_every == 0:
                self.agent.optimize()
        return total_steps

    @staticmethod
    def _get_pass_rates(history: Dict[int, deque]) -> Dict[int, float]:
        return {
            sid: (sum(h) / len(h) if h else 0.0)
            for sid, h in history.items()
        }

    @staticmethod
    def _tier_mastered(
        tier: TutorialTier,
        rates: Dict[int, float],
        threshold: float,
    ) -> bool:
        return all(rates.get(sc.id, 0.0) >= threshold for sc in tier.scenarios)

    # ── Logging ───────────────────────────────────────────────────────

    def _print_tier_plan(self, tiers: List[TutorialTier]) -> None:
        print(f"\n  Tutorial tiers ({len(tiers)} total):", flush=True)
        for t in tiers:
            ids = ", ".join(f"S{s.id}" for s in t.scenarios)
            print(f"    Tier {t.index}: {t.name} [{ids}]", flush=True)
        print(flush=True)

    def _log_progress(
        self,
        epoch: int,
        max_epochs: int,
        rates: Dict[int, float],
        tiers: List[TutorialTier],
        current_tier: int,
        threshold: float,
        log_every: int,
        force: bool = False,
    ) -> None:
        if not self.verbose or (not force and epoch % log_every != 0):
            return

        tier_label = tiers[min(current_tier, len(tiers) - 1)].name
        print(f"\n  Epoch {epoch}/{max_epochs} — "
              f"Tier {current_tier}: {tier_label}", flush=True)

        for tier in tiers:
            tier_ok = self._tier_mastered(tier, rates, threshold)
            active = tier.index == current_tier
            locked = tier.index > current_tier

            if tier_ok:
                status = "✓"
            elif active:
                status = "►"
            elif locked:
                status = "🔒"
            else:
                status = " "
            print(f"    {status} Tier {tier.index}: {tier.name}", flush=True)

            # Show bars for active + mastered tiers; hide locked
            if locked:
                continue

            for sc in tier.scenarios:
                rate = rates.get(sc.id, 0.0)
                filled = int(rate * 20)
                bar = "=" * filled + "-" * (20 - filled)
                tag = "OK" if rate >= threshold else "  "
                print(f"      [{bar}] {rate:5.1%} {tag} "
                      f"S{sc.id}: {sc.name}", flush=True)

        active_scs = [sc for t in tiers[:current_tier + 1] for sc in t.scenarios]
        active_rates = [rates.get(sc.id, 0.0) for sc in active_scs]
        avg = sum(active_rates) / max(len(active_rates), 1)
        n_ok = sum(1 for r in rates.values() if r >= threshold)
        eps = (f"{self.agent._get_epsilon():.3f}"
               if hasattr(self.agent, "_get_epsilon") else "?")
        print(f"    Average (active): {avg:.1%} | "
              f"Mastered: {n_ok}/{len(ALL_SCENARIOS)} | ε={eps}", flush=True)

    # ── Environment reset ─────────────────────────────────────────────

    def _reset_tutorial(self, env) -> None:
        """Clear env for a tutorial scenario."""
        from simulation.env.env import CraneState
        from simulation.config.crane_config import CraneDefaults

        env.current_time = TUTORIAL_TIME
        env.day_index = 0
        env.day_plan = _StubDayPlan(TUTORIAL_TIME)
        env._scheduled_trains = []

        env.trains.clear()
        env.trucks.clear()
        env.terminal_trucks.clear()
        env._tt_busy_until.clear()
        env._admitted_truck_ids.clear()
        env._departed_cache.clear()
        env.reward_engine.reset_train_tracking()

        _clear_yard(env.yard)
        _clear_parking(env.parking)
        _clear_rail(env.rail)

        env.cranes = [CraneState(i, None) for i in range(env.num_cranes)]
        env.crane_zones = env._make_crane_zones(
            overlap_bays=CraneDefaults.ZONE_OVERLAP_BAYS,
        )

        env.rmgc.set_layout(
            yard=env.yard, rail=env.rail, num_tracks=env.num_tracks,
        )
        env._train_heat_bays = set()


# ══════════════════════════════════════════════════════════════════════════
# Tutorial helpers (previously in tutorial_runner.py)
# ══════════════════════════════════════════════════════════════════════════

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


class _StubDayPlan:
    """Minimal day plan that keeps _check_day_end happy.

    Sets end-of-day far in the future so tutorials don't time-expire.
    """

    def __init__(self, base_time):
        self.date = base_time.replace(hour=0, minute=0, second=0)
        self.todays_trains = []
        self.trucks_today = []