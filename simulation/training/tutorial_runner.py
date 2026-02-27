# simulation/training/tutorial_runner.py
"""Tiered tutorial runner for the unified spatial agent.

The agent must master each tier before advancing to the next.
Previously mastered tiers are replayed every ``maintenance_interval``
epochs to prevent catastrophic forgetting.

**One-shot graduation**: scenarios marked ``repeatable=False`` (the
deterministic primitives and chains S1–S8, S25) are dropped from
maintenance replays once their tier is mastered.  Only ``repeatable``
scenarios continue to be replayed, saving significant training time.

Tier definitions live in ``simulation.training.scenarios._registry.TIER_DEFS``.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

from simulation.rl.multihead_dqn.config import MultiHeadDQNConfig
from simulation.rl.multihead_dqn.agent import DQNAgent
from simulation.rl.multihead_dqn.replay_buffer import Transition
from simulation.training.scenarios import (
    ALL_SCENARIOS,
    MoveRecord,
    StepEvent,
    TIER_DEFS,
    TUTORIAL_TIME,
    TutorialResult,
    TutorialScenario,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────

TUTORIAL_REWARD_TIMEOUT: float = -5.0
TUTORIAL_REWARD_SUCCESS: float = 5.0

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
    """Build tiers from ALL_SCENARIOS using TIER_DEFS (from _registry)."""
    by_id = {sc.id: sc for sc in ALL_SCENARIOS}
    assigned: set = set()
    tiers: List[TutorialTier] = []

    for idx, (name, ids) in enumerate(TIER_DEFS):
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

class TutorialRunner:
    """Execute tutorials with tiered curriculum progression."""

    def __init__(
        self,
        env_factory: Callable,
        agent_or_config: Union[DQNAgent, MultiHeadDQNConfig],
        verbose: bool = True,
    ):
        self.env = env_factory()
        self.verbose = verbose

        if hasattr(agent_or_config, "act"):
            self.agent = agent_or_config
        else:
            self.agent = DQNAgent(agent_or_config)
            if self.verbose:
                logger.info("TutorialRunner created agent from config")

    # ── Public API ────────────────────────────────────────────────────

    def run_scenario(self, scenario: TutorialScenario) -> TutorialResult:
        """Run one scenario, return a rich TutorialResult with structured logs."""
        env = self.env
        self._reset_tutorial(env)
        scenario.setup(env)

        env._update_train_heat()
        env.rmgc.set_layout(
            yard=env.yard, rail=env.rail, num_tracks=env.num_tracks,
        )

        total_reward = 0.0
        agent_moves = 0
        move_records: List[MoveRecord] = []
        step_events: List[StepEvent] = []
        move_log_raw: List[Dict] = []
        last_transition: Optional[Transition] = None

        for step in range(scenario.max_steps):
            state, reward, done, info = env.step_all_cranes(self.agent)

            executed = info.get("executed", [])
            transitions = info.get("transitions", [])

            # ── Separate move vs IDLE transitions ─────────────────────
            _IDLE_SENTINEL = (-1, -1, -1)
            move_transitions = []
            idle_transitions = []
            for t in transitions:
                if getattr(t, "source_pos_down", None) == _IDLE_SENTINEL:
                    idle_transitions.append(t)
                else:
                    move_transitions.append(t)

            # ── Build MoveRecords from executed + move transitions ─────
            for i, ex in enumerate(executed):
                agent_moves += 1
                move_log_raw.append(ex)

                move_reward = 0.0
                if i < len(move_transitions):
                    move_reward = move_transitions[i].reward

                move_records.append(MoveRecord(
                    step=step,
                    move_num=agent_moves,
                    move_type=ex.get("move_type", ""),
                    container_id=ex.get("container_id", ex.get("truck_id", "")),
                    distance_m=ex.get("distance_m", 0.0),
                    time_s=ex.get("time_s", 0.0),
                    reward=move_reward,
                    proximity_bonus=ex.get("proximity_bonus"),
                    src_row=ex.get("src_row"),
                    src_bay=ex.get("src_bay"),
                    dst_row=ex.get("dst_row"),
                    dst_bay=ex.get("dst_bay"),
                ))

            # ── Track IDLE penalties as visible events ─────────────────
            for idle_t in idle_transitions:
                step_events.append(StepEvent(
                    step=step,
                    event_type="IDLE",
                    reward=idle_t.reward,
                    detail=(f"Crane chose IDLE"
                            f" (penalty={idle_t.reward:+.1f})"
                            if idle_t.reward != 0.0 else "Crane chose IDLE"),
                ))

            # ── Accumulate step reward ────────────────────────────────
            total_reward += reward

            # ── Track non-move reward (departures, waiting, etc.) ─────
            accounted = (sum(t.reward for t in move_transitions)
                         + sum(t.reward for t in idle_transitions))
            non_move_reward = reward - accounted
            if abs(non_move_reward) > 1e-6:
                step_events.append(StepEvent(
                    step=step,
                    event_type="non_move",
                    reward=non_move_reward,
                    detail=f"step {step}: non-move reward component",
                ))

            # ── Remember transitions for agent learning ───────────────
            for t in transitions:
                self.agent.remember(t)
                last_transition = t

            if scenario.check_success(env):
                total_reward += TUTORIAL_REWARD_SUCCESS
                step_events.append(StepEvent(
                    step=step,
                    event_type="SUCCESS_BONUS",
                    reward=TUTORIAL_REWARD_SUCCESS,
                    detail=f"Scenario completed at step {step + 1}",
                ))
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
                    move_log=move_records,
                    step_events=step_events,
                    move_log_raw=move_log_raw,
                )

            if done:
                break

        total_reward += TUTORIAL_REWARD_TIMEOUT
        step_events.append(StepEvent(
            step=scenario.max_steps - 1,
            event_type="TIMEOUT_PENALTY",
            reward=TUTORIAL_REWARD_TIMEOUT,
            detail=f"Scenario timed out after {scenario.max_steps} steps",
        ))
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
            move_log=move_records,
            step_events=step_events,
            move_log_raw=move_log_raw,
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
        early_stop_check: Optional[Callable[[int, int], bool]] = None,
    ) -> Dict[str, object]:
        """Train with tiered progression — master each tier before advancing.

        One-shot graduation: when a tier is mastered, any scenario in that
        tier with ``repeatable=False`` is permanently graduated and skipped
        during future maintenance replays.  This saves training time on
        deterministic primitives (S1–S8, S25) that don't benefit from
        ongoing practice.

        Args:
            early_stop_check: Optional callback ``(epoch, current_tier) -> bool``.
                If it returns ``True``, training is aborted early.
        """
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

        # One-shot graduation: track which scenario IDs have graduated.
        # A one-shot scenario graduates when its tier is first mastered.
        graduated: Set[int] = set()

        if self.verbose:
            self._print_tier_plan(tiers)

        for epoch in range(epochs):
            self.agent.set_tutorial_epsilon(epoch)
            self.agent.set_tutorial_noise(epoch)     # anneal NoisyNet (no-op for others)
            steps_this_epoch = 0

            # ── Run current tier (every epoch) ───────────────────────
            steps_this_epoch += self._run_tier(
                tiers[current_tier], history, learn_every,
            )

            # ── Maintenance: replay mastered tiers periodically ──────
            # Uses _run_tier_maintenance which skips graduated one-shots.
            if current_tier > 0 and epoch % maintenance_interval == 0:
                for ti in range(current_tier):
                    steps_this_epoch += self._run_tier_maintenance(
                        tiers[ti], history, learn_every, graduated,
                    )

            # Extra optimization passes
            for _ in range(min(10, steps_this_epoch // 4)):
                self.agent.optimize()

            # ── Check tier mastery ───────────────────────────────────
            rates = self._get_pass_rates(history)
            self._log_progress(epoch + 1, epochs, rates, tiers, current_tier,
                               mastery_threshold, log_every,
                               graduated=graduated)

            epochs_in_tier = epoch - tier_start_epoch.get(current_tier, 0)
            if epochs_in_tier >= min_epochs and epochs_in_tier >= window_size:
                if self._tier_mastered(tiers[current_tier], rates, mastery_threshold):
                    # Graduate one-shot scenarios in the mastered tier
                    newly_graduated = self._graduate_one_shots(
                        tiers[current_tier], graduated,
                    )

                    if self.verbose:
                        print(f"\n  ✓ TIER {current_tier} MASTERED "
                              f"({tiers[current_tier].name}) at epoch {epoch + 1}",
                              flush=True)
                        if newly_graduated:
                            grad_ids = ", ".join(f"S{sid}" for sid in sorted(newly_graduated))
                            print(f"    → Graduated one-shots: {grad_ids}", flush=True)

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
                                graduated=graduated,
                            )
                        break

                    tier_start_epoch[current_tier] = epoch + 1
                    if self.verbose:
                        print(f"  → Advancing to Tier {current_tier}: "
                              f"{tiers[current_tier].name} "
                              f"({len(tiers[current_tier].scenarios)} scenarios)",
                              flush=True)

            # ── Early stopping callback ────────────────────────────
            if early_stop_check and early_stop_check(epoch + 1, current_tier):
                if self.verbose:
                    print(f"\n  EARLY STOPPED at epoch {epoch + 1} "
                          f"(stuck at tier {current_tier}: "
                          f"{tiers[current_tier].name})", flush=True)
                break

        final_rates = self._get_pass_rates(history)
        self.agent.clear_epsilon_override()

        if self.verbose and not mastered:
            print(f"\n  Training ended after {epoch + 1} epochs "
                  f"(Tier {current_tier}/{n_tiers - 1})", flush=True)
            self._log_progress(
                epoch + 1, epochs, final_rates, tiers,
                current_tier, mastery_threshold,
                log_every=1, force=True,
                graduated=graduated,
            )

        early_stopped = (
            not mastered
            and early_stop_check is not None
            and early_stop_check(epoch + 1, current_tier)
        )

        return {
            "epochs_completed": epoch + 1,
            "mastered": mastered,
            "early_stopped": early_stopped,
            "current_tier": current_tier,
            "n_tiers": n_tiers,
            "pass_rates": {sid: final_rates.get(sid, 0) for sid in all_sids},
            "scenario_names": scenario_names,
            "graduated": sorted(graduated),
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

    def _run_tier_maintenance(
        self,
        tier: TutorialTier,
        history: Dict[int, deque],
        learn_every: int,
        graduated: Set[int],
    ) -> int:
        """Run a tier for maintenance, **skipping** graduated one-shot scenarios.

        Repeatable scenarios always run.  One-shot scenarios that have
        been graduated (mastered and no longer need practice) are skipped.
        If all scenarios in the tier are graduated, returns 0.
        """
        total_steps = 0
        for sc in tier.scenarios:
            if sc.id in graduated:
                continue
            result = self.run_scenario(sc)
            history[sc.id].append(1 if result.passed else 0)
            total_steps += result.steps
            if total_steps % learn_every == 0:
                self.agent.optimize()
        return total_steps

    @staticmethod
    def _graduate_one_shots(
        tier: TutorialTier,
        graduated: Set[int],
    ) -> Set[int]:
        """Graduate non-repeatable scenarios in a mastered tier.

        Returns the set of newly graduated scenario IDs.
        """
        newly = set()
        for sc in tier.scenarios:
            if not sc.repeatable and sc.id not in graduated:
                graduated.add(sc.id)
                newly.add(sc.id)
        return newly

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
        graduated: Optional[Set[int]] = None,
    ) -> None:
        if not self.verbose or (not force and epoch % log_every != 0):
            return

        graduated = graduated or set()
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
                if sc.id in graduated:
                    tag = "🎓"  # graduated one-shot
                elif rate >= threshold:
                    tag = "OK"
                else:
                    tag = "  "
                print(f"      [{bar}] {rate:5.1%} {tag} "
                      f"S{sc.id}: {sc.name}", flush=True)

        active_scs = [sc for t in tiers[:current_tier + 1] for sc in t.scenarios]
        active_rates = [rates.get(sc.id, 0.0) for sc in active_scs]
        avg = sum(active_rates) / max(len(active_rates), 1)
        n_ok = sum(1 for r in rates.values() if r >= threshold)
        n_grad = len(graduated)
        eps = (f"{self.agent._get_epsilon():.3f}"
               if hasattr(self.agent, "_get_epsilon") else "?")
        print(f"    Average (active): {avg:.1%} | "
              f"Mastered: {n_ok}/{len(ALL_SCENARIOS)} | "
              f"Graduated: {n_grad} | ε={eps}", flush=True)

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
    """Remove all containers from an StorageYard."""
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
    from simulation.core.facilities.base import EMPTY_SLOT as _EMPTY
    # Bay-level arrays (existing)
    parking.occupied[:] = False
    parking.truck_ids.fill(None)
    parking._truck_spots.clear()
    # Flat spatial grid (new)
    parking.occupancy_flat[:] = False
    parking.position_grid[:] = _EMPTY
    for i in range(len(parking._records)):
        parking._records[i] = None
    parking._id_to_idx.clear()
    parking._free_indices = list(range(len(parking._records) - 1, -1, -1))


def _clear_rail(rail) -> None:
    """Remove all trains from rail yard."""
    from simulation.core.facilities.base import EMPTY_SLOT as _EMPTY
    # Dict-based slot tracking (existing)
    rail._train_to_slot.clear()
    rail._track_trains.clear()
    # Spatial grid (new)
    rail.occupancy_mask[:] = False
    rail.position_grid[:] = _EMPTY
    for i in range(len(rail._records)):
        rail._records[i] = None
    rail._id_to_idx.clear()
    rail._free_indices = list(range(len(rail._records) - 1, -1, -1))
    rail._train_records.clear()


class _StubDayPlan:
    """Minimal day plan that keeps _check_day_end happy.

    Sets end-of-day far in the future so tutorials don't time-expire.
    """

    def __init__(self, base_time):
        self.date = base_time.replace(hour=0, minute=0, second=0)
        self.todays_trains = []
        self.trucks_today = []