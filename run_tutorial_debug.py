#!/usr/bin/env python3
"""Run full tutorial debug (all tiers) — mirrors notebook logic.

Trains each tier sequentially, then runs greedy debug pass.
Outputs structured move logs for analysis.
"""
import sys, os, time, random
import numpy as np
import torch
from collections import defaultdict, deque
from typing import Optional, List, Dict

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

from simulation.rl.agent_registry import create_agent
from simulation.training.curriculum_trainer import (
    create_env_factory, build_agent_config,
)
from simulation.training.scenarios import (
    ALL_SCENARIOS, SCENARIO_BY_ID, TIER_DEFS,
    ONE_SHOT_SCENARIOS, REPEATABLE_SCENARIOS,
    TutorialResult, MoveRecord, StepEvent,
)
from simulation.training.tutorial_runner import (
    TutorialRunner, TUTORIAL_TIERS,
    TUTORIAL_REWARD_SUCCESS, TUTORIAL_REWARD_TIMEOUT,
)

try:
    from simulation.rl.variants.networks.noisy_linear import NoisyLinear
    _HAS_NOISY = True
except ImportError:
    _HAS_NOISY = False

# ── Config ────────────────────────────────────────────────────────────
VARIANT = "noisynet"
BACKBONE = "baseline"
MAX_EPOCHS_PER_TIER = 75
MASTERY_THRESHOLD = 0.9
WINDOW_SIZE = 20
MIN_EPOCHS = 10
LOG_EVERY = 10

# ── Setup ─────────────────────────────────────────────────────────────
env_factory = create_env_factory(rows=5, bays=58, tiers=5, tracks=7, split_factor=20)
base_cfg = build_agent_config(rows=5, bays=58, tiers=5, split_factor=20, tracks=7)
agent = create_agent(VARIANT, base_cfg, backbone_variant=BACKBONE)

n_params = sum(p.numel() for p in agent.q_net.parameters())
print(f"{VARIANT} agent created (from scratch)")
print(f"  Parameters: {n_params:,}")
print(f"  Device:     {base_cfg.device}")
print(f"  Scenarios:  {len(ALL_SCENARIOS)} total")
print(f"  Tiers:      {len(TUTORIAL_TIERS)}")

runner = TutorialRunner(
    env_factory=env_factory,
    agent_or_config=agent,
    verbose=False,
)

ALL_RESULTS: dict = {}
TIER_TRAINING: dict = {}


# ── Functions ─────────────────────────────────────────────────────────

def train_single_tier(tier_idx):
    tier = TUTORIAL_TIERS[tier_idx]
    history = {s.id: deque(maxlen=WINDOW_SIZE) for s in tier.scenarios}
    t0 = time.time()
    mastered = False
    epoch = 0

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Training Tier {tier_idx}: {tier.name}")
    print(f"  Scenarios: {', '.join(f'S{s.id}' for s in tier.scenarios)}")
    print(f"  Max epochs: {MAX_EPOCHS_PER_TIER}  |  Mastery: {MASTERY_THRESHOLD:.0%}")
    print(sep)

    for epoch in range(MAX_EPOCHS_PER_TIER):
        agent.set_tutorial_epsilon(epoch)
        steps = 0
        for sc in tier.scenarios:
            result = runner.run_scenario(sc)
            history[sc.id].append(1 if result.passed else 0)
            steps += result.steps

        n_optim = min(10, max(1, steps // 4))
        for _ in range(n_optim):
            agent.optimize()

        pass_rates = {}
        for sid, hist in history.items():
            pass_rates[sid] = sum(hist) / len(hist) if len(hist) > 0 else 0.0
        avg_rate = np.mean(list(pass_rates.values()))

        if (epoch + 1) % LOG_EVERY == 0 or epoch == 0:
            rates_str = "  ".join(f"S{sid}={r:.0%}" for sid, r in sorted(pass_rates.items()))
            print(f"  Epoch {epoch+1:>3d}/{MAX_EPOCHS_PER_TIER}  avg={avg_rate:.0%}  {rates_str}")

        if epoch + 1 >= MIN_EPOCHS and epoch + 1 >= WINDOW_SIZE:
            if all(r >= MASTERY_THRESHOLD for r in pass_rates.values()):
                mastered = True
                rates_str = "  ".join(f"S{sid}={r:.0%}" for sid, r in sorted(pass_rates.items()))
                print(f"  Epoch {epoch+1:>3d}/{MAX_EPOCHS_PER_TIER}  avg={avg_rate:.0%}  {rates_str}")
                print(f"  >>> MASTERED at epoch {epoch+1}")
                break

    wall = time.time() - t0
    status = "MASTERED" if mastered else "NOT MASTERED"
    print(f"\n  Result: {status} in {epoch+1} epochs ({wall:.1f}s)")
    print(sep)

    return {
        "tier": tier_idx,
        "tier_name": tier.name,
        "epochs": epoch + 1,
        "mastered": mastered,
        "pass_rates": pass_rates,
        "wall_time": wall,
    }


def debug_tier(tier_idx):
    saved_override = agent.epsilon_override
    agent.epsilon_override = 0.0

    noisy_overridden = False
    if _HAS_NOISY:
        noisy_mods = [m for m in agent.q_net.modules() if isinstance(m, NoisyLinear)]
        if noisy_mods:
            noisy_overridden = True
            agent._pre_act_hook = lambda: None
            agent._post_optimize_hook = lambda: None
            for m in noisy_mods:
                m.weight_epsilon.zero_()
                m.bias_epsilon.zero_()

    tier = TUTORIAL_TIERS[tier_idx]
    results = []
    for sc in tier.scenarios:
        result = runner.run_scenario(sc)
        results.append(result)
        ALL_RESULTS[sc.id] = result

    agent.epsilon_override = saved_override
    if noisy_overridden:
        for attr in ("_pre_act_hook", "_post_optimize_hook"):
            if attr in agent.__dict__:
                del agent.__dict__[attr]

    # Display
    n_pass = sum(1 for r in results if r.passed)
    line = chr(9472) * 60
    print(f"\n{line}")
    print(f"  Tier {tier_idx}: {tier.name}: {n_pass}/{len(results)} passed")
    print(line)
    for r in results:
        sc = SCENARIO_BY_ID[r.scenario_id]
        tag = "one-shot" if not sc.repeatable else "repeat"
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] S{r.scenario_id:>2d} ({tag:>8s})  {r.name:<35s} "
              f"moves={r.agent_moves:>2d}  R={r.total_reward:>+8.2f}")

    # Detailed per-scenario
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        sc = SCENARIO_BY_ID[r.scenario_id]
        tag = "repeatable" if sc.repeatable else "one-shot"
        sep2 = "=" * 90
        print(f"\n{sep2}")
        print(f"S{r.scenario_id}: {r.name}  [{status}]  ({tag})")
        print(f"  total_reward={r.total_reward:+.4f}  "
              f"steps={r.steps}  moves={r.agent_moves}")
        if r.move_type_counts:
            types_str = ", ".join(f"{k}={v}" for k, v in sorted(r.move_type_counts.items()))
            print(f"  move types: {types_str}")
        if r.reward_breakdown:
            bd_str = ", ".join(f"{k}={v:+.2f}" for k, v in sorted(r.reward_breakdown.items()))
            print(f"  reward breakdown: {bd_str}")
        print(sep2)

        if not r.move_log and not r.step_events:
            print("  (no moves or events)")
            continue

        timeline = []
        for m in r.move_log:
            timeline.append((m.step, 1, "move", m))
        for e in r.step_events:
            timeline.append((e.step, 0, "event", e))
        timeline.sort(key=lambda x: (x[0], x[1]))

        hdr = "  {:>3s}  {:>4s}  {:>6s}  {:<28s}  {:<14s}  {:>7s}  {:>7s}  {:>9s}  {}"
        print(hdr.format("#", "Step", "Kind", "Move Type", "ID", "Dist", "Time", "Reward", "Notes"))
        print("  {}  {}  {}  {}  {}  {}  {}  {}  {}".format(
            "-"*3, "-"*4, "-"*6, "-"*28, "-"*14, "-"*7, "-"*7, "-"*9, "-"*20))

        for step_val, _, kind, data in timeline:
            if kind == "move":
                m = data
                flags = []
                productive = {"TRAIN_TO_YARD", "TRUCK_TO_YARD", "YARD_TO_TRAIN",
                              "YARD_TO_TRUCK", "TRAIN_TO_TRUCK", "TRUCK_TO_TRAIN",
                              "YARD_TO_TERMINAL_TRUCK"}
                if m.move_type in productive and m.reward < 0:
                    flags.append("NEG_PROD")
                if m.move_type == "YARD_TO_YARD" and m.reward > 0:
                    flags.append("POS_RESTACK")
                if abs(m.reward) > 15:
                    flags.append("LARGE_R")
                if m.proximity_bonus and m.proximity_bonus > 0:
                    flags.append(f"prox={m.proximity_bonus:.2f}")
                if m.src_bay is not None and m.dst_bay is not None:
                    flags.append(f"r{m.src_row}b{m.src_bay}->r{m.dst_row}b{m.dst_bay}")
                flag_str = "  ".join(flags)
                print(f"  {m.move_num:>3d}  {step_val:>4d}  {'move':>6s}  "
                      f"{m.move_type:<28s}  {m.container_id:<14s}  "
                      f"{m.distance_m:>7.1f}  {m.time_s:>7.1f}  "
                      f"{m.reward:>+9.4f}  {flag_str}")
            else:
                e = data
                print(f"  {'':>3s}  {step_val:>4d}  {'event':>6s}  "
                      f"{e.event_type:<28s}  {'':14s}  {'':>7s}  {'':>7s}  "
                      f"{e.reward:>+9.4f}  {e.detail}")

        move_r = sum(m.reward for m in r.move_log)
        event_r = sum(e.reward for e in r.step_events)
        print(f"\n  Reward: moves={move_r:+.4f}  events={event_r:+.4f}  "
              f"total={r.total_reward:+.4f}")
        residual = r.total_reward - move_r - event_r
        if abs(residual) > 0.01:
            print(f"  *** RESIDUAL = {residual:+.4f} (unaccounted reward!) ***")

    return results


# ══════════════════════════════════════════════════════════════════════
# Run all tiers
# ══════════════════════════════════════════════════════════════════════

total_start = time.time()

for tier_idx in range(len(TUTORIAL_TIERS)):
    summary = train_single_tier(tier_idx)
    TIER_TRAINING[tier_idx] = summary
    print("\n--- Greedy Debug (epsilon=0, noise off) ---")
    debug_tier(tier_idx)

total_wall = time.time() - total_start

# ══════════════════════════════════════════════════════════════════════
# Global Summary
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("GLOBAL SUMMARY")
print("=" * 90)

n_pass = sum(1 for r in ALL_RESULTS.values() if r.passed)
n_total = len(ALL_RESULTS)
print(f"\nOverall: {n_pass}/{n_total} passed ({n_pass/n_total*100:.0f}%)")

print(f"\n{'ID':<6s} {'Name':<35s} {'Type':<10s} {'Pass':<6s} {'Steps':>6s} {'Moves':>6s} {'Total R':>9s}  Move Types")
print("-" * 120)
for sid in sorted(ALL_RESULTS.keys()):
    r = ALL_RESULTS[sid]
    sc = SCENARIO_BY_ID[sid]
    tag = "one-shot" if not sc.repeatable else "repeat"
    status = "PASS" if r.passed else "FAIL"
    types_str = ", ".join(f"{k}={v}" for k, v in sorted(r.move_type_counts.items()))
    print(f"S{sid:<5d} {r.name:<35s} {tag:<10s} {status:<6s} {r.steps:>6d} {r.agent_moves:>6d} {r.total_reward:>+9.2f}  {types_str}")

print(f"\nTraining Summary:")
print(f"  {'Tier':<6s}  {'Name':<25s}  {'Epochs':>7s}  {'Status':<12s}  {'Time':>8s}")
print(f"  {'-'*6}  {'-'*25}  {'-'*7}  {'-'*12}  {'-'*8}")
for tidx in sorted(TIER_TRAINING.keys()):
    t = TIER_TRAINING[tidx]
    status = "MASTERED" if t["mastered"] else "not mastered"
    print(f"  {tidx:<6d}  {t['tier_name']:<25s}  {t['epochs']:>7d}  {status:<12s}  {t['wall_time']:>7.1f}s")

print(f"\nTotal wall time: {total_wall:.1f}s ({total_wall/60:.1f}min)")

# Cleanup
import gc
agent.q_net.cpu()
agent.target_net.cpu()
agent.replay = None
del agent, runner
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
