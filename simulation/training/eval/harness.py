# simulation/training/eval/harness.py
"""Evaluation harness: run a parameter matrix and collect metrics."""
from __future__ import annotations

import time
from typing import Callable, List, Optional

import pandas as pd

from simulation.training.eval.metrics import EvalRunResult, SubTaskProgress
from simulation.training.eval.scenario import EvalParams, ScalableEvalScenario
from simulation.training.tutorial_runner import TutorialRunner


# ================================================================
# Core runner
# ================================================================

def _warmup_on_scenario(
    agent,
    runner: TutorialRunner,
    params: EvalParams,
    warmup_epochs: int,
    warmup_epsilon: float,
    grad_steps_per_epoch: int = 4,
    verbose: bool = True,
) -> None:
    """Run warmup training epochs on a single scenario before evaluation.

    The agent learns on varied seeds of the same parameter combo so it can
    adapt to the scenario's characteristics.  Transitions are stored in the
    replay buffer (via ``run_scenario`` → ``remember``) and gradient steps
    are taken after each epoch.

    Training is cumulative — weights are NOT restored between combos.
    """
    if not getattr(agent, "is_trainable", True):
        return  # heuristic or non-trainable agent — skip

    old_eps = agent.epsilon_override
    agent.epsilon_override = warmup_epsilon

    for epoch in range(warmup_epochs):
        wp = EvalParams(
            n_imports=params.n_imports,
            n_exports=params.n_exports,
            n_delivery_trucks=params.n_delivery_trucks,
            n_pickup_trucks=params.n_pickup_trucks,
            yard_fill_pct=params.yard_fill_pct,
            seed=params.seed + epoch * 137,  # offset from eval seeds
            safety_factor=params.safety_factor,
        )
        scenario = ScalableEvalScenario(wp)
        runner.run_scenario(scenario)  # remember() called inside

        # Gradient steps
        if hasattr(agent, "optimize"):
            for _ in range(grad_steps_per_epoch):
                agent.optimize()

    agent.epsilon_override = old_eps

    if verbose:
        print(f"  ⚡ Warmup: {warmup_epochs} epochs on {params.label}")


def run_eval_matrix(
    agent,
    env_factory: Callable,
    param_grid: List[EvalParams],
    n_repeats: int = 5,
    warmup_epochs: int = 0,
    warmup_epsilon: float = 0.20,
    warmup_grad_steps: int = 4,
    eval_epsilon: float = 0.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run evaluation across a parameter matrix with optional warmup.

    For each parameter combo the agent can first train for
    ``warmup_epochs`` episodes (with exploration at ``warmup_epsilon``
    and ``warmup_grad_steps`` gradient updates per episode) before
    switching to ``eval_epsilon`` for the measured runs.

    Training is **cumulative** across combos — weights are not restored.

    Args:
        agent: Trained DQN agent (or heuristic).
        env_factory: Callable that creates a TerminalEnv instance.
        param_grid: List of :class:`EvalParams` to evaluate.
        n_repeats: Number of eval runs per parameter set (different seeds).
        warmup_epochs: Training episodes per combo before eval (0 = skip).
        warmup_epsilon: Exploration rate during warmup.
        warmup_grad_steps: Gradient steps per warmup episode.
        eval_epsilon: Exploration rate during measured eval runs.
        verbose: Print progress.

    Returns:
        :class:`pd.DataFrame` with one row per (param_set, repeat).
    """
    old_eps = getattr(agent, "epsilon_override", None)

    runner = TutorialRunner(
        env_factory=env_factory,
        agent_or_config=agent,
        verbose=False,
    )

    rows: List[dict] = []
    total = len(param_grid) * n_repeats

    for p_idx, params in enumerate(param_grid):
        # ── Warmup phase (learn on this scenario) ──────────────────
        if warmup_epochs > 0:
            _warmup_on_scenario(
                agent, runner, params,
                warmup_epochs=warmup_epochs,
                warmup_epsilon=warmup_epsilon,
                grad_steps_per_epoch=warmup_grad_steps,
                verbose=verbose,
            )

        # ── Eval phase ─────────────────────────────────────────────
        if hasattr(agent, "epsilon_override"):
            agent.epsilon_override = eval_epsilon

        for rep in range(n_repeats):
            # Vary seed per repeat for layout variance
            rep_params = EvalParams(
                n_imports=params.n_imports,
                n_exports=params.n_exports,
                n_delivery_trucks=params.n_delivery_trucks,
                n_pickup_trucks=params.n_pickup_trucks,
                yard_fill_pct=params.yard_fill_pct,
                seed=params.seed + rep * 1000,
                safety_factor=params.safety_factor,
            )
            scenario = ScalableEvalScenario(rep_params)

            t0 = time.time()
            result = runner.run_scenario(scenario)
            wall_time = time.time() - t0

            # Get granular progress from the env after run
            progress = scenario.check_progress(runner.env)

            eval_result = EvalRunResult(
                params=rep_params,
                repeat=rep,
                passed=result.passed,
                steps=result.steps,
                agent_moves=result.agent_moves,
                total_reward=result.total_reward,
                sub_tasks=progress,
                move_type_counts=dict(result.move_type_counts),
                wall_time_s=wall_time,
            )

            # Flatten to dict for DataFrame
            row = {
                "n_imports": params.n_imports,
                "n_exports": params.n_exports,
                "n_delivery_trucks": params.n_delivery_trucks,
                "n_pickup_trucks": params.n_pickup_trucks,
                "yard_fill_pct": params.yard_fill_pct,
                "total_containers": rep_params.total_containers,
                "seed": rep_params.seed,
                "repeat": rep,
                "passed": eval_result.passed,
                "completion_pct": eval_result.completion_pct,
                "steps": eval_result.steps,
                "agent_moves": eval_result.agent_moves,
                "total_reward": eval_result.total_reward,
                "moves_per_container": eval_result.moves_per_container,
                "reshuffle_ratio": eval_result.reshuffle_ratio,
                "productive_move_ratio": eval_result.productive_move_ratio,
                "wall_time_s": wall_time,
                "departure_window_min": scenario._feasible_seconds / 60.0,
                "max_steps": scenario.max_steps,
            }

            # Per-sub-task columns
            for st in progress:
                row[f"{st.name}_completed"] = st.completed
                row[f"{st.name}_total"] = st.total
                row[f"{st.name}_pct"] = st.pct

            # Per-move-type counts
            for mt, count in result.move_type_counts.items():
                row[f"moves_{mt}"] = count

            rows.append(row)

            if verbose:
                run_num = p_idx * n_repeats + rep + 1
                tag = "PASS" if eval_result.passed else "FAIL"
                pcts = " ".join(f"{st.name}={st.pct:.0%}" for st in progress)
                print(
                    f"  [{run_num:3d}/{total}] {tag} "
                    f"{params.label} rep={rep} "
                    f"compl={eval_result.completion_pct:.0%} "
                    f"moves={eval_result.agent_moves} "
                    f"R={eval_result.total_reward:+.1f} "
                    f"[{pcts}]"
                )

    # Restore epsilon
    if old_eps is not None and hasattr(agent, "epsilon_override"):
        agent.epsilon_override = old_eps
    elif hasattr(agent, "clear_epsilon_override"):
        agent.clear_epsilon_override()

    return pd.DataFrame(rows)


# ================================================================
# Default parameter grids
# ================================================================

def default_eval_grid(base_seed: int = 7000) -> List[EvalParams]:
    """Primary evaluation matrix: load sizes x yard fill levels.

    Train-only (imports + exports), no trucks.

    Returns:
        15 :class:`EvalParams` (5 load sizes x 3 fill levels).
    """
    grid: List[EvalParams] = []
    load_sizes = [10, 20, 40, 60, 80]
    yard_fills = [0.0, 0.25, 0.50]

    for i, n in enumerate(load_sizes):
        for j, fill in enumerate(yard_fills):
            n_imp = n // 2
            n_exp = n - n_imp
            grid.append(EvalParams(
                n_imports=n_imp,
                n_exports=n_exp,
                yard_fill_pct=fill,
                seed=base_seed + i * 100 + j * 10,
            ))
    return grid


def default_eval_grid_with_trucks(base_seed: int = 8000) -> List[EvalParams]:
    """Extended grid including truck operations.

    Load is split roughly ⅓ imports, ⅓ exports, ⅓ trucks.

    Returns:
        12 :class:`EvalParams` (4 load sizes x 3 fill levels).
    """
    grid: List[EvalParams] = []
    load_sizes = [12, 24, 48, 60]
    yard_fills = [0.0, 0.25, 0.50]

    for i, n in enumerate(load_sizes):
        for j, fill in enumerate(yard_fills):
            n_imp = n // 3
            n_exp = n // 3
            n_trucks = n - n_imp - n_exp
            n_del = n_trucks // 2
            n_pick = n_trucks - n_del
            grid.append(EvalParams(
                n_imports=n_imp,
                n_exports=n_exp,
                n_delivery_trucks=n_del,
                n_pickup_trucks=n_pick,
                yard_fill_pct=fill,
                seed=base_seed + i * 100 + j * 10,
            ))
    return grid


# ================================================================
# Aggregation
# ================================================================

def summarize_eval(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate eval results across repeats.

    Groups by parameter columns and computes mean/std for key metrics.
    """
    group_cols = [
        "n_imports", "n_exports", "n_delivery_trucks",
        "n_pickup_trucks", "yard_fill_pct", "total_containers",
    ]
    # Only group by columns that exist
    group_cols = [c for c in group_cols if c in df.columns]

    agg = df.groupby(group_cols, dropna=False).agg(
        n_runs=("passed", "count"),
        pass_rate=("passed", "mean"),
        completion_mean=("completion_pct", "mean"),
        completion_std=("completion_pct", "std"),
        moves_mean=("agent_moves", "mean"),
        moves_per_container_mean=("moves_per_container", "mean"),
        reshuffle_ratio_mean=("reshuffle_ratio", "mean"),
        reward_mean=("total_reward", "mean"),
        reward_std=("total_reward", "std"),
        steps_mean=("steps", "mean"),
    ).reset_index()
    return agg
