# simulation/training/eval/__init__.py
"""Parameterized evaluation scenario system.

Usage::

    from simulation.training.eval import (
        ScalableEvalScenario, EvalParams,
        run_eval_matrix, default_eval_grid, summarize_eval,
    )

    # Build grid
    grid = default_eval_grid()

    # Run evaluation
    df = run_eval_matrix(agent, env_factory, grid, n_repeats=5)

    # Aggregate
    summary = summarize_eval(df)
"""

from simulation.training.eval.metrics import EvalRunResult, SubTaskProgress
from simulation.training.eval.scenario import EvalParams, ScalableEvalScenario
from simulation.training.eval.timing import WorkloadEstimate, estimate_feasible_time
from simulation.training.eval.harness import (
    default_eval_grid,
    default_eval_grid_with_trucks,
    run_eval_matrix,
    summarize_eval,
)
from simulation.training.eval.heuristic import HeuristicAgent

__all__ = [
    "EvalParams",
    "EvalRunResult",
    "HeuristicAgent",
    "ScalableEvalScenario",
    "SubTaskProgress",
    "WorkloadEstimate",
    "default_eval_grid",
    "default_eval_grid_with_trucks",
    "estimate_feasible_time",
    "run_eval_matrix",
    "summarize_eval",
]
