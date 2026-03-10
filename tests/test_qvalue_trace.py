"""Diagnostic: dump source Q-values per region at each step.

Captures the agent's Q-values for RAIL vs YARD sources to understand
why the agent prefers YARD_TO_YARD over TRAIN_TO_YARD.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.training.curriculum_trainer import create_env_factory, build_agent_config
from simulation.training.eval import ScalableEvalScenario, EvalParams
from simulation.training.tutorial_runner import TutorialRunner
from simulation.rl.agent_registry import create_agent
from simulation.rl.multihead_dqn.config import Dims
from simulation.training.scenarios._base import TUTORIAL_TIME

ROWS, BAYS, TIERS = 5, 58, 5
TRACKS, SPLIT_FACTOR = 7, 20

env_factory = create_env_factory(
    rows=ROWS, bays=BAYS, tiers=TIERS,
    tracks=TRACKS, split_factor=SPLIT_FACTOR,
    export_ratio=1.0, max_retries=2,
)
cfg = build_agent_config(
    rows=ROWS, bays=BAYS, tiers=TIERS,
    split_factor=SPLIT_FACTOR, tracks=TRACKS,
)
dims: Dims = cfg.unified


def _load_agent():
    agent = create_agent('baseline', cfg, backbone_variant='baseline')
    p = ROOT / 'runs' / 'tutorial_checkpoints' / 'baseline_best.pt'
    ckpt = torch.load(str(p), map_location='cpu', weights_only=False)
    agent.q_net.load_state_dict(ckpt['q_net'])
    agent.target_net.load_state_dict(ckpt['target_net'])
    agent.step_count = ckpt['step_count']
    return agent


def _region_q_stats(q_src_np, src_down):
    """Compute Q-value statistics per region."""
    R, S_down, T = src_down.shape
    spatial_q = q_src_np[:R * S_down * T].reshape(R, S_down, T)
    idle_q = float(q_src_np[-1])

    stats = {}
    for name, r_start, r_end in [
        ('RAIL', dims.rail_row_start, dims.rail_row_end),
        ('YARD', dims.yard_row_start, dims.yard_row_end),
        ('PARKING', dims.parking_row_start, dims.parking_row_end),
        ('QUEUE', dims.queue_row_start, dims.queue_row_end),
    ]:
        mask = src_down[r_start:r_end]
        q = spatial_q[r_start:r_end]
        if mask.any():
            valid = q[mask]
            stats[name] = {
                'n': int(mask.sum()),
                'max': float(valid.max()),
                'mean': float(valid.mean()),
                'min': float(valid.min()),
            }

    # Global argmax (including IDLE)
    valid_flat = np.append(src_down.flatten(), True)
    masked_q = np.where(valid_flat, q_src_np, -1e9)
    best_idx = int(masked_q.argmax())
    if best_idx == len(q_src_np) - 1:
        argmax_region = 'IDLE'
    else:
        best_row = best_idx // (S_down * T)
        argmax_region = dims.region_of(best_row)

    return stats, idle_q, argmax_region


def _capture_q_before_step(agent, env):
    """Capture source Q-values for the current state."""
    state = env._encode_state()
    source_mask = env.encoder.get_source_mask(
        env.trains, env.trucks, env.current_time,
    )
    if not source_mask.any():
        return None

    state_t = agent._to_tensor(state).unsqueeze(0)
    agent.q_net.eval()
    with torch.no_grad():
        encoded = agent.q_net.encode_state(state_t)
        src_down = agent._downsample_mask(source_mask)
        src_mask_t = agent._to_bool_tensor(src_down).unsqueeze(0)
        q_src = agent.q_net.q_source(encoded.feat_map, src_mask_t)[0]

    return _region_q_stats(q_src.cpu().numpy(), src_down)


def run_traced(agent, fill_pct, n_imp=10, n_exp=10, seed=42, max_print=30):
    """Run scenario, printing Q-values and moves at each step."""
    agent.epsilon_override = 0.0  # Pure greedy
    if hasattr(agent, 'set_noise_scale'):
        agent.set_noise_scale(0.0)  # No noise

    params = EvalParams(n_imports=n_imp, n_exports=n_exp,
                        yard_fill_pct=fill_pct, seed=seed)
    sc = ScalableEvalScenario(params)

    runner = TutorialRunner(env_factory=env_factory, agent_or_config=agent,
                            verbose=False)
    env = runner.env
    runner._reset_tutorial(env)
    sc.setup(env)
    env._update_train_heat()
    env.rmgc.set_layout(yard=env.yard, rail=env.rail, num_tracks=env.num_tracks)

    print(f'\n{"="*80}')
    print(f'Q-VALUE TRACE: {n_imp}+{n_exp}, fill={fill_pct:.0%}, '
          f'greedy (eps=0, noise=0)')
    print(f'{"="*80}')

    for step in range(sc.max_steps):
        # Capture Q-values BEFORE stepping
        result = _capture_q_before_step(agent, env)

        if result and step < max_print:
            stats, idle_q, argmax = result

            rail = stats.get('RAIL')
            yard = stats.get('YARD')

            line = f'  step {step:3d}: '
            if rail:
                line += (f'RAIL({rail["n"]:2d}) '
                         f'max={rail["max"]:+7.2f} mean={rail["mean"]:+7.2f}  ')
            else:
                line += 'RAIL( 0)  --                          '

            if yard:
                line += (f'YARD({yard["n"]:3d}) '
                         f'max={yard["max"]:+7.2f} mean={yard["mean"]:+7.2f}  ')
            else:
                line += 'YARD(  0)  --                          '

            # Train/rail diagnostics
            n_trains = len(env.trains)
            n_rail_recs = sum(1 for _ in env.rail.iter_records())
            t_sec = (env.current_time - TUTORIAL_TIME).total_seconds()
            line += f'IDLE={idle_q:+6.2f}  → {argmax}'
            line += f'  [trains={n_trains} recs={n_rail_recs} t={t_sec:.0f}s]'
            print(line)

        # Step
        state, reward, done, info = env.step_all_cranes(agent)
        executed = info.get('executed', [])

        if executed and step < max_print:
            types = [ex.get('move_type', '') for ex in executed]
            cids = [ex.get('container_id', ex.get('truck_id', ''))
                    for ex in executed]
            move_str = '  '.join(f'{t}({c})' for t, c in zip(types, cids))
            print(f'           → {move_str}')

        if sc.check_success(env):
            print(f'  ✓ SUCCESS at step {step}')
            break

    if step >= max_print - 1:
        # Print final state
        result = _capture_q_before_step(agent, env)
        if result:
            stats, idle_q, argmax = result
            rail = stats.get('RAIL')
            yard = stats.get('YARD')
            print(f'  ...')
            line = f'  step {step:3d} (final): '
            if rail:
                line += f'RAIL({rail["n"]:2d}) max={rail["max"]:+7.2f}  '
            else:
                line += 'RAIL( 0)  '
            if yard:
                line += f'YARD({yard["n"]:3d}) max={yard["max"]:+7.2f}  '
            line += f'→ {argmax}'
            print(line)

    progress = sc.check_progress(env)
    print(f'  Progress: {dict((p.name, f"{p.completed}/{p.total}") for p in progress)}')

    agent.epsilon_override = None
    return progress


def main():
    agent = _load_agent()

    # Key scenarios
    run_traced(agent, fill_pct=0.0, n_imp=10, n_exp=10, seed=42)
    run_traced(agent, fill_pct=0.25, n_imp=10, n_exp=10, seed=42)
    run_traced(agent, fill_pct=0.50, n_imp=10, n_exp=10, seed=42)

    # Also: 0 exports, only imports (no YARD export sources to compete)
    print('\n\n*** IMPORT-ONLY (no exports to distract) ***')
    run_traced(agent, fill_pct=0.50, n_imp=10, n_exp=0, seed=42)


if __name__ == '__main__':
    main()
