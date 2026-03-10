"""Diagnostic: trace what the DQN agent actually does during eval scenarios.

Runs a single eval scenario and logs every move to see whether the agent
ever selects RAIL sources (TRAIN_TO_YARD) or only YARD sources.
"""
import sys
from collections import Counter
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.training.curriculum_trainer import create_env_factory, build_agent_config
from simulation.training.eval import ScalableEvalScenario, EvalParams
from simulation.training.tutorial_runner import TutorialRunner
from simulation.rl.agent_registry import create_agent

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


def _load_agent(variant='baseline', backbone='baseline'):
    agent = create_agent(variant, cfg, backbone_variant=backbone)
    ckpt_dir = ROOT / 'runs' / 'tutorial_checkpoints'
    for name in [f'{variant}_best.pt', f'{variant}.pt']:
        p = ckpt_dir / name
        if p.exists():
            ckpt = torch.load(str(p), map_location='cpu', weights_only=False)
            agent.q_net.load_state_dict(ckpt['q_net'])
            agent.target_net.load_state_dict(ckpt['target_net'])
            agent.step_count = ckpt['step_count']
            print(f'Loaded {p.name} (step {agent.step_count})')
            return agent
    raise FileNotFoundError(f'No checkpoint for {variant}')


def run_traced_scenario(agent, fill_pct, n_imp=10, n_exp=10, seed=42,
                        epsilon=0.05):
    """Run one eval scenario and return detailed move trace."""
    agent.epsilon_override = epsilon
    if hasattr(agent, 'set_noise_scale'):
        agent.set_noise_scale(1.0)

    params = EvalParams(
        n_imports=n_imp, n_exports=n_exp,
        yard_fill_pct=fill_pct, seed=seed,
    )
    sc = ScalableEvalScenario(params)
    runner = TutorialRunner(env_factory=env_factory, agent_or_config=agent,
                            verbose=False)
    result = runner.run_scenario(sc)

    # Collect progress
    progress = sc.check_progress(runner.env)
    progress_dict = {p.name: f'{p.completed}/{p.total}' for p in progress}

    agent.epsilon_override = None
    return result, progress_dict


def analyze_trace(result, label=''):
    """Print move-by-move analysis."""
    move_types = Counter()
    failed_moves = Counter()  # moves where agent tried but nothing happened
    move_details = []

    for rec in result.move_log:
        mt = rec.move_type
        move_types[mt] += 1
        move_details.append({
            'step': rec.step,
            'type': mt,
            'container': rec.container_id,
            'src_bay': rec.src_bay,
            'dst_bay': rec.dst_bay,
            'reward': rec.reward,
        })

    print(f'\n{"="*60}')
    print(f'Trace: {label}')
    print(f'{"="*60}')
    print(f'Steps: {result.steps}, Moves: {result.agent_moves}, '
          f'Passed: {result.passed}, Reward: {result.total_reward:+.1f}')

    print(f'\nMove type counts:')
    for mt, count in move_types.most_common():
        print(f'  {mt:25s}: {count}')

    # Show first 30 moves
    print(f'\nFirst 30 moves:')
    for i, m in enumerate(move_details[:30]):
        print(
            f'  [{m["step"]:3d}] {m["type"]:25s} '
            f'{m["container"]:20s} '
            f'bay {m["src_bay"]}→{m["dst_bay"]} '
            f'R={m["reward"]:+.2f}'
        )

    # Check for IDLE events
    idle_events = [e for e in result.step_events if e.event_type == 'IDLE']
    if idle_events:
        print(f'\nIDLE events: {len(idle_events)} '
              f'(total penalty: {sum(e.reward for e in idle_events):+.1f})')

    return move_types


def main():
    agent = _load_agent()

    # Test 3 fill levels for 10+10 scenario
    for fill in [0.0, 0.25, 0.50]:
        result, progress = run_traced_scenario(
            agent, fill_pct=fill, n_imp=10, n_exp=10, seed=42, epsilon=0.05,
        )
        mt = analyze_trace(result, f'10+10, fill={fill:.0%}')
        print(f'  Progress: {progress}')

        # Key diagnostic: does the agent ever pick RAIL sources?
        rail_moves = mt.get('TRAIN_TO_YARD', 0)
        yard_exports = mt.get('YARD_TO_TRAIN', 0)
        yard_reshuffles = mt.get('YARD_TO_YARD', 0)
        print(f'\n  RAIL→YARD (imports):   {rail_moves}')
        print(f'  YARD→RAIL (exports):   {yard_exports}')
        print(f'  YARD→YARD (reshuffles): {yard_reshuffles}')

    # Also test a scenario where DQN passes: 20+20 fill=0%
    print('\n' + '='*60)
    print('Control: 20+20 fill=0% (DQN passes this)')
    print('='*60)
    result, progress = run_traced_scenario(
        agent, fill_pct=0.0, n_imp=20, n_exp=20, seed=42, epsilon=0.05,
    )
    mt = analyze_trace(result, '20+20, fill=0%')
    print(f'  Progress: {progress}')


if __name__ == '__main__':
    main()
