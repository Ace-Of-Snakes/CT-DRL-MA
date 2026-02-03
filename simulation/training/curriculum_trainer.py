# simulation/training/curriculum_trainer.py
"""
Curriculum-based training for Multi-Head DQN agent.

Adapted for:
- Unified ContainerTerminalEnv (no HierarchicalEnvWrapper)
- OptimizedStorageYard / OptimizedRailYard / OptimizedParkingArea
- 8-channel split-level state encoder (C, R, S, T)
- MultiHeadDQNAgent with spatial masks
"""
import csv
import json
import random
import sys
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch

from simulation.rl.multihead_dqn.config import (
    MultiHeadDQNConfig, YardDims, BackboneConfig, HeadConfig, DQNConfig,
)
from simulation.env.state_encoder import NUM_CHANNELS
from simulation.rl.multihead_dqn.agent import MultiHeadDQNAgent


# -- Safety limits --
MAX_STEPS_PER_DAY: int = 5_000
DEBUG_INTERVAL_STEPS: int = 100


# -- Curriculum schedule (separate from network config) --
@dataclass
class CurriculumSchedule:
    """Curriculum scaling parameters."""
    start_imports: int = 20
    increment: int = 20
    max_imports: int = 220
    days_per_stage: int = 365
    epsilon_reset_per_stage: bool = True
    log_interval_days: int = 10

    @property
    def num_stages(self) -> int:
        return (self.max_imports - self.start_imports) // self.increment + 1

    def imports_for_stage(self, stage: int) -> int:
        return min(self.start_imports + stage * self.increment, self.max_imports)


# 
# -- Metrics --
# 

@dataclass
class DayMetrics:
    """Metrics for a single simulation day."""
    day_index: int
    stage: int
    date: str
    total_reward: float
    moves_executed: int
    containers_in_yard: int = 0
    trains_active: int = 0
    trucks_active: int = 0
    avg_loss: float = 0.0
    epsilon: float = 0.0
    steps: int = 0


@dataclass
class StageMetrics:
    """Aggregate metrics for a curriculum stage."""
    stage: int
    import_cap: int
    total_days: int
    total_reward: float
    avg_reward_per_day: float
    total_moves: int
    avg_moves_per_day: float
    final_epsilon: float


# 
# -- Logger --
# 

class MetricsLogger:
    """CSV + JSON logging for training runs."""

    _DAILY_COLS = [
        "day_index", "stage", "date", "total_reward", "moves_executed",
        "containers_in_yard", "trains_active", "trucks_active",
        "avg_loss", "epsilon", "steps",
    ]
    _STAGE_COLS = [
        "stage", "import_cap", "total_days", "total_reward",
        "avg_reward_per_day", "total_moves", "avg_moves_per_day",
        "final_epsilon",
    ]

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.daily_csv = self.log_dir / "daily_metrics.csv"
        self.stage_csv = self.log_dir / "stage_metrics.csv"
        self.episodes_dir = self.log_dir / "episodes"
        self.episodes_dir.mkdir(exist_ok=True)
        self._init_csv(self.daily_csv, self._DAILY_COLS)
        self._init_csv(self.stage_csv, self._STAGE_COLS)

    @staticmethod
    def _init_csv(path: Path, cols: List[str]):
        if not path.exists():
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(cols)

    def log_day(self, m: DayMetrics):
        with open(self.daily_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                m.day_index, m.stage, m.date,
                f"{m.total_reward:.4f}", m.moves_executed,
                m.containers_in_yard, m.trains_active, m.trucks_active,
                f"{m.avg_loss:.6f}", f"{m.epsilon:.4f}", m.steps,
            ])

    def log_stage(self, m: StageMetrics):
        with open(self.stage_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                m.stage, m.import_cap, m.total_days,
                f"{m.total_reward:.4f}", f"{m.avg_reward_per_day:.4f}",
                m.total_moves, f"{m.avg_moves_per_day:.2f}",
                f"{m.final_epsilon:.4f}",
            ])

    def log_episode_detail(self, stage: int, day: int, data: Dict[str, Any]):
        path = self.episodes_dir / f"stage{stage}_day{day}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)


# 
# -- Trainer --
# 

class CurriculumTrainer:
    """Trains MultiHead DQN agent through curriculum stages."""

    def __init__(
        self,
        env_factory,
        schedule: CurriculumSchedule,
        agent_config: MultiHeadDQNConfig,
        output_dir: str,
        seed: int = 42,
        verbose: bool = True,
        skip_tutorials: bool = False,
        tutorial_epochs: int = 500,
        tutorial_mastery: float = 0.9,
    ):
        self.env_factory = env_factory
        self.schedule = schedule
        self.agent_cfg = agent_config
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.verbose = verbose
        self.skip_tutorials = skip_tutorials
        self.tutorial_epochs = tutorial_epochs
        self.tutorial_mastery = tutorial_mastery

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.logger = MetricsLogger(str(self.output_dir / "logs"))

        self.env = None
        self.agent: Optional[MultiHeadDQNAgent] = None

    def _print(self, msg: str):
        if self.verbose:
            if hasattr(self, '_pbar') and self._pbar is not None:
                self._pbar.write(msg)
            else:
                print(msg, flush=True)

    def _run_tutorial_phase(self):
        """Run tutorial scenarios before curriculum stages."""
        from simulation.training.tutorial_runner import TutorialRunner
        
        self._print("\n" + "=" * 60)
        self._print("PHASE 0: Tutorial Scenarios")
        self._print(f"Target: {self.tutorial_mastery:.0%} pass rate on all scenarios")
        self._print(f"Max epochs: {self.tutorial_epochs}")
        self._print("=" * 60)
        
        runner = TutorialRunner(
            env_factory=self.env_factory,
            agent_or_config=self.agent,
            verbose=self.verbose,
        )
        
        summary = runner.train_all(
            epochs=self.tutorial_epochs,
            mastery_threshold=self.tutorial_mastery,
            window_size=20,
            min_epochs=10,
            log_every=10,
        )
        
        # Summary output
        self._print("\n" + "-" * 40)
        if summary.get('mastered', False):
            self._print(f"TUTORIALS MASTERED in {summary['epochs_completed']} epochs!")
        else:
            self._print(f"Tutorial phase ended after {summary['epochs_completed']} epochs")
            self._print(f"(Mastery not achieved - continuing anyway)")
        
        self._print("\nFinal pass rates:")
        names = summary.get('scenario_names', {})
        for sid, rate in summary['pass_rates'].items():
            name = names.get(sid, f"scenario_{sid}")
            status = "OK" if rate >= self.tutorial_mastery else "LOW"
            self._print(f"  S{sid} {name}: {rate:.0%} [{status}]")
        
        ckpt = self.ckpt_dir / "tutorial_complete.pt"
        self.agent.save(str(ckpt))
        self._print(f"\nSaved tutorial checkpoint: {ckpt}")
        self._print("=" * 60 + "\n")

    # -- Main entry point --

    def train(self, start_stage: int = 0, load_checkpoint: Optional[str] = None):
        """Run full curriculum training."""
        sched = self.schedule
        self._print("=== Curriculum Training (MultiHead DQN) ===")
        self._print(f"Stages: {sched.num_stages}  |  Days/stage: {sched.days_per_stage}")
        self._print(f"Output: {self.output_dir}  |  Device: {self.agent_cfg.device}\n")

        # -- Create environment --
        self._print("Creating environment ...")
        self.env = self.env_factory()
        yard = self.env.yard
        self._print(f"  Yard : {yard.n_rows}R x {yard.n_bays}B x {yard.n_tiers}T "
                     f"(sf={yard.split_factor}, splits={yard.total_splits})")
        self._print(f"  Cranes: {self.env.num_cranes}")

        # -- Create agent --
        self._print("Creating agent ...")
        self.agent = MultiHeadDQNAgent(self.agent_cfg)
        params = sum(p.numel() for p in self.agent.q_net.parameters())
        self._print(f"  Parameters: {params:,}\n")

        if load_checkpoint:
            self._print(f"Loading checkpoint: {load_checkpoint}")
            self.agent.load(load_checkpoint)

        # -- Tutorial Phase (Phase 0) --
        if not self.skip_tutorials:
            self._run_tutorial_phase()

        # -- Stage loop --
        global_day = 0
        for stage in range(start_stage, sched.num_stages):
            cap = sched.imports_for_stage(stage)
            self._print(f"\n{'=' * 60}")
            self._print(f"STAGE {stage}: {cap} imports/day")
            self._print(f"{'=' * 60}")

            self.env.lm.daily_train_import_cap = cap

            if sched.epsilon_reset_per_stage:
                self.agent.step_count = 0  # reset epsilon schedule

            stage_m = self._train_stage(stage)
            self.logger.log_stage(stage_m)

            ckpt = self.ckpt_dir / f"stage{stage}_complete.pt"
            self.agent.save(str(ckpt))
            self._print(f"Saved checkpoint: {ckpt}")
            global_day += sched.days_per_stage

        final = self.ckpt_dir / "final.pt"
        self.agent.save(str(final))
        self._print(f"\n=== Training Complete ===\nFinal: {final}")

    # -- Stage --

    def _train_stage(self, stage: int) -> StageMetrics:
        total_reward = 0.0
        total_moves = 0
        days = self.schedule.days_per_stage
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        carry_trains, carry_trucks = {}, {}

        pbar = tqdm(range(days), desc=f"Stage {stage}", unit="day")
        self._pbar = pbar
        for day in pbar:
            day_date = start_date + timedelta(days=day)
            try:
                self.env.reset(
                    day_start=day_date, day_index=day,
                    carryover_trains=carry_trains,
                    carryover_trucks=carry_trucks,
                )
            except Exception as e:
                self._print(f"ERROR reset Day {day}: {e}")
                import traceback; traceback.print_exc()
                raise

            if day == 0 and self.verbose:
                self._print_day_zero()

            day_m = self._run_day(stage, day, day_date)
            total_reward += day_m.total_reward
            total_moves += day_m.moves_executed
            self.logger.log_day(day_m)
            pbar.set_postfix(
                R=f"{day_m.total_reward:.1f}", mv=day_m.moves_executed,
                st=day_m.steps, eps=f"{day_m.epsilon:.3f}",
            )
            carry_trains, carry_trucks = self.env.get_carryover()

            if day % self.schedule.log_interval_days == 0:
                self.logger.log_episode_detail(stage, day, {
                    "metrics": asdict(day_m),
                    "replay_size": len(self.agent.replay),
                    "step_count": self.agent.step_count,
                })

        self._pbar = None
        return StageMetrics(
            stage=stage,
            import_cap=self.schedule.imports_for_stage(stage),
            total_days=days,
            total_reward=total_reward,
            avg_reward_per_day=total_reward / max(days, 1),
            total_moves=total_moves,
            avg_moves_per_day=total_moves / max(days, 1),
            final_epsilon=self.agent._get_epsilon(),
        )

    # -- Day --

    def _run_day(self, stage: int, day: int, date: datetime) -> DayMetrics:
        day_reward = 0.0
        day_moves = 0
        day_losses: List[float] = []
        step = 0
        done = False

        while not done and step < MAX_STEPS_PER_DAY:
            state, reward, done, info = self.env.step_all_cranes(self.agent)
            day_reward += reward
            day_moves += len(info.get("executed", []))

            if self.agent.replay.is_ready(self.agent_cfg.training.batch_size):
                loss = self.agent.optimize()
                if loss > 0:
                    day_losses.append(loss)

            step += 1

            if step % DEBUG_INTERVAL_STEPS == 0 and self.verbose and day == 0:
                self._print_step_debug(step, day_moves, day_reward)

        if step >= MAX_STEPS_PER_DAY:
            self._print(f"WARNING: Day {day} hit step limit ({MAX_STEPS_PER_DAY})")

        if self.verbose and day < 3:
            self._print(f"    Day {day}: {day_moves} moves, {step} steps, "
                        f"reward={day_reward:.2f}, yard={self.env.yard.container_count}")

        return DayMetrics(
            day_index=day, stage=stage,
            date=date.strftime("%Y-%m-%d"),
            total_reward=day_reward, moves_executed=day_moves,
            containers_in_yard=self.env.yard.container_count,
            trains_active=len(self.env.trains),
            trucks_active=len(self.env.trucks),
            avg_loss=float(np.mean(day_losses)) if day_losses else 0.0,
            epsilon=self.agent._get_epsilon(),
            steps=step,
        )

    # -- Debug --

    def _print_day_zero(self):
        env = self.env
        self._print(f"\n  After reset:")
        self._print(f"    Cranes={len(env.cranes)}, Trains={len(env.trains)}, "
                     f"Trucks={len(env.trucks)}, Yard={env.yard.container_count}")
        self._print(f"    Scheduled trains: {len(getattr(env, '_scheduled_trains', []))}")
        dp = env.day_plan
        self._print(f"    Trucks in plan: {len(dp.trucks_today) if dp else 0}")

    def _print_step_debug(self, step: int, moves: int, reward: float):
        env = self.env
        containers = env.move_gen.list_moveable_containers(
            env.trains, env.trucks, env.current_time,
        )
        parkings = env.move_gen.list_parking_actions(env.trucks)
        self._print(f"    Step {step}: time={env.current_time.strftime('%H:%M')}, "
                    f"t={len(env.trains)}, tk={len(env.trucks)}, "
                    f"yard={env.yard.container_count}, "
                    f"pool=[{len(containers)}c,{len(parkings)}p], "
                    f"mv={moves}, R={reward:.2f}")


# 
# -- Environment factory --
# 

def create_env_factory(
    rows: int = 5, bays: int = 58, tiers: int = 5,
    tracks: int = 7, export_ratio: float = 0.75,
    max_retries: int = 10, no_destination_penalty: float = -1.0,
):
    """Return a callable that creates a unified ContainerTerminalEnv."""
    def factory():
        from simulation.core.facilities.yard import OptimizedStorageYard
        from simulation.core.facilities.railyard import OptimizedRailYard
        from simulation.core.facilities.parking import OptimizedParkingArea
        from simulation.core.facilities.constants import BAY_SPLIT_FACTOR
        from simulation.core.factories.container_factory import ContainerFactory
        from simulation.core.factories.truck_factory import TruckFactory
        from simulation.operations.terminal_manager import TerminalLogisticsManager
        from simulation.operations.gate import TerminalGate
        from simulation.planning.logistics_manager import LogisticsManager
        from simulation.planning.driving_plan_parser import DrivingPlanParser
        from simulation.planning.train_scheduler import TrainScheduler
        from simulation.planning.train_loader import TrainLoader
        from simulation.env.env import ContainerTerminalEnv
        from simulation.config.yard_config import YardZoneConfig

        coordinates = YardZoneConfig.generate_special_coordinates(n_rows=rows, n_bays=bays)
        yard = OptimizedStorageYard(n_rows=rows, n_bays=bays, n_tiers=tiers,
                                    coordinates=coordinates, validate=False)
        rail = OptimizedRailYard(n_tracks=tracks)
        parking = OptimizedParkingArea(n_bays=bays, split_factor=BAY_SPLIT_FACTOR)

        container_factory = ContainerFactory()
        truck_factory = TruckFactory()
        gate = TerminalGate(container_factory, truck_factory)
        scheduler = TrainScheduler(num_tracks=tracks)
        loader = TrainLoader(container_factory, overgeneration_factor=3.0)
        parser = DrivingPlanParser()

        lm = LogisticsManager(yard, gate, loader, scheduler, parser,
                               export_per_import=export_ratio, daily_train_import_cap=20)
        tlm = TerminalLogisticsManager(yard, rail, parking)

        return ContainerTerminalEnv(
            yard=yard, rail=rail, parking=parking, tlm=tlm, lm=lm,
            num_tracks=tracks, step_minutes=5, auto_park=False,
            max_retries=max_retries, no_destination_penalty=no_destination_penalty,
        )
    return factory


def build_agent_config(
    rows: int = 5, bays: int = 58, tiers: int = 5, split_factor: int = 20,
) -> MultiHeadDQNConfig:
    """Build a MultiHeadDQNConfig for the given yard geometry."""
    yard = YardDims(
        n_rows=rows,
        n_splits=bays * split_factor,
        n_tiers=tiers,
        n_bays=bays,
        split_factor=split_factor,
    )
    return MultiHeadDQNConfig(
        yard=yard,
        backbone=BackboneConfig(in_channels=NUM_CHANNELS),
        heads=HeadConfig(),
        training=DQNConfig(),
    )


# 
# -- CLI --
# 

def main():
    import argparse
    p = argparse.ArgumentParser(description="Curriculum training - MultiHead DQN")
    p.add_argument("--output-dir", type=str, default="runs/curriculum")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start-stage", type=int, default=0)
    p.add_argument("--load-checkpoint", type=str, default=None)
    p.add_argument("--rows", type=int, default=5)
    p.add_argument("--bays", type=int, default=58)
    p.add_argument("--tiers", type=int, default=5)
    p.add_argument("--tracks", type=int, default=7)
    p.add_argument("--split-factor", type=int, default=20)
    p.add_argument("--days-per-stage", type=int, default=365)
    p.add_argument("--start-imports", type=int, default=20)
    p.add_argument("--max-imports", type=int, default=220)
    p.add_argument("--increment", type=int, default=20)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--skip-tutorials", action="store_true", help="Skip tutorial phase")
    p.add_argument("--tutorial-epochs", type=int, default=500, help="Max tutorial epochs (runs until mastery)")
    p.add_argument("--tutorial-mastery", type=float, default=0.9)
    args = p.parse_args()

    schedule = CurriculumSchedule(
        start_imports=args.start_imports, increment=args.increment,
        max_imports=args.max_imports, days_per_stage=args.days_per_stage,
    )
    agent_cfg = build_agent_config(
        rows=args.rows, bays=args.bays, tiers=args.tiers,
        split_factor=args.split_factor,
    )
    env_factory = create_env_factory(
        rows=args.rows, bays=args.bays, tiers=args.tiers, tracks=args.tracks,
    )

    trainer = CurriculumTrainer(
        env_factory=env_factory, schedule=schedule, agent_config=agent_cfg,
        output_dir=args.output_dir, seed=args.seed, verbose=not args.quiet,
        skip_tutorials=args.skip_tutorials,
        tutorial_epochs=args.tutorial_epochs,
        tutorial_mastery=args.tutorial_mastery,
    )
    trainer.train(start_stage=args.start_stage, load_checkpoint=args.load_checkpoint)


if __name__ == "__main__":
    main()