# simulation/training/curriculum_trainer_fixed.py
"""
Curriculum-based training for hierarchical DQN agent.

FIXED VERSION - addresses:
1. Better debug output to see what's happening
2. Safety timeouts for infinite loops
3. Proper progress tracking
"""
import os
import csv
import json
import random
import sys
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch

from simulation.config.curriculum_config import CurriculumConfig, HierarchicalDQNConfig
from simulation.rl.agents.hierarchical_dqn_agent import HierarchicalDQNAgent
from simulation.env.hierarchical_env import HierarchicalEnvWrapper


# Safety limits
MAX_STEPS_PER_DAY = 5000  # Prevent infinite loops
DEBUG_INTERVAL_STEPS = 100  # Print debug info every N steps


@dataclass
class DayMetrics:
    """Metrics for a single simulation day."""
    day_index: int
    stage: int
    date: str
    total_reward: float
    moves_executed: int
    containers_imported: int = 0
    containers_exported: int = 0
    trains_departed: int = 0
    trucks_served: int = 0
    avg_loss: float = 0.0
    epsilon: float = 0.0
    steps: int = 0  # Track step count


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


class MetricsLogger:
    """Handles logging of training metrics."""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.daily_csv_path = self.log_dir / "daily_metrics.csv"
        self._init_daily_csv()
        
        self.stage_csv_path = self.log_dir / "stage_metrics.csv"
        self._init_stage_csv()
        
        self.episodes_dir = self.log_dir / "episodes"
        self.episodes_dir.mkdir(exist_ok=True)
    
    def _init_daily_csv(self):
        """Initialize daily metrics CSV."""
        if not self.daily_csv_path.exists():
            with open(self.daily_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "day_index", "stage", "date", "total_reward", "moves_executed",
                    "containers_imported", "containers_exported", "trains_departed",
                    "trucks_served", "avg_loss", "epsilon", "steps"
                ])
    
    def _init_stage_csv(self):
        """Initialize stage metrics CSV."""
        if not self.stage_csv_path.exists():
            with open(self.stage_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "stage", "import_cap", "total_days", "total_reward",
                    "avg_reward_per_day", "total_moves", "avg_moves_per_day",
                    "final_epsilon"
                ])
    
    def log_day(self, metrics: DayMetrics):
        """Log daily metrics."""
        with open(self.daily_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.day_index,
                metrics.stage,
                metrics.date,
                f"{metrics.total_reward:.4f}",
                metrics.moves_executed,
                metrics.containers_imported,
                metrics.containers_exported,
                metrics.trains_departed,
                metrics.trucks_served,
                f"{metrics.avg_loss:.6f}",
                f"{metrics.epsilon:.4f}",
                metrics.steps
            ])
    
    def log_stage(self, metrics: StageMetrics):
        """Log stage summary metrics."""
        with open(self.stage_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.stage,
                metrics.import_cap,
                metrics.total_days,
                f"{metrics.total_reward:.4f}",
                f"{metrics.avg_reward_per_day:.4f}",
                metrics.total_moves,
                f"{metrics.avg_moves_per_day:.2f}",
                f"{metrics.final_epsilon:.4f}"
            ])
    
    def log_episode_detail(self, stage: int, day: int, data: Dict[str, Any]):
        """Log detailed episode data."""
        filepath = self.episodes_dir / f"stage{stage}_day{day}.json"
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)


class CurriculumTrainer:
    """Trains hierarchical DQN agent through curriculum stages."""
    
    def __init__(
        self,
        env_factory,
        curriculum_config: CurriculumConfig,
        network_config: HierarchicalDQNConfig,
        output_dir: str,
        seed: int = 42,
        verbose: bool = True
    ):
        self.env_factory = env_factory
        self.curriculum = curriculum_config
        self.net_config = network_config
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.verbose = verbose
        
        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Create output directories
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = self.output_dir / "logs"
        self.logger = MetricsLogger(str(self.log_dir))
        
        # Will be initialized in train()
        self.env = None
        self.wrapper = None
        self.agent = None
    
    def _print(self, msg: str):
        """Print if verbose mode."""
        if self.verbose:
            print(msg)
            sys.stdout.flush()
    
    def train(self, start_stage: int = 0, load_checkpoint: Optional[str] = None):
        """Run full curriculum training."""
        self._print(f"=== Curriculum Training ===")
        self._print(f"Stages: {self.curriculum.num_stages}")
        self._print(f"Days per stage: {self.curriculum.days_per_stage}")
        self._print(f"Output: {self.output_dir}")
        self._print(f"Device: {self.net_config.device}")
        self._print("")
        
        # Create environment
        self._print("Creating environment...")
        self.env, self.wrapper = self.env_factory()
        self._print(f"  Yard: {self.env.yard.n_rows}x{self.env.yard.n_bays}x{self.env.yard.n_tiers}")
        # Note: cranes list is populated in reset(), so use num_cranes
        self._print(f"  Cranes: {self.env.num_cranes}")
        
        # Get yard dimensions
        yard_dims = (
            self.env.yard.n_rows,
            self.env.yard.n_bays,
            self.env.yard.n_tiers,
            self.env.yard.split_factor
        )
        
        # Create agent
        self._print("Creating agent...")
        self.agent = HierarchicalDQNAgent(yard_dims, self.net_config)
        self._print(f"  Parameters: {sum(p.numel() for p in self.agent.q_net.parameters()):,}")
        
        # Load checkpoint if provided
        if load_checkpoint:
            self._print(f"Loading checkpoint: {load_checkpoint}")
            self.agent.load(load_checkpoint)
        
        # Training loop
        global_day = 0
        for stage in range(start_stage, self.curriculum.num_stages):
            import_cap = self.curriculum.imports_for_stage(stage)
            
            self._print(f"\n{'='*60}")
            self._print(f"STAGE {stage}: {import_cap} imports/day")
            self._print(f"{'='*60}")
            
            # Update environment import cap
            self.env.lm.daily_train_import_cap = import_cap
            
            # Reset epsilon if configured
            if self.curriculum.epsilon_reset_per_stage:
                self.agent.reset_epsilon()
            
            # Train this stage
            stage_metrics = self._train_stage(stage, global_day)
            
            # Log stage completion
            self.logger.log_stage(stage_metrics)
            
            # Save checkpoint
            ckpt_path = self.ckpt_dir / f"stage{stage}_complete.pt"
            self.agent.save(str(ckpt_path))
            self._print(f"Saved checkpoint: {ckpt_path}")
            
            global_day += self.curriculum.days_per_stage
        
        self._print(f"\n=== Training Complete ===")
        self._print(f"Final checkpoint: {self.ckpt_dir / 'final.pt'}")
        self.agent.save(str(self.ckpt_dir / "final.pt"))
    
    def _train_stage(self, stage: int, global_day_offset: int) -> StageMetrics:
        """Train a single curriculum stage."""
        total_reward = 0.0
        total_moves = 0
        losses: List[float] = []
        
        start_date = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        
        # Carryover tracking
        carry_trains, carry_trucks = {}, {}
        
        pbar = tqdm(range(self.curriculum.days_per_stage), desc=f"Stage {stage}")
        
        for day in pbar:
            day_date = start_date + timedelta(days=day)
            
            # Debug: starting new day
            if self.verbose and day < 5:
                print(f"\n  Starting Day {day}...")
            
            # Reset environment for new day
            try:
                self.env.reset(
                    day_start=day_date,
                    day_index=day,
                    carryover_trains=carry_trains,
                    carryover_trucks=carry_trucks
                )
            except Exception as e:
                print(f"ERROR in reset for Day {day}: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            # Debug: check initial state
            if day == 0:
                self._print(f"\n  Initial state (after reset):")
                self._print(f"    Cranes: {len(self.env.cranes)}")
                self._print(f"    Trains: {len(self.env.trains)}")
                self._print(f"    Trucks: {len(self.env.trucks)}")
                self._print(f"    Scheduled trains: {len(self.env._scheduled_trains)}")
                self._print(f"    Trucks in day plan: {len(self.env.day_plan.trucks_today) if self.env.day_plan else 0}")
                self._print(f"    Current time: {self.env.current_time}")
            
            # Run day
            day_metrics = self._run_day(stage, day, day_date)
            
            total_reward += day_metrics.total_reward
            total_moves += day_metrics.moves_executed
            if day_metrics.avg_loss > 0:
                losses.append(day_metrics.avg_loss)
            
            # Log
            self.logger.log_day(day_metrics)
            
            # Update progress bar
            pbar.set_postfix({
                "R": f"{day_metrics.total_reward:.1f}",
                "moves": day_metrics.moves_executed,
                "steps": day_metrics.steps,
                "ε": f"{day_metrics.epsilon:.3f}"
            })
            
            # Get carryover
            carry_trains, carry_trucks = self.env.get_carryover()
            
            # Detailed logging every N days
            if day % self.curriculum.log_interval_days == 0:
                self.logger.log_episode_detail(stage, day, {
                    "metrics": asdict(day_metrics),
                    "replay_size": len(self.agent.replay),
                    "step_count": self.agent.step_count
                })
        
        # Compute stage metrics
        days = self.curriculum.days_per_stage
        return StageMetrics(
            stage=stage,
            import_cap=self.curriculum.imports_for_stage(stage),
            total_days=days,
            total_reward=total_reward,
            avg_reward_per_day=total_reward / days,
            total_moves=total_moves,
            avg_moves_per_day=total_moves / days,
            final_epsilon=self.agent._get_epsilon()
        )
    
    def _run_day(self, stage: int, day: int, date: datetime) -> DayMetrics:
        """Run a single simulation day."""
        day_reward = 0.0
        day_moves = 0
        day_losses: List[float] = []
        
        # Track failure reasons for debugging
        parking_success = 0
        parking_fail = 0
        move_success = 0
        move_fail = 0
        no_action_steps = 0
        retry_reasons: Dict[str, int] = {}
        
        done = False
        step = 0
        
        while not done and step < MAX_STEPS_PER_DAY:
            # Step all cranes
            state, reward, done, info = self.wrapper.step_all_cranes(self.agent)
            
            day_reward += reward
            executed = info.get("executed", [])
            day_moves += len(executed)
            
            # Track success/failure
            for ex in executed:
                if ex.get("type") == "PARKING":
                    parking_success += 1
                else:
                    move_success += 1
            
            # Track when no moves were made
            if len(executed) == 0:
                no_action_steps += 1
            
            # Track retry reasons from crane results
            for cr in info.get("crane_results", []):
                pass  # Could extract more info here
            
            # Track retry reasons (if available in transitions)
            for trans_info in info.get("transitions", []):
                pass  # Info is in the HierarchicalStepResult, not transition
            
            # Track retry reasons from the info dict
            for reason in info.get("retry_reasons", []):
                retry_reasons[reason] = retry_reasons.get(reason, 0) + 1
            
            # Optimize agent
            if len(self.agent.replay) >= self.net_config.batch_size:
                loss = self.agent.optimize()
                if loss > 0:
                    day_losses.append(loss)
            
            step += 1
            
            # Debug output periodically
            if step % DEBUG_INTERVAL_STEPS == 0 and self.verbose and day == 0:
                # Count action pool sizes
                containers = self.wrapper.move_gen.list_moveable_containers(
                    self.env.trains, self.env.trucks, self.env.current_time
                )
                parkings = self.wrapper.move_gen.list_parking_actions(self.env.trucks)
                
                print(f"    Step {step}: time={self.env.current_time.strftime('%H:%M')}, "
                      f"trains={len(self.env.trains)}, trucks={len(self.env.trucks)}, "
                      f"yard={len(self.env.yard.containers)}, "
                      f"pool=[{len(containers)}c,{len(parkings)}p], "
                      f"moves={day_moves}, reward={day_reward:.2f}")
        
        if step >= MAX_STEPS_PER_DAY:
            print(f"WARNING: Day {day} hit step limit ({MAX_STEPS_PER_DAY})")
        
        # Print day summary for first few days
        if self.verbose and day < 3:
            retry_str = ", ".join(f"{k}:{v}" for k, v in sorted(retry_reasons.items()))
            print(f"    Day {day} summary: {day_moves} moves ({parking_success} park, {move_success} container), "
                  f"{no_action_steps} idle steps, reward={day_reward:.2f}")
            if retry_reasons:
                print(f"      Retry reasons: {retry_str}")
            print(f"    Day {day} complete, getting carryover...")
        
        return DayMetrics(
            day_index=day,
            stage=stage,
            date=date.strftime("%Y-%m-%d"),
            total_reward=day_reward,
            moves_executed=day_moves,
            avg_loss=np.mean(day_losses) if day_losses else 0.0,
            epsilon=self.agent._get_epsilon(),
            steps=step
        )


def create_env_factory(
    rows: int = 5,
    bays: int = 58,
    tiers: int = 5,
    tracks: int = 7,
    export_ratio: float = 0.75
):
    """Create a factory function for environment creation."""
    def factory():
        from simulation.core.facilities.yard import BooleanStorageYard
        from simulation.core.facilities.railyard import BooleanRailYard
        from simulation.core.facilities.parking import ParkingArea
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
        
        # Create coordinates for special zones
        coordinates = YardZoneConfig.generate_special_coordinates(
            n_rows=rows, n_bays=bays
        )
        
        # Create facilities
        yard = BooleanStorageYard(
            n_rows=rows,
            n_bays=bays,
            n_tiers=tiers,
            coordinates=coordinates,
            validate=False
        )
        rail = BooleanRailYard()
        parking = ParkingArea(ParkingArea.make_grid(
            n_bays=bays,
            split_factor=20,
            prefix="P"
        ))
        
        # Create factories
        container_factory = ContainerFactory()
        truck_factory = TruckFactory()
        
        # Create logistics components
        gate = TerminalGate(container_factory, truck_factory)
        scheduler = TrainScheduler(num_tracks=tracks)
        loader = TrainLoader(container_factory, overgeneration_factor=3.0)
        parser = DrivingPlanParser()
        
        lm = LogisticsManager(
            yard, gate, loader, scheduler, parser,
            export_per_import=export_ratio,
            daily_train_import_cap=20
        )
        tlm = TerminalLogisticsManager(yard, rail, parking)
        
        # Create environment
        env = ContainerTerminalEnv(
            yard=yard,
            rail=rail,
            parking=parking,
            tlm=tlm,
            lm=lm,
            num_tracks=tracks,
            step_minutes=5,
            auto_park=False
        )
        
        # Create hierarchical wrapper
        wrapper = HierarchicalEnvWrapper(
            env,
            max_retries=10,
            no_destination_penalty=-1.0
        )
        
        return env, wrapper
    
    return factory


def main():
    """Main entry point for curriculum training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Curriculum training for hierarchical DQN")
    parser.add_argument("--output-dir", type=str, default="runs/curriculum",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--start-stage", type=int, default=0,
                        help="Stage to start from")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="Checkpoint to load")
    parser.add_argument("--rows", type=int, default=5, help="Yard rows")
    parser.add_argument("--bays", type=int, default=58, help="Yard bays")
    parser.add_argument("--tiers", type=int, default=5, help="Yard tiers")
    parser.add_argument("--tracks", type=int, default=7, help="Rail tracks")
    parser.add_argument("--days-per-stage", type=int, default=365,
                        help="Days per curriculum stage")
    parser.add_argument("--start-imports", type=int, default=20,
                        help="Starting import count")
    parser.add_argument("--max-imports", type=int, default=220,
                        help="Maximum import count")
    parser.add_argument("--increment", type=int, default=20,
                        help="Import increment per stage")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    # Create configs
    curriculum_config = CurriculumConfig(
        start_imports=args.start_imports,
        increment=args.increment,
        max_imports=args.max_imports,
        days_per_stage=args.days_per_stage
    )
    
    network_config = HierarchicalDQNConfig()
    
    # Create environment factory
    env_factory = create_env_factory(
        rows=args.rows,
        bays=args.bays,
        tiers=args.tiers,
        tracks=args.tracks
    )
    
    # Create trainer
    trainer = CurriculumTrainer(
        env_factory=env_factory,
        curriculum_config=curriculum_config,
        network_config=network_config,
        output_dir=args.output_dir,
        seed=args.seed,
        verbose=not args.quiet
    )
    
    # Run training
    trainer.train(
        start_stage=args.start_stage,
        load_checkpoint=args.load_checkpoint
    )


if __name__ == "__main__":
    main()