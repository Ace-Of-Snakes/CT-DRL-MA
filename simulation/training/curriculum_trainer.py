# simulation/training/curriculum_trainer.py
"""
Curriculum-based training for hierarchical DQN agent.

Features:
- Gradual scaling of container throughput (20 → 220 imports)
- 365 days per curriculum stage
- Checkpoint saving after each stage
- Logging of metrics for visualization
"""
import os
import csv
import json
import random
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
        
        # CSV for daily metrics
        self.daily_csv_path = self.log_dir / "daily_metrics.csv"
        self._init_daily_csv()
        
        # CSV for stage summaries
        self.stage_csv_path = self.log_dir / "stage_metrics.csv"
        self._init_stage_csv()
        
        # JSON for detailed episode data
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
                    "trucks_served", "avg_loss", "epsilon"
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
                f"{metrics.epsilon:.4f}"
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
        """Log detailed episode data (sparse - only every N days)."""
        filepath = self.episodes_dir / f"stage{stage}_day{day}.json"
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)


class CurriculumTrainer:
    """
    Trains hierarchical DQN agent through curriculum stages.
    """
    
    def __init__(
        self,
        env_factory,  # Callable that returns (env, hierarchical_wrapper)
        curriculum_config: CurriculumConfig,
        network_config: HierarchicalDQNConfig,
        output_dir: str,
        seed: int = 42
    ):
        """
        Initialize trainer.
        
        Args:
            env_factory: Factory function to create environment
            curriculum_config: Curriculum configuration
            network_config: Network architecture configuration
            output_dir: Directory for outputs (checkpoints, logs)
            seed: Random seed
        """
        self.env_factory = env_factory
        self.curriculum = curriculum_config
        self.net_config = network_config
        self.output_dir = Path(output_dir)
        self.seed = seed
        
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
    
    def train(self, start_stage: int = 0, load_checkpoint: Optional[str] = None):
        """
        Run full curriculum training.
        
        Args:
            start_stage: Which stage to start from (for resuming)
            load_checkpoint: Path to checkpoint to load (for resuming)
        """
        print(f"=== Curriculum Training ===")
        print(f"Stages: {self.curriculum.num_stages}")
        print(f"Days per stage: {self.curriculum.days_per_stage}")
        print(f"Output: {self.output_dir}")
        print()
        
        # Create environment
        self.env, self.wrapper = self.env_factory()
        
        # Get yard dimensions
        yard_dims = (
            self.env.yard.n_rows,
            self.env.yard.n_bays,
            self.env.yard.n_tiers,
            self.env.yard.split_factor
        )
        
        # Create agent
        self.agent = HierarchicalDQNAgent(yard_dims, self.net_config)
        
        # Load checkpoint if provided
        if load_checkpoint:
            print(f"Loading checkpoint: {load_checkpoint}")
            self.agent.load(load_checkpoint)
        
        # Training loop
        global_day = 0
        for stage in range(start_stage, self.curriculum.num_stages):
            import_cap = self.curriculum.imports_for_stage(stage)
            
            print(f"\n{'='*60}")
            print(f"STAGE {stage}: {import_cap} imports/day")
            print(f"{'='*60}")
            
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
            print(f"Saved checkpoint: {ckpt_path}")
            
            global_day += self.curriculum.days_per_stage
        
        print(f"\n=== Training Complete ===")
        print(f"Final checkpoint: {self.ckpt_dir / 'final.pt'}")
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
            global_day = global_day_offset + day
            
            # Reset environment for new day
            self.env.reset(
                day_start=day_date,
                day_index=day,
                carryover_trains=carry_trains,
                carryover_trucks=carry_trucks
            )
            
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
        
        done = False
        step = 0
        
        while not done:
            # Step all cranes
            state, reward, done, info = self.wrapper.step_all_cranes(self.agent)
            
            day_reward += reward
            day_moves += len(info.get("executed", []))
            
            # Optimize agent
            if len(self.agent.replay) >= self.net_config.batch_size:
                loss = self.agent.optimize()
                if loss > 0:
                    day_losses.append(loss)
            
            step += 1
            
            # Safety: prevent infinite loops
            if step > 10000:
                print(f"Warning: Day {day} exceeded 10000 steps")
                break
        
        return DayMetrics(
            day_index=day,
            stage=stage,
            date=date.strftime("%Y-%m-%d"),
            total_reward=day_reward,
            moves_executed=day_moves,
            avg_loss=np.mean(day_losses) if day_losses else 0.0,
            epsilon=self.agent._get_epsilon()
        )


def create_env_factory(
    rows: int = 5,
    bays: int = 58,
    tiers: int = 5,
    tracks: int = 7,
    export_ratio: float = 0.75
):
    """
    Create a factory function for environment creation.
    
    This allows lazy environment creation with specified parameters.
    """
    def factory():
        # These imports are done inside to avoid circular imports
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
            daily_train_import_cap=20  # Will be overridden by curriculum
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
            auto_park=False  # Agent controls parking
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
                        help="Stage to start from (for resuming)")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="Checkpoint to load (for resuming)")
    parser.add_argument("--rows", type=int, default=5, help="Yard rows")
    parser.add_argument("--bays", type=int, default=58, help="Yard bays")
    parser.add_argument("--tiers", type=int, default=5, help="Yard tiers")
    parser.add_argument("--tracks", type=int, default=7, help="Number of rail tracks")
    parser.add_argument("--days-per-stage", type=int, default=365,
                        help="Days per curriculum stage")
    parser.add_argument("--start-imports", type=int, default=20,
                        help="Starting import count")
    parser.add_argument("--max-imports", type=int, default=220,
                        help="Maximum import count")
    parser.add_argument("--increment", type=int, default=20,
                        help="Import increment per stage")
    
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
        seed=args.seed
    )
    
    # Run training
    trainer.train(
        start_stage=args.start_stage,
        load_checkpoint=args.load_checkpoint
    )


if __name__ == "__main__":
    main()
