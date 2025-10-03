# simulation/training/train_two_modules.py
import os
import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple, Set, Any, Optional
from datetime import datetime, timedelta

import numpy as np
from tqdm import tqdm

# Core terminal imports
from simulation.core.facilities.yard import BooleanStorageYard
from simulation.core.facilities.railyard import BooleanRailYard
from simulation.core.facilities.parking import ParkingArea
from simulation.core.factories.container_factory import ContainerFactory
from simulation.core.factories.truck_factory import TruckFactory

from simulation.planning.logistics_manager import LogisticsManager
from simulation.planning.driving_plan_parser import DrivingPlanParser
from simulation.planning.train_scheduler import TrainScheduler
from simulation.planning.train_loader import TrainLoader

from simulation.operations.terminal_manager import TerminalLogisticsManager
from simulation.operations.gate import TerminalGate

from simulation.analytics.stats_tracker import StatsTracker
from simulation.env.env import ContainerTerminalEnv

# Agents
from simulation.rl.agents.dqn_agent import DQNAgent, DQNConfig
from simulation.rl.agents.ppo_agent import PPOAgent, PPOConfig

# Config
from simulation.config.paths import OutputPaths, DataPaths
from simulation.config.operations_config import OperationsDefaults
from simulation.config.yard_config import YardZoneConfig
from simulation.core.enums import Direction


@dataclass
class Module:
    """Training module configuration."""
    name: str
    env: ContainerTerminalEnv
    yard: BooleanStorageYard
    scheduler: TrainScheduler
    tracker: StatsTracker
    agent: Optional[PPOAgent | DQNAgent]


class DualActRecorder:
    """Records agent actions and outcomes for training."""
    
    def __init__(self, agent):
        self.agent = agent
        self.records = []
        self._last_record_idx = -1
    
    def reset(self):
        """Reset records for new episode."""
        self.records.clear()
        self._last_record_idx = -1
    
    def act(self, state_np, moves):
        """Record action selection."""
        if isinstance(self.agent, DQNAgent):
            a_idx = self.agent.act(state_np, moves)
            record = {
                "algo": "dqn",
                "state": state_np,
                "moves": moves,
                "a": a_idx,
                "reward": 0.0,
                "success": False
            }
            self.records.append(record)
            self._last_record_idx = len(self.records) - 1
            return a_idx
        else:  # PPO
            a_idx, logp, value = self.agent.act(state_np, moves)
            record = {
                "algo": "ppo",
                "state": state_np,
                "moves": moves,
                "a": a_idx,
                "logp": logp,
                "value": value,
                "reward": 0.0,
                "success": False
            }
            self.records.append(record)
            self._last_record_idx = len(self.records) - 1
            return a_idx
    
    def record_outcome(self, success: bool, reward: float):
        """Record the outcome of the most recent action."""
        if self._last_record_idx >= 0:
            self.records[self._last_record_idx]["reward"] = reward
            self.records[self._last_record_idx]["success"] = success


def split_trains_evenly(
    trains: List,
    tracks_m1: int = 7,
    tracks_m2: int = 6
) -> Tuple[List, List]:
    """Split trains evenly between two modules."""
    keyed = [(t.schedule_encoded['arrival']['seconds'], t) for t in trains]
    keyed.sort(key=lambda x: x[0])
    
    m1, m2 = [], []
    load1 = 0.0
    load2 = 0.0
    
    for _, t in keyed:
        stay = float(t.schedule_encoded['stay_duration']['hours'])
        if (load1 / max(1, tracks_m1)) <= (load2 / max(1, tracks_m2)):
            m1.append(t)
            load1 += stay
        else:
            m2.append(t)
            load2 += stay
    
    return m1, m2


class FilteringDrivingPlanParser(DrivingPlanParser):
    """Filters trains by train_id whitelist."""
    
    def __init__(self, whitelist_ids: Set[str], json_path: str = None):
        super().__init__(json_path)
        self.whitelist = set(whitelist_ids)
    
    def create_trains(self) -> List:
        all_trains = super().create_trains()
        return [t for t in all_trains if t.train_id in self.whitelist]


def build_module(
    name: str,
    rows: int,
    bays: int,
    tiers: int,
    tracks: int,
    parser: DrivingPlanParser,
    container_factory: ContainerFactory,
    truck_factory: TruckFactory,
    train_import_cap: Optional[int] = 220,
    export_per_import: float = 0.75,
    overgen: float = 3.0,
    logdir: str = "",
    algo: str = "dqn"
) -> Module:
    """Build a training module."""
    # Generate special coordinates
    coordinates = YardZoneConfig.generate_special_coordinates(
        n_rows=rows,
        n_bays=bays
    )
    
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
        prefix=f"P_{name}"
    ))
    
    gate = TerminalGate(container_factory, truck_factory)
    scheduler = TrainScheduler(num_tracks=tracks)
    loader = TrainLoader(container_factory, overgeneration_factor=overgen)
    
    lm = LogisticsManager(
        yard, gate, loader, scheduler, parser,
        export_per_import=export_per_import,
        daily_train_import_cap=train_import_cap
    )
    tlm = TerminalLogisticsManager(yard, rail, parking)
    
    # Create logs directory
    module_dir = os.path.join(logdir, name)
    os.makedirs(module_dir, exist_ok=True)
    
    tracker = StatsTracker(
        moves_path=os.path.join(module_dir, "moves.ndjson"),
        daily_csv_path=os.path.join(module_dir, "daily.csv"),
        yard=yard
    )
    
    env = ContainerTerminalEnv(
        yard=yard,
        rail=rail,
        parking=parking,
        tlm=tlm,
        lm=lm,
        num_tracks=tracks,
        step_minutes=5,
        stats=tracker
    )
    
    # Create agent
    dims = (yard.n_rows, yard.n_bays, yard.n_tiers, yard.split_factor)
    if algo == "dqn":
        agent = DQNAgent(dims, DQNConfig())
    else:
        agent = PPOAgent(dims, PPOConfig())
    
    return Module(
        name=name,
        env=env,
        yard=yard,
        scheduler=scheduler,
        tracker=tracker,
        agent=agent
    )


def main():
    parser = argparse.ArgumentParser(description="Train dual module terminal agents")
    parser.add_argument("--algo", choices=["dqn", "ppo"], default="dqn", help="Algorithm to use")
    parser.add_argument("--days", type=int, default=30, help="Number of days to train")
    parser.add_argument("--logdir", type=str, default="runs/dual_modules", help="Log directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-import-cap", type=int, default=220, help="Daily train import cap per module")
    parser.add_argument("--export-per-import", type=float, default=0.75, help="Exports per import ratio")
    parser.add_argument("--overgen", type=float, default=3.0, help="Train loader overgeneration factor")
    
    # NEW: Weight loading arguments
    parser.add_argument("--load-m1", type=str, default=None, help="Path to M1 agent weights")
    parser.add_argument("--load-m2", type=str, default=None, help="Path to M2 agent weights")
    
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Set proximity
    import simulation.operations.terminal_manager as TM
    TM.PROXIMITY = OperationsDefaults.PROXIMITY_SEARCH_BAYS
    
    # Create output directory
    outdir = OutputPaths.create_run_dir(base_name="dual_modules")
    print(f"Logging to: {outdir}")
    
    ckpt_dir = os.path.join(outdir, OutputPaths.CHECKPOINTS_SUBDIR)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Parse and split trains
    base_parser = DrivingPlanParser()
    all_trains = base_parser.create_trains()
    m1_trains, m2_trains = split_trains_evenly(all_trains, tracks_m1=7, tracks_m2=6)
    m1_ids = {t.train_id for t in m1_trains}
    m2_ids = {t.train_id for t in m2_trains}
    
    # Create factories
    container_factory = ContainerFactory()
    truck_factory = TruckFactory()
    
    # Build modules
    parser_m1 = FilteringDrivingPlanParser(whitelist_ids=m1_ids)
    parser_m2 = FilteringDrivingPlanParser(whitelist_ids=m2_ids)
    
    m1 = build_module(
        "M1", rows=5, bays=58, tiers=5, tracks=7,
        parser=parser_m1,
        container_factory=container_factory,
        truck_factory=truck_factory,
        train_import_cap=args.train_import_cap,
        export_per_import=args.export_per_import,
        overgen=args.overgen,
        logdir=outdir,
        algo=args.algo
    )
    
    m2 = build_module(
        "M2", rows=3, bays=58, tiers=3, tracks=6,
        parser=parser_m2,
        container_factory=container_factory,
        truck_factory=truck_factory,
        train_import_cap=args.train_import_cap,
        export_per_import=args.export_per_import,
        overgen=args.overgen,
        logdir=outdir,
        algo=args.algo
    )
    
    # Load weights if provided
    if args.load_m1:
        m1.agent.load(args.load_m1, map_location="cpu")
        print(f"[M1] Loaded weights from: {args.load_m1}")
    
    if args.load_m2:
        m2.agent.load(args.load_m2, map_location="cpu")
        print(f"[M2] Loaded weights from: {args.load_m2}")
    
    # Training loop
    start_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    best_sum_reward = -1e18
    
    carry_trains_m1, carry_trucks_m1 = {}, {}
    carry_trains_m2, carry_trucks_m2 = {}, {}
    
    try:
        for day in range(args.days):
            day_start = start_day + timedelta(days=day)
            
            # Reset environments
            m1.env.reset(
                day_start,
                day_index=day,
                carryover_trains=carry_trains_m1,
                carryover_trucks=carry_trucks_m1
            )
            m2.env.reset(
                day_start,
                day_index=day,
                carryover_trains=carry_trains_m2,
                carryover_trucks=carry_trucks_m2
            )
            
            m1.tracker.reset_day_aggregates()
            m2.tracker.reset_day_aggregates()
            
            # Create recorders
            rec1 = DualActRecorder(m1.agent)
            rec2 = DualActRecorder(m2.agent)
            
            done1 = False
            done2 = False
            day_reward_m1 = 0.0
            day_reward_m2 = 0.0
            steps = 0
            
            pbar = tqdm(total=1, desc=f"Day {day+1}/{args.days}", leave=False)
            
            # Simulation loop
            while not (done1 and done2):
                t1 = None if done1 else m1.env.current_time
                t2 = None if done2 else m2.env.current_time
                
                # Process M1
                if t2 is None or (t1 is not None and t1 <= t2):
                    rec1.reset()
                    ns, nm, rew, dn, info = m1.env.step_dual_agent(rec1)
                    day_reward_m1 += rew
                    
                    # Store experiences
                    if rec1.records:
                        for k, r in enumerate(rec1.records):
                            r_k = r["reward"]
                            if isinstance(m1.agent, DQNAgent):
                                # Determine next state
                                if k + 1 < len(rec1.records):
                                    ns_k = rec1.records[k+1]["state"]
                                    nm_k = rec1.records[k+1]["moves"]
                                    dn_k = False
                                else:
                                    ns_k = ns
                                    nm_k = nm
                                    dn_k = dn
                                m1.agent.remember(
                                    r["state"], r["moves"], r["a"],
                                    r_k, ns_k, nm_k, dn_k
                                )
                            else:  # PPO
                                m1.agent.remember(
                                    r["state"], r["moves"], r["a"],
                                    r.get("logp", 0.0), r_k,
                                    r.get("value", 0.0), dn
                                )
                        
                        # Train DQN online
                        if isinstance(m1.agent, DQNAgent):
                            m1.agent.optimize()
                    
                    done1 = dn
                else:
                    # Process M2 (same logic)
                    rec2.reset()
                    ns, nm, rew, dn, info = m2.env.step_dual_agent(rec2)
                    day_reward_m2 += rew
                    
                    if rec2.records:
                        for k, r in enumerate(rec2.records):
                            r_k = r["reward"]
                            if isinstance(m2.agent, DQNAgent):
                                if k + 1 < len(rec2.records):
                                    ns_k = rec2.records[k+1]["state"]
                                    nm_k = rec2.records[k+1]["moves"]
                                    dn_k = False
                                else:
                                    ns_k = ns
                                    nm_k = nm
                                    dn_k = dn
                                m2.agent.remember(
                                    r["state"], r["moves"], r["a"],
                                    r_k, ns_k, nm_k, dn_k
                                )
                            else:  # PPO
                                m2.agent.remember(
                                    r["state"], r["moves"], r["a"],
                                    r.get("logp", 0.0), r_k,
                                    r.get("value", 0.0), dn
                                )
                        
                        if isinstance(m2.agent, DQNAgent):
                            m2.agent.optimize()
                    
                    done2 = dn
                
                steps += 1
                pbar.set_postfix_str(f"steps={steps} M1R={day_reward_m1:.1f} M2R={day_reward_m2:.1f}")
                
                if done1 and done2:
                    pbar.update(1.0)
                    pbar.close()
            
            # Update PPO agents
            if isinstance(m1.agent, PPOAgent):
                m1.agent.update()
            if isinstance(m2.agent, PPOAgent):
                m2.agent.update()
            
            # Write stats
            m1.tracker.write_day_summary(day_index=day, date=day_start)
            m2.tracker.write_day_summary(day_index=day, date=day_start)
            
            # Get carryover for next day
            carry_trains_m1, carry_trucks_m1 = m1.env.get_carryover()
            carry_trains_m2, carry_trucks_m2 = m2.env.get_carryover()
            
            # Save checkpoints
            m1.agent.save(os.path.join(ckpt_dir, "m1_last.pt"))
            m2.agent.save(os.path.join(ckpt_dir, "m2_last.pt"))
            
            sum_r = day_reward_m1 + day_reward_m2
            if sum_r > best_sum_reward:
                best_sum_reward = sum_r
                m1.agent.save(os.path.join(ckpt_dir, "m1_best.pt"))
                m2.agent.save(os.path.join(ckpt_dir, "m2_best.pt"))
            
            print(f"Day {day+1}: M1R={day_reward_m1:.2f} M2R={day_reward_m2:.2f} (saved)")
        
        print(f"\nTraining finished!")
        print(f"Logs: {outdir}")
        print(f"Checkpoints: {ckpt_dir}")
    
    finally:
        m1.tracker.close()
        m2.tracker.close()


if __name__ == "__main__":
    main()