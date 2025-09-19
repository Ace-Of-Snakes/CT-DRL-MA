# scripts/train_two_modules.py
import os
import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict, Any, Optional
from datetime import datetime, timedelta

import numpy as np
from tqdm import tqdm

# Core terminal imports
from simulation.terminal_components.storage.BooleanStorage import BooleanStorageYard, PlacementResult
from simulation.terminal_components.systems.TerminalManager import TerminalLogisticsManager
from simulation.terminal_components.systems import TerminalManager as TM  # to set PROXIMITY
from simulation.terminal_components.systems.railyard import BooleanRailYard
from simulation.terminal_components.systems.parking import ParkingArea
from simulation.terminal_components.systems.LogisticsManager import LogisticsManager
from simulation.terminal_components.systems.train_tools.DPParser import DrivingPlanParser
from simulation.terminal_components.systems.train_tools.TrainScheduler import TrainScheduler
from simulation.terminal_components.systems.train_tools.TrainLoader import TrainLoader
from simulation.terminal_components.storage_units.ContainerFactory import ContainerFactory
from simulation.terminal_components.vehicles.TruckFactory import TruckFactory
from simulation.terminal_components.systems.TerminalGate import TerminalGate
from simulation.environment.CTEnv import ContainerTerminalEnv
from simulation.analytics.stats_tracker import StatsTracker

# Agents
from simulation.rl.agents.dqn_agent import DQNAgent, DQNConfig
from simulation.rl.agents.ppo_agent import PPOAgent, PPOConfig

@dataclass
class Module:
    name: str
    env: ContainerTerminalEnv
    yard: BooleanStorageYard
    scheduler: TrainScheduler
    tracker: StatsTracker
    agent: Any  # DQNAgent or PPOAgent

class DualActRecorder:
    def __init__(self, agent):
        self.agent = agent
        self.records = []

    def reset(self):
        self.records.clear()

    def act(self, state_np, moves):
        if isinstance(self.agent, DQNAgent):
            a_idx = self.agent.act(state_np, moves)
            self.records.append({"algo": "dqn", "state": state_np, "moves": moves, "a": a_idx})
            return a_idx
        else:
            a_idx, logp, value = self.agent.act(state_np, moves)
            self.records.append({"algo": "ppo", "state": state_np, "moves": moves,
                                 "a": a_idx, "logp": logp, "value": value})
            return a_idx

def jsonable(obj):
    if isinstance(obj, PlacementResult):
        return {"row": int(obj.row), "bay": int(obj.bay), "tier": int(obj.tier),
                "start_split": int(obj.start_split), "score": float(obj.score)}
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, (set, tuple)):
        return [jsonable(x) for x in obj]
    if isinstance(obj, list):
        return [jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    return obj

def split_trains_evenly(trains: List, tracks_m1: int = 7, tracks_m2: int = 6) -> Tuple[List, List]:
    # Sort by arrival, balance by total stay per track
    keyed = [(t.schedule_encoded['arrival']['seconds'], t) for t in trains]
    keyed.sort(key=lambda x: x[0])
    m1, m2 = [], []
    load1 = 0.0
    load2 = 0.0
    for _, t in keyed:
        stay = float(t.schedule_encoded['stay_duration']['hours'])
        if (load1 / max(1, tracks_m1)) <= (load2 / max(1, tracks_m2)):
            m1.append(t); load1 += stay
        else:
            m2.append(t); load2 += stay
    return m1, m2

class FilteringDrivingPlanParser(DrivingPlanParser):
    """Filters trains by train_id whitelist."""
    def __init__(self, whitelist_ids: Set[str], json_path: str = None):
        super().__init__()
        self.whitelist = set(whitelist_ids)

    def create_trains(self) -> List:
        all_tr = super().create_trains()
        return [t for t in all_tr if t.train_id in self.whitelist]

def build_module(name: str,
                 rows: int, bays: int, tiers: int, tracks: int,
                 parser, container_factory: ContainerFactory, truck_factory: TruckFactory,
                 import_cap: int = 400, export_per_import: float = 0.75,
                 overgen: float = 3.0,
                 logdir: str = "", algo: str = "dqn") -> Module:
    yard = BooleanStorageYard(n_rows=rows, n_bays=bays, n_tiers=tiers, coordinates=[], validate=False)
    rail = BooleanRailYard()
    parking = ParkingArea(ParkingArea.make_grid(n_bays=bays, split_factor=20, prefix=f"P_{name}"))
    gate = TerminalGate(container_factory, truck_factory)
    scheduler = TrainScheduler(num_tracks=tracks)
    loader = TrainLoader(container_factory, overgeneration_factor=overgen)
    lm = LogisticsManager(yard, gate, loader, scheduler, parser,
                          daily_import_cap=import_cap, export_per_import=export_per_import)
    tlm = TerminalLogisticsManager(yard, rail, parking)

    mdir = os.path.join(logdir, name)
    os.makedirs(mdir, exist_ok=True)
    tracker = StatsTracker(moves_path=os.path.join(mdir, "moves.ndjson"),
                           daily_csv_path=os.path.join(mdir, "daily.csv"),
                           yard=yard)

    # NEW: pass stats to env
    env = ContainerTerminalEnv(yard=yard, rail=rail, parking=parking,
                               tlm=tlm, lm=lm, num_tracks=tracks,
                               step_minutes=5, stats=tracker)

    dims = (yard.n_rows, yard.n_bays, yard.n_tiers, yard.split_factor)
    agent = DQNAgent(dims, DQNConfig()) if algo == "dqn" else PPOAgent(dims, PPOConfig())
    return Module(name=name, env=env, yard=yard, scheduler=scheduler, tracker=tracker, agent=agent)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["dqn", "ppo"], default="dqn")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--logdir", type=str, default="runs/dual_modules")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--import-cap", type=int, default=400, help="Per-module daily import cap")
    ap.add_argument("--export-per-import", type=float, default=0.75, help="Exports per import (e.g., 0.6 or 0.75)")
    ap.add_argument("--overgen", type=float, default=3.0, help="TrainLoader overgeneration factor")
    ap.add_argument("--load-m1", type=str, default=None)
    ap.add_argument("--load-m2", type=str, default=None)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    TM.PROXIMITY = 5

    outdir = os.path.join(args.logdir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(outdir, exist_ok=True)
    ckpt_dir = os.path.join(outdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    base_parser = DrivingPlanParser()
    all_trains = base_parser.create_trains()
    m1_trains, m2_trains = split_trains_evenly(all_trains, tracks_m1=7, tracks_m2=6)
    m1_ids = {t.train_id for t in m1_trains}
    m2_ids = {t.train_id for t in m2_trains}

    container_factory = ContainerFactory()
    truck_factory = TruckFactory()

    parser_m1 = FilteringDrivingPlanParser(whitelist_ids=m1_ids)
    parser_m2 = FilteringDrivingPlanParser(whitelist_ids=m2_ids)
    m1 = build_module("M1", rows=5, bays=58, tiers=5, tracks=7,
                      parser=parser_m1, container_factory=container_factory, truck_factory=truck_factory,
                      import_cap=args.import_cap, export_per_import=args.export_per_import,
                      overgen=args.overgen, logdir=outdir, algo=args.algo)
    m2 = build_module("M2", rows=3, bays=58, tiers=3, tracks=6,
                      parser=parser_m2, container_factory=container_factory, truck_factory=truck_factory,
                      import_cap=args.import_cap, export_per_import=args.export_per_import,
                      overgen=args.overgen, logdir=outdir, algo=args.algo)

    if args.load_m1:
        m1.agent.load(args.load_m1, map_location="cpu"); print(f"[M1] Loaded: {args.load_m1}")
    if args.load_m2:
        m2.agent.load(args.load_m2, map_location="cpu"); print(f"[M2] Loaded: {args.load_m2}")

    start_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    best_sum_reward = -1e18

    try:
        for d in range(args.days):
            day_start = start_day + timedelta(days=d)

            m1.env.reset(day_start, day_index=d)
            m2.env.reset(day_start, day_index=d)
            m1.tracker.reset_day_aggregates()
            m2.tracker.reset_day_aggregates()

            rec1 = DualActRecorder(m1.agent)
            rec2 = DualActRecorder(m2.agent)

            done1 = False; done2 = False
            day_reward_m1 = 0.0; day_reward_m2 = 0.0
            steps = 0
            pbar = tqdm(total=1, desc=f"Day {d+1}/{args.days}", leave=False)

            while not (done1 and done2):
                t1 = None if done1 else m1.env.current_time
                t2 = None if done2 else m2.env.current_time

                if t2 is None or (t1 is not None and t1 <= t2):
                    rec1.reset()
                    ns, nm, rew, dn, info = m1.env.step_dual_agent(rec1)  # no log_cb needed
                    day_reward_m1 += rew

                    if rec1.records:
                        r_exec = [e.get("reward", 0.0) for e in info.get("executed", [])]
                        for k, r in enumerate(rec1.records):
                            r_k = r_exec[k] if k < len(r_exec) else rew / max(1, len(rec1.records))
                            if isinstance(m1.agent, DQNAgent):
                                if k + 1 < len(rec1.records):
                                    ns_k = rec1.records[k+1]["state"]; nm_k = rec1.records[k+1]["moves"]; dn_k = False
                                else:
                                    ns_k = ns; nm_k = nm; dn_k = dn
                                m1.agent.remember(r["state"], r["moves"], r["a"], r_k, ns_k, nm_k, dn_k)
                            else:
                                m1.agent.remember(r["state"], r["moves"], r["a"], r.get("logp", 0.0),
                                                  r_k, r.get("value", 0.0), dn)
                        if isinstance(m1.agent, DQNAgent):
                            m1.agent.optimize()

                    done1 = dn
                else:
                    rec2.reset()
                    ns, nm, rew, dn, info = m2.env.step_dual_agent(rec2)
                    day_reward_m2 += rew

                    if rec2.records:
                        r_exec = [e.get("reward", 0.0) for e in info.get("executed", [])]
                        for k, r in enumerate(rec2.records):
                            r_k = r_exec[k] if k < len(r_exec) else rew / max(1, len(rec2.records))
                            if isinstance(m2.agent, DQNAgent):
                                if k + 1 < len(rec2.records):
                                    ns_k = rec2.records[k+1]["state"]; nm_k = rec2.records[k+1]["moves"]; dn_k = False
                                else:
                                    ns_k = ns; nm_k = nm; dn_k = dn
                                m2.agent.remember(r["state"], r["moves"], r["a"], r_k, ns_k, nm_k, dn_k)
                            else:
                                m2.agent.remember(r["state"], r["moves"], r["a"], r.get("logp", 0.0),
                                                  r_k, r.get("value", 0.0), dn)
                        if isinstance(m2.agent, DQNAgent):
                            m2.agent.optimize()

                    done2 = dn

                steps += 1
                pbar.set_postfix_str(f"steps={steps} M1R={day_reward_m1:.1f} M2R={day_reward_m2:.1f}")

                if done1 and done2:
                    pbar.update(1.0)
                    pbar.close()

            if isinstance(m1.agent, PPOAgent):
                m1.agent.update()
            if isinstance(m2.agent, PPOAgent):
                m2.agent.update()

            # Fill daily imports_unloaded from TRAIN_TO_YARD move counts
            m1.tracker.imports_unloaded = m1.tracker.move_counts.get("TRAIN_TO_YARD", 0)
            m2.tracker.imports_unloaded = m2.tracker.move_counts.get("TRAIN_TO_YARD", 0)

            m1.tracker.write_day_summary(day_index=d, date=day_start)
            m2.tracker.write_day_summary(day_index=d, date=day_start)

            m1.agent.save(os.path.join(ckpt_dir, "m1_last.pt"))
            m2.agent.save(os.path.join(ckpt_dir, "m2_last.pt"))
            sum_r = day_reward_m1 + day_reward_m2
            if sum_r > best_sum_reward:
                best_sum_reward = sum_r
                m1.agent.save(os.path.join(ckpt_dir, "m1_best.pt"))
                m2.agent.save(os.path.join(ckpt_dir, "m2_best.pt"))
            print(f"Day {d+1}: M1R={day_reward_m1:.2f} M2R={day_reward_m2:.2f} (ckpt saved)")

        print(f"Training finished. Logs at: {outdir}; checkpoints in {ckpt_dir}")

    finally:
        m1.tracker.close()
        m2.tracker.close()

if __name__ == "__main__":
    main()