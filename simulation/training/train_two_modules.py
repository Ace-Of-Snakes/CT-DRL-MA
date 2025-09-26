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
from simulation.core.facilities.yard import BooleanStorageYard, PlacementResult
from simulation.core.facilities.railyard import BooleanRailYard
from simulation.core.facilities.parking import ParkingArea
from simulation.core.factories.container_factory import ContainerFactory
from simulation.core.factories.truck_factory import TruckFactory

from simulation.planning.logistics_manager import LogisticsManager
from simulation.planning.driving_plan_parser import DrivingPlanParser
from simulation.planning.train_scheduler import TrainScheduler
from simulation.planning.train_loader import TrainLoader

from simulation.operations.terminal_manager import TerminalLogisticsManager
from simulation.operations import terminal_manager as TM  # to set PROXIMITY
from simulation.operations.gate import TerminalGate

from simulation.analytics.stats_tracker import StatsTracker
from simulation.env.env import ContainerTerminalEnv

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

def make_special_coordinates(n_rows: int,
                             n_bays: int,
                             sb_row_1b: int = 1,
                             dg_rows_big: Tuple[int, ...] = (3, 4, 5),
                             dg_rows_small: Tuple[int, ...] = (2, 3),
                             dg_halfwidth_bays: int = 3,
                             reefers_include_sb_row: bool = True) -> List[Tuple[int, int, str]]:
    """
    Build special-position coordinates for BooleanStorageYard.
    (bay, row, type) are 1-based; types: 'sb_t', 'r', 'dg'.
    """
    coords: List[Tuple[int, int, str]] = []

    sb_row = max(1, min(n_rows, sb_row_1b))
    center_bay = (n_bays + 1) // 2

    # Swap bodies/trailers row
    for bay in range(1, n_bays + 1):
        coords.append((bay, sb_row, "sb_t"))

    # Reefers on outer bays (include SB row allowed)
    for row in range(1, n_rows + 1):
        if not reefers_include_sb_row and row == sb_row:
            continue
        coords.append((1, row, "r"))
        coords.append((n_bays, row, "r"))

    # DG rows: big -> (3,4,5); small -> (2,3); exclude SB row
    yard_is_big = n_rows >= max(dg_rows_big) if dg_rows_big else False
    dg_rows = [r for r in (dg_rows_big if yard_is_big else dg_rows_small) if 1 <= r <= n_rows and r != sb_row]
    for row in dg_rows:
        for delta in range(-dg_halfwidth_bays, dg_halfwidth_bays + 1):
            b = center_bay + delta
            if 1 <= b <= n_bays:
                coords.append((b, row, "dg"))

    return coords

def build_module(name: str,
                 rows: int, bays: int, tiers: int, tracks: int,
                 parser, container_factory: ContainerFactory, truck_factory: TruckFactory,
                 train_import_cap: Optional[int] = 220,
                 export_per_import: float = 0.75,
                 overgen: float = 3.0,
                 logdir: str = "", algo: str = "dqn") -> Module:
    # Special zones derived from yard shape
    coordinates = make_special_coordinates(n_rows=rows, n_bays=bays)

    yard = BooleanStorageYard(n_rows=rows, n_bays=bays, n_tiers=tiers, coordinates=coordinates, validate=False)
    rail = BooleanRailYard()
    parking = ParkingArea(ParkingArea.make_grid(n_bays=bays, split_factor=20, prefix=f"P_{name}"))
    gate = TerminalGate(container_factory, truck_factory)
    scheduler = TrainScheduler(num_tracks=tracks)
    loader = TrainLoader(container_factory, overgeneration_factor=overgen)
    lm = LogisticsManager(yard, gate, loader, scheduler, parser,
                          export_per_import=export_per_import,
                          daily_train_import_cap=train_import_cap)
    tlm = TerminalLogisticsManager(yard, rail, parking)

    mdir = os.path.join(logdir, name)
    os.makedirs(mdir, exist_ok=True)
    tracker = StatsTracker(moves_path=os.path.join(mdir, "moves.ndjson"),
                           daily_csv_path=os.path.join(mdir, "daily.csv"),
                           yard=yard)

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
    ap.add_argument("--train-import-cap", type=int, default=None, help="Per-module cap on import containers on trains per day")
    ap.add_argument("--export-per-import", type=float, default=0.75, help="Exports per import ratio")
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
                    train_import_cap=args.train_import_cap,
                      export_per_import=args.export_per_import,
                      overgen=args.overgen, logdir=outdir, algo=args.algo)
    m2 = build_module("M2", rows=3, bays=58, tiers=3, tracks=6,
                      parser=parser_m2, container_factory=container_factory, truck_factory=truck_factory,
                      train_import_cap=args.train_import_cap,
                      export_per_import=args.export_per_import,
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
            # m1.tracker.imports_unloaded = m1.tracker.move_counts.get("TRAIN_TO_YARD", 0)
            # m2.tracker.imports_unloaded = m2.tracker.move_counts.get("TRAIN_TO_YARD", 0)

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