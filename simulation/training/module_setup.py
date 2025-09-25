# simulation/multi/module_setup.py
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Dict, Optional

from simulation.core.facilities.yard import BooleanStorageYard
from simulation.core.facilities.railyard import BooleanRailYard
from simulation.core.facilities.parking import ParkingArea
from simulation.planning.driving_plan_parser import DrivingPlanParser
from simulation.planning.train_scheduler import TrainScheduler
from simulation.planning.train_loader import TrainLoader
from simulation.planning.logistics_manager import LogisticsManager
from simulation.operations.gate import TerminalGate
from simulation.core.factories.container_factory import ContainerFactory
from simulation.core.factories.truck_factory import TruckFactory
from simulation.env.env import ContainerTerminalEnv
from simulation.core.vehicles.train import Train

@dataclass
class TerminalModule:
    name: str
    yard: BooleanStorageYard
    rail: BooleanRailYard
    parking: ParkingArea
    gate: TerminalGate
    parser: DrivingPlanParser
    scheduler: TrainScheduler
    loader: TrainLoader
    lm: LogisticsManager
    env: ContainerTerminalEnv

def make_special_coordinates(n_rows: int,
                             n_bays: int,
                             sb_row_1b: int = 1,
                             dg_rows_big: Tuple[int, ...] = (3, 4, 5),
                             dg_rows_small: Tuple[int, ...] = (2, 3),
                             dg_halfwidth_bays: int = 3,
                             reefers_include_sb_row: bool = True) -> List[Tuple[int, int, str]]:
    """
    Build special-position coordinates for BooleanStorageYard.
    Coordinates are (bay, row, type) in 1-based indexing:
      - 'sb_t': full swap-body/trailer row (tier 0 handled by yard)
      - 'r': reefers on outer bays (1 and n_bays), all rows (optionally including SB row)
      - 'dg': center bay ± dg_halfwidth_bays, on designated DG rows, excluding SB row
    """
    coords: List[Tuple[int, int, str]] = []

    # Clamp helpers
    sb_row = max(1, min(n_rows, sb_row_1b))
    first_bay = 1
    last_bay = max(1, n_bays)
    center_bay = (n_bays + 1) // 2

    # 1) Swap bodies/trailers: full row across all bays
    for bay in range(1, n_bays + 1):
        coords.append((bay, sb_row, "sb_t"))

    # 2) Reefers: outer bays on all rows (include SB row if requested)
    for row in range(1, n_rows + 1):
        if not reefers_include_sb_row and row == sb_row:
            continue
        coords.append((first_bay, row, "r"))
        coords.append((last_bay, row, "r"))

    # 3) Dangerous Goods: big yard -> rows (3,4,5); small yard -> rows (2,3); always exclude SB row
    yard_is_big = n_rows >= max(dg_rows_big) if dg_rows_big else False
    dg_rows = [r for r in (dg_rows_big if yard_is_big else dg_rows_small) if 1 <= r <= n_rows and r != sb_row]

    for row in dg_rows:
        for delta in range(-dg_halfwidth_bays, dg_halfwidth_bays + 1):
            b = center_bay + delta
            if 1 <= b <= n_bays:
                coords.append((b, row, "dg"))

    return coords

def build_module(name: str,
                 rows: int, bays: int, tiers: int,
                 tracks: int, num_cranes: int = 2) -> TerminalModule:
    # Apply special zones per yard size
    coordinates = make_special_coordinates(n_rows=rows, n_bays=bays)

    yard = BooleanStorageYard(n_rows=rows, n_bays=bays, n_tiers=tiers, coordinates=coordinates, validate=False)
    rail = BooleanRailYard()
    parking = ParkingArea(ParkingArea.make_grid(n_bays=bays, split_factor=20, prefix=f"P_{name}"))
    cf = ContainerFactory()
    tf = TruckFactory()
    gate = TerminalGate(cf, tf)
    parser = DrivingPlanParser()
    scheduler = TrainScheduler(num_tracks=tracks)
    loader = TrainLoader(cf)
    lm = LogisticsManager(yard, gate, loader, scheduler, parser)
    env = ContainerTerminalEnv(yard, rail, parking, tlm=None, lm=lm, num_tracks=tracks, step_minutes=5, num_cranes=num_cranes)
    return TerminalModule(name, yard, rail, parking, gate, parser, scheduler, loader, lm, env)

def split_trains_evenly(trains: List[Train], tracks_m1: int = 7, tracks_m2: int = 6) -> Tuple[List[Train], List[Train]]:
    # sort by arrival; balance by normalized stay load per track
    keyed = [(t.schedule_encoded['arrival']['seconds'], t) for t in trains]
    keyed.sort(key=lambda x: x[0])
    m1, m2 = [], []
    load1 = 0.0
    load2 = 0.0
    for _, t in keyed:
        stay = float(t.schedule_encoded['stay_duration']['hours'])
        # assign to lighter module (stay/tracks)
        if (load1 / max(1, tracks_m1)) <= (load2 / max(1, tracks_m2)):
            m1.append(t); load1 += stay
        else:
            m2.append(t); load2 += stay
    return m1, m2

class MultiModuleRunner:
    """
    Shared-clock orchestrator for two independent modules (no conflicts).
    - reset(day_start): split trains (7/6), reset both envs with overrides.
    - step(agent_m1, agent_m2): step the module with the earlier current_time.
    """
    def __init__(self, mod1: TerminalModule, mod2: TerminalModule, parser: Optional[DrivingPlanParser] = None):
        self.m1 = mod1
        self.m2 = mod2
        self.parser = parser or DrivingPlanParser()
        self.done1 = False
        self.done2 = False

    def reset(self, day_start: datetime) -> None:
        self.done1 = False; self.done2 = False
        all_trains = self.parser.create_trains()
        t1, t2 = split_trains_evenly(all_trains, tracks_m1=self.m1.scheduler.num_tracks, tracks_m2=self.m2.scheduler.num_tracks)
        self.m1.env.reset(day_start, day_index=0, trains_override=t1)
        self.m2.env.reset(day_start, day_index=0, trains_override=t2)

    def step(self, agent_m1, agent_m2,
             log_cb1=None, log_cb2=None) -> Dict[str, any]:
        # pick the env that is earlier in time (shared-clock behavior)
        if self.done1 and self.done2:
            return {"done": True}
        e1_t = self.m1.env.current_time if not self.done1 else None
        e2_t = self.m2.env.current_time if not self.done2 else None

        # choose the earlier one
        if e2_t is None or (e1_t is not None and e1_t <= e2_t):
            state, moves, reward, done, info = self.m1.env.step_dual_agent(agent_m1, log_cb=log_cb1)
            self.done1 = done
            return {"module": self.m1.name, "done": self.done1 and self.done2, "reward": reward, "info": info}
        else:
            state, moves, reward, done, info = self.m2.env.step_dual_agent(agent_m2, log_cb=log_cb2)
            self.done2 = done
            return {"module": self.m2.name, "done": self.done1 and self.done2, "reward": reward, "info": info}