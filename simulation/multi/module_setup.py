from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Dict, Optional

from simulation.terminal_components.storage.BooleanStorage import BooleanStorageYard
from simulation.terminal_components.systems.railyard import BooleanRailYard
from simulation.terminal_components.systems.parking import ParkingArea
from simulation.terminal_components.systems.train_tools.DPParser import DrivingPlanParser
from simulation.terminal_components.systems.train_tools.TrainScheduler import TrainScheduler
from simulation.terminal_components.systems.train_tools.TrainLoader import TrainLoader
from simulation.terminal_components.systems.LogisticsManager import LogisticsManager
from simulation.terminal_components.systems.TerminalGate import TerminalGate
from simulation.terminal_components.storage_units.ContainerFactory import ContainerFactory
from simulation.terminal_components.vehicles.TruckFactory import TruckFactory
from simulation.environment.CTEnv import ContainerTerminalEnv
from simulation.terminal_components.vehicles.Train import Train

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

def build_module(name: str,
                 rows: int, bays: int, tiers: int,
                 tracks: int, num_cranes: int = 2) -> TerminalModule:
    yard = BooleanStorageYard(n_rows=rows, n_bays=bays, n_tiers=tiers, coordinates=[], validate=False)
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
    # tlm is instantiated inside your code when building env; keep as in your project if different
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