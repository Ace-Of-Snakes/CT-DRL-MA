# scripts/train_terminal.py
import os, argparse, random
from datetime import datetime, timedelta

import numpy as np
from tqdm import tqdm

# Terminal imports
from simulation.terminal_components.storage.BooleanStorage import BooleanStorageYard, PlacementResult
from simulation.terminal_components.systems.TerminalManager import TerminalLogisticsManager
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


def build_terminal(args):
    # Yard: ggf. Koordinaten anpassen
    yard = BooleanStorageYard(
        n_rows=args.rows,
        n_bays=args.bays,
        n_tiers=args.tiers,
        coordinates=[],
        validate=False
    )
    rail = BooleanRailYard()
    parking_spots = ParkingArea.make_grid(n_bays=args.bays, split_factor=20, prefix="P") if hasattr(ParkingArea, "make_grid") else set()
    parking = ParkingArea(parking_spots)

    # Factories + Gate
    container_factory = ContainerFactory()
    truck_factory = TruckFactory()
    gate = TerminalGate(container_factory, truck_factory)

    # Parser/Scheduler/Loader
    parser = DrivingPlanParser()
    scheduler = TrainScheduler(num_tracks=args.tracks)
    loader = TrainLoader(container_factory)

    lm = LogisticsManager(yard, gate, loader, scheduler, parser)
    tlm = TerminalLogisticsManager(yard, rail, parking)

    env = ContainerTerminalEnv(
        yard=yard, rail=rail, parking=parking, tlm=tlm, lm=lm, num_tracks=args.tracks,
        step_minutes=args.step_minutes
    )
    return env, yard


class DualActRecorder:
    """
    Wrapper um Agent.act, damit step_dual_agent die getroffenen Entscheidungen
    mitsamt (State, Kandidatenliste) zurückliefert und das Training (Replay/GAE) möglich bleibt.
    - Für DQN speichert: state, moves, action_idx
    - Für PPO speichert zusätzlich: logp, value
    Rückgabewert an Env ist immer nur der Index (int), damit Env generisch bleibt.
    """
    def __init__(self, agent, yard_dims):
        self.agent = agent
        self.records = []
        self.yard_dims = yard_dims

    def reset(self):
        self.records.clear()

    def act(self, state_np, moves):
        # DQNAgent.act(state, moves) -> int
        # PPOAgent.act(state, moves) -> (idx, logp, value)
        if isinstance(self.agent, DQNAgent):
            a_idx = self.agent.act(state_np, moves)
            self.records.append({"algo": "dqn", "state": state_np, "moves": moves, "a": a_idx})
            return a_idx
        else:
            a_idx, logp, value = self.agent.act(state_np, moves)
            # Nur Index an Env zurückgeben, aber Logp/Value speichern
            self.records.append({"algo": "ppo", "state": state_np, "moves": moves, "a": a_idx, "logp": logp, "value": value})
            return a_idx


def jsonable(obj):
    # PlacementResult -> dict, datetime -> iso, numpy scalar -> python, sets/tuples->list
    if isinstance(obj, PlacementResult):
        return {
            "row": int(obj.row),
            "bay": int(obj.bay),
            "tier": int(obj.tier),
            "start_split": int(obj.start_split),
            "score": float(obj.score),
        }
    try:
        from datetime import datetime as _dt
        if isinstance(obj, _dt):
            return obj.isoformat()
    except Exception:
        pass
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, (set, tuple)):
        return [jsonable(x) for x in obj]
    if isinstance(obj, list):
        return [jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["dqn", "ppo"], default="dqn")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--bays", type=int, default=60)
    ap.add_argument("--tiers", type=int, default=5)
    ap.add_argument("--tracks", type=int, default=6)
    ap.add_argument("--step-minutes", type=int, default=5)
    ap.add_argument("--logdir", type=str, default="runs/terminal")
    ap.add_argument("--seed", type=int, default=42)
    # optional checkpoints
    ap.add_argument("--ckpt-dir", type=str, default=None)
    ap.add_argument("--save-every-days", type=int, default=1)
    ap.add_argument("--load", type=str, default=None)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--epsilon-eval", type=float, default=0.0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    outdir = os.path.join(args.logdir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(outdir, exist_ok=True)
    ckpt_dir = args.ckpt_dir or os.path.join(outdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    moves_path = os.path.join(outdir, "moves.ndjson")
    daily_path = os.path.join(outdir, "daily.csv")

    env, yard = build_terminal(args)
    tracker = StatsTracker(moves_path, daily_path, yard)

    # Agent
    yard_dims = (yard.n_rows, yard.n_bays, yard.n_tiers, yard.split_factor)
    if args.algo == "dqn":
        agent = DQNAgent(yard_dims, DQNConfig())
    else:
        agent = PPOAgent(yard_dims, PPOConfig())

    if args.load:
        if args.algo == "dqn":
            agent.load(args.load, map_location="cpu")
        else:
            agent.load(args.load, map_location="cpu")
        print(f"Loaded checkpoint: {args.load}")

    start_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    event_counter = 0
    best_day_reward = -1e18

    def log_cb(rec):
        # rec kommt aus env.step_dual_agent; säubern und an Tracker geben
        safe = {
            "day_index": current_day_index,
            "time": rec.get("timestamp"),
            "crane_id": rec.get("crane_id"),
            "move_type": rec.get("move_type"),
            "args": jsonable(rec.get("args")),
            "distance_m": rec.get("distance_m"),
            "time_s": rec.get("time_s"),
            "reward": rec.get("reward"),
            "event_idx": event_counter
        }
        tracker.log_move(safe)

    try:
        for d in range(args.days):
            current_day_index = d
            day_start = start_day + timedelta(days=d)
            state, moves = env.reset(day_start, day_index=d)
            tracker.reset_day_aggregates()

            pbar = tqdm(total=1, desc=f"Day {d+1}/{args.days}", leave=False)
            day_done = False
            day_reward = 0.0
            event_counter = 0

            # Recorder für duale Kranentscheidungen
            recorder = DualActRecorder(agent, yard_dims)

            while not day_done:
                recorder.reset()
                # duales, ereignisgetriebenes Step – env ruft recorder.act intern (bis zu 2x)
                next_state, next_moves, reward, done, info = env.step_dual_agent(recorder, log_cb=log_cb)
                day_reward += reward

                # Training: für jede Entscheidung im Recorder eine Transition speichern
                # Zuordnung des Teilrewards je Entscheidung: info["executed"][k]["reward"]
                # next-States:
                #   - für erste Entscheidung: next_state_part = recorder.records[1]["state"] (falls vorhanden)
                #   - für zweite Entscheidung (oder wenn nur 1): next_state (Rückgabewert)
                # next-Moves analog
                if recorder.records:
                    # Rewards pro Move (falls Env sie liefert)
                    rewards_list = []
                    if info.get("executed"):
                        rewards_list = [e.get("reward", 0.0) for e in info["executed"]]
                    # DQN/PPO speichern
                    for k, rec in enumerate(recorder.records):
                        r_k = rewards_list[k] if k < len(rewards_list) else reward / max(1, len(recorder.records))
                        if args.algo == "dqn":
                            # Next state/moves für k
                            if k + 1 < len(recorder.records):
                                ns_k = recorder.records[k+1]["state"]
                                nm_k = recorder.records[k+1]["moves"]
                                dn_k = False
                            else:
                                ns_k = next_state
                                nm_k = next_moves
                                dn_k = done
                            # speichern + optimieren
                            agent.remember(rec["state"], rec["moves"], rec["a"], r_k, ns_k, nm_k, dn_k)
                        else:
                            # PPO speichert Trajektorie; Update am Tagesende
                            agent.remember(rec["state"], rec["moves"], rec["a"], rec.get("logp", 0.0), r_k, rec.get("value", 0.0), done)

                    if args.algo == "dqn":
                        agent.optimize()

                # Abfahrten in KPI übernehmen
                for dep in info.get("train_departures", []):
                    # wenn Env dict statt id liefert:
                    if isinstance(dep, dict):
                        tracker.on_train_departure(leftover_ids_count=dep.get("leftover_ids", 0),
                                                   imports_unloaded_count=dep.get("imports_unloaded", 0))
                    else:
                        tracker.on_train_departure(leftover_ids_count=0, imports_unloaded_count=0)

                # Lkw-Abfahrtenlog
                for tinfo in info.get("truck_departures", []):
                    tracker.on_truck_departure(wait_minutes=tinfo.get("wait_min", 0.0))

                state, moves = next_state, next_moves
                event_counter += 1

                # Fortschritt
                pbar.set_postfix_str(f"events={event_counter} dayR={day_reward:.2f}")
                if done:
                    pbar.update(1.0)
                    day_done = True

            pbar.close()

            # PPO Update am Ende des Tages
            if args.algo == "ppo" and not args.eval_only:
                agent.update()

            # Tageszusammenfassung
            tracker.write_day_summary(day_index=d, date=day_start)

            # Checkpointing
            if not args.eval_only and (d + 1) % max(1, args.save_every_days) == 0:
                last_path = os.path.join(ckpt_dir, f"{args.algo}_last.pt")
                best_path = os.path.join(ckpt_dir, f"{args.algo}_best.pt")
                agent.save(last_path)
                if day_reward > best_day_reward:
                    best_day_reward = day_reward
                    agent.save(best_path)
                print(f"Day {d+1}: saved checkpoints to {ckpt_dir} (last/best)")

        print(f"Training finished. Logs at: {outdir}; checkpoints at: {ckpt_dir}")

    finally:
        tracker.close()


if __name__ == "__main__":
    main()