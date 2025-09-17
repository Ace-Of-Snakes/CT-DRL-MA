# scripts/train_terminal.py
import os, sys, argparse, random, time
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
from tqdm import tqdm

# Imports from your codebase
from simulation.terminal_components.storage.BooleanStorage import BooleanStorageYard
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
    # Yard: adjust coordinates as per your site config
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

    # Factories and gate
    container_factory = ContainerFactory()
    truck_factory = TruckFactory()
    gate = TerminalGate(container_factory, truck_factory)

    # Schedulers and loaders
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["dqn","ppo"], default="dqn")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--bays", type=int, default=60)
    ap.add_argument("--tiers", type=int, default=5)
    ap.add_argument("--tracks", type=int, default=6)
    ap.add_argument("--step-minutes", type=int, default=5)
    ap.add_argument("--logdir", type=str, default="runs/terminal")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    outdir = os.path.join(args.logdir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(outdir, exist_ok=True)
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

    start_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    try:
        for d in range(args.days):
            day_start = start_day + timedelta(days=d)
            state, moves = env.reset(day_start, day_index=d)
            tracker.reset_day_aggregates()

            pbar = tqdm(total=1, desc=f"Day {d+1}/{args.days}", leave=False)
            step = 0
            day_done = False
            day_reward = 0.0

            while not day_done:
                # Select action
                if args.algo == "dqn":
                    a = agent.act(state, moves)
                else:
                    a, logp, value = agent.act(state, moves)

                next_state, next_moves, reward, done, info = env.step(a, moves)
                day_reward += reward

                # Log executed move
                if info.get("executed"):
                    rec = info["executed"]
                    rec.update({
                        "day_index": d,
                        "step": step,
                        "time": rec["timestamp"],
                        "move_idx": a
                    })
                    tracker.log_move(rec)

                # Log train departures (for KPI aggregation)
                for tid in info.get("train_departures", []):
                    # leftover count and imports unloaded are observable from the train / yard if needed
                    # here we zero-fill; if you keep a train ref in env._departed_cache, you can compute.
                    tracker.on_train_departure(leftover_ids_count=0, imports_unloaded_count=0)

                # Log truck departures (waiting times)
                for tinfo in info.get("truck_departures", []):
                    tracker.on_truck_departure(wait_minutes=tinfo["wait_min"])

                # Store in replay/update
                if args.algo == "dqn":
                    agent.remember(state, moves, a, reward, next_state, next_moves, done)
                    agent.optimize()
                else:
                    # PPO trajectory
                    if moves:
                        agent.remember(state, moves, a, logp, reward, value, done)

                state, moves = next_state, next_moves
                step += 1

                # progress bar (reward running avg)
                pbar.set_postfix_str(f"steps={step} dayR={day_reward:.2f}")
                pbar.update(0.0 if not done else 1.0)
                if done:
                    day_done = True

            pbar.close()

            # PPO epoch update at end of day
            if args.algo == "ppo":
                agent.update()

            # write day summary
            tracker.write_day_summary(day_index=d, date=day_start)

        print(f"Training finished. Logs at: {outdir}")

    finally:
        tracker.close()

if __name__ == "__main__":
    main()