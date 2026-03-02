# simulation/training/scenarios/9_complex_ops/multi_truck_to_train.py
"""S30 — Multiple delivery trucks → single train.

3 delivery trucks each bring 2 export containers.  1 train wants all 6.
The agent must park trucks, unload them (to yard or direct-transfer),
then load the train.  Tests cross-modal coordination across all three
facilities (queue → parking → yard → rail).
"""
import random
from datetime import timedelta

from simulation.core.enums import Direction
from simulation.training.scenarios._base import (
    TutorialScenario,
    _make_container,
    _make_train,
    _make_truck,
    _slot_train,
    ANCHOR_BAY,
    TUTORIAL_TIME,
)


class MultiTruckToTrain(TutorialScenario):
    id = 30
    name = "multi_truck_to_train"
    description = "Park 3 delivery trucks, transfer 6 exports to a single train"
    max_steps = 80
    repeatable = True

    def setup(self, env) -> None:
        rng = random.Random(30)

        export_ids = []

        # 3 delivery trucks, each carrying 2 export containers
        for t in range(3):
            truck_containers = []
            for c in range(2):
                cid = f"S30_EXP{t * 2 + c}"
                cont = _make_container(
                    cid, Direction.EXPORT,
                    departure=TUTORIAL_TIME + timedelta(days=5),
                    rng=rng,
                )
                truck_containers.append(cont)
                export_ids.append(cid)

            tk = _make_truck(f"S30_DTK{t}", containers=truck_containers)
            env.trucks[tk.truck_id] = tk

        # Train wanting all 6 exports
        tr = _make_train("S30_TR1", pickup_ids=export_ids, num_wagons=10)
        _slot_train(env, tr)

    def check_success(self, env) -> bool:
        tr = env.trains.get("S30_TR1")
        if tr is None:
            return False
        # All 6 exports loaded
        exports_loaded = all(
            tr.has_container(f"S30_EXP{i}") for i in range(6)
        )
        # All 3 delivery trucks departed
        trucks_gone = all(
            f"S30_DTK{t}" not in env.trucks for t in range(3)
        )
        return exports_loaded and trucks_gone
