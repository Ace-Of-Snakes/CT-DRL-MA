# simulation/training/scenarios/9_complex_ops/full_train_unload.py
"""S28 — Fully unload a train (6 import containers → yard).

Validates the agent can sustain sequential TRAIN_TO_YARD moves to
completely clear a train.  No trucks — pure unload operation.
"""
import random

from simulation.core.enums import Direction
from simulation.training.scenarios._base import (
    TutorialScenario,
    _make_container,
    _make_train,
    _place_distractors,
    _random_container_spec,
    _slot_train,
    ANCHOR_BAY,
)


class FullTrainUnload(TutorialScenario):
    id = 28
    name = "full_train_unload"
    description = "Unload 6 import containers from a single train into the yard"
    max_steps = 40
    repeatable = True

    def setup(self, env) -> None:
        rng = random.Random(28)

        # 6 import containers on the train (diverse sizes)
        imports = []
        for i in range(6):
            c = _make_container(f"S28_IMP{i}", Direction.IMPORT, rng=rng)
            imports.append(c)

        tr = _make_train("S28_TR1", containers=imports, num_wagons=10)
        _slot_train(env, tr)

        # Light clutter so the agent must find free yard space
        _place_distractors(env, rng, 6, "S28", exclude_bays={ANCHOR_BAY})

    def check_success(self, env) -> bool:
        tr = env.trains.get("S28_TR1")
        if tr is None:
            return False
        # All 6 imports must be in the yard (off the train)
        for i in range(6):
            cid = f"S28_IMP{i}"
            if tr.has_container(cid):
                return False  # still on train
            if env.yard.get_container(cid) is None:
                return False  # not in yard either
        return True
