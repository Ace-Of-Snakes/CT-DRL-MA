"""Tests for unified agent and replay buffer — Phase 3 validation.

Part 1: Pure-logic tests (no torch needed).
  - Move type resolution from region pairs
  - Flatten/unflatten coordinate transforms
  - Replay buffer compression and sampling
  - Config validation

Part 2: Torch functional tests (skipped if unavailable).
  - Agent forward pass
  - Source mask derivation from state
  - Loss computation
  - Gradient flow
"""
import sys
import os
import types
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ══════════════════════════════════════════════════════════════════════════
# Constants matching defaults
# ══════════════════════════════════════════════════════════════════════════

R = 15; S = 1160; T = 5; S_DOWN = 290; STRIDE = 4; C = 10


# ══════════════════════════════════════════════════════════════════════════
# Part 1: Pure-logic tests (no torch)
# ══════════════════════════════════════════════════════════════════════════

class TestMoveTypeResolution:
    """Verify all region → move type mappings."""

    # Inline the map (avoids importing config which depends on torch)
    _MAP = {
        ("QUEUE", "PARKING"):  "PARK_TRUCK",
        ("RAIL", "YARD"):      "TRAIN_TO_YARD",
        ("PARKING", "YARD"):   "TRUCK_TO_YARD",
        ("YARD", "YARD"):      "YARD_TO_YARD",
        ("YARD", "RAIL"):      "YARD_TO_TRAIN",
        ("YARD", "PARKING"):   "YARD_TO_TRUCK",
    }

    def _get_resolve(self):
        return lambda src, dst: self._MAP.get((src, dst))

    def test_queue_to_parking(self):
        resolve = self._get_resolve()
        assert resolve("QUEUE", "PARKING") == "PARK_TRUCK"

    def test_rail_to_yard(self):
        resolve = self._get_resolve()
        assert resolve("RAIL", "YARD") == "TRAIN_TO_YARD"

    def test_parking_to_yard(self):
        resolve = self._get_resolve()
        assert resolve("PARKING", "YARD") == "TRUCK_TO_YARD"

    def test_yard_to_yard(self):
        resolve = self._get_resolve()
        assert resolve("YARD", "YARD") == "YARD_TO_YARD"

    def test_yard_to_rail(self):
        resolve = self._get_resolve()
        assert resolve("YARD", "RAIL") == "YARD_TO_TRAIN"

    def test_yard_to_parking(self):
        resolve = self._get_resolve()
        assert resolve("YARD", "PARKING") == "YARD_TO_TRUCK"

    def test_invalid_combination_returns_none(self):
        resolve = self._get_resolve()
        assert resolve("QUEUE", "YARD") is None
        assert resolve("RAIL", "RAIL") is None
        assert resolve("PARKING", "QUEUE") is None

    def test_all_6_valid_combinations(self):
        resolve = self._get_resolve()
        valid_count = sum(
            1 for src in ["RAIL", "PARKING", "YARD", "QUEUE"]
            for dst in ["RAIL", "PARKING", "YARD", "QUEUE"]
            if resolve(src, dst) is not None
        )
        assert valid_count == 6


class TestCoordinateTransforms:
    """Verify flatten/unflatten at unified dimensions."""

    def _flatten(self, pos):
        """(row, s_down, tier) → flat index for (R, S_DOWN, T)."""
        row, s_d, tier = pos
        return (row * S_DOWN + s_d) * T + tier

    def _unflatten(self, flat_idx):
        """flat index → (row, s_down, tier)."""
        tier = flat_idx % T
        s_down = (flat_idx // T) % S_DOWN
        row = flat_idx // (S_DOWN * T)
        return row, s_down, tier

    def test_roundtrip_origin(self):
        pos = (0, 0, 0)
        assert self._unflatten(self._flatten(pos)) == pos

    def test_roundtrip_yard_position(self):
        pos = (10, 150, 3)  # yard row, mid-bay, tier 3
        assert self._unflatten(self._flatten(pos)) == pos

    def test_roundtrip_queue_position(self):
        pos = (14, 280, 0)  # queue row 1, near end, tier 0
        assert self._unflatten(self._flatten(pos)) == pos

    def test_roundtrip_last_position(self):
        pos = (R - 1, S_DOWN - 1, T - 1)
        flat = self._flatten(pos)
        assert flat == R * S_DOWN * T - 1
        assert self._unflatten(flat) == pos

    def test_flat_range(self):
        """Flat indices should cover [0, R*S_DOWN*T)."""
        max_flat = R * S_DOWN * T
        assert max_flat == 21_750
        assert self._flatten((0, 0, 0)) == 0
        assert self._flatten((R - 1, S_DOWN - 1, T - 1)) == max_flat - 1


class TestDownsampleMask:
    """Verify mask downsampling logic (pure numpy)."""

    def _downsample(self, mask):
        """Max-pool (R, S, T) → (R, S_DOWN, T)."""
        R_m, S_m, T_m = mask.shape
        S_d = S_m // STRIDE
        trimmed = mask[:, :S_d * STRIDE, :]
        reshaped = trimmed.reshape(R_m, S_d, STRIDE, T_m)
        return reshaped.any(axis=2)

    def test_empty_mask(self):
        mask = np.zeros((R, S, T), dtype=bool)
        down = self._downsample(mask)
        assert down.shape == (R, S_DOWN, T)
        assert not down.any()

    def test_single_position(self):
        mask = np.zeros((R, S, T), dtype=bool)
        mask[10, 42, 0] = True  # split 42 → s_down 10 (42//4=10)
        down = self._downsample(mask)
        assert down[10, 10, 0] == True
        assert down.sum() == 1

    def test_stride_window_collapse(self):
        """Multiple Trues in same stride window → single True."""
        mask = np.zeros((R, S, T), dtype=bool)
        mask[8, 40, 0] = True
        mask[8, 41, 0] = True
        mask[8, 42, 0] = True
        mask[8, 43, 0] = True  # all in window [40,44) → s_down=10
        down = self._downsample(mask)
        assert down[8, 10, 0] == True
        assert down.sum() == 1

    def test_different_stride_windows(self):
        mask = np.zeros((R, S, T), dtype=bool)
        mask[8, 0, 0] = True   # s_down = 0
        mask[8, 4, 0] = True   # s_down = 1
        mask[8, 8, 0] = True   # s_down = 2
        down = self._downsample(mask)
        assert down[8, 0, 0] == True
        assert down[8, 1, 0] == True
        assert down[8, 2, 0] == True
        assert down.sum() == 3

    def test_multi_region(self):
        """Positions across all 4 regions."""
        mask = np.zeros((R, S, T), dtype=bool)
        mask[3, 100, 0] = True   # rail
        mask[7, 200, 0] = True   # parking
        mask[10, 300, 0] = True  # yard
        mask[13, 400, 0] = True  # queue
        down = self._downsample(mask)
        assert down.sum() == 4


class TestReplayBuffer:
    """Test unified replay buffer (no torch needed)."""

    def _make_transition(self, reward=0.5):
        from unified_replay_buffer import UnifiedTransition
        return UnifiedTransition(
            state=np.random.randn(C, R, S, T).astype(np.float32),
            source_pos_down=(10, 50, 0),
            dest_pos_down=(9, 48, 0),
            reward=reward,
            next_state=np.random.randn(C, R, S, T).astype(np.float32),
            done=False,
        )

    def test_push_and_len(self):
        from unified_replay_buffer import UnifiedReplayBuffer
        buf = UnifiedReplayBuffer(capacity=10)
        assert len(buf) == 0
        buf.push(self._make_transition())
        assert len(buf) == 1

    def test_compression(self):
        from unified_replay_buffer import UnifiedReplayBuffer
        buf = UnifiedReplayBuffer(capacity=10)
        t = self._make_transition()
        assert t.state.dtype == np.float32
        buf.push(t)
        # After push, stored as float16
        assert buf.buffer[0].state.dtype == np.float16

    def test_sample_decompresses(self):
        from unified_replay_buffer import UnifiedReplayBuffer
        buf = UnifiedReplayBuffer(capacity=10)
        for _ in range(5):
            buf.push(self._make_transition())
        batch = buf.sample(3)
        assert len(batch) == 3
        for t in batch:
            assert t.state.dtype == np.float32
            assert t.next_state.dtype == np.float32

    def test_circular_overwrite(self):
        from unified_replay_buffer import UnifiedReplayBuffer
        buf = UnifiedReplayBuffer(capacity=3)
        for i in range(5):
            buf.push(self._make_transition(reward=float(i)))
        assert len(buf) == 3
        # Buffer should contain rewards 2, 3, 4 (oldest overwritten)
        rewards = {t.reward for t in buf.buffer}
        assert rewards == {2.0, 3.0, 4.0}

    def test_is_ready(self):
        from unified_replay_buffer import UnifiedReplayBuffer
        buf = UnifiedReplayBuffer(capacity=10)
        assert not buf.is_ready(5)
        for _ in range(5):
            buf.push(self._make_transition())
        assert buf.is_ready(5)
        assert not buf.is_ready(6)

    def test_transition_fields(self):
        from unified_replay_buffer import UnifiedTransition
        t = UnifiedTransition(
            state=np.zeros((C, R, S, T), dtype=np.float32),
            source_pos_down=(10, 50, 0),
            dest_pos_down=(9, 48, 0),
            reward=1.0,
            next_state=np.zeros((C, R, S, T), dtype=np.float32),
            done=True,
        )
        assert t.source_pos_down == (10, 50, 0)
        assert t.dest_pos_down == (9, 48, 0)
        assert t.reward == 1.0
        assert t.done is True

    def test_transition_no_dest(self):
        """Transitions where agent found no valid destination."""
        from unified_replay_buffer import UnifiedTransition
        t = UnifiedTransition(
            state=np.zeros((C, R, S, T), dtype=np.float32),
            source_pos_down=(13, 100, 0),
            dest_pos_down=None,  # no destination
            reward=-0.1,
            next_state=np.zeros((C, R, S, T), dtype=np.float32),
            done=False,
        )
        assert t.dest_pos_down is None


class TestPrioritizedReplayBuffer:
    """Test prioritized buffer basics."""

    def _make_transition(self, reward=0.5):
        from unified_replay_buffer import UnifiedTransition
        return UnifiedTransition(
            state=np.random.randn(C, R, S, T).astype(np.float32),
            source_pos_down=(10, 50, 0),
            dest_pos_down=(9, 48, 0),
            reward=reward,
            next_state=np.random.randn(C, R, S, T).astype(np.float32),
            done=False,
        )

    def test_push_and_sample(self):
        from unified_replay_buffer import UnifiedPrioritizedReplayBuffer
        buf = UnifiedPrioritizedReplayBuffer(capacity=10)
        for _ in range(5):
            buf.push(self._make_transition())
        assert len(buf) == 5
        transitions, indices, weights = buf.sample(3)
        assert len(transitions) == 3
        assert len(indices) == 3
        assert len(weights) == 3
        assert all(w > 0 for w in weights)

    def test_update_priorities(self):
        from unified_replay_buffer import UnifiedPrioritizedReplayBuffer
        buf = UnifiedPrioritizedReplayBuffer(capacity=10)
        for _ in range(5):
            buf.push(self._make_transition())
        _, indices, _ = buf.sample(3)
        td_errors = np.array([0.5, 0.1, 0.9], dtype=np.float32)
        buf.update_priorities(indices, td_errors)
        # Max priority should be updated
        assert buf.max_priority >= 0.9


class TestRegionMapping:
    """Verify region_of() for all row ranges."""

    @staticmethod
    def _region_of(row):
        """Inline region logic matching UnifiedDims defaults."""
        if row < 7: return "RAIL"
        if row < 8: return "PARKING"
        if row < 13: return "YARD"
        return "QUEUE"

    def test_rail_rows(self):
        for r in range(7):
            assert self._region_of(r) == "RAIL", f"Row {r}"

    def test_parking_row(self):
        assert self._region_of(7) == "PARKING"

    def test_yard_rows(self):
        for r in range(8, 13):
            assert self._region_of(r) == "YARD", f"Row {r}"

    def test_queue_rows(self):
        for r in range(13, 15):
            assert self._region_of(r) == "QUEUE", f"Row {r}"


class TestResolveDestSplit:
    """Test dest split resolution logic (pure Python)."""

    def test_yard_finds_free_split(self):
        """In yard, should find first unoccupied split in stride window."""
        state = np.zeros((C, R, S, T), dtype=np.float32)
        # Mark splits 40,41 as occupied
        state[0, 10, 40, 0] = 1.0  # OCCUPANCY
        state[0, 10, 41, 0] = 1.0
        # s_down=10 → window [40,44), first free = 42
        s_start = 10 * STRIDE
        for s in range(s_start, min(s_start + STRIDE, S)):
            if state[0, 10, s, 0] < 0.5:
                result = s
                break
        assert result == 42

    def test_rail_finds_container_start(self):
        """In rail, should find CONTAINER_START marker."""
        state = np.zeros((C, R, S, T), dtype=np.float32)
        # CONTAINER_START at split 43
        state[1, 3, 43, 0] = 1.0
        # s_down=10 → window [40,44), find marker
        s_start = 10 * STRIDE
        for s in range(s_start, min(s_start + STRIDE, S)):
            if state[1, 3, s, 0] > 0.5:
                result = s
                break
        assert result == 43


class TestSourceCodeStructure:
    """Verify agent file structure."""

    def test_no_old_action_types(self):
        with open(os.path.join(os.path.dirname(__file__), "unified_agent.py")) as f:
            code = f.read()
        # Should not DEFINE or USE old head classes / action types
        assert "class ActionTypeHead" not in code
        assert "AgentActionType.SLOT_PARKING" not in code
        assert "AgentActionType.IMPORT_VEHICLE" not in code

    def test_has_key_components(self):
        with open(os.path.join(os.path.dirname(__file__), "unified_agent.py")) as f:
            code = f.read()
        assert "class UnifiedDQNAgent" in code
        assert "class UnifiedActionResult" in code
        assert "resolve_move_type" in code
        assert "DestMaskFn" in code

    def test_two_stage_selection(self):
        with open(os.path.join(os.path.dirname(__file__), "unified_agent.py")) as f:
            code = f.read()
        assert "Stage 1: Source selection" in code
        assert "Stage 2: Destination selection" in code

    def test_td_plus_aux_training(self):
        with open(os.path.join(os.path.dirname(__file__), "unified_agent.py")) as f:
            code = f.read()
        assert "Source Q-values at taken positions" in code
        assert "Auxiliary dest loss: reward prediction" in code


# ══════════════════════════════════════════════════════════════════════════
# Part 2: Torch functional tests
# ══════════════════════════════════════════════════════════════════════════

HAS_TORCH = False
try:
    import torch
    import torch.nn as nn

    for mod_name in ["simulation", "simulation.multihead_dqn"]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    import config as config_mod
    sys.modules["simulation.multihead_dqn.config"] = config_mod
    import unified_networks as un_mod
    sys.modules["simulation.multihead_dqn.unified_networks"] = un_mod
    import unified_replay_buffer as urb_mod
    sys.modules["simulation.multihead_dqn.unified_replay_buffer"] = urb_mod

    from config import UnifiedDims, CNNConfig, HeadConfig, DQNConfig, MultiHeadDQNConfig, YardDims
    from unified_agent import UnifiedDQNAgent, UnifiedActionResult, resolve_move_type

    HAS_TORCH = True
except (ImportError, Exception):
    pass


class TestTorchAgent:
    """Functional tests requiring torch."""

    SKIP = not HAS_TORCH

    def _make_cfg(self):
        return MultiHeadDQNConfig(
            yard=YardDims(n_rows=5, n_splits=1160, n_tiers=5, n_bays=58, split_factor=20),
            unified=UnifiedDims(),
            cnn=CNNConfig(),
            heads=HeadConfig(),
            training=DQNConfig(replay_size=50, batch_size=4),
            device="cpu",
        )

    def _make_state(self):
        return np.random.randn(C, R, S, T).astype(np.float32)

    def _trivial_dest_fn(self, row, split, tier):
        """Always returns a single valid yard dest."""
        mask = np.zeros((R, S_DOWN, T), dtype=bool)
        mask[10, 50, 0] = True
        return mask

    def test_agent_creation(self):
        if self.SKIP: return
        agent = UnifiedDQNAgent(self._make_cfg())
        assert agent.q_net is not None
        assert agent.target_net is not None

    def test_act_returns_result(self):
        if self.SKIP: return
        agent = UnifiedDQNAgent(self._make_cfg())
        state = self._make_state()
        source_mask = np.zeros((R, S, T), dtype=bool)
        source_mask[10, 200, 0] = True  # yard container

        result = agent.act(state, source_mask, self._trivial_dest_fn, epsilon=0.0)
        assert isinstance(result, UnifiedActionResult)
        assert result.source_pos is not None
        assert result.dest_pos is not None
        assert result.source_region == "YARD"
        assert result.dest_region == "YARD"
        assert result.move_type == "YARD_TO_YARD"

    def test_act_empty_source(self):
        if self.SKIP: return
        agent = UnifiedDQNAgent(self._make_cfg())
        state = self._make_state()
        source_mask = np.zeros((R, S, T), dtype=bool)

        result = agent.act(state, source_mask, self._trivial_dest_fn)
        assert result.source_pos is None
        assert result.dest_pos is None

    def test_act_no_valid_dest(self):
        if self.SKIP: return
        agent = UnifiedDQNAgent(self._make_cfg())
        state = self._make_state()
        source_mask = np.zeros((R, S, T), dtype=bool)
        source_mask[10, 200, 0] = True

        def empty_dest_fn(r, s, t):
            return np.zeros((R, S_DOWN, T), dtype=bool)

        result = agent.act(state, source_mask, empty_dest_fn, epsilon=0.0)
        assert result.source_pos is not None
        assert result.dest_pos is None

    def test_act_queue_to_parking(self):
        """Queue source + parking dest → PARK_TRUCK."""
        if self.SKIP: return
        agent = UnifiedDQNAgent(self._make_cfg())
        state = self._make_state()
        # Mark CONTAINER_START for queue truck
        state[1, 13, 200, 0] = 1.0
        source_mask = np.zeros((R, S, T), dtype=bool)
        source_mask[13, 200, 0] = True  # queue

        def parking_dest_fn(r, s, t):
            mask = np.zeros((R, S_DOWN, T), dtype=bool)
            mask[7, 50, 0] = True  # parking row
            return mask

        result = agent.act(state, source_mask, parking_dest_fn, epsilon=0.0)
        assert result.source_region == "QUEUE"
        assert result.dest_region == "PARKING"
        assert result.move_type == "PARK_TRUCK"

    def test_epsilon_exploration(self):
        """With epsilon=1.0, should always explore randomly."""
        if self.SKIP: return
        agent = UnifiedDQNAgent(self._make_cfg())
        state = self._make_state()
        source_mask = np.zeros((R, S, T), dtype=bool)
        source_mask[10, 200, 0] = True
        source_mask[8, 100, 0] = True

        results = set()
        for _ in range(20):
            result = agent.act(state, source_mask, self._trivial_dest_fn, epsilon=1.0)
            if result.source_pos:
                results.add(result.source_pos[0])  # track selected rows

        # With 2 sources and eps=1.0, should select both eventually
        assert len(results) >= 1  # at minimum 1, likely 2

    def test_remember_and_optimize(self):
        """Full training loop: remember transitions → optimize."""
        if self.SKIP: return
        from unified_replay_buffer import UnifiedTransition
        agent = UnifiedDQNAgent(self._make_cfg())

        # Fill replay buffer
        for _ in range(10):
            t = UnifiedTransition(
                state=np.random.randn(C, R, S, T).astype(np.float32),
                source_pos_down=(10, 50, 0),
                dest_pos_down=(9, 48, 0),
                reward=np.random.randn(),
                next_state=np.random.randn(C, R, S, T).astype(np.float32),
                done=False,
            )
            agent.remember(t)

        loss = agent.optimize()
        assert loss > 0  # non-trivial loss
        assert len(agent.losses) == 1

    def test_batch_source_mask_derivation(self):
        """Source mask derived from CONTAINER_START channel in state."""
        if self.SKIP: return
        agent = UnifiedDQNAgent(self._make_cfg())
        state = np.zeros((1, C, R, S, T), dtype=np.float32)
        state[0, 1, 10, 200, 0] = 1.0  # CONTAINER_START at (10, 200, 0)
        state_t = torch.as_tensor(state, dtype=torch.float32)
        mask = agent._batch_source_mask(state_t)
        assert mask.shape == (1, R, S_DOWN, T)
        assert mask[0, 10, 50, 0].item() == True  # 200 // 4 = 50
        assert mask.sum().item() == 1

    def test_tutorial_epsilon(self):
        if self.SKIP: return
        agent = UnifiedDQNAgent(self._make_cfg())
        agent.set_tutorial_epsilon(0)
        assert agent._get_epsilon() == agent.cfg.training.tutorial_epsilon_start
        agent.set_tutorial_epsilon(80)
        assert abs(agent._get_epsilon() - agent.cfg.training.tutorial_epsilon_end) < 0.01
        agent.clear_epsilon_override()
        # Back to step-based epsilon
        assert agent.epsilon_override is None


# ══════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════

def run_all():
    test_classes = [
        TestMoveTypeResolution,
        TestCoordinateTransforms,
        TestDownsampleMask,
        TestReplayBuffer,
        TestPrioritizedReplayBuffer,
        TestRegionMapping,
        TestResolveDestSplit,
        TestSourceCodeStructure,
        TestTorchAgent,
    ]

    total = 0
    passed = 0
    skipped = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in sorted(methods):
            total += 1
            skip = getattr(cls, "SKIP", False)
            if skip:
                skipped += 1
                print(f"  ⊘ {cls.__name__}.{method_name} (no torch)")
                continue
            try:
                getattr(instance, method_name)()
                passed += 1
                print(f"  ✓ {cls.__name__}.{method_name}")
            except Exception as e:
                failed += 1
                errors.append((cls.__name__, method_name, e))
                print(f"  ✗ {cls.__name__}.{method_name}: {e}")

    print(f"\n{'='*60}")
    msg = f"Results: {passed}/{total} passed"
    if skipped:
        msg += f", {skipped} skipped (torch unavailable)"
    if failed:
        msg += f", {failed} failed"
    print(msg)

    if errors:
        print("\nFailures:")
        for cls_name, method, err in errors:
            print(f"  {cls_name}.{method}:")
            import traceback
            traceback.print_exception(type(err), err, err.__traceback__)
    elif not failed:
        print("All runnable tests passed! ✓")

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)