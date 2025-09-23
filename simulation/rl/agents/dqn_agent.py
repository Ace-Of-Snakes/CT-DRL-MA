# simulation/rl/agents/dqn_agent.py
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

from simulation.terminal_components.systems.TerminalManager import Move
from simulation.terminal_components.systems.StateEncoder import TerminalStateEncoder
from simulation.rl.policy.cnn_policy import CNN3DBackbone, MoveFeaturizer, MoveEncoder, QScorer

@dataclass
# simulation/rl/agents/dqn_agent.py
@dataclass
class DQNConfig:
    in_channels: int = 21     # 11 base + 10 forecast channels (3,6,12,24,48 for trains and trucks)
    state_hidden: int = 128
    move_hidden: int = 128
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 32
    replay_size: int = 50000
    target_tau: float = 0.005
    epsilon_start: float = 0.2
    epsilon_end: float = 0.02
    epsilon_decay_steps: int = 50000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.ptr = 0
        self.full = False
        self.data = []

    def push(self,
             state: np.ndarray,
             move_feats: np.ndarray,
             action_idx: int,
             reward: float,
             next_state: np.ndarray,
             next_move_feats: np.ndarray,
             done: bool):
        item = (state, move_feats, action_idx, reward, next_state, next_move_feats, done)
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.ptr] = item
            self.full = True
        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size: int):
        n = len(self.data)
        idxs = np.random.choice(n, size=min(batch_size, n), replace=False)
        batch = [self.data[i] for i in idxs]
        return batch

    def __len__(self):
        return len(self.data)

class DQNAgent:
    """
    CNN+DQN agent:
    - State: 4D yard tensor [R, B, T, C] (converted to [1, C, R, B, T])
    - Moves: dynamic list; featurized and scored against state embedding.
    - Training: vanilla DQN with soft target updates, Huber loss.
    """
    def __init__(self, yard_dims: Tuple[int, int, int, int], cfg: DQNConfig = DQNConfig()):
        self.R, self.B, self.T, self.SF = yard_dims
        self.cfg = cfg

        self.backbone = CNN3DBackbone(cfg.in_channels, cfg.state_hidden).to(cfg.device)
        self.move_enc = MoveEncoder(MoveFeaturizer.feat_dim(), cfg.move_hidden).to(cfg.device)
        self.q_head = QScorer(cfg.state_hidden, cfg.move_hidden).to(cfg.device)

        self.tgt_backbone = CNN3DBackbone(cfg.in_channels, cfg.state_hidden).to(cfg.device)
        self.tgt_move_enc = MoveEncoder(MoveFeaturizer.feat_dim(), cfg.move_hidden).to(cfg.device)
        self.tgt_q_head = QScorer(cfg.state_hidden, cfg.move_hidden).to(cfg.device)
        self._hard_update_targets()

        self.optim = optim.Adam(
            list(self.backbone.parameters()) +
            list(self.move_enc.parameters()) +
            list(self.q_head.parameters()),
            lr=cfg.lr
        )
        self.replay = ReplayBuffer(cfg.replay_size)
        self.step_count = 0

    def _hard_update_targets(self):
        self.tgt_backbone.load_state_dict(self.backbone.state_dict())
        self.tgt_move_enc.load_state_dict(self.move_enc.state_dict())
        self.tgt_q_head.load_state_dict(self.q_head.state_dict())

    def _soft_update_targets(self):
        tau = self.cfg.target_tau
        for tgt, src in [
            (self.tgt_backbone, self.backbone),
            (self.tgt_move_enc, self.move_enc),
            (self.tgt_q_head, self.q_head),
        ]:
            for tp, sp in zip(tgt.parameters(), src.parameters()):
                tp.data.copy_(tp.data * (1 - tau) + sp.data * tau)

    def _to_tensor_state(self, state_np: np.ndarray) -> torch.Tensor:
        # state_np: [R, B, T, C] -> torch [1, C, R, B, T]
        x = torch.from_numpy(np.transpose(state_np, (3, 0, 1, 2))).float().unsqueeze(0)
        return x.to(self.cfg.device)

    def _to_tensor_moves(self, moves: List[Move]) -> torch.Tensor:
        feats = MoveFeaturizer.featurize(moves, (self.R, self.B, self.T, self.SF))
        return feats.to(self.cfg.device)

    def act(self, state_np: np.ndarray, moves: List[Move], epsilon: Optional[float] = None) -> int:
        if not moves:
            return -1
        self.step_count += 1
        if epsilon is None:
            # linear decay
            eps = self.cfg.epsilon_end + max(0.0, (self.cfg.epsilon_start - self.cfg.epsilon_end) *
                                             (1 - min(1.0, self.step_count / self.cfg.epsilon_decay_steps)))
        else:
            eps = epsilon
        if random.random() < eps:
            return random.randrange(len(moves))
        with torch.no_grad():
            s = self._to_tensor_state(state_np)
            mf = self._to_tensor_moves(moves)
            zs = self.backbone(s)            # [1, Hs]
            za = self.move_enc(mf)           # [K, Hm]
            q = self.q_head(zs, za)          # [K]
            return int(torch.argmax(q).item())

    def remember(self,
                 state_np: np.ndarray, moves: List[Move], action_idx: int, reward: float,
                 next_state_np: np.ndarray, next_moves: List[Move], done: bool):
        # store featurized moves to keep buffer compact
        mf = MoveFeaturizer.featurize(moves, (self.R, self.B, self.T, self.SF)).numpy()
        nmf = MoveFeaturizer.featurize(next_moves, (self.R, self.B, self.T, self.SF)).numpy()
        self.replay.push(state_np.copy(), mf.copy(), action_idx, float(reward),
                         next_state_np.copy(), nmf.copy(), bool(done))

    def optimize(self) -> float:
        if len(self.replay) < self.cfg.batch_size:
            return 0.0
        batch = self.replay.sample(self.cfg.batch_size)
        loss_total = 0.0
        count = 0

        for (s_np, mf_np, a_idx, r, ns_np, nmf_np, d) in batch:
            if mf_np.shape[0] == 0:
                continue
            s = self._to_tensor_state(s_np)
            mf = torch.from_numpy(mf_np).float().to(self.cfg.device)
            ns = self._to_tensor_state(ns_np)
            nmf = torch.from_numpy(nmf_np).float().to(self.cfg.device)

            # Current Q(s, a)
            zs = self.backbone(s)
            za = self.move_enc(mf)
            q_all = self.q_head(zs, za)  # [K]
            if a_idx < 0 or a_idx >= q_all.size(0):
                # skip invalid index
                continue
            q_sa = q_all[a_idx]

            # Target: r + gamma * max_a' Q'(s', a')
            with torch.no_grad():
                zsn = self.tgt_backbone(ns)
                zan = self.tgt_move_enc(nmf) if nmf.size(0) > 0 else None
                if zan is not None and zan.size(0) > 0:
                    qn_all = self.tgt_q_head(zsn, zan)  # [K']
                    qn_max = torch.max(qn_all)
                else:
                    qn_max = torch.tensor(0.0, device=self.cfg.device)
                target = r + (0.0 if d else self.cfg.gamma * qn_max.item())

            target_t = torch.tensor(target, dtype=torch.float32, device=self.cfg.device)
            loss = nn.SmoothL1Loss()(q_sa, target_t)
            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(self.backbone.parameters()) +
                                     list(self.move_enc.parameters()) +
                                     list(self.q_head.parameters()), 1.0)
            self.optim.step()
            self._soft_update_targets()
            loss_total += float(loss.item())
            count += 1

        return loss_total / max(1, count)

    def save(self, path: str):
        torch.save({
            "cfg": self.cfg.__dict__,
            "backbone": self.backbone.state_dict(),
            "move_enc": self.move_enc.state_dict(),
            "q_head": self.q_head.state_dict(),
            "tgt_backbone": self.tgt_backbone.state_dict(),
            "tgt_move_enc": self.tgt_move_enc.state_dict(),
            "tgt_q_head": self.tgt_q_head.state_dict(),
            "step_count": self.step_count
        }, path)

    def load(self, path: str, map_location: Optional[str] = None):
        ckpt = torch.load(path, map_location=map_location or self.cfg.device)
        self.backbone.load_state_dict(ckpt["backbone"])
        self.move_enc.load_state_dict(ckpt["move_enc"])
        self.q_head.load_state_dict(ckpt["q_head"])
        self.tgt_backbone.load_state_dict(ckpt["tgt_backbone"])
        self.tgt_move_enc.load_state_dict(ckpt["tgt_move_enc"])
        self.tgt_q_head.load_state_dict(ckpt["tgt_q_head"])
        self.step_count = ckpt.get("step_count", 0)