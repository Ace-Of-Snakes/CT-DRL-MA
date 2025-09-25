# simulation/rl/agents/ppo_agent.py
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from simulation.operations.terminal_manager import Move
from simulation.rl.policy.cnn_policy import CNN3DBackbone, MoveFeaturizer, MoveEncoder, PolicyHeads

@dataclass
# simulation/rl/agents/ppo_agent.py
@dataclass
class PPOConfig:
    in_channels: int = 21     # 11 base + 10 forecast channels (trains & trucks 3/6/12/24/48h)
    state_hidden: int = 128
    move_hidden: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    batch_size: int = 64
    epochs: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class TrajectoryBuffer:
    def __init__(self):
        self.data = []

    def add(self, state, move_feats, action_idx, logp, reward, value, done):
        self.data.append((state, move_feats, action_idx, logp, reward, value, done))

    def compute_advantages(self, gamma: float, lam: float):
        # GAE-lambda on trajectory (single episode for simplicity)
        adv, ret = [], []
        gae = 0.0
        returns = 0.0
        values = [v for (_,_,_,_,_,v,_) in self.data] + [0.0]
        rewards = [r for (_,_,_,_,r,_,_) in self.data]
        dones = [d for (_,_,_,_,_,_,d) in self.data]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + (0 if dones[t] else gamma * values[t+1]) - values[t]
            gae = delta + (0 if dones[t] else gamma * lam) * gae
            adv.insert(0, gae)
            returns = rewards[t] + (0 if dones[t] else gamma * (returns if t < len(rewards)-1 else values[t+1]))
            ret.insert(0, returns)
        return np.array(adv, dtype=np.float32), np.array(ret, dtype=np.float32)

    def clear(self):
        self.data.clear()

class PPOAgent:
    """
    CNN+PPO policy over the same dynamic move set:
    - logits per move from [z_s, z_a], value from z_s
    """
    def __init__(self, yard_dims: Tuple[int,int,int,int], cfg: PPOConfig = PPOConfig()):
        self.R, self.B, self.T, self.SF = yard_dims
        self.cfg = cfg

        self.backbone = CNN3DBackbone(cfg.in_channels, cfg.state_hidden).to(cfg.device)
        self.move_enc = MoveEncoder(MoveFeaturizer.feat_dim(), cfg.move_hidden).to(cfg.device)
        self.heads = PolicyHeads(cfg.state_hidden, cfg.move_hidden).to(cfg.device)
        self.optim = optim.Adam(list(self.backbone.parameters()) +
                                list(self.move_enc.parameters()) +
                                list(self.heads.parameters()),
                                lr=cfg.lr)
        self.buf = TrajectoryBuffer()

    def _to_tensor_state(self, state_np: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.transpose(state_np, (3,0,1,2))).float().unsqueeze(0)
        return x.to(self.cfg.device)

    def _to_tensor_moves(self, moves: List[Move]) -> torch.Tensor:
        feats = MoveFeaturizer.featurize(moves, (self.R, self.B, self.T, self.SF))
        return feats.to(self.cfg.device)

    def act(self, state_np: np.ndarray, moves: List[Move]) -> Tuple[int, float, float]:
        # returns (action_idx, logp, value)
        if not moves:
            return -1, 0.0, 0.0
        with torch.no_grad():
            s = self._to_tensor_state(state_np)
            mf = self._to_tensor_moves(moves)
            zs = self.backbone(s)
            za = self.move_enc(mf)
            logits = self.heads.logits(zs, za)   # [K]
            probs = torch.softmax(logits, dim=0)
            action = torch.multinomial(probs, 1).item()
            logp = torch.log(probs[action] + 1e-12).item()
            value = self.heads.value(zs).item()
            return action, logp, value

    def remember(self, state_np: np.ndarray, moves: List[Move], action_idx: int, logp: float,
                 reward: float, value: float, done: bool):
        mf = MoveFeaturizer.featurize(moves, (self.R, self.B, self.T, self.SF)).numpy()
        self.buf.add(state_np.copy(), mf.copy(), int(action_idx), float(logp), float(reward), float(value), bool(done))

    def update(self):
        if not self.buf.data:
            return 0.0
        adv, ret = self.buf.compute_advantages(self.cfg.gamma, self.cfg.lam)
        # normalize adv
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        loss_total = 0.0
        N = len(self.buf.data)
        idxs = np.arange(N)

        for _ in range(self.cfg.epochs):
            np.random.shuffle(idxs)
            for start in range(0, N, self.cfg.batch_size):
                batch_idx = idxs[start:start+self.cfg.batch_size]
                if len(batch_idx) == 0:
                    continue

                policy_loss = 0.0
                value_loss = 0.0
                entropy = 0.0

                self.optim.zero_grad(set_to_none=True)
                for j in batch_idx:
                    s_np, mf_np, a_idx, old_logp, r, v, d = self.buf.data[j]
                    s = self._to_tensor_state(s_np)
                    mf = torch.from_numpy(mf_np).float().to(self.cfg.device)

                    zs = self.backbone(s)
                    za = self.move_enc(mf)
                    logits = self.heads.logits(zs, za)  # [K]
                    probs = torch.softmax(logits, dim=0)
                    dist_logp = torch.log(probs[a_idx] + 1e-12)
                    ratio = torch.exp(dist_logp - torch.tensor(old_logp, device=self.cfg.device))
                    a = torch.tensor(adv[j], device=self.cfg.device)
                    surr1 = ratio * a
                    surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * a
                    policy_loss += -torch.min(surr1, surr2)

                    v_pred = self.heads.value(zs).squeeze(0)
                    ret_t = torch.tensor(ret[j], dtype=torch.float32, device=self.cfg.device)
                    value_loss += 0.5 * (ret_t - v_pred).pow(2)

                    entropy += -(probs * torch.log(probs + 1e-12)).sum()

                loss = (policy_loss / len(batch_idx) +
                        self.cfg.value_coef * (value_loss / len(batch_idx)) -
                        self.cfg.entropy_coef * (entropy / len(batch_idx)))
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.backbone.parameters()) +
                                         list(self.move_enc.parameters()) +
                                         list(self.heads.parameters()), 1.0)
                self.optim.step()
                loss_total += float(loss.item())

        self.buf.clear()
        return loss_total / max(1, (N / self.cfg.batch_size))