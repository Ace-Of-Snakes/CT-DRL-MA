# simulation/rl/policy/cnn_policy.py
import torch
import torch.nn as nn
from typing import List, Tuple
from simulation.operations.terminal_manager import Move
from simulation.core.facilities.yard import PlacementResult

MOVE_TYPES = [
    "YARD_TO_YARD",
    "TRAIN_TO_YARD",
    "YARD_TO_TRAIN",
    "TRUCK_TO_YARD",
    "YARD_TO_TRUCK",
    "TRAIN_TO_TRUCK",
    "TRUCK_TO_TRAIN",
    "YARD_TO_TERMINAL_TRUCK",
    "SLOT_TRUCK_PARKING",   
]
TYPE_TO_IDX = {t: i for i, t in enumerate(MOVE_TYPES)}

class CNN3DBackbone(nn.Module):
    """
    3D CNN over [C, R, B, T] (we use Conv3d treating Tiers as Depth).
    Output: fixed-dim state embedding.
    """
    def __init__(self, in_channels: int, hidden: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.proj = nn.Linear(64, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, R, BAYS, T]
        h = self.conv(x).flatten(1)
        return torch.tanh(self.proj(h))  # [B, hidden]


class MoveFeaturizer:
    """
    Cheap features from Move:
    - type one-hot (len(MOVE_TYPES))
    - placement row/bay/tier/start_split normalized (if present), else zeros
    - parking_delta (SLOT_TRUCK_PARKING only): {-1,0,+1} normalized, else 0
    Total dim: len(MOVE_TYPES) + 5
    """
    @staticmethod
    def feat_dim() -> int:
        return len(MOVE_TYPES) + 5

    @staticmethod
    def featurize(moves: List[Move],
                  yard_dims: Tuple[int, int, int, int]) -> torch.Tensor:
        n_rows, n_bays, n_tiers, split_factor = yard_dims
        out = []
        for mv in moves:
            f = [0.0] * len(MOVE_TYPES)
            f[TYPE_TO_IDX.get(mv.type, 0)] = 1.0

            row = bay = tier = start = 0.0
            parking_delta = 0.0

            pl: PlacementResult = mv.args.get("placement", None)
            if pl is not None:
                row = pl.row / max(1, n_rows - 1)
                bay = pl.bay / max(1, n_bays - 1)
                tier = pl.tier / max(1, n_tiers - 1)
                start = pl.start_split / max(1, split_factor - 1)
            elif mv.type == "SLOT_TRUCK_PARKING":
                spot = mv.args.get("spot", "")
                try:
                    parts = spot.split("_")
                    if len(parts) >= 3:
                        b, s = parts[-2], parts[-1]
                        bay = int(b) / max(1, n_bays - 1)
                        start = int(s) / max(1, split_factor - 1)
                except Exception:
                    pass
                try:
                    d = int(mv.args.get("delta_bay", 0))
                    if d < -1: d = -1
                    if d > +1: d = +1
                    parking_delta = float(d)
                except Exception:
                    parking_delta = 0.0

            out.append(f + [row, bay, tier, start, parking_delta])
        if not out:
            return torch.zeros((0, MoveFeaturizer.feat_dim()), dtype=torch.float32)
        return torch.tensor(out, dtype=torch.float32)


class MoveEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [K, F]
        return self.net(feats)  # [K, H]


class QScorer(nn.Module):
    """
    Scores moves with state embedding: Q(s, a) = f([z_s, z_a]).
    """
    def __init__(self, state_dim: int, move_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + move_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, state_emb: torch.Tensor, move_emb: torch.Tensor) -> torch.Tensor:
        # state_emb: [B, Hs]; move_emb: [K, Hm] (per-sample K varies; we call per-sample)
        Hs = state_emb.shape[-1]
        B = state_emb.shape[0]
        assert B == 1, "Call per-sample; batch mode requires padding."
        st = state_emb[0].expand(move_emb.size(0), Hs)
        q = self.net(torch.cat([st, move_emb], dim=-1)).squeeze(-1)  # [K]
        return q


class PolicyHeads(nn.Module):
    """
    PPO heads:
    - logits for Categorical over moves: w([z_s, z_a])
    - value head V(s): from z_s
    """
    def __init__(self, state_dim: int, move_dim: int, hidden: int = 128):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(state_dim + move_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        self.v = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def logits(self, state_emb: torch.Tensor, move_emb: torch.Tensor) -> torch.Tensor:
        Hs = state_emb.shape[-1]
        st = state_emb[0].expand(move_emb.size(0), Hs)
        return self.pi(torch.cat([st, move_emb], dim=-1)).squeeze(-1)  # [K]

    def value(self, state_emb: torch.Tensor) -> torch.Tensor:
        return self.v(state_emb).squeeze(-1)  # [B] or [1]