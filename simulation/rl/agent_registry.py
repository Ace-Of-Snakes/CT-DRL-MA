# simulation/rl/agent_registry.py
"""Registry mapping DQN algorithm names to agent classes.

The registry is *algorithm-only*: CNN backbone selection is orthogonal and
handled via the ``backbone_variant`` / ``spectral_norm`` arguments of
:func:`create_agent`.

Usage:
    from simulation.rl.agent_registry import create_agent

    # Baseline DQN with default (baseline) backbone
    agent = create_agent("baseline", cfg)

    # Munchausen DQN with wider backbone + spectral norm
    agent = create_agent("munchausen", cfg,
                         backbone_variant="wider", spectral_norm=True)
"""
from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import Dict, Type

from simulation.rl.base_agent import BaseSpatialDQNAgent


# ── Public constants ──────────────────────────────────────────────────

DQN_ALGORITHMS = [
    "baseline",
    "munchausen",
    "spectral_norm",
    "noisynet",
    "dueling",
    "qrdqn",
    "iqn",
    "kitchen_sink",
    "gru",
]


def _lazy_registry() -> Dict[str, Type[BaseSpatialDQNAgent]]:
    """Build registry on first access to avoid circular imports."""
    from simulation.rl.multihead_dqn.agent import DQNAgent
    from simulation.rl.variants import (
        MunchausenDQNAgent,
        SpectralNormDQNAgent,
        NoisyNetDQNAgent,
        DuelingDQNAgent,
        QRDQNAgent,
        IQNAgent,
        KitchenSinkDQNAgent,
        GRUDQNAgent,
    )
    return {
        "baseline": DQNAgent,
        "munchausen": MunchausenDQNAgent,
        "spectral_norm": SpectralNormDQNAgent,
        "noisynet": NoisyNetDQNAgent,
        "dueling": DuelingDQNAgent,
        "qrdqn": QRDQNAgent,
        "iqn": IQNAgent,
        "kitchen_sink": KitchenSinkDQNAgent,
        "gru": GRUDQNAgent,
    }


class _RegistryProxy:
    """Dict-like lazy proxy so imports happen only when accessed."""

    def __init__(self):
        self._reg = None

    def _ensure(self):
        if self._reg is None:
            self._reg = _lazy_registry()

    def __getitem__(self, key: str) -> Type[BaseSpatialDQNAgent]:
        self._ensure()
        return self._reg[key]

    def __contains__(self, key: str) -> bool:
        self._ensure()
        return key in self._reg

    def keys(self):
        self._ensure()
        return self._reg.keys()

    def values(self):
        self._ensure()
        return self._reg.values()

    def items(self):
        self._ensure()
        return self._reg.items()

    def __iter__(self):
        self._ensure()
        return iter(self._reg)

    def __len__(self):
        self._ensure()
        return len(self._reg)

    def __repr__(self):
        self._ensure()
        return repr(self._reg)


AGENT_REGISTRY = _RegistryProxy()


# ── Factory function ──────────────────────────────────────────────────

def create_agent(
    variant: str,
    cfg,
    backbone_variant: str = "baseline",
    spectral_norm: bool = False,
) -> BaseSpatialDQNAgent:
    """Create an agent by DQN algorithm name, optionally with a non-default backbone.

    The function first adjusts channel widths in ``cfg.cnn`` if needed
    (e.g. *wider* doubles channels), builds the agent normally, then
    swaps in the requested backbone architecture.

    Args:
        variant: DQN algorithm — one of 'baseline', 'munchausen',
                 'spectral_norm', 'noisynet', 'dueling', 'qrdqn', 'iqn',
                 'kitchen_sink', 'gru'.
        cfg: MultiHeadDQNConfig instance.
        backbone_variant: CNN backbone architecture — one of 'baseline',
                          'wider', 'deeper', 'residual', 'narrow_deep'.
                          Default ``'baseline'`` uses the standard 4-layer
                          FactoredCNNBackbone.
        spectral_norm: If True, apply spectral normalization to all
                       Conv3d layers in the backbone.  This is *additive*
                       with the ``spectral_norm`` DQN variant (which
                       already applies SN via its own ``_build_networks``).

    Returns:
        Instantiated agent ready for training.
    """
    if variant not in AGENT_REGISTRY:
        available = ", ".join(sorted(AGENT_REGISTRY.keys()))
        raise ValueError(f"Unknown variant {variant!r}. Available: {available}")

    # 1) Adjust CNN config for backbone variants that change channel widths
    if backbone_variant != "baseline":
        from simulation.rl.backbone_factory import get_cnn_config
        cfg = dc_replace(cfg, cnn=get_cnn_config(backbone_variant, cfg.cnn))

    # 2) Build agent — calls _build_networks() with the (possibly adjusted) cfg
    agent = AGENT_REGISTRY[variant](cfg)

    # 3) If a non-default backbone was requested, swap it into both networks.
    #    The spectral_norm DQN variant already applies SN in _build_networks,
    #    so we only need to swap when the backbone *architecture* changes.
    needs_swap = backbone_variant != "baseline"
    # Also swap if the user explicitly asked for SN and the agent doesn't
    # already apply it (i.e. variant != "spectral_norm")
    if spectral_norm and variant != "spectral_norm":
        needs_swap = True

    if needs_swap:
        import torch
        from simulation.rl.backbone_factory import build_backbone, build_pool

        # For spectral_norm DQN variant + non-baseline backbone, SN is needed
        apply_sn = spectral_norm or (variant == "spectral_norm")

        def _make_bb():
            return build_backbone(
                cfg.cnn, variant=backbone_variant, spectral_norm=apply_sn,
                dims=cfg.unified,
            )

        # Swap backbone modules in both Q-net and target-net
        agent.q_net.backbone = _make_bb()
        agent.target_net.backbone = _make_bb()

        # Swap pool if the backbone variant provides a custom one
        pool = build_pool(cfg.cnn, variant=backbone_variant, dims=cfg.unified)
        if pool is not None:
            agent.q_net.pool = pool
            agent.target_net.pool = build_pool(
                cfg.cnn, variant=backbone_variant, dims=cfg.unified,
            )

        # Move everything to the correct device
        agent.q_net.to(agent.device)
        agent.target_net.to(agent.device)

        # Sync target network weights
        agent.target_net.load_state_dict(agent.q_net.state_dict())

        # Rebuild optimizer so it tracks the new backbone parameters
        agent.optimizer = torch.optim.Adam(
            agent.q_net.parameters(), lr=cfg.training.lr,
        )

    return agent
