# simulation/rl/backbone_factory.py
"""CNN Backbone Factory — builds any backbone variant from config + flags.

This is the single entry point for constructing CNN backbones. It cleanly
separates the CNN feature extractor from the DQN decision-making heads,
enabling any CNN × DQN combination.

Usage:
    from simulation.rl.backbone_factory import build_backbone, get_cnn_config

    # Build a wider backbone with spectral normalization
    cnn_cfg = get_cnn_config("wider", base_cnn_cfg)
    backbone = build_backbone(cnn_cfg, variant="wider", spectral_norm=True)

    # Pass to UnifiedQNetwork
    q_net = UnifiedQNetwork(dims, cnn_cfg, head_cfg, backbone=backbone)
"""
from __future__ import annotations

from dataclasses import replace

import torch.nn as nn
from torch.nn.utils import spectral_norm

from simulation.rl.multihead_dqn.config import CNNConfig


# ── Backbone class registry (lazy to avoid circular imports) ────────────

BACKBONE_VARIANTS = ["baseline", "wider", "deeper", "residual", "narrow_deep", "kitchen_sink"]

_BACKBONE_CLASSES = None


def _get_backbone_classes():
    """Build the backbone class mapping on first access."""
    global _BACKBONE_CLASSES
    if _BACKBONE_CLASSES is None:
        from simulation.rl.multihead_dqn.unified_networks import FactoredCNNBackbone
        from simulation.rl.variants.backbones.wider_backbone import WiderCNNBackbone
        from simulation.rl.variants.backbones.deeper_backbone import DeeperCNNBackbone
        from simulation.rl.variants.backbones.residual_backbone import ResidualCNNBackbone
        from simulation.rl.variants.backbones.narrow_deep_backbone import NarrowDeepCNNBackbone
        from simulation.rl.variants.backbones.kitchen_sink_backbone import KitchenSinkCNNBackbone

        _BACKBONE_CLASSES = {
            "baseline": FactoredCNNBackbone,
            "wider": WiderCNNBackbone,
            "deeper": DeeperCNNBackbone,
            "residual": ResidualCNNBackbone,
            "narrow_deep": NarrowDeepCNNBackbone,
            "kitchen_sink": KitchenSinkCNNBackbone,
        }
    return _BACKBONE_CLASSES


# ── Config adjustments ───────────────────────────────────────────────────

def get_cnn_config(variant: str, base_cfg: CNNConfig) -> CNNConfig:
    """Return a CNNConfig with channel widths adjusted for the given variant.

    Wider and narrow_deep change channel counts; all others use base config.

    Args:
        variant: One of 'baseline', 'wider', 'deeper', 'residual', 'narrow_deep'.
        base_cfg: The default CNNConfig.

    Returns:
        CNNConfig (possibly modified copy).
    """
    if variant == "wider":
        return replace(base_cfg, stage1_channels=64, feat_channels=128, global_dim=256)
    elif variant == "narrow_deep":
        return replace(base_cfg, stage1_channels=16, feat_channels=32, global_dim=64)
    else:
        return base_cfg


# ── Spectral norm application ────────────────────────────────────────────

def _apply_spectral_norm(backbone: nn.Module) -> nn.Module:
    """Apply spectral normalization to all Conv3d layers in-place.

    Constrains the Lipschitz constant of each convolution, improving
    training stability during long curriculum runs.
    """
    for name, module in list(backbone.named_modules()):
        if isinstance(module, nn.Conv3d):
            # Navigate to parent and replace the Conv3d with spectral-normed version
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = dict(backbone.named_modules())[parts[0]]
                attr = parts[1]
            else:
                parent = backbone
                attr = parts[0]
            setattr(parent, attr, spectral_norm(module))
    return backbone


# ── Main factory function ────────────────────────────────────────────────

def build_backbone(
    cnn_cfg: CNNConfig,
    variant: str = "baseline",
    spectral_norm: bool = False,
) -> nn.Module:
    """Build a CNN backbone from config and variant flags.

    Args:
        cnn_cfg: CNNConfig (use get_cnn_config() to adjust channels first).
        variant: Architecture variant — one of:
            'baseline'      — 4-layer factored CNN (default)
            'wider'         — same 4 layers, 2× channel width
            'deeper'        — 5th conv layer (2nd cross-region pass)
            'residual'      — skip connections around conv2+conv3
            'narrow_deep'   — half channels, 6 layers
            'kitchen_sink'  — deeper + residual (5 layers with skips)
        spectral_norm: If True, wrap all Conv3d layers with spectral norm.

    Returns:
        nn.Module with forward: (B, C_in, R, S, T) → (B, feat_ch, R, S_down, T)
    """
    classes = _get_backbone_classes()
    if variant not in classes:
        available = ", ".join(sorted(classes.keys()))
        raise ValueError(f"Unknown backbone variant {variant!r}. Available: {available}")

    backbone = classes[variant](cnn_cfg)

    if spectral_norm:
        backbone = _apply_spectral_norm(backbone)

    return backbone
