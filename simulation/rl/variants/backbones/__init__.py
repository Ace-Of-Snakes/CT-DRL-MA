# simulation/rl/variants/backbones/__init__.py
"""CNN backbone variants — drop-in replacements for FactoredCNNBackbone."""

from simulation.rl.variants.backbones.wider_backbone import WiderCNNBackbone
from simulation.rl.variants.backbones.deeper_backbone import DeeperCNNBackbone
from simulation.rl.variants.backbones.residual_backbone import ResidualCNNBackbone
from simulation.rl.variants.backbones.narrow_deep_backbone import NarrowDeepCNNBackbone

__all__ = [
    "WiderCNNBackbone",
    "DeeperCNNBackbone",
    "ResidualCNNBackbone",
    "NarrowDeepCNNBackbone",
]
