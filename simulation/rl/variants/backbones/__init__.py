# simulation/rl/variants/backbones/__init__.py
"""CNN backbone variants — drop-in replacements for FactoredCNNBackbone."""

from simulation.rl.variants.backbones.wider_backbone import WiderCNNBackbone
from simulation.rl.variants.backbones.deeper_backbone import DeeperCNNBackbone
from simulation.rl.variants.backbones.residual_backbone import ResidualCNNBackbone
from simulation.rl.variants.backbones.narrow_deep_backbone import NarrowDeepCNNBackbone
from simulation.rl.variants.backbones.kitchen_sink_backbone import KitchenSinkCNNBackbone
from simulation.rl.variants.backbones.region_aware_backbone import (
    RegionAwareCNNBackbone,
    RegionAwarePooling,
)

__all__ = [
    "WiderCNNBackbone",
    "DeeperCNNBackbone",
    "ResidualCNNBackbone",
    "NarrowDeepCNNBackbone",
    "KitchenSinkCNNBackbone",
    "RegionAwareCNNBackbone",
    "RegionAwarePooling",
]
