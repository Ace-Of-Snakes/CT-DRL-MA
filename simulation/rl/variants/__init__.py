# simulation/rl/variants/__init__.py
"""DQN variant agents — drop-in replacements for DQNAgent.

Each agent uses the standard QNetwork with swappable heads
and/or backbone.  CNN backbone selection is handled orthogonally
via :func:`simulation.rl.agent_registry.create_agent`.
"""

from simulation.rl.variants.munchausen_agent import MunchausenDQNAgent
from simulation.rl.variants.spectralnorm_agent import SpectralNormDQNAgent
from simulation.rl.variants.noisynet_agent import NoisyNetDQNAgent
from simulation.rl.variants.dueling_agent import DuelingDQNAgent
from simulation.rl.variants.qrdqn_agent import QRDQNAgent
from simulation.rl.variants.iqn_agent import IQNAgent
from simulation.rl.variants.kitchen_sink_agent import KitchenSinkDQNAgent

__all__ = [
    "MunchausenDQNAgent",
    "SpectralNormDQNAgent",
    "NoisyNetDQNAgent",
    "DuelingDQNAgent",
    "QRDQNAgent",
    "IQNAgent",
    "KitchenSinkDQNAgent",
]
