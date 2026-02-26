# simulation/rl/variants/networks/noisy_linear.py
"""Factorized NoisyLinear layer (Fortunato et al., 2018).

Replaces nn.Linear with a layer that adds learned parametric noise
to weights and biases, enabling state-dependent exploration without
epsilon-greedy.

Supports an external ``noise_scale`` multiplier (default 1.0) that
uniformly attenuates the noise contribution.  Set to 0.0 for a fully
deterministic (greedy) forward pass.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Linear layer with factorized Gaussian noise.

    The ``noise_scale`` attribute (default 1.0) multiplies the noise
    contribution at forward time.  External callers (e.g. the agent's
    ``set_noise_scale``) can anneal it from 1.0 → 0.0 over training to
    smoothly transition from exploration to exploitation.
    """

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma0 = sigma0
        self.noise_scale: float = 1.0

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma0 / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.sigma0 / math.sqrt(self.in_features))

    def reset_noise(self):
        device = self.weight_epsilon.device
        eps_in = self._scale_noise(self.in_features, device)
        eps_out = self._scale_noise(self.out_features, device)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    @staticmethod
    def _scale_noise(size: int, device: torch.device = None) -> torch.Tensor:
        x = torch.randn(size, device=device)
        return x.sign() * x.abs().sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ns = self.noise_scale
        weight = self.weight_mu + ns * self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + ns * self.bias_sigma * self.bias_epsilon
        return F.linear(x, weight, bias)
