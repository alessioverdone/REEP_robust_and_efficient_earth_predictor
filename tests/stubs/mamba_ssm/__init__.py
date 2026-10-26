"""
CPU stand-in for mamba_ssm, used ONLY by the tests when the real package (CUDA
kernels) cannot run: tests/conftest.py puts tests/stubs on the path in that
case. Never on the path of a real run.

Mamba here is a small causal, order-dependent sequence model with the same
constructor and the same (B, L, D) -> (B, L, D) contract as mamba_ssm.Mamba:
token t sees the running mean of tokens 0..t, followed by a linear map. It is
NOT a state space model: it only lets the code around the Mamba blocks run.
"""

import torch
import torch.nn as nn


class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, **kwargs):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        steps = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        return self.proj(torch.cumsum(x, dim=1) / steps)
