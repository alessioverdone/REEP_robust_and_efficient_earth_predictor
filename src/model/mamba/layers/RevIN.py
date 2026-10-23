"""
Reversible Instance Normalization (RevIN).

Reference: T. Kim et al., "Reversible Instance Normalization for Accurate
Time-Series Forecasting against Distribution Shift", ICLR 2022.
"""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Normalizes every series of the batch with the statistics of its own input
    window ('norm'), and maps the forecast back to the original scale with the
    same statistics ('denorm'). The statistics computed by 'norm' are stored in
    the module and used by the following 'denorm'.

    Args:
        num_features: number of variates N (channels of the last axis).
        eps: added to the variance for numerical stability.
        affine: if True, a learnable per-variate scale and bias are applied
            after the normalization (and removed before the de-normalization).
        subtract_last: if True, the last value of the window is subtracted
            instead of its mean.
    """

    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        """
        Args:
            x: (B, ..., N) tensor: the input window for 'norm', the forecast for 'denorm'.
            mode: 'norm' or 'denorm'.
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        # statistics over every axis except the batch and the variates
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
