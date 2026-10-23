"""
PatchMixer: a patch-mixing architecture for time series forecasting.

Reference: Z. Gong, Y. Tang, J. Liang, "PatchMixer: A Patch-Mixing Architecture
for Long-Term Time Series Forecasting", 2023, https://arxiv.org/abs/2310.00655.

Channel independent: every variate is forecast separately (the variates are
folded into the batch), with shared weights. For each variate:
  1. RevIN normalizes the input window;
  2. the window is padded at the end (last value repeated `patch_stride` times)
     and cut into `patch_num` overlapping patches of `patch_len` values;
  3. every patch is embedded linearly into d_model values;
  4. two heads are summed: a linear head on the patch embeddings, and an MLP
     head on the output of the PatchMixer blocks;
  5. RevIN maps the forecast back to the original scale.

Shapes: B batch, L seq_len, S pred_len, N variates, P patch_num, E d_model.
"""

import torch
from torch import nn

from src.model.mamba.layers.RevIN import RevIN


class PatchMixerLayer(nn.Module):
    """
    One PatchMixer block: (B*N, P, E) -> (B*N, a, E).

    A depthwise convolution along the embedding of every patch (each patch is a
    channel), with residual connection, followed by a 1x1 convolution that
    mixes the patches.

    Args:
        dim: number of patches P (input channels).
        a: number of output channels (Backbone uses a = P).
        kernel_size: kernel of the depthwise convolution.
    """

    def __init__(self, dim, a, kernel_size=8):
        super().__init__()
        self.Resnet = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim, padding='same'),
            nn.GELU(),
            nn.BatchNorm1d(dim)
        )
        self.Conv_1x1 = nn.Sequential(
            nn.Conv1d(dim, a, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(a)
        )

    def forward(self, x):
        x = x + self.Resnet(x)                 # x: [batch * n_val, patch_num, d_model]
        x = self.Conv_1x1(x)                   # x: [batch * n_val, a, d_model]
        return x


class Model(nn.Module):
    """
    PatchMixer forecaster: (B, L, N) input window -> (B, S, N) forecast.

    Args:
        configs: namespace with `enc_in` (number of variates N), the window
            lengths `seq_len`, `pred_len` (from the data) and the architecture
            of cfg.model.params: `patch_len`, `patch_stride`,
            `mixer_kernel_size`, `d_model`, `e_layers`, `dropout`, `head_dropout`.
    """

    def __init__(self, configs):
        super().__init__()
        self.model = Backbone(configs)

    def forward(self, x, x_mark=None):
        """x_mark (covariates) is accepted for a common interface with S-Mamba, and ignored."""
        x = self.model(x)
        return x


class Backbone(nn.Module):
    """
    The PatchMixer network (see the module docstring).

    Args:
        configs: see Model.
        revin: apply RevIN to the input and the forecast.
        affine: learnable affine parameters of RevIN.
        subtract_last: RevIN subtracts the last value instead of the mean.
    """

    def __init__(self, configs, revin=True, affine=True, subtract_last=False):
        super().__init__()

        self.nvals = configs.enc_in
        self.lookback = configs.seq_len
        self.forecasting = configs.pred_len
        self.patch_size = configs.patch_len
        self.stride = configs.patch_stride
        self.kernel_size = configs.mixer_kernel_size

        self.PatchMixer_blocks = nn.ModuleList([])
        # the window is padded with `stride` copies of its last value, so the
        # last patch ends on the last observation
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num = int((self.lookback - self.patch_size) / self.stride + 1) + 1
        self.a = self.patch_num
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.head_dropout = configs.head_dropout
        self.depth = configs.e_layers
        for _ in range(self.depth):
            self.PatchMixer_blocks.append(PatchMixerLayer(dim=self.patch_num, a=self.a, kernel_size=self.kernel_size))
        # patch embedding: patch_len values -> d_model
        self.W_P = nn.Linear(self.patch_size, self.d_model)
        # linear head on the patch embeddings
        self.head0 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.patch_num * self.d_model, self.forecasting),
            nn.Dropout(self.head_dropout)
        )
        # MLP head on the output of the PatchMixer blocks
        self.head1 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.a * self.d_model, int(self.forecasting * 2)),
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.Linear(int(self.forecasting * 2), self.forecasting),
            nn.Dropout(self.head_dropout)
        )
        self.dropout = nn.Dropout(self.dropout)
        # RevIn
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(self.nvals, affine=affine, subtract_last=subtract_last)

    def forward(self, x):
        """(B, L, N) input window -> (B, S, N) forecast."""
        bs = x.shape[0]
        nvars = x.shape[-1]
        if self.revin:
            x = self.revin_layer(x, 'norm')
        x = x.permute(0, 2, 1)                                                       # x: [batch, n_val, seq_len]

        x_lookback = self.padding_patch_layer(x)
        x = x_lookback.unfold(dimension=-1, size=self.patch_size, step=self.stride)  # x: [batch, n_val, patch_num, patch_size]

        x = self.W_P(x)                                                              # x: [batch, n_val, patch_num, d_model]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))      # x: [batch * n_val, patch_num, d_model]
        x = self.dropout(x)
        u = self.head0(x)

        for PatchMixer_block in self.PatchMixer_blocks:
            x = PatchMixer_block(x)
        x = self.head1(x)
        x = u + x
        x = torch.reshape(x, (bs, nvars, -1))                                        # x: [batch, n_val, pred_len]
        x = x.permute(0, 2, 1)
        if self.revin:
            x = self.revin_layer(x, 'denorm')
        return x
