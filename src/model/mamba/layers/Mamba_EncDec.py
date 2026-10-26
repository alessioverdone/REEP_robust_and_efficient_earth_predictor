"""
Encoder of S-Mamba (src/model/S_Mamba.py): a stack of layers, each made of a
bidirectional Mamba over the variate tokens followed by a position-wise
feed-forward block, both with residual connection and LayerNorm.

Shapes: B batch, N tokens (variates + covariates), E d_model.
"""

import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    """
    One S-Mamba layer: (B, N, E) -> (B, N, E).

    A Mamba block is causal: the output of a token depends only on the tokens
    before it. Two blocks are therefore combined: `mamba` scans the tokens in
    order, `mamba_r` scans them reversed, and their outputs are summed, so every
    token sees all the others.

    Args:
        mamba: Mamba block for the forward scan, (B, N, E) -> (B, N, E).
        mamba_r: Mamba block for the backward scan, same signature.
        d_model: token dimension E.
        d_ff: hidden dimension of the feed-forward block (default 4 * d_model).
        dropout: dropout of the feed-forward block.
        activation: "relu" or "gelu" (any other value means gelu).
    """

    def __init__(self, mamba, mamba_r, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.mamba = mamba
        self.mamba_r = mamba_r
        # Position-wise feed-forward as two 1x1 convolutions over the token dimension
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        # Bidirectional scan: forward + (flip -> scan -> flip back)
        new_x = self.mamba(x) + self.mamba_r(x.flip(dims=[1])).flip(dims=[1])

        # Residual + norm, then feed-forward on every token, residual + norm
        x = x + new_x
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class Encoder(nn.Module):
    """
    Stack of EncoderLayer, with an optional final normalization: (B, N, E) -> (B, N, E).

    Args:
        layers: the EncoderLayer modules, applied in order.
        norm_layer: optional module applied to the output (S-Mamba: LayerNorm(E)).
    """

    def __init__(self, layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x):
        # x [B, N, E]
        for layer in self.layers:
            x = layer(x)

        if self.norm is not None:
            x = self.norm(x)

        return x
