"""
S-Mamba: Mamba for multivariate time series forecasting.

Reference: Z. Wang et al., "Is Mamba Effective for Time Series Forecasting?",
2024, https://arxiv.org/abs/2403.11144. The computation is the one of the
official implementation (https://github.com/wzhwzhwzh0921/S-D-Mamba), without
the modules it built and never used.

The model is *inverted*, as iTransformer (https://arxiv.org/abs/2310.06625):
every variate is ONE token, embedded from its whole input window. As a
consequence:
  - the correlations between variates are modelled by a bidirectional Mamba
    that scans the sequence of tokens (one per variate, plus covariates);
  - the temporal dynamics are modelled by the linear embedding
    (seq_len -> d_model), the feed-forward block of every layer and the linear
    projector (d_model -> pred_len).

Shapes: B batch, L seq_len, S pred_len, N variates, M covariates, E d_model.
"""

import torch
import torch.nn as nn
from mamba_ssm import Mamba

from src.model.mamba.layers.Embed import DataEmbedding_inverted
from src.model.mamba.layers.Mamba_EncDec import Encoder, EncoderLayer


class Model(nn.Module):
    """
    S-Mamba forecaster: (B, L, N) input window -> (B, S, N) forecast.

    Args:
        configs: namespace with the window lengths `seq_len`, `pred_len` (from
            the data) and the architecture of cfg.model.params: `d_model`,
            `d_state`, `d_conv`, `expand` (Mamba blocks), `d_ff`, `e_layers`,
            `dropout`, `activation` and `use_norm`.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm

        # Tokenization: each variate's window (L values) -> one E-dim token
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)

        # Encoder-only architecture: e_layers blocks of bidirectional Mamba + feed-forward.
        # The two Mamba blocks of a layer scan the tokens in opposite directions.
        self.encoder = Encoder(
            [
                EncoderLayer(
                    Mamba(
                        d_model=configs.d_model,  # model dimension
                        d_state=configs.d_state,  # SSM state expansion factor
                        d_conv=configs.d_conv,  # local convolution width
                        expand=configs.expand,  # block expansion factor
                    ),
                    Mamba(
                        d_model=configs.d_model,
                        d_state=configs.d_state,
                        d_conv=configs.d_conv,
                        expand=configs.expand,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Each token (E values) -> the S forecast values of its variate
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc=None):
        """
        Args:
            x_enc: (B, L, N) input window.
            x_mark_enc: optional (B, L, M) covariates, embedded as M extra
                tokens; they take part in the scan but are not forecast.

        Returns:
            (B, S, N) forecast, in the units of x_enc.
        """
        if self.use_norm:
            # Instance normalization of every variate over its window
            # (Non-stationary Transformer); undone on the forecast below
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N

        # Embedding: B L N -> B (N + M) E
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Encoder: B (N + M) E -> B (N + M) E
        enc_out = self.encoder(enc_out)

        # Projection: B (N + M) E -> B (N + M) S -> B S (N + M), then the covariate tokens are dropped
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            # De-normalization with the statistics of the input window
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc=None):
        """(B, L, N) input window [+ (B, L, M) covariates] -> (B, S, N) forecast."""
        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, S, N]
