"""Input embedding of S-Mamba (src/model/S_Mamba.py)."""

import torch
import torch.nn as nn


class DataEmbedding_inverted(nn.Module):
    """
    Inverted embedding (iTransformer): every variate is ONE token, obtained by
    a linear map of its whole input window, L values -> E values. Covariates,
    when given, become extra tokens embedded with the same map.

    Args:
        c_in: length L of the input window.
        d_model: token dimension E.
        dropout: dropout applied to the tokens.
    """

    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        """
        Args:
            x: (B, L, N) input window of N variates.
            x_mark: optional (B, L, M) covariates over the same window.

        Returns:
            (B, N, E) tokens, or (B, N + M, E) with covariates (variates first).
        """
        x = x.permute(0, 2, 1)

        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        return self.dropout(x)
