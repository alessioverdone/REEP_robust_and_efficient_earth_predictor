"""Tests of the building blocks of the Mamba models, src/model/mamba/layers/."""

import pytest
import torch
from mamba_ssm import Mamba

from src.model.mamba.layers.Embed import DataEmbedding_inverted
from src.model.mamba.layers.Mamba_EncDec import Encoder, EncoderLayer
from src.model.mamba.layers.RevIN import RevIN

B, L, N, M, E = 4, 10, 5, 3, 16


# ================================== RevIN ==================================

@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("subtract_last", [True, False])
def test_revin_denorm_inverts_norm(affine, subtract_last):
    revin = RevIN(N, affine=affine, subtract_last=subtract_last)
    if affine:
        # non-trivial affine parameters, as after some training
        with torch.no_grad():
            revin.affine_weight.uniform_(0.5, 2.0)
            revin.affine_bias.normal_()
    x = 5.0 + 3.0 * torch.randn(B, L, N)
    assert torch.allclose(revin(revin(x, "norm"), "denorm"), x, atol=1e-4)


def test_revin_normalizes_every_series():
    x = 5.0 + 3.0 * torch.randn(B, L, N)
    out = RevIN(N, affine=False)(x, "norm")
    assert torch.allclose(out.mean(dim=1), torch.zeros(B, N), atol=1e-5)
    assert torch.allclose(out.std(dim=1, unbiased=False), torch.ones(B, N), atol=1e-3)


def test_revin_rejects_an_unknown_mode():
    with pytest.raises(NotImplementedError):
        RevIN(N)(torch.randn(B, L, N), "other")


# ================================= EMBEDDING =================================

def test_inverted_embedding_makes_one_token_per_variate():
    emb = DataEmbedding_inverted(L, E)
    assert emb(torch.randn(B, L, N)).shape == (B, N, E)


def test_inverted_embedding_adds_covariate_tokens():
    emb = DataEmbedding_inverted(L, E)
    assert emb(torch.randn(B, L, N), torch.randn(B, L, M)).shape == (B, N + M, E)


def test_inverted_embedding_maps_each_window_independently():
    emb = DataEmbedding_inverted(L, E, dropout=0.0)
    x = torch.randn(B, L, N)
    out = emb(x)
    x_changed = x.clone()
    x_changed[:, :, 0] += 1.0
    changed = emb(x_changed)
    assert not torch.allclose(changed[:, 0], out[:, 0])
    assert torch.allclose(changed[:, 1:], out[:, 1:])


# ================================== ENCODER ==================================

def make_layer(**kwargs):
    return EncoderLayer(Mamba(d_model=E, d_state=8, d_conv=2, expand=1),
                        Mamba(d_model=E, d_state=8, d_conv=2, expand=1),
                        E, d_ff=2 * E, dropout=0.0, **kwargs)


@pytest.mark.parametrize("activation", ["gelu", "relu"])
def test_encoder_layer_keeps_the_shape(device, activation):
    layer = make_layer(activation=activation).to(device)
    assert layer(torch.randn(B, N, E, device=device)).shape == (B, N, E)


def test_encoder_layer_scans_in_both_directions(device):
    """A single Mamba is causal; with the reversed scan the FIRST token also sees the LAST one."""
    layer = make_layer().to(device).eval()
    x = torch.randn(B, N, E, device=device)
    x_changed = x.clone()
    x_changed[:, -1] += 1.0
    with torch.no_grad():
        first, first_changed = layer(x)[:, 0], layer(x_changed)[:, 0]
        forward_only = layer.mamba(x)[:, 0], layer.mamba(x_changed)[:, 0]
    assert torch.allclose(*forward_only), "the stand-in / Mamba block is expected to be causal"
    assert not torch.allclose(first, first_changed)


def test_encoder_stacks_the_layers_and_normalizes(device):
    encoder = Encoder([make_layer() for _ in range(3)], norm_layer=torch.nn.LayerNorm(E)).to(device)
    out = encoder(torch.randn(B, N, E, device=device))
    assert out.shape == (B, N, E)
    assert len(encoder.layers) == 3
    # LayerNorm at initialization: every token has mean 0 and std 1
    assert torch.allclose(out.mean(-1), torch.zeros(B, N, device=device), atol=1e-5)
