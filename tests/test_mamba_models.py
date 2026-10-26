"""
Tests of the forecasting models of the Condition Predictor, src/model/mamba/:
shapes, normalization, gradients and the ability to learn a simple signal.

The windows have the size of the weather series of an EarthNet cube: 10 steps
of context, 20 to forecast, 5 variates.
"""

import math
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from src.config import CONFIGS_DIR, PatchMixerParams, SMambaParams, load_config
from src.model.condition_utils import build_from_arch, model_arch
from src.model.mamba.PatchMixer import Model as PatchMixer
from src.model.mamba.S_Mamba import Model as SMamba

SEQ_LEN, PRED_LEN, N_VARIATES, BATCH = 10, 20, 5, 8
MODEL_NAMES = ["S_Mamba", "PatchMixer"]


def model_configs(name: str, **overrides) -> SimpleNamespace:
    """The namespace a model is built from: the schema defaults, small, plus the data dimensions."""
    if name == "S_Mamba":
        params = OmegaConf.to_container(OmegaConf.structured(SMambaParams))
        params.update(d_model=32, d_state=8, d_ff=64, e_layers=2)
    else:
        params = OmegaConf.to_container(OmegaConf.structured(PatchMixerParams))
        params.update(d_model=32, patch_len=4, patch_stride=2)
    params.update(overrides)
    return SimpleNamespace(seq_len=SEQ_LEN, pred_len=PRED_LEN, enc_in=N_VARIATES, **params)


def build(name: str, device, **overrides) -> torch.nn.Module:
    model_class = SMamba if name == "S_Mamba" else PatchMixer
    return model_class(model_configs(name, **overrides)).to(device)


# ================================== SHAPES ==================================

@pytest.mark.parametrize("name", MODEL_NAMES)
def test_forecast_shape(name, device):
    model = build(name, device)
    assert model(torch.randn(BATCH, SEQ_LEN, N_VARIATES, device=device)).shape == (BATCH, PRED_LEN, N_VARIATES)


def test_s_mamba_uses_the_covariates_but_does_not_forecast_them(device):
    model = build("S_Mamba", device).eval()
    x = torch.randn(BATCH, SEQ_LEN, N_VARIATES, device=device)
    covariates = torch.randn(BATCH, SEQ_LEN, 3, device=device)
    with torch.no_grad():
        out = model(x, covariates)
        out_changed = model(x, covariates + 1.0)
    assert out.shape == (BATCH, PRED_LEN, N_VARIATES)
    assert not torch.allclose(out, out_changed)


def test_s_mamba_builds_the_configured_layers(device):
    model = build("S_Mamba", device, e_layers=3)
    assert len(model.encoder.layers) == 3
    assert model.projector.out_features == PRED_LEN


@pytest.mark.parametrize("patch_stride", [1, 2, 3])
def test_patchmixer_cuts_patches_with_patch_stride(patch_stride, device):
    """Before the move to src/ PatchMixer read the stride of the data windows instead of patch_stride."""
    model = build("PatchMixer", device, patch_stride=patch_stride)
    backbone = model.model
    assert backbone.stride == patch_stride
    # patches of the window padded with patch_stride copies of its last value
    assert backbone.patch_num == (SEQ_LEN + patch_stride - backbone.patch_size) // patch_stride + 1
    assert model(torch.randn(BATCH, SEQ_LEN, N_VARIATES, device=device)).shape == (BATCH, PRED_LEN, N_VARIATES)


# =============================== NORMALIZATION ===============================

@pytest.mark.parametrize("name", MODEL_NAMES)
def test_forecast_follows_a_shift_of_the_input(name, device):
    """
    S_Mamba (use_norm) and PatchMixer (RevIN) normalize every input window and
    de-normalize the forecast: adding a constant to a variate adds it to its forecast.
    """
    model = build(name, device).eval()
    x = torch.randn(BATCH, SEQ_LEN, N_VARIATES, device=device)
    shift = torch.linspace(-10, 10, N_VARIATES, device=device)
    with torch.no_grad():
        assert torch.allclose(model(x + shift), model(x) + shift, atol=1e-4)


def test_s_mamba_without_norm_does_not_follow_a_shift(device):
    model = build("S_Mamba", device, use_norm=False).eval()
    x = torch.randn(BATCH, SEQ_LEN, N_VARIATES, device=device)
    with torch.no_grad():
        assert not torch.allclose(model(x + 10.0), model(x) + 10.0, atol=1e-2)


# ================================ TRAINING ================================

@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_parameter_receives_a_gradient(name, device):
    """Catches modules that are built but never used in forward (as the old Mamba_EncDec had)."""
    model = build(name, device)
    model(torch.randn(BATCH, SEQ_LEN, N_VARIATES, device=device)).pow(2).mean().backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert not no_grad, f"parameters outside the computation: {no_grad}"
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_eval_mode_is_deterministic(name, device):
    model = build(name, device).eval()
    x = torch.randn(BATCH, SEQ_LEN, N_VARIATES, device=device)
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


def sine_batch(n: int, generator: torch.Generator, device) -> tuple:
    """Sinusoids of period 8 with random phase and amplitude: (context, continuation)."""
    t = torch.arange(SEQ_LEN + PRED_LEN, dtype=torch.float32)
    phase = 2 * math.pi * torch.rand(n, 1, N_VARIATES, generator=generator)
    amplitude = 0.5 + torch.rand(n, 1, N_VARIATES, generator=generator)
    series = amplitude * torch.sin(2 * math.pi * t.view(1, -1, 1) / 8 + phase)
    series = series.to(device)
    return series[:, :SEQ_LEN], series[:, SEQ_LEN:]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_learns_to_continue_a_sinusoid(name, device):
    """A few hundred Adam steps must cut the error on new series by a large factor."""
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    model = build(name, device, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    x_test, y_test = sine_batch(64, generator, device)

    def test_loss():
        model.eval()
        with torch.no_grad():
            loss = torch.nn.functional.mse_loss(model(x_test), y_test).item()
        model.train()
        return loss

    initial = test_loss()
    for _ in range(300):
        x, y = sine_batch(32, generator, device)
        optimizer.zero_grad()
        torch.nn.functional.mse_loss(model(x), y).backward()
        optimizer.step()
    assert test_loss() < 0.1 * initial


# ============================ CONFIGURATION ============================

@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models_build_from_the_default_configuration(name, device):
    """configuration_0.yaml describes, for both architectures, a model that can be built and run."""
    cfg = load_config(CONFIGS_DIR / "configuration_0.yaml", overrides=[f"condition.model.name={name}"])
    arch = model_arch(cfg)
    assert (arch["n_context"], arch["n_predict"], arch["n_channels"]) == (SEQ_LEN, PRED_LEN, N_VARIATES)
    model = build_from_arch(arch).to(device)
    assert model(torch.randn(2, SEQ_LEN, N_VARIATES, device=device)).shape == (2, PRED_LEN, N_VARIATES)


# ============================ REAL MAMBA KERNELS ============================

@pytest.mark.cuda
def test_s_mamba_with_the_real_mamba_kernels(device):
    """Forward and backward with mamba_ssm on the GPU (skipped without it)."""
    model = build("S_Mamba", device)
    out = model(torch.randn(BATCH, SEQ_LEN, N_VARIATES, device=device))
    assert out.shape == (BATCH, PRED_LEN, N_VARIATES)
    out.pow(2).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
