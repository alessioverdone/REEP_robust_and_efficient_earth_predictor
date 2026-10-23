"""Tests of src/utils/metrics.py (metrics of the Condition Predictor) against reference implementations and known values."""

import numpy as np
import pytest

from src.utils import metrics


@pytest.fixture
def series():
    """(prediction, ground truth) with shape (samples, steps, variates) and correlation ~0.8."""
    rng = np.random.default_rng(0)
    true = rng.normal(size=(40, 20, 5))
    pred = 0.8 * true + 0.6 * rng.normal(size=true.shape)
    return pred, true


def test_errors_match_their_definition(series):
    pred, true = series
    assert metrics.MAE(pred, true) == pytest.approx(np.abs(pred - true).mean())
    assert metrics.MSE(pred, true) == pytest.approx(((pred - true) ** 2).mean())
    assert metrics.RMSE(pred, true) == pytest.approx(np.sqrt(((pred - true) ** 2).mean()))


def test_metric_returns_mae_mse_rmse(series):
    pred, true = series
    assert metrics.metric(pred, true) == pytest.approx(
        (metrics.MAE(pred, true), metrics.MSE(pred, true), metrics.RMSE(pred, true)))


def test_perfect_forecast_has_zero_error(series):
    _, true = series
    assert metrics.metric(true, true) == (0.0, 0.0, 0.0)
    assert metrics.RSE(true, true) == 0.0


def test_rse_of_the_mean_forecast_is_one(series):
    _, true = series
    assert metrics.RSE(np.full_like(true, true.mean()), true) == pytest.approx(1.0)


@pytest.mark.parametrize("shape", [(200, 3), (40, 20, 5)])
def test_corr_is_the_mean_pearson_correlation(shape):
    rng = np.random.default_rng(1)
    true = rng.normal(size=shape)
    pred = 0.8 * true + 0.6 * rng.normal(size=shape)
    t, p = true.reshape(-1, shape[-1]), pred.reshape(-1, shape[-1])
    expected = np.mean([np.corrcoef(t[:, i], p[:, i])[0, 1] for i in range(shape[-1])])
    assert metrics.CORR(pred, true) == pytest.approx(expected)


def test_corr_limits(series):
    _, true = series
    assert metrics.CORR(true, true) == pytest.approx(1.0)
    assert metrics.CORR(-true, true) == pytest.approx(-1.0)
    # invariant to an affine change of scale of the prediction
    assert metrics.CORR(3.0 * true + 7.0, true) == pytest.approx(1.0)


def test_corr_is_bounded(series):
    pred, true = series
    assert -1.0 <= metrics.CORR(pred, true) <= 1.0


def test_corr_of_a_constant_variate_is_nan():
    true = np.ones((10, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        assert np.isnan(metrics.CORR(np.random.default_rng(2).normal(size=(10, 2)), true))
