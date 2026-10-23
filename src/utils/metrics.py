"""
Deterministic forecast metrics.

Every function takes the prediction and the ground truth as numpy arrays of
the same shape, e.g. (n_samples, pred_len, n_variates), and returns a float.
The variates are expected on the LAST axis (used by CORR only).
"""

import numpy as np


def RSE(pred, true):
    """Root relative squared error: error norm / norm of the deviations of `true` from its mean."""
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    """
    Pearson correlation of every variate (last axis) over all the other axes,
    averaged over the variates. nan when a variate is constant.
    """
    pred = pred.reshape(-1, pred.shape[-1])
    true = true.reshape(-1, true.shape[-1])
    p = pred - pred.mean(0)
    t = true - true.mean(0)
    return np.mean((p * t).sum(0) / np.sqrt((p ** 2).sum(0) * (t ** 2).sum(0)))


def MAE(pred, true):
    """Mean absolute error."""
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    """Mean squared error."""
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    """Root mean squared error, in the units of the data."""
    return np.sqrt(MSE(pred, true))


def metric(pred, true):
    """The three scale-dependent errors: (MAE, MSE, RMSE)."""
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)

    return mae, mse, rmse
