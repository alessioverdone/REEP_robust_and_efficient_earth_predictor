"""
Figures of a training run: metric curves, reconstruction examples (phase 1)
and forecast examples (phase 2).

Only matplotlib's object-oriented API (Figure, no pyplot): no global figure
state and no GUI backend, so it is safe to call from a training loop running
on a headless machine.
"""

import csv
import os

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator


def read_history(history_csv: str) -> dict:
    """history.csv as {column: list of floats}; empty cells become nan."""
    columns = {}
    with open(history_csv, newline="") as f:
        for row in csv.DictReader(f):
            for key, value in row.items():
                columns.setdefault(key, []).append(float(value) if value not in ("", None)
                                                   else float("nan"))
    return columns


def plot_history(history_csv: str,
                 out_dir: str,
                 metrics: tuple,
                 best_epoch: int = None,
                 test_metrics: dict = None):
    """
    One figure per metric (<metric>.png): train and val per epoch, the best
    epoch as a dotted vertical line and the test value of the best checkpoint
    as a dashed horizontal line. Plus lr.png with the learning rate.
    """
    if not os.path.isfile(history_csv):
        return
    history = read_history(history_csv)
    if not history.get("epoch"):
        return
    epochs = history["epoch"]

    for metric in metrics:
        fig = Figure(figsize=(7, 4.5), layout="constrained")
        ax = fig.add_subplot()
        for split in ("train", "val"):
            key = f"{split}_{metric}"
            if key in history:
                ax.plot(epochs,
                        history[key],
                        marker="o",
                        markersize=3,
                        label=split)
        if best_epoch is not None:
            ax.axvline(best_epoch,
                       color="gray",
                       linestyle=":",
                       label=f"best (epoch {best_epoch})")
        if test_metrics and metric in test_metrics:
            ax.axhline(test_metrics[metric],
                       color="black",
                       linestyle="--",
                       label=f"test, best weights ({test_metrics[metric]:.4g})")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.savefig(os.path.join(out_dir, f"{metric}.png"),
                    dpi=120)

    if "lr" in history:
        fig = Figure(figsize=(7, 3.5), layout="constrained")
        ax = fig.add_subplot()
        ax.plot(epochs,
                history["lr"],
                marker="o",
                markersize=3)
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.grid(alpha=0.3)
        fig.savefig(os.path.join(out_dir, "lr.png"),
                    dpi=120)


def _stretch(img: np.ndarray,
             lo: np.ndarray,
             hi: np.ndarray) -> np.ndarray:
    """Linear stretch to [0, 1] with the given per-channel bounds."""
    return np.clip((img - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)


def save_reconstruction_grid(inputs: np.ndarray,
                             recons: np.ndarray,
                             path: str,
                             title: str = None,
                             rgb_bands: tuple = (2, 1, 0),
                             nir_band: int = 3):
    """
    One row per example, columns: input RGB | reconstruction RGB | input NIR |
    reconstruction NIR | |error| averaged over the bands.

    inputs, recons: (N, C, H, W) in [-1, 1]. Each row is stretched with the
    2-98 percentiles of its input, applied identically to the reconstruction,
    so the two are directly comparable. The error maps share one color scale.
    The default bands follow src/dataset/preprocessing.py: blue, green, red, NIR.
    """
    n = inputs.shape[0]
    error = np.abs(recons - inputs).mean(axis=1)
    error_max = float(np.percentile(error, 99)) or 1.0
    columns = ("input RGB", "recon RGB", "input NIR", "recon NIR", "|error|")

    fig = Figure(figsize=(2.2 * len(columns), 2.2 * n + 0.6), layout="constrained")
    axes = fig.subplots(n, len(columns), squeeze=False)
    for i in range(n):
        rgb_in = np.transpose(inputs[i, list(rgb_bands)], (1, 2, 0))
        rgb_out = np.transpose(recons[i, list(rgb_bands)], (1, 2, 0))
        lo = np.percentile(rgb_in, 2, axis=(0, 1))
        hi = np.percentile(rgb_in, 98, axis=(0, 1))

        nir_in = inputs[i, nir_band]
        nir_lo, nir_hi = np.percentile(nir_in, 2), np.percentile(nir_in, 98)

        axes[i, 0].imshow(_stretch(rgb_in, lo, hi))
        axes[i, 1].imshow(_stretch(rgb_out, lo, hi))
        axes[i, 2].imshow(_stretch(nir_in, nir_lo, nir_hi),
                          cmap="gray")
        axes[i, 3].imshow(_stretch(recons[i, nir_band], nir_lo, nir_hi),
                          cmap="gray")
        err = axes[i, 4].imshow(error[i],
                                cmap="magma",
                                vmin=0.0,
                                vmax=error_max)
        for j, ax in enumerate(axes[i]):
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(columns[j], fontsize=9)

    fig.colorbar(err,
                 ax=axes[:, 4],
                 shrink=0.6,
                 label="mean |error| over bands")
    if title:
        fig.suptitle(title)
    fig.savefig(path,
                dpi=110)


# ============================== PHASE 2: DIFFUSION ==============================

def _with_values(epochs: list,
                 values: list) -> tuple:
    """The epochs that have a value and their values: validation columns are
    empty on the epochs without a validation."""
    pairs = [(e, v) for e, v in zip(epochs, values) if np.isfinite(v)]
    return [e for e, _ in pairs], [v for _, v in pairs]


def _epoch_figure(history: dict,
                  series: tuple,
                  ylabel: str,
                  title: str,
                  path: str,
                  best_epoch: int = None):
    """One curve per (column, label, style) of `series` that history.csv has."""
    plotted = False
    fig = Figure(figsize=(7, 4.5), layout="constrained")
    ax = fig.add_subplot()
    for column, label, style in series:
        if column not in history:
            continue
        epochs, values = _with_values(history["epoch"],
                                      history[column])
        if not values:
            continue
        ax.plot(epochs,
                values,
                style,
                marker="o",
                markersize=3,
                label=label)
        plotted = True
    if not plotted:
        return
    if best_epoch is not None:
        ax.axvline(best_epoch,
                   color="gray",
                   linestyle=":",
                   label=f"best (epoch {best_epoch})")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(path,
                dpi=120)


def plot_diffusion_history(history_csv: str,
                           out_dir: str,
                           best_epoch: int = None):
    """
    The curves of a diffusion run, from its history.csv:

        loss.png         training loss, mean over the steps of each epoch
        latent_rmse.png  latent RMSE: training (one denoising step from a noised
                         target) and validation (full sampling), with the
                         context-mean baseline
        pixel_rmse.png   validation RMSE in pixel space on the two halves of the
                         predicted frames; the second half selects the best model
        lr.png           learning rate at the end of each epoch
        ens.png          EarthNetScore, validation and training sample (with `ens`)
        ens_subscores.png  MAD, OLS, EMD, SSIM
        ens_mae.png      P-MAE and N-MAE
    """
    if not os.path.isfile(history_csv):
        return
    history = read_history(history_csv)
    if not history.get("epoch"):
        return

    _epoch_figure(history,
                  (("train_loss", "train", "-"),),
                  ylabel="loss",
                  title="diffusion loss",
                  path=os.path.join(out_dir, "loss.png"))
    _epoch_figure(history,
                  (("train_latent_rmse", "train (one denoising step)", "-"),
                   ("val_predict_latent_rmse", "val (sampled)", "-"),
                   ("val_mean_of_k_latent_rmse", "val, mean of K samples", "-"),
                   ("val_best_of_k_latent_rmse", "val, best of K samples", "-"),
                   ("val_past_mean_latent_rmse", "val, context mean (baseline)", "--")),
                  ylabel="RMSE (scaled latent)",
                  title="latent RMSE on the predicted frames",
                  path=os.path.join(out_dir, "latent_rmse.png"),
                  best_epoch=best_epoch)
    _epoch_figure(history,
                  (("val_fromer_rmse", "val, first half of the predicted frames", "-"),
                   ("val_latter_rmse", "val, second half (selection criterion)", "-")),
                  ylabel="RMSE (pixels in [-1, 1])",
                  title="pixel RMSE of the forecast",
                  path=os.path.join(out_dir, "pixel_rmse.png"),
                  best_epoch=best_epoch)

    # ENS columns (config section `ens`): val_* on the validation epochs, train_* on the
    # epochs where a sample of training cubes was scored
    _epoch_figure(history,
                  (("val_ENS", "val", "-"),
                   ("train_ENS", "train (sample of cubes)", "-")),
                  ylabel="ENS (higher is better)",
                  title="EarthNetScore of the forecast",
                  path=os.path.join(out_dir, "ens.png"),
                  best_epoch=best_epoch)
    _epoch_figure(history,
                  (("val_MAD", "val MAD", "-"),
                   ("val_OLS", "val OLS", "-"),
                   ("val_EMD", "val EMD", "-"),
                   ("val_SSIM", "val SSIM", "-"),
                   ("train_MAD", "train MAD", "--"),
                   ("train_OLS", "train OLS", "--"),
                   ("train_EMD", "train EMD", "--"),
                   ("train_SSIM", "train SSIM", "--")),
                  ylabel="sub-score (higher is better)",
                  title="ENS sub-scores",
                  path=os.path.join(out_dir, "ens_subscores.png"),
                  best_epoch=best_epoch)
    _epoch_figure(history,
                  (("val_P_MAE", "val P-MAE (bands)", "-"),
                   ("val_N_MAE", "val N-MAE (NDVI)", "-"),
                   ("train_P_MAE", "train P-MAE (bands)", "--"),
                   ("train_N_MAE", "train N-MAE (NDVI)", "--")),
                  ylabel="mean absolute error (reflectance, valid pixels)",
                  title="MAE of the forecast",
                  path=os.path.join(out_dir, "ens_mae.png"),
                  best_epoch=best_epoch)

    if "lr" in history:
        fig = Figure(figsize=(7, 3.5), layout="constrained")
        ax = fig.add_subplot()
        ax.plot(history["epoch"],
                history["lr"],
                marker="o",
                markersize=3)
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.grid(alpha=0.3)
        fig.savefig(os.path.join(out_dir, "lr.png"),
                    dpi=120)


def plot_lead_time_rmse(lead_time_csv: str,
                        path: str,
                        best_epoch: int = None):
    """
    Pixel RMSE as a function of the lead time (1 = first predicted frame): the
    last validation, the best one, and the context-mean baseline of the last
    validation. Read from lead_time.csv (epoch, global_step, lead, model_rmse,
    baseline_rmse), one row per validation and lead time.
    """
    if not os.path.isfile(lead_time_csv):
        return
    rows = read_history(lead_time_csv)
    if not rows.get("epoch"):
        return
    epochs = np.asarray(rows["epoch"])
    leads = np.asarray(rows["lead"])
    model = np.asarray(rows["model_rmse"])
    baseline = np.asarray(rows["baseline_rmse"])
    last = epochs.max()

    fig = Figure(figsize=(7, 4.5), layout="constrained")
    ax = fig.add_subplot()
    shown = [(last, f"model, last validation (epoch {last:.0f})")]
    if best_epoch is not None and best_epoch != last and np.any(epochs == best_epoch):
        shown.append((best_epoch, f"model, best (epoch {best_epoch})"))
    for epoch, label in shown:
        sel = epochs == epoch
        ax.plot(leads[sel],
                model[sel],
                marker="o",
                markersize=3,
                label=label)
    sel = epochs == last
    ax.plot(leads[sel],
            baseline[sel],
            "--",
            color="gray",
            label="context mean (baseline)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("lead time (predicted frame)")
    ax.set_ylabel("RMSE (pixels in [-1, 1])")
    ax.set_title("forecast error by lead time, validation")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(path,
                dpi=120)


def forecast_lead_indices(n_predict: int,
                          n_columns: int) -> list:
    """n_columns lead times evenly spread over the predicted frames, first and last included."""
    return sorted(set(np.linspace(0, n_predict - 1, num=min(n_columns, n_predict))
                      .round().astype(int).tolist()))


def save_forecast_grid(context: np.ndarray,
                       target: np.ndarray,
                       pred: np.ndarray,
                       path: str,
                       n_columns: int,
                       names: list = None,
                       title: str = None,
                       rgb_bands: tuple = (2, 1, 0)):
    """
    Three rows per cube: ground truth, prediction, |error| averaged over the
    bands. The first column is the last context frame, the others are
    n_columns lead times evenly spread over the predicted frames.

    context: (N, T_context, C, H, W); target, pred: (N, T_predict, C, H, W),
    all in [-1, 1]. Each column is stretched with the 2-98 percentiles of its
    ground-truth frame, applied identically to the prediction below it: a
    cloudy frame does not darken the clear ones, and the prediction stays
    directly comparable to its target. The error maps share one color scale. The default bands follow
    src/dataset/preprocessing.py: blue, green, red, NIR.
    """
    n, n_context = context.shape[0], context.shape[1]
    leads = forecast_lead_indices(target.shape[1],
                                  n_columns)
    error = np.abs(pred - target).mean(axis=2)
    error_max = float(np.percentile(error[:, leads], 99)) or 1.0
    titles = [f"context t={n_context}"] + [f"t+{i + 1}" for i in leads]

    def rgb(frame):
        return np.transpose(frame[list(rgb_bands)], (1, 2, 0))

    fig = Figure(figsize=(1.9 * len(titles), 1.9 * 3 * n + 0.6), layout="constrained")
    axes = fig.subplots(3 * n, len(titles), squeeze=False)
    for i in range(n):
        gt_row, pred_row, err_row = axes[3 * i], axes[3 * i + 1], axes[3 * i + 2]

        def bounds(frame):
            return (np.percentile(frame, 2, axis=(0, 1)),
                    np.percentile(frame, 98, axis=(0, 1)))

        gt_row[0].imshow(_stretch(rgb(context[i, -1]), *bounds(rgb(context[i, -1]))))
        for j, t in enumerate(leads, start=1):
            lo, hi = bounds(rgb(target[i, t]))
            gt_row[j].imshow(_stretch(rgb(target[i, t]), lo, hi))
            pred_row[j].imshow(_stretch(rgb(pred[i, t]), lo, hi))
            err = err_row[j].imshow(error[i, t],
                                    cmap="magma",
                                    vmin=0.0,
                                    vmax=error_max)

        name = f"{names[i][:28]}\n" if names else f"cube {i}\n"
        for row, label in ((gt_row, f"{name}ground truth"),
                           (pred_row, "prediction"),
                           (err_row, "|error|")):
            for ax in row:
                ax.set_xticks([])
                ax.set_yticks([])
            # the context column of the prediction and error rows stays empty
            if row is not gt_row:
                for spine in row[0].spines.values():
                    spine.set_visible(False)
            row[0].set_ylabel(label,
                              fontsize=8)
        if i == 0:
            for j, ax in enumerate(gt_row):
                ax.set_title(titles[j],
                             fontsize=9)

    fig.colorbar(err,
                 ax=axes[:, -1],
                 shrink=0.4,
                 label="mean |error| over bands")
    if title:
        fig.suptitle(title)
    fig.savefig(path,
                dpi=100)
