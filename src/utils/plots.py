"""
Figures of a training run: metric curves and the per-band example grids
(reconstructions of phase 1, forecasts of phase 2).

Only matplotlib's object-oriented API (Figure, no pyplot): no global figure
state and no GUI backend, so it is safe to call from a training loop running
on a headless machine.
"""

import csv
import os

import numpy as np
from matplotlib import colormaps
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
                 best_epoch: int = None):
    """
    One figure per metric (<metric>.png): train and val per epoch, the best
    epoch as a dotted vertical line. Plus lr.png with the learning rate.
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


BAND_NAMES = ("blue", "green", "red", "nir")      # order of src/dataset/preprocessing.py
MAX_COLUMNS = 20                                  # columns of one figure; more are subsampled


def column_indices(n: int,
                   max_columns: int = MAX_COLUMNS) -> list:
    """All the n columns up to max_columns, otherwise max_columns evenly spread (first and last included)."""
    if n <= max_columns:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, num=max_columns).round().astype(int).tolist()))


def _stretch(img: np.ndarray,
             lo: float,
             hi: float) -> np.ndarray:
    """Linear stretch to [0, 1] with the given bounds."""
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def save_band_grids(truth: np.ndarray,
                    pred: np.ndarray,
                    prob: np.ndarray,
                    out_dirs,
                    col_titles: list,
                    valid: np.ndarray = None,
                    pred_label: str = "reconstruction",
                    title: str = None,
                    band_names: tuple = BAND_NAMES,
                    cell: float = 1.6,
                    dpi: int = 100):
    """
    One PNG per band, <folder>/<band>.png in every folder of out_dirs (the same figure
    is written to all of them). One column per image, four rows:

        ground truth        the band, stretched with the 2-98 percentiles of that
                            column's ground truth
        reconstruction      (pred_label) the same band with the same stretch, so the
                            two rows are directly comparable
        cloud probability   0 (clear) to 1 (cloud)
        |error| / 2         |pred - truth| / 2: the values are in [-1, 1], so the
                            range is 2 and the error falls in [0, 1]. Same color
                            scale as the cloud probability, one colorbar on the right

    truth, pred: (N, C, H, W) in [-1, 1]. prob: (N, H, W) in [0, 1], NaN = unknown (grey).
    valid: (N, H, W) bool, False pixels are grey in the error row (the ones the metrics
    leave out); None = every pixel counts.
    """
    if isinstance(out_dirs, (str, os.PathLike)):
        out_dirs = [out_dirs]
    for folder in out_dirs:
        os.makedirs(folder, exist_ok=True)

    n, n_bands = truth.shape[0], truth.shape[1]
    error = np.clip(np.abs(pred - truth) / 2.0, 0.0, 1.0)
    cmap = colormaps["magma"].copy()
    cmap.set_bad("0.6")
    prob_masked = np.ma.masked_invalid(prob)
    row_labels = ("ground truth", pred_label, "cloud probability", "|error| / 2")

    for c in range(n_bands):
        name = band_names[c] if c < len(band_names) else f"band{c}"
        error_c = error[:, c] if valid is None else np.ma.masked_where(~valid, error[:, c])

        fig = Figure(figsize=(cell * n + 1.4, cell * 4 + 0.9), layout="constrained")
        axes = fig.subplots(4, n, squeeze=False)
        for j in range(n):
            lo = float(np.percentile(truth[j, c], 2))
            hi = float(np.percentile(truth[j, c], 98))
            axes[0, j].imshow(_stretch(truth[j, c], lo, hi), cmap="gray", vmin=0.0, vmax=1.0)
            axes[1, j].imshow(_stretch(pred[j, c], lo, hi), cmap="gray", vmin=0.0, vmax=1.0)
            image = axes[2, j].imshow(prob_masked[j], cmap=cmap, vmin=0.0, vmax=1.0)
            axes[3, j].imshow(error_c[j], cmap=cmap, vmin=0.0, vmax=1.0)
            axes[0, j].set_title(col_titles[j], fontsize=8)
        for i, ax_row in enumerate(axes):
            for ax in ax_row:
                ax.set_xticks([])
                ax.set_yticks([])
            ax_row[0].set_ylabel(row_labels[i], fontsize=8)
        fig.colorbar(image,
                     ax=axes[2:, -1],
                     shrink=0.8,
                     label="cloud probability  |  |error| / 2   (0-1)")
        fig.suptitle(f"{title} - {name}" if title else name)
        for folder in out_dirs:
            fig.savefig(os.path.join(folder, f"{name}.png"),
                        dpi=dpi)


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
                  (("train_loss", "train", "-"),
                   ("val_loss", "val (diffusion.save_best_by=loss)", "-")),
                  ylabel="loss",
                  title="diffusion loss",
                  path=os.path.join(out_dir, "loss.png"),
                  best_epoch=best_epoch)
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
