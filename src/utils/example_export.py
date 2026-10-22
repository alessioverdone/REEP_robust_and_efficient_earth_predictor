"""
Per-band example figures of the forecasts (phase 2), see plots.save_band_grids.

    <results>/forecasts/epoch_XXX/<cube>/<band>.png   during the validation
    <results>/forecasts/<split>/<cube>/<band>.png     at the end (val, iid, ood, ...)

<cube> is the stem of the .npz the cube comes from. The cubes of a split are always the
same ones: pick_cubes() draws them with a fixed seed from the sorted file names, so every
epoch, the final validation and a later evaluate_ens.py run show the same cubes.

The cloud probability is not in the tensors the model works with (it keeps the first four
bands of the .npz), so it is read from the original file, see ens_metrics.load_cloud_frames.
"""

import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from src.utils.ens_metrics import load_cloud_frames
from src.utils.plots import column_indices, save_band_grids

logger = logging.getLogger(__name__)


def pick_cubes(files: Sequence,
               n: int,
               seed: int) -> list:
    """n cubes drawn at random (fixed seed) from the sorted `files`, as strings; [] when n <= 0."""
    ordered = sorted(str(f) for f in files)
    if n <= 0 or not ordered:
        return []
    rng = np.random.RandomState(seed)
    chosen = np.sort(rng.choice(len(ordered), size=min(n, len(ordered)), replace=False))
    return [ordered[i] for i in chosen]


def dataset_files(dataset) -> list:
    """The cube files of a diffusion dataset, also when wrapped (MaskedCubeDataset.inner)."""
    while not hasattr(dataset, "files") and hasattr(dataset, "inner"):
        dataset = dataset.inner
    return list(dataset.files)


def save_cube_forecast(folders,
                       cube_path,
                       target,
                       pred,
                       source,
                       ens_cfg,
                       resize,
                       masked: bool = True,
                       title: str = None) -> str:
    """
    Writes <folder>/<cube>/<band>.png for every folder in `folders`. Returns <cube>.

    target, pred: (T, C, H, W) in [-1, 1], the predicted frames. source: (.npz holding the
    cloud channel, index of the first predicted frame in it). Windows longer than
    plots.MAX_COLUMNS frames are shown at evenly spread lead times. masked: the pixels
    that are not valid (cloud probability above ens.cloud_prob_threshold) are grey in the
    error row; False = no pixel is left out. The cloud probability and the validity follow
    the rule of the metrics, with the image resize (ens_metrics.load_cloud_frames).
    """
    stem = Path(str(cube_path)).stem
    columns = column_indices(target.shape[0])
    truth = target[columns].float().cpu().numpy()
    forecast = pred[columns].float().clamp(-1.0, 1.0).cpu().numpy()

    path, offset = source
    try:
        prob, valid = load_cloud_frames(path,
                                        [offset + t for t in columns],
                                        ens_cfg.cloud_prob_threshold,
                                        ens_cfg.cloud_prob_scale,
                                        resize)
    except (OSError, KeyError, IndexError, ValueError) as error:
        logger.warning(f"[examples] cloud probability of {stem} not readable from {path}: "
                       f"{error}; the row is left grey")
        prob, valid = np.full(truth[:, 0].shape, np.nan, dtype=np.float32), None
    if not masked:
        valid = None

    save_band_grids(truth,
                    forecast,
                    prob,
                    [os.path.join(str(folder), stem) for folder in folders],
                    [f"t+{t + 1}" for t in columns],
                    valid=valid,
                    pred_label="prediction",
                    title=title or stem)
    return stem
