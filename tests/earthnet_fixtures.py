"""
Synthetic EarthNet2021 cubes for the tests, in the format of the real ones:

    highresdynamic  (128, 128, 7, T)   4 reflectance bands + 3 mask channels, one frame every 5 days
    highresstatic   (128, 128, 1)      elevation
    mesodynamic     (80, 80, 5, days)  weather, one value per day
    mesostatic      (80, 80, 1)

The images are smooth gradients (no frame is empty or constant, so the REEP
preprocessing keeps every cube) and the weather is a known function of
(cell, variable, day), so the extracted series can be checked exactly.
Arrays are small once compressed.
"""

from pathlib import Path

import numpy as np

HW, MESO_HW, N_MESO, STEPS = 128, 80, 5, 5


def highres(n_frames: int, seed: int) -> np.ndarray:
    ramp = np.linspace(0.05, 0.6, HW, dtype=np.float32)
    x = np.zeros((HW, HW, 7, n_frames), dtype=np.float32)
    for b in range(4):
        for t in range(n_frames):
            x[:, :, b, t] = ramp[None, :] * (1 + 0.1 * b) + 0.002 * t + 0.01 * seed
    return x


def weather(n_days: int, seed: int, first_day: int = 0) -> np.ndarray:
    """(80, 80, 5, n_days): variable c on day d = smooth seasonal curve + a spatial gradient."""
    days = np.arange(first_day, first_day + n_days, dtype=np.float32)
    rows = np.arange(MESO_HW, dtype=np.float32)
    md = np.zeros((MESO_HW, MESO_HW, N_MESO, n_days), dtype=np.float32)
    base = [2.0, 1010.0, 12.0, 6.0, 18.0]
    amplitude = [2.0, 6.0, 8.0, 7.0, 9.0]
    for c in range(N_MESO):
        curve = base[c] + amplitude[c] * np.sin(2 * np.pi * days / 60.0 + seed + c)
        if c == 0:
            curve = np.maximum(curve, 0.0)          # precipitation >= 0
        md[:, :, c, :] = curve[None, None, :] + 0.01 * rows[:, None, None]
    return md


def write_cube(path: Path, n_frames: int, seed: int, meso_days: int = None, **overrides):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"highresdynamic": highres(n_frames, seed),
              "highresstatic": np.full((HW, HW, 1), 300.0 + seed, dtype=np.float32)
              + np.linspace(0, 50, HW, dtype=np.float32)[None, :, None],
              "mesodynamic": weather(meso_days if meso_days is not None else n_frames * STEPS, seed),
              "mesostatic": np.ones((MESO_HW, MESO_HW, 1), dtype=np.float32)}
    arrays.update(overrides)
    np.savez_compressed(path, **arrays)


def expected_series(md: np.ndarray, n_frames: int, crop: int = 2, agg: str = "mean") -> np.ndarray:
    """What the extraction must produce: mean of the central crop x crop cells, then per frame."""
    y0 = (MESO_HW - crop) // 2
    center = md[y0:y0 + crop, y0:y0 + crop, :, :n_frames * STEPS].mean(axis=(0, 1))   # (C, days)
    blocks = center.reshape(N_MESO, n_frames, STEPS)
    frames = {"mean": blocks.mean(axis=2), "last": blocks[:, :, -1],
              "first": blocks[:, :, 0], "sum": blocks.sum(axis=2)}[agg]
    return frames.T                                                                        # (frames, C)


def make_earthnet(root: Path, tiles: int = 3, cubes_per_tile: int = 2) -> Path:
    """
    root/raw/train/<tile>/<cube>.npz, plus the test splits:
      root/raw/iid_test_split  context files with all 150 days of weather
      root/raw/ood_test_split  weather split between context (50 days) and target (100 days)
    Returns root/raw/train.
    """
    train = root / "raw" / "train"
    seed = 0
    for t in range(tiles):
        for c in range(cubes_per_tile):
            write_cube(train / f"tile{t}" / f"cube{t}{c}.npz", n_frames=30, seed=seed)
            seed += 1
    for split, ctx_days in (("iid", 150), ("ood", 50)):
        test = root / "raw" / f"{split}_test_split"
        for c in range(3):
            name = f"{split}{c}.npz"
            write_cube(test / "context" / "tileX" / f"context_{name}", n_frames=10, seed=100 + c,
                       mesodynamic=weather(ctx_days, 100 + c))
            target = {"mesodynamic": weather(150 - ctx_days, 100 + c, first_day=ctx_days)} if ctx_days < 150 else {}
            path = test / "target" / "tileX" / f"target_{name}"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, highresdynamic=highres(20, 100 + c), **target)
    return train
