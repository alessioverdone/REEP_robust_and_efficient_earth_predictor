"""
Weather series of the Condition Predictor (Mamba): one table (frames x
variables) per cube, extracted once from the EarthNet cubes.

    python src/dataset/meso_series.py                                  (or Run from the IDE)
    python src/dataset/meso_series.py --cfgdir configuration_0.yaml --overwrite

For every cube, mesodynamic (80x80 cells x 5 variables x days) becomes one
value per frame and per variable, with the same code the DiT dataloader uses
(load_meso_dynamic / aggregate_meso in src/dataset/dataloader_s2_dif.py):
  1. mean of the condition.center_crop x center_crop central cells, the area
     covered by the satellite image;
  2. aggregation (condition.meso_agg) of the steps_per_frame days of every frame.

Cubes: the train / val splits of dataset.json (src/dataset/preprocessing.py must
have run), plus the EarthNet test splits of condition.test_splits whose folder
data/raw/<split>_test_split/{context,target} exists.

Writes condition.series_dir (data/preprocessed/<name>/condition_<meso_agg>/):
    <split>.npz     series (n_cubes, n_frames, n_channels) float32 in the units of
                    the cubes, cubes (path of every cube, relative to its split folder)
    stats.json      min / max of every variable over the train series: the
                    normalization of the model, x_norm = 2 (x - lo) / (hi - lo) - 1,
                    clipped to [-1, 1] (the formula and statistics of cond_stats.json)
    series.json     parameters, dataset id, cubes per split, cubes left out. Written
                    last: its presence means the folder is complete
    meso_series.log

A cube is left out (and listed in series.json) when it cannot be read, or when a
frame has no value at all for a variable (all the central cells missing on all
its days): the model needs a complete table.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    # launched as a file (or Run from the IDE): the project root goes on the path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import datetime
import json
import logging
import multiprocessing as mp
import os
import time
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import TrainCfg, load_config, project_path, resolve_config_path, save_config, use_project_dir
from src.dataset.dataloader_s2_dif import MESO_DYNAMIC_KEY, aggregate_meso, minmax_pm1
from src.dataset.dataset_info import dataset_identity, load_dataset_info, split_files

logger = logging.getLogger(__name__)

# (context, target) frames of the EarthNet test splits, as in
# src/scripts/inference_visualization.py
TEST_SPLIT_FRAMES = {
    "iid": (10, 20),
    "ood": (10, 20),
    "extreme": (20, 40),
    "seasonal": (70, 140),
}


# ================================== PARAMETERS ==================================

def series_parameters(cfg: TrainCfg) -> dict:
    """What defines the content of the series: stored in series.json and compared at every use."""
    cond = cfg.condition
    return {"T": cond.T,
            "n_channels": cond.n_channels,
            "steps_per_frame": cond.steps_per_frame,
            "center_crop": cond.center_crop,
            "meso_agg": cond.meso_agg}


def series_dir(cfg: TrainCfg) -> Path:
    return project_path(cfg.condition.series_dir)


def variable_names(cfg: TrainCfg) -> list:
    """
    The labels of the weather channels: condition.variables, or generic names
    (channel_0, ...) when their number differs from the channels actually read.
    They only name metrics and plots.
    """
    cond = cfg.condition
    if len(cond.variables) == cond.n_channels:
        return list(cond.variables)
    logger.warning(f"[meso_series] condition.variables has {len(cond.variables)} names for "
                   f"{cond.n_channels} channels: generic names are used")
    return [f"channel_{i}" for i in range(cond.n_channels)]


def _write_json(path: Path, payload: dict) -> None:
    """Atomic write: a half-written file never looks like a complete one."""
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


# =================================== WORKERS ===================================

_WORKER_CFG: dict = {}


def _init_worker(params: dict) -> None:
    _WORKER_CFG.update(params)
    if not logging.getLogger().handlers:
        logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                            level=logging.INFO)


def _reduce(md: np.ndarray, n_frames: int, source) -> np.ndarray:
    """
    mesodynamic -> (n_frames, n_channels) with aggregate_meso, the computation
    of the DiT dataloader. A frame whose days are all missing for a variable is
    NaN whatever the aggregation: 'sum' (nansum) would turn it into a 0, a value
    the cube does not have, and the cube must be left out instead.
    """
    p = _WORKER_CFG
    kwargs = dict(n_channels=p["n_channels"], n_time=n_frames, steps_per_frame=p["steps_per_frame"],
                  center_crop=p["center_crop"], source=source)
    series = aggregate_meso(md, agg=p["meso_agg"], **kwargs)
    if p["meso_agg"] != "mean":
        series[np.isnan(aggregate_meso(md, agg="mean", **kwargs))] = np.nan
    return series


def _cube_series(path_str: str) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Series of a cube of the train / val splits: (n_frames, n_channels), or (None, error)."""
    try:
        with np.load(path_str) as z:
            md = z[MESO_DYNAMIC_KEY]
        return _reduce(md, _WORKER_CFG["T"], path_str), None
    except Exception as error:
        return None, str(error)


def _test_cube_series(task: tuple[str, str, int]) -> tuple[Optional[np.ndarray], Optional[str]]:
    """
    Series of a cube of an EarthNet test split, whose weather may be split in
    two files: the context one, and the target one. The days of the context
    are used alone when they cover all the frames, otherwise followed by the
    days of the target (the rule of merge_pair in inference_visualization.py).
    """
    ctx_path, tgt_path, n_frames = task
    p = _WORKER_CFG
    need = n_frames * p["steps_per_frame"]
    try:
        with np.load(ctx_path) as ctx:
            md = ctx[MESO_DYNAMIC_KEY]
        if md.shape[3] < need:
            with np.load(tgt_path) as tgt:
                if MESO_DYNAMIC_KEY not in tgt.files:
                    raise ValueError(f"{Path(ctx_path).name}: {md.shape[3]} days of {MESO_DYNAMIC_KEY} "
                                     f"in the context and none in the target, {need} needed")
                md = np.concatenate([md, tgt[MESO_DYNAMIC_KEY]], axis=3)
        return _reduce(md, n_frames, ctx_path), None
    except Exception as error:
        return None, str(error)


def _run(fn, tasks: Sequence, params: dict, workers: int, label: str) -> list:
    """fn over tasks, in order, on `workers` processes, with a progress line every 5%."""
    results = []
    every = max(1, len(tasks) // 20)
    t0 = time.time()

    def progress(done):
        if done % every == 0 or done == len(tasks):
            elapsed = time.time() - t0
            eta = elapsed / done * (len(tasks) - done)
            logger.info(f"[meso_series] {label}: {done}/{len(tasks)} cubes "
                        f"| {elapsed / 60:.1f} min | ETA {eta / 60:.1f} min")

    if workers > 1 and len(tasks) > 1:
        with mp.Pool(processes=workers, initializer=_init_worker, initargs=(params,)) as pool:
            for done, result in enumerate(pool.imap(fn, tasks, chunksize=8), 1):
                results.append(result)
                progress(done)
    else:
        _init_worker(params)
        for done, task in enumerate(tasks, 1):
            results.append(fn(task))
            progress(done)
    return results


# =================================== SPLITS ===================================

def list_test_pairs(test_dir: Path) -> list[tuple[str, Path, Path]]:
    """
    (cube name, context file, target file) of an EarthNet test split: the two
    files live in context/<tile>/ and target/<tile>/ with the same name except
    for the prefix (context_<cube>.npz, target_<cube>.npz).
    """
    ctx_root, tgt_root = test_dir / "context", test_dir / "target"

    def key(path: Path, root: Path) -> tuple[str, str]:
        rel = path.relative_to(root)
        return rel.parent.as_posix(), rel.name.split("_", 1)[-1]

    targets = {key(p, tgt_root): p for p in tgt_root.rglob("*.npz")}
    pairs, missing = [], 0
    for ctx in sorted(ctx_root.rglob("*.npz")):
        tile, cube = key(ctx, ctx_root)
        tgt = targets.get((tile, cube))
        if tgt is None:
            missing += 1
            continue
        pairs.append((f"{tile}/{cube}", ctx, tgt))
    if missing:
        logger.warning(f"[meso_series] {missing} context files without their target in {tgt_root}")
    return pairs


def _collect(names: Sequence[str], results: list) -> tuple[list, list, dict]:
    """Splits the results of a split into (kept names, kept series, left out {name: reason})."""
    kept_names, kept, left_out = [], [], {}
    for name, (series, error) in zip(names, results):
        if series is None:
            left_out[name] = f"unreadable: {error}"
        elif not np.isfinite(series).all():
            left_out[name] = f"{int((~np.isfinite(series)).sum())} missing values"
        else:
            kept_names.append(name)
            kept.append(series)
    return kept_names, kept, left_out


def _save_split(out_dir: Path, split: str, names: list, series: list, n_frames: int, n_channels: int):
    array = (np.stack(series).astype(np.float32) if series
             else np.zeros((0, n_frames, n_channels), dtype=np.float32))
    np.savez(out_dir / f"{split}.npz", series=array, cubes=np.array(names, dtype=str))
    return array


# ================================== BUILD ==================================

def build_series(cfg: TrainCfg, overwrite: bool = False) -> dict:
    """Extracts the series of every split into condition.series_dir. Returns series.json."""
    cond = cfg.condition
    out_dir = series_dir(cfg)
    info_path = out_dir / "series.json"
    if info_path.is_file() and not overwrite:
        raise SystemExit(f"{out_dir} already holds the series: pass --overwrite to extract them again")
    out_dir.mkdir(parents=True, exist_ok=True)
    info_path.unlink(missing_ok=True)

    dataset = load_dataset_info(cfg.data.params.dataset_json)
    raw_dir = project_path(cfg.data.params.data_root)
    params = series_parameters(cfg)
    logger.info(f"[meso_series] dataset {dataset['name']} (id {dataset['id']}), "
                f"aggregation {cond.meso_agg}, {cond.center_crop}x{cond.center_crop} central cells "
                f"-> {out_dir}")

    splits, left_out, train_lo_hi = {}, {}, None
    for split in ("train", "val"):
        files = split_files(dataset, split, raw_dir)
        names = [Path(f).relative_to(raw_dir).as_posix() for f in files]
        results = _run(_cube_series, [str(f) for f in files], params, cond.extract_workers, split)
        if split == "train":
            # min / max over every readable train cube, missing values ignored: the
            # same statistics as the meso part of cond_stats.json
            readable = [s for s, _ in results if s is not None]
            if not readable:
                raise SystemExit(f"[meso_series] no readable train cube in {raw_dir}")
            stacked = np.stack(readable)
            train_lo_hi = (np.nanmin(stacked, axis=(0, 1)), np.nanmax(stacked, axis=(0, 1)))
        kept_names, kept, left_out[split] = _collect(names, results)
        splits[split] = len(_save_split(out_dir, split, kept_names, kept, cond.T, cond.n_channels))

    for split in cond.test_splits:
        test_dir = project_path(cfg.paths.raw_dir).parent / f"{split}_test_split"
        if not (test_dir / "context").is_dir() or not (test_dir / "target").is_dir():
            logger.info(f"[meso_series] {test_dir} not found: test split {split} skipped")
            continue
        pairs = list_test_pairs(test_dir)
        n_frames = sum(TEST_SPLIT_FRAMES[split])
        results = _run(_test_cube_series, [(str(c), str(t), n_frames) for _, c, t in pairs],
                       params, cond.extract_workers, split)
        kept_names, kept, left_out[split] = _collect([n for n, _, _ in pairs], results)
        splits[split] = len(_save_split(out_dir, split, kept_names, kept, n_frames, cond.n_channels))

    lo, hi = train_lo_hi
    stats = {**params,
             "variables": variable_names(cfg),
             "n_train_cubes": splits["train"] + len(left_out["train"]),
             "normalization": "x_norm = 2*(x - lo)/(hi - lo) - 1, clip [-1,1]",
             "lo": [float(v) for v in lo],
             "hi": [float(v) for v in hi]}
    _write_json(out_dir / "stats.json", stats)
    compare_with_cond_stats(cfg, stats)
    save_config(cfg, out_dir / "config.yaml", resolve=True)

    info = {"created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "dataset": dataset_identity(dataset),
            "parameters": params,
            "variables": variable_names(cfg),
            "splits": {split: {"cubes": n, "left_out": len(left_out.get(split, {})),
                               "frames": cond.T if split in ("train", "val") else sum(TEST_SPLIT_FRAMES[split])}
                       for split, n in splits.items()},
            "left_out": left_out}
    _write_json(info_path, info)
    for split, entry in info["splits"].items():
        logger.info(f"[meso_series] {split:8s}: {entry['cubes']} cubes, {entry['frames']} frames"
                    + (f", {entry['left_out']} left out" if entry["left_out"] else ""))
    logger.info(f"[meso_series] lo={np.round(lo, 4).tolist()} hi={np.round(hi, 4).tolist()}")
    return info


def compare_with_cond_stats(cfg: TrainCfg, stats: dict) -> None:
    """
    When the DiT aggregates the weather the same way (data.params.meso_agg),
    cond_stats.json and stats.json must hold the same min / max: they are the
    same statistics of the same cubes. Logged, so a difference is visible.
    """
    path = project_path(cfg.data.params.cond_stats_json)
    if not path.is_file():
        return
    cond_stats = json.loads(path.read_text())
    if (cond_stats.get("meso_agg") != stats["meso_agg"]
            or cond_stats.get("T") != stats["T"]
            or cond_stats.get("meso_center_crop") != stats["center_crop"]
            or cond_stats.get("n_train_files") != stats["n_train_cubes"]):
        logger.info(f"[meso_series] {path} uses other parameters (meso_agg "
                    f"{cond_stats.get('meso_agg')!r}): no comparison")
        return
    same = (np.allclose(cond_stats["meso"]["lo"], stats["lo"])
            and np.allclose(cond_stats["meso"]["hi"], stats["hi"]))
    log = logger.info if same else logger.warning
    log(f"[meso_series] min / max {'equal to' if same else 'DIFFERENT from'} those of {path}")


def ensure_series(cfg: TrainCfg) -> dict:
    """
    series.json of condition.series_dir, extracting the series when missing. An
    extraction made with other parameters, or on another dataset, stops the
    launch instead of being used silently.
    """
    path = series_dir(cfg) / "series.json"
    if not path.is_file():
        logger.info(f"[meso_series] {path} not found: extracting the series")
        return build_series(cfg)
    info = json.loads(path.read_text())
    current = dataset_identity(load_dataset_info(cfg.data.params.dataset_json))
    problems = []
    if info["parameters"] != series_parameters(cfg):
        problems.append(f"parameters {info['parameters']} instead of {series_parameters(cfg)}")
    if info["dataset"]["id"] != current["id"]:
        problems.append(f"dataset id {info['dataset']['id']} instead of {current['id']}")
    if problems:
        raise SystemExit(f"{path}: extracted with other {'; '.join(problems)}.\n"
                         f"Extract them again: python src/dataset/meso_series.py --overwrite")
    return info


def check_series_fit(cfg: TrainCfg, ckpt: dict) -> None:
    """
    The series now in condition.series_dir must be the ones the checkpoint was
    trained on: same dataset (id) and same reduction of the weather. They can
    differ after a new preprocessing with the same name, or an extraction redone
    with --overwrite and other parameters: evaluating on them would mix cubes
    or weather the model never saw that way.
    """
    path = series_dir(cfg) / "series.json"
    if not path.is_file():
        raise SystemExit(f"{path} not found: extract the series (python src/dataset/meso_series.py)")
    info = json.loads(path.read_text())
    trained = ckpt["series"]
    problems = [f"{key} {info['parameters'][key]!r} instead of {trained[key]!r}"
                for key in ("meso_agg", "T", "steps_per_frame", "center_crop")
                if info["parameters"].get(key) != trained.get(key)]
    if info["dataset"]["id"] != (ckpt.get("dataset") or {}).get("id"):
        problems.append(f"dataset id {info['dataset']['id']} instead of {(ckpt.get('dataset') or {}).get('id')}")
    if problems:
        raise SystemExit(f"the series in {path.parent} are not the ones the model was trained on: "
                         f"{'; '.join(problems)}")


# =================================== LOADING ===================================

def load_split(cfg: TrainCfg, split: str) -> tuple[np.ndarray, list]:
    """(series (n_cubes, n_frames, n_channels) in the units of the cubes, cube names) of a split."""
    path = series_dir(cfg) / f"{split}.npz"
    if not path.is_file():
        hint = (f"the folder data/raw/{split}_test_split was missing at extraction time"
                if split in TEST_SPLIT_FRAMES else "extract the series first")
        raise SystemExit(f"{path} not found: {hint} (python src/dataset/meso_series.py --overwrite)")
    with np.load(path) as z:
        return z["series"], z["cubes"].tolist()


def load_stats(cfg: TrainCfg) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) of every variable over the train series, from stats.json."""
    stats = json.loads((series_dir(cfg) / "stats.json").read_text())
    return np.asarray(stats["lo"], dtype=np.float32), np.asarray(stats["hi"], dtype=np.float32)


def normalize(series: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Units of the cubes -> [-1, 1], the formula of cond_stats.json (minmax_pm1)."""
    return minmax_pm1(np.asarray(series, dtype=np.float32), lo, hi).astype(np.float32)


def denormalize(series: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """[-1, 1] -> units of the cubes, the inverse of normalize (without the clip)."""
    span = np.maximum(np.asarray(hi) - np.asarray(lo), 1e-8)
    return (np.asarray(series, dtype=np.float64) + 1.0) / 2.0 * span + np.asarray(lo)


class MesoSeriesDataset(Dataset):
    """
    The normalized series of a split as (context, target) pairs:
        context (n_context, n_channels): the first n_context frames, read by the model
        target  (n_predict, n_channels): the next n_predict frames, to forecast
    """

    def __init__(self, series: np.ndarray, n_context: int, n_predict: int):
        if series.shape[1] < n_context + n_predict:
            raise ValueError(f"series of {series.shape[1]} frames, {n_context} + {n_predict} needed")
        self.context = torch.from_numpy(np.ascontiguousarray(series[:, :n_context]))
        self.target = torch.from_numpy(np.ascontiguousarray(series[:, n_context:n_context + n_predict]))

    def __len__(self):
        return len(self.context)

    def __getitem__(self, idx):
        return self.context[idx], self.target[idx]


def make_loaders(cfg: TrainCfg, normalized: dict) -> dict:
    """DataLoaders of the normalized series {split: array}: shuffled for train, in order otherwise."""
    cond = cfg.condition
    loaders = {}
    for split, series in normalized.items():
        dataset = MesoSeriesDataset(series, cond.n_context, cond.n_predict)
        train = split == "train"
        loaders[split] = DataLoader(dataset,
                                    batch_size=cond.batch_size,
                                    shuffle=train,
                                    # a last batch of one cube would break BatchNorm (PatchMixer)
                                    drop_last=train and len(dataset) > cond.batch_size,
                                    num_workers=cond.num_workers)
    return loaders


# ==================================== CLI ====================================

def main(config_file: str = "configuration_0.yaml"):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfgdir", default=config_file,
                        help="config: a path or the name of a file in registry/configurations/")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="config overrides, e.g. --set condition.extract_workers=16")
    parser.add_argument("--overwrite", action="store_true",
                        help="extract again the series already in condition.series_dir")
    args = parser.parse_args()

    config_path = resolve_config_path(args.cfgdir).resolve()
    use_project_dir()
    cfg = load_config(config_path, overrides=args.set)

    out_dir = series_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(),
                    logging.FileHandler(out_dir / "meso_series.log", mode="w", encoding="utf-8")):
        handler.setFormatter(formatter)
        root.addHandler(handler)
    build_series(cfg, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
