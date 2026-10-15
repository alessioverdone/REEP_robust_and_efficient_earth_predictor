"""
dataset.json: the description of one preprocessed dataset.

src/dataset/preprocessing.py is the only place where cubes are selected and
split. It writes the outcome into <preprocessed_dir>/dataset.json: the
parameters it ran with, how the cubes were selected (limit), and the explicit
list of cubes of each split (train / val). Both training phases read
their cubes from there and never split again, so the autoencoder and the
diffusion model see the same cubes and the same locations in each split.

The dataset id is a hash of the split lists and of the normalization: two runs
on the same id saw exactly the same data. The autoencoder saves it in its
checkpoints, and phase 2 refuses a VAE trained on a different dataset.
"""

import hashlib
import json
from pathlib import Path


def dataset_info_path(preprocessed_dir):
    return Path(preprocessed_dir) / "dataset.json"


def compute_dataset_id(splits,
                       stats):
    """
    Short hash of the split lists and of the per-band normalization (lo/hi of
    every temporal version): the two things that define what a model sees.
    """
    payload = {
        "splits": {name: sorted(splits.get(name, [])) for name in ("train", "val")},
        "normalization": {t: {"lo": s["lo"], "hi": s["hi"]} for t, s in sorted(stats.items())},
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()[:12]


def load_dataset_info(path):
    """dataset.json as a dict, with an explicit error when it is missing."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found: run src/dataset/preprocessing.py with the same "
            f"configuration first (it writes dataset.json, stats.json and the "
            f"split files into paths.preprocessed_dir)."
        )
    info = json.loads(path.read_text())
    missing = [key for key in ("name", "id", "splits") if key not in info]
    if missing:
        raise ValueError(f"{path} is incomplete, missing {missing}: rerun the preprocessing.")
    return info


def write_dataset_info(path,
                       info):
    """Atomic write: a half-written dataset.json would look like a valid dataset."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(info, indent=2))
    tmp.replace(path)


def dataset_identity(info):
    """The part of dataset.json that identifies the data: saved in checkpoints."""
    return {"name": info["name"],
            "id": info["id"]}


def split_files(info,
                split,
                raw_dir):
    """
    The .npz files of a split, as paths under raw_dir. dataset.json stores them
    relative to raw_dir, so the dataset survives moving the raw data.
    """
    if split not in info["splits"]:
        raise KeyError(f"dataset.json has no split {split!r}, available: {sorted(info['splits'])}")
    return [Path(raw_dir) / rel for rel in info["splits"][split]]
