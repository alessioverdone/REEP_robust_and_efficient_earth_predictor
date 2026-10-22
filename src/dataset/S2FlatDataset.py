import os
import csv
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL


class S2FlatDataset(Dataset):
    """
    Dataset piatto sopra un .npy prodotto da preprocessing.py.

    L'array ha shape (N, C, H, W): ogni riga e' una singola immagine con le sue
    bande, gia' normalizzata in [-1, 1]. Non c'e' nessuna struttura a minicubo
    da ricostruire.

    Il file viene aperto in memmap, non caricato in RAM: le pagine le gestisce
    la cache del sistema operativo, che le condivide tra i worker invece di
    duplicarle. L'apertura e' lazy perche' un memmap non sopravvive al pickling
    verso i worker.

    Con indices si usa solo un sottoinsieme di righe.
    """

    def __init__(self, npy_path, indices=None):
        self.npy_path = str(npy_path)
        self._x = None
        # La shape si legge dall'header, senza toccare i dati.
        shape = np.load(self.npy_path, mmap_mode="r").shape
        self.n_channels = int(shape[1])
        self.image_size = tuple(int(v) for v in shape[2:])

        if indices is None:
            self.indices = None
            self._len = int(shape[0])
        else:
            self.indices = np.asarray(indices, dtype=np.int64)
            if self.indices.size and self.indices.max() >= shape[0]:
                raise IndexError(
                    f"indice {self.indices.max()} fuori dall'array "
                    f"({shape[0]} righe)"
                )
            self._len = int(self.indices.size)

    def _array(self):
        if self._x is None:
            self._x = np.load(self.npy_path, mmap_mode="r")
        return self._x

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if self.indices is not None:
            idx = int(self.indices[idx])
        # copia esplicita: le righe del memmap sono read-only e torch.from_numpy
        # su un array non scrivibile e' undefined behavior.
        row = np.array(self._array()[idx])
        return {"tif": torch.from_numpy(row).float()}


def rows_by_location(manifest_path,
                     npy_path):
    """
    Row ranges of npy_path grouped by *location*: {location: [(start, end), ...]}.

    Il manifest registra, per ogni cubo, quanti istanti sono finiti nell'array
    e in che ordine: le righe di un cubo sono consecutive, quindi basta
    scorrere i cubi accumulando gli offset.
    """
    man = json.loads(Path(manifest_path).read_text())

    target = Path(npy_path).name
    entry = next(
        (v for v in man.values() if Path(v["path"]).name == target), None
    )
    if entry is None:
        raise KeyError(
            f"{target} non e' presente in {manifest_path}. "
            f"Voci disponibili: {sorted(man)}"
        )

    per_loc = {}
    row = 0
    for cube in entry["cubes"]:
        n = len(cube["frames"])
        per_loc.setdefault(Path(cube["path"]).parent.name, []).append((row, row + n))
        row += n
    return per_loc


def locate_rows(manifest_path,
                npy_path,
                rows):
    """
    The cube each row of npy_path comes from: [(cube .npz path, time index in it), ...],
    one per row of `rows`, in the same order (the time index is the one of
    highresdynamic, which is where the cloud probability lives). The rows of a cube are
    consecutive and `frames` lists their time indices (see rows_by_location).
    """
    man = json.loads(Path(manifest_path).read_text())
    target = Path(npy_path).name
    entry = next((v for v in man.values() if Path(v["path"]).name == target), None)
    if entry is None:
        raise KeyError(f"{target} non e' presente in {manifest_path}")

    cubes = entry["cubes"]
    ends = np.cumsum([len(c["frames"]) for c in cubes])
    located = []
    for row in rows:
        k = int(np.searchsorted(ends, int(row), side="right"))
        start = int(ends[k - 1]) if k else 0
        located.append((cubes[k]["path"], int(cubes[k]["frames"][int(row) - start])))
    return located


def split_composition(per_loc,
                      locations=None):
    """Images, cubes and locations of a split (all of per_loc if locations is None)."""
    locations = sorted(per_loc) if locations is None else sorted(locations)
    spans = [span for loc in locations for span in per_loc[loc]]
    return {
        "n_images": int(sum(b - a for a, b in spans)),
        "n_cubes": len(spans),
        "n_locations": len(locations),
    }


def file_composition(npy_path,
                     dataset,
                     manifest=None):
    """
    Composition of a split that is a whole file: images from the dataset,
    cubes and locations from the manifest when the file is listed there.
    """
    composition = {"n_images": len(dataset)}
    if manifest is not None:
        try:
            composition = split_composition(rows_by_location(manifest,
                                                             npy_path))
        except KeyError:
            pass
    return composition


def make_dataloaders(train_npy,
                     val_npy,
                     manifest=None,
                     batch_size=32,
                     num_workers=4,
                     prefetch_factor=2,
                     persistent_workers=True,
                     pin_memory=True):
    """
    Costruisce i DataLoader sui due file scritti dal preprocessing, uno per
    split: the split is decided there (dataset.json), nothing is split here.

    Returns (loaders, composition): loaders is {"train", "val"}, composition
    holds images, cubes and locations of each split, plus the file each split
    comes from.
    """
    for name, path in (("train", train_npy), ("val", val_npy)):
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"{name} split not found: {path}. Run src/dataset/preprocessing.py "
                f"with the same configuration (without --stats-only: phase 1 needs the .npy files)."
            )

    train_ds = S2FlatDataset(train_npy)
    val_ds = S2FlatDataset(val_npy)
    composition = {
        "train": file_composition(train_npy,
                                  train_ds,
                                  manifest),
        "val": file_composition(val_npy,
                                val_ds,
                                manifest),
    }
    composition["train"]["source"] = str(train_npy)
    composition["val"]["source"] = str(val_npy)

    if val_ds.n_channels != train_ds.n_channels:
        raise ValueError(
            f"train e val hanno un numero di bande diverso: "
            f"{train_ds.n_channels} vs {val_ds.n_channels}"
        )
    if len(val_ds) == 0:
        raise ValueError("il set di val e' vuoto")

    common = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    loaders = {
        "train": DataLoader(train_ds,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True,
                            **common),
        "val": DataLoader(val_ds,
                          batch_size=batch_size,
                          shuffle=False,
                          drop_last=False,
                          **common),
    }

    print(f"Train: {len(train_ds)} immagini {train_ds.image_size}, "
          f"{train_ds.n_channels} bande, {len(loaders['train'])} batch")
    print(f"Val  : {len(val_ds)} immagini, {len(loaders['val'])} batch")

    return loaders, composition
