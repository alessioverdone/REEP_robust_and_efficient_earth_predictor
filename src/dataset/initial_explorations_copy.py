"""Initial exploration of the EarthNet2021 dataset.

Logs, for the train split and each test split (iid, ood, extreme, seasonal),
how many tiles (MGRS grid cells) and how many .npz minicube files are present.
"""

import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import earthnet as en
import os
from concurrent.futures import ProcessPoolExecutor

DATASET_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

TRAIN_SPLITS = ["train"]
TEST_SPLITS = ["iid_test_split", "ood_test_split", "extreme_test_split", "seasonal_test_split"]


def process_cube(npz_path):
    """Funzione worker per elaborare un singolo minicube."""
    with np.load(npz_path) as cube:
        # Slice delle bande desiderate (es. 0:3 per b, g, r oppure 0:4 per includere il nif)
        data = cube["highresdynamic"][:, :, 0:3, :]

        local_min = np.nanmin(data)
        local_max = np.nanmax(data)
        nan_count = np.isnan(data).sum()
        total_elems = data.size

        return local_min, local_max, nan_count, total_elems


def main():
    split_dir = DATASET_DIR / "train"
    npz_list = list(split_dir.rglob("*.npz"))
    total_files = len(npz_list)  # Conteggio dei file totali
    min0 = float("inf")
    max0 = float("-inf")
    nan_tot = 0
    total_elements = 0
    # Parallelizza l'elaborazione sfruttando tutti i core della CPU
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_cube, npz_list, chunksize=64)
        for local_min, local_max, nan_count, elems in results:
            if local_min < min0:
                min0 = local_min
            if local_max > max0:
                max0 = local_max
            nan_tot += nan_count
            total_elements += elems
    # Print dei risultati
    print(f"File totali elaborati: {total_files}")
    print(f"Max: {max0}")
    print(f"Min: {min0}")
    print(
        f"Proporzione NaN:"
        f" {nan_tot / total_elements if total_elements > 0 else 0}"
    )


if __name__ == "__main__":
    main()
