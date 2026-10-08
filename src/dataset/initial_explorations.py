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

log = logging.getLogger(__name__)

logging.basicConfig(
    filename=Path(__file__).parent / 'exploration.log', 
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

DATASET_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

TRAIN_SPLITS = ["train"]
TEST_SPLITS = ["iid_test_split", "ood_test_split", "extreme_test_split", "seasonal_test_split"]


def count_tiles_and_npz(tile_root: Path) -> tuple[int, int]:
    """Count tile subdirectories and total .npz files under tile_root."""
    if not tile_root.is_dir():
        return 0, 0
    tiles = [d for d in tile_root.iterdir() if d.is_dir()]
    n_npz = sum(1 for _ in tile_root.rglob("*.npz"))
    return len(tiles), n_npz


def explore_train_split(split: str) -> None:
    split_dir = DATASET_DIR / split
    n_tiles, n_npz = count_tiles_and_npz(split_dir)
    log.info("[%s] tiles=%d npz_files=%d\n", split, n_tiles, n_npz)


def explore_test_split(split: str) -> None:
    split_dir = DATASET_DIR / split
    for subset in ["context", "target"]:
        n_tiles, n_npz = count_tiles_and_npz(split_dir / subset)
        if split == "seasonal_test_split" and subset == "target":
            log.info("[%s/%s] tiles=%d npz_files=%d\n", split, subset, n_tiles, n_npz)
        else:
            log.info("[%s/%s] tiles=%d npz_files=%d", split, subset, n_tiles, n_npz)


def explore_npz(split: str) -> None:
    split_dir = DATASET_DIR / split
    npz = list(split_dir.rglob("*.npz"))[0] # All files access
    #npz = next((f for f in next((d for d in split_dir.iterdir() if d.is_dir()), None).iterdir()), None)
    #npz = split_dir.rglob("*.npz")[0] # Get the first .npz file
    cube = np.load(npz)
    for key in cube.keys():
        data = cube[key]
        log.info(f"Variable {key}, shape: {data.shape}, dtype: {data.dtype}")

def show_img(npz_file: Path) -> None:
    cube = np.load(npz_file)
    high_res = cube["highresdynamic"]
    rgb_img = high_res[:, :, [2, 1, 0], 0] # red, green, blue
    rgb_clean = np.nan_to_num(rgb_img, nan=0.0)
    rgb_vis = np.clip(rgb_clean, 0, 1)
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_vis)
    plt.title("Img Sentinel-2 (T=0)")
    plt.axis("off")
    plt.show()
    plt.close()

def get_npz_coordinates(npz_file: Path, split: str = None) -> None:
    if split is None:
        cubename = os.path.basename(npz_file)  # es: "29SND_2017-06-14_cube.npz"
    else:
        cubename = os.path.basename(npz_file)[len(split)+1:]
    coords = en.get_coords_from_cube(cubename, return_meso=False)
    log.info("Coordinates of the cube %s: %s", cubename, coords)
    coords = en.get_coords_from_cube(cubename, return_meso=True)
    log.info("Coordinates of the cube %s: %s", cubename, coords)

def get_tile_coordinates(tile_path: str) -> None:
    tilename = os.path.basename(tile_path)  # es: "29SND"
    coords = en.get_coords_from_tile(tilename)
    log.info("Coordinates of the tile %s: %s", tilename, coords)

def main() -> None:
    log.info("Dataset root: %s", DATASET_DIR)

    log.info("== Train ==")
    for split in TRAIN_SPLITS:
        explore_train_split(split)

    log.info("== Test splits ==")
    for split in TEST_SPLITS:
        explore_test_split(split)
    
    log.info("== NPZ file exploration ==\n")
    log.info("== Train ==")
    for split in TRAIN_SPLITS:
        explore_npz(split)
    log.info("== Test ==")
    for split in TEST_SPLITS:
        log.info("== %s ==", split)
        log.info("== Context ==")
        explore_npz(split+"/context" )
        log.info("== Target ==")
        explore_npz(split+"/target" )

    for split in TRAIN_SPLITS:
        tile = [f.name for f in (DATASET_DIR / split).iterdir() if f.is_dir()][3]
        get_tile_coordinates(tile)
        npz = list((DATASET_DIR / split).rglob("*.npz"))[100]
        get_npz_coordinates(npz)
        #show_img(npz)

    for split in TEST_SPLITS:
        for subset in ["context", "target"]:
            tile = [f.name for f in (DATASET_DIR / split / subset).iterdir() if f.is_dir()][3]
            get_tile_coordinates(tile)
            npz = list((DATASET_DIR / split / subset).rglob("*.npz"))[100]
            get_npz_coordinates(npz, subset)
            #show_img(npz)

    # Funzione time series prende in input anche le predizioni
    #en.en.cube_ndvi_timeseries(predpath, targpath)
    #plt.show
    #plt.close()

    npz_path = list((DATASET_DIR / "train").rglob("*.npz"))[100]
    fig = en.cube_gallery(npz_path, variable = "ndvi") #variable = rgb, ndvi, rr, pp, tg, tn, tx; vegetation_mask Bool per coprire zone senza vegetazione; cloud_mask per coprire zone nuvolese; save_path Bool;
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    main()
