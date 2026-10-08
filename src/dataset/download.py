"""
Scarica i minicubi EarthNet2021 in data/raw/, la cartella che il config indica
come sola lettura (paths.raw_dir). Da lanciare una volta sola: dopo, tutto
quello che si produce va in data/preprocessed/ (vedi src/dataset/preprocessing.py).
"""

import earthnet as en

from src.config import PROJECT_DIR

RAW_DIR = PROJECT_DIR / "data" / "raw"

if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Scarico EarthNet2021 in {RAW_DIR}")
    en.Downloader.get(RAW_DIR, "all")
