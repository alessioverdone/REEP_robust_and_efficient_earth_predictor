#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataLoader del modello diffusivo video, direttamente sui minicubi EarthNet2021.

Ogni item del dataset e' un minicubo .npz completo, cioe' una sequenza di 30
istanti: i primi 10 sono il contesto passato, i 20 successivi sono il target da
predire. Il .npz contiene quattro array:

    highresdynamic : (128, 128, 7, 30)  riflettanze Sentinel-2, passo 5 giorni
    highresstatic  : (128, 128, 1)      DEM, invariante nel tempo
    mesodynamic    : (80, 80, 5, 150)   variabili meteo, passo 1 giorno
    mesostatic     : (80, 80, 1)        non usato

Il .npy prodotto da preprocessing.py contiene solo i primi 10 istanti di ogni
cubo e serve al training dell'autoencoder; qui servono tutti e 30 gli istanti,
quindi si legge dai .npz e si riapplica *la stessa* normalizzazione, in modo che
i latenti prodotti dal VAE restino nella distribuzione su cui e' stato allenato.

Cosa fa questo file, variabile per variabile
--------------------------------------------
highresdynamic
    Prime 4 bande (blue, green, red, NIR), primi 30 istanti, layout portato a
    (T, C, H, W). I valori mancanti sono riempiti con la media dell'intorno 3x3
    iterata (stessa fill_spatial del preprocessing) e la normalizzazione usa i
    lo/hi per banda salvati in stats.json:

        x_norm = 2 * (x - lo) / (hi - lo) - 1,  poi clip in [-1, 1]

mesodynamic
    Allineamento temporale 1:5. I 150 passi giornalieri sono divisi in 30
    blocchi da 5 giorni: il blocco i copre i giorni [5i, 5i+5) e corrisponde
    all'istante i di highresdynamic.

    Dentro il blocco, l'acquisizione Sentinel-2 cade sull'ULTIMO giorno, cioe'
    5i+4 (meso_agg="last", il default). Due riscontri:
      - la copertura nuvolosa del frame i correla con il meteo del giorno 5i+j
        e il massimo, per tutte e 5 le variabili, e' a j=4 (su 300 cubi:
        precipitazione +0.25, pressione -0.24, temperatura massima -0.24);
      - le date nel nome del file distano sempre 149 giorni, quindi il giorno
        149 e' la data di fine: con 5i+4 il frame 29 ci cade esattamente,
        con 5i avanzerebbero 4 giorni di meteo non associati a nessun frame.
    Con meso_agg="mean"/"sum" si aggregano invece i 5 giorni del blocco (dopo
    la normalizzazione min-max le due opzioni danno lo stesso risultato, visto
    che sum = 5*mean); "first" prende il giorno 5i.
    Allineamento spaziale: dei 80x80 pixel si tengono solo i 4 centrali (crop
    2x2), mediati tra loro, perche' e' quella la porzione che copre il minicubo
    ad alta risoluzione. Risultato: (T, 5).
    Con meso_time_span="future" si restituiscono solo i 20 passi allineati ai 20
    frame target; il default "all" restituisce tutti e 30 i passi, che e' quello
    che il DiT si aspetta (climate_var deve avere T = image_time_length).

highresstatic
    Gia' allineata spazialmente (128x128) e costante nel tempo: si porta a
    (1, H, W) e si normalizza min-max. I buchi sono chiusi dalla stessa
    fill_spatial di highresdynamic: la quota e' una superficie continua, quindi
    i vicini la stimano meglio di qualunque statistica globale della tile.

Split
-----
Nothing is selected or split here. The cubes of each split (train / val /
test, by location) are listed in dataset.json, written by preprocessing.py
after dropping the cubes with entirely missing images: they are the same
cubes the autoencoder was trained, validated and tested on (see
src/dataset/dataset_info.py). With max_bad_frames the cubes with too many
rejected frames can also be excluded (see keep_map in stats.json).

Normalizzazione delle condizioni
--------------------------------
mesodynamic e highresstatic non hanno statistiche precalcolate, quindi si
scandisce una volta il solo split di *train* e si salvano i min/max per banda in
un JSON (cond_stats.json), riusato da tutti gli split. I min/max di mesodynamic
sono calcolati sulla grandezza effettivamente data alla rete (crop 2x2 mediato e
aggregato a 5 giorni), non sul cubo grezzo: cosi' il range normalizzato copre
davvero [-1, 1].

preprocessing.py writes cond_stats.json already. To recompute it by hand:

    python src/dataset/dataloader_s2_dif.py --compute-cond-stats --workers 8 \\
        --dataset-dir data/preprocessed/<name>

Smoke test su un batch:

    python src/dataset/dataloader_s2_dif.py --smoke-test --dataset-dir data/preprocessed/<name>

Il modulo non stampa nulla: scrive tutto su logging.getLogger(__name__). Lanciato
da riga di comando il log va in console e in logs/dataloader_s2_dif.log
(--log-file per cambiarlo); lanciato dal trainer finisce nel file di log del run.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.dataset.dataset_info import load_dataset_info, split_files

# Niente print: tutto passa da qui, cosi' finisce anche nel file di log che il
# run aggancia al logger root (artifacts/train.log, vedi src/training/runtime.py).
logger = logging.getLogger(__name__)

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

try:
    import pytorch_lightning as pl
    _DataModuleBase = pl.LightningDataModule
except Exception:  # pytorch_lightning non indispensabile per usare il Dataset
    class _DataModuleBase:  # type: ignore[no-redef]
        def __init__(self):
            pass


# Chiavi degli array dentro i .npz EarthNet2021
HIGHRES_DYNAMIC_KEY = "highresdynamic"
HIGHRES_STATIC_KEY = "highresstatic"
MESO_DYNAMIC_KEY = "mesodynamic"

EXPECTED_HW = (128, 128)


# ---------------------------------------------------------------------------
# Elenco dei cubi e split per location
# ---------------------------------------------------------------------------
def list_cubes(root: Path) -> List[Path]:
    """Tutti i .npz sotto root, in ordine deterministico."""
    return sorted(Path(root).rglob("*.npz"))


def location_of(path: Path) -> str:
    """La location e' la cartella che contiene il cubo (es. '29SND')."""
    return Path(path).parent.name


# ---------------------------------------------------------------------------
# Riempimento dei valori mancanti (identico a preprocessing.py)
# ---------------------------------------------------------------------------
def _box_sum3(a: np.ndarray) -> np.ndarray:
    """Somma sull'intorno 3x3 (centro incluso) di uno stack (K, H, W)."""
    _, h, w = a.shape
    p = np.pad(a, ((0, 0), (1, 1), (1, 1)))
    out = np.zeros_like(a)
    for dy in range(3):
        for dx in range(3):
            out += p[:, dy : dy + h, dx : dx + w]
    return out


def fill_spatial(x: np.ndarray, max_iter: int = 256) -> int:
    """
    Riempie in place i valori non finiti di x (T, B, H, W): ogni pixel mancante
    prende la media dei pixel validi nel suo intorno 3x3, iterando finche' le
    chiazze non si chiudono. Ritorna il numero di valori rimasti non riempibili
    (slice interamente mancanti), che il chiamante azzera.
    """
    k = x.reshape(-1, x.shape[2], x.shape[3])  # vista (T*B, H, W)
    bad = ~np.isfinite(k)
    idx = np.where(bad.reshape(k.shape[0], -1).any(axis=1))[0]
    if idx.size == 0:
        return 0

    sub = np.array(k[idx], dtype=np.float32)
    m = ~np.isfinite(sub)

    vals = np.where(m, 0.0, sub).astype(np.float32)
    valid = (~m).astype(np.float32)

    for _ in range(max_iter):
        if not m.any():
            break
        s = _box_sum3(vals)
        n = _box_sum3(valid)
        fillable = m & (n > 0)
        if not fillable.any():
            break  # nessun vicino valido: slice interamente mancante
        vals[fillable] = s[fillable] / n[fillable]
        valid[fillable] = 1.0
        m &= ~fillable

    unfixable = int(m.sum())
    vals[m] = np.nan  # buchi non chiusi: li azzera il chiamante
    k[idx] = vals
    return unfixable


def minmax_pm1(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """x -> 2*(x - lo)/(hi - lo) - 1, clippato in [-1, 1]. lo/hi broadcastabili."""
    span = np.maximum(hi - lo, 1e-8)
    out = 2.0 * (x - lo) / span - 1.0
    np.clip(out, -1.0, 1.0, out=out)
    return out


# ---------------------------------------------------------------------------
# Lettura ed elaborazione delle singole variabili
# ---------------------------------------------------------------------------
def load_highres_dynamic(
    path: Path,
    n_bands: int = 4,
    n_time: int = 30,
) -> Tuple[np.ndarray, int]:
    """
    highresdynamic (H, W, C, T) -> (T, n_bands, H, W) float32, piu' il numero di
    valori che fill_spatial non e' riuscito a riempire.

    I buchi sparsi sono chiusi dall'interpolazione 3x3, esattamente come nel
    preprocessing. Quello che resta non finito (immagini interamente vuote) NON
    viene toccato qui: lo azzera il chiamante *dopo* la normalizzazione, cosi'
    un buco residuo finisce a 0 in [-1,1], cioe' a meta' scala, che e' la stessa
    convenzione di write_chunk() nel preprocessing. Azzerarlo prima lo manderebbe
    invece a -1, il fondo scala.

    Nessuna normalizzazione: la applica il chiamante con i lo/hi di stats.json.
    """
    with np.load(path) as z:
        hd = z[HIGHRES_DYNAMIC_KEY]

    if hd.ndim != 4 or hd.shape[:2] != EXPECTED_HW:
        raise ValueError(f"{path}: {HIGHRES_DYNAMIC_KEY} shape inattesa {hd.shape}")
    if hd.shape[2] < n_bands or hd.shape[3] < n_time:
        raise ValueError(
            f"{path}: {HIGHRES_DYNAMIC_KEY} ha {hd.shape[2]} bande e {hd.shape[3]} "
            f"istanti, ne servono almeno {n_bands} e {n_time}"
        )

    x = hd[:, :, :n_bands, :n_time].astype(np.float32)      # (H, W, B, T)
    x = np.ascontiguousarray(np.transpose(x, (3, 2, 0, 1)))  # (T, B, H, W)

    unfixable = fill_spatial(x)
    return x, unfixable


def load_highres_static(path: Path) -> Tuple[np.ndarray, int]:
    """
    highresstatic (H, W, C) -> (C, H, W) float32, piu' il numero di valori che
    fill_spatial non e' riuscito a riempire.

    I buchi del DEM sono chiusi con la stessa interpolazione 3x3 iterata usata
    per highresdynamic: la quota e' una superficie continua, quindi i pixel
    vicini la stimano molto meglio della media della banda, che appiattirebbe un
    buco su un versante al valore medio dell'intera tile.

    Quello che resta non finito (banda senza nemmeno un pixel valido, quindi
    niente da cui propagare) NON viene toccato qui: lo azzera il chiamante
    *dopo* la normalizzazione, cosi' finisce a 0 in [-1,1], cioe' a meta' scala.
    Azzerarlo prima lo manderebbe a fondo scala, visto che lo del DEM e' ben
    sopra 0.
    """
    with np.load(path) as z:
        hs = z[HIGHRES_STATIC_KEY]

    if hs.ndim == 2:
        hs = hs[:, :, None]
    if hs.ndim != 3 or hs.shape[:2] != EXPECTED_HW:
        raise ValueError(f"{path}: {HIGHRES_STATIC_KEY} shape inattesa {hs.shape}")

    x = np.ascontiguousarray(np.transpose(hs.astype(np.float32), (2, 0, 1)))

    # fill_spatial vuole (T, B, H, W) e lavora in place: l'asse temporale che il
    # DEM non ha si aggiunge come vista, cosi' la scrittura torna su x.
    unfixable = fill_spatial(x[None])
    return x, unfixable


def center_crop_2d(x: np.ndarray, crop: int) -> np.ndarray:
    """
    Ritaglio centrale crop x crop sulle prime due dimensioni di x (H, W, ...).
    Con H pari e crop pari il ritaglio e' esattamente centrato.
    """
    h, w = x.shape[0], x.shape[1]
    if crop > h or crop > w:
        raise ValueError(f"crop {crop} piu' grande dell'array {h}x{w}")
    y0 = (h - crop) // 2
    x0 = (w - crop) // 2
    return x[y0 : y0 + crop, x0 : x0 + crop, ...]


def load_meso_dynamic(
    path: Path,
    n_channels: int = 5,
    n_time: int = 30,
    steps_per_frame: int = 5,
    center_crop: int = 2,
    agg: str = "last",
) -> np.ndarray:
    """
    mesodynamic (H, W, C, T_daily) -> (n_time, n_channels) float32.

    - spazio: crop centrale center_crop x center_crop, poi media dei pixel;
    - tempo : i primi n_time*steps_per_frame giorni sono divisi in n_time
      blocchi consecutivi; il blocco i copre i giorni [5i, 5i+5) e corrisponde
      all'istante i di highresdynamic. agg decide cosa si prende dentro il
      blocco: "last" (default, il giorno dell'acquisizione S2, 5i+4), "first"
      (5i), "mean"/"sum" (aggregazione dei 5 giorni).

    I NaN si propagano: se dei 4 pixel centrali qualcuno e' valido, la media li
    ignora; se sono tutti mancanti resta NaN, e ad azzerarlo e' il chiamante
    dopo la normalizzazione (quindi a meta' scala, non a fondo scala).

    Nessuna normalizzazione: la applica il chiamante con i lo/hi di cond_stats.
    """
    with np.load(path) as z:
        md = z[MESO_DYNAMIC_KEY]

    if md.ndim != 4:
        raise ValueError(f"{path}: {MESO_DYNAMIC_KEY} shape inattesa {md.shape}")
    if md.shape[2] < n_channels:
        raise ValueError(
            f"{path}: {MESO_DYNAMIC_KEY} ha {md.shape[2]} variabili, ne servono {n_channels}"
        )

    need = n_time * steps_per_frame
    if md.shape[3] < need:
        raise ValueError(
            f"{path}: {MESO_DYNAMIC_KEY} ha {md.shape[3]} passi giornalieri, "
            f"ne servono {need} per allineare {n_time} frame a passo {steps_per_frame}"
        )

    x = md[:, :, :n_channels, :need].astype(np.float32)   # (H, W, C, need)
    x = center_crop_2d(x, center_crop)                    # (crop, crop, C, need)

    # nanmean su una fetta tutta NaN e' legittimo qui (da' NaN, gestito a valle),
    # ma numpy lo segnala con un RuntimeWarning per ogni chiamata: si silenzia,
    # altrimenti il log si riempirebbe a ogni item.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")

        # Media sui 4 pixel centrali, ignorando eventuali NaN.
        x = np.nanmean(x.reshape(-1, n_channels, need), axis=0)  # (C, need)

        x = x.reshape(n_channels, n_time, steps_per_frame)
        if agg == "mean":
            x = np.nanmean(x, axis=2)
        elif agg == "sum":
            x = np.nansum(x, axis=2)
        elif agg == "last":
            x = x[:, :, -1]
        elif agg == "first":
            x = x[:, :, 0]
        else:
            raise ValueError(f"meso_agg sconosciuto: {agg}")

    return np.ascontiguousarray(x.T, dtype=np.float32)  # (n_time, C)


# ---------------------------------------------------------------------------
# Statistiche di normalizzazione
# ---------------------------------------------------------------------------
def load_highres_stats(
    stats_json: str,
    time_key: str = "10",
    n_bands: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    lo/hi per banda di highresdynamic, dal stats.json del preprocessing.

    time_key sceglie la versione temporale: va usata la stessa su cui e' stato
    allenato l'autoencoder (train_T10_b4_128.npy -> "10"), cosi' i 30 frame
    finiscono nella stessa scala dei 10 visti dal VAE.
    """
    payload = json.loads(Path(stats_json).read_text())
    stats = payload["stats"]
    if time_key not in stats:
        raise KeyError(
            f"{stats_json} non contiene le statistiche per T={time_key}. "
            f"Disponibili: {sorted(stats)}"
        )
    lo = np.asarray(stats[time_key]["lo"], dtype=np.float32)[:n_bands]
    hi = np.asarray(stats[time_key]["hi"], dtype=np.float32)[:n_bands]
    if lo.size != n_bands or hi.size != n_bands:
        raise ValueError(
            f"{stats_json}: attese {n_bands} bande, trovate {lo.size}"
        )
    return lo.reshape(1, n_bands, 1, 1), hi.reshape(1, n_bands, 1, 1)


def load_keep_map(payload: Dict[str, Any]) -> Dict[str, List[int]]:
    """
    keep_map del preprocessing, riclassificato per nome file: dice quali istanti
    di ogni cubo hanno superato i controlli di qualita'. Serve solo per filtrare
    i cubi troppo rovinati (vedi max_bad_frames).
    """
    keep = payload.get("keep_map", {}) or {}
    return {Path(k).name: v for k, v in keep.items()}


# --- scansione per i min/max di mesodynamic e highresstatic -----------------
_SCAN_CFG: Dict[str, Any] = {}


def _init_scan(cfg: Dict[str, Any]) -> None:
    _SCAN_CFG.update(cfg)
    # I worker di multiprocessing partono con il logging non configurato (su
    # Windows sono processi nuovi): senza questo i loro warning sparirebbero.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            level=logging.INFO,
        )


def _scan_one(path_str: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """min/max per banda di (meso aggregato, static) su un singolo cubo."""
    cfg = _SCAN_CFG
    p = Path(path_str)
    try:
        meso = load_meso_dynamic(
            p,
            n_channels=cfg["meso_channels"],
            n_time=cfg["T"],
            steps_per_frame=cfg["steps_per_frame"],
            center_crop=cfg["meso_center_crop"],
            agg=cfg["meso_agg"],
        )
        static, _ = load_highres_static(p)
    except Exception as exc:
        logger.warning(f"[stats] cubo saltato, {p.name}: {exc}")
        return None

    # nanmin/nanmax: sia meso (4 pixel centrali tutti mancanti) sia static
    # (banda senza pixel validi) possono arrivare qui con dei NaN, e un min
    # normale li propagherebbe dentro cond_stats.json rovinando la scala di
    # tutti i cubi. Su una banda interamente NaN restano NaN, che _accumulate
    # ignora con fmin/fmax.
    flat = static.reshape(static.shape[0], -1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        return (
            np.nanmin(meso, axis=0),
            np.nanmax(meso, axis=0),
            np.nanmin(flat, axis=1),
            np.nanmax(flat, axis=1),
        )


def compute_condition_stats(
    files: Sequence[Path],
    meso_channels: int = 5,
    T: int = 30,
    steps_per_frame: int = 5,
    meso_center_crop: int = 2,
    meso_agg: str = "last",
    workers: int = 0,
) -> Dict[str, Any]:
    """
    Min/max per banda di mesodynamic (dopo crop 2x2 + aggregazione a 5 giorni) e
    di highresstatic, su un elenco di cubi (va passato il solo split di train).
    """
    cfg = dict(
        meso_channels=meso_channels,
        T=T,
        steps_per_frame=steps_per_frame,
        meso_center_crop=meso_center_crop,
        meso_agg=meso_agg,
    )

    meso_lo = np.full(meso_channels, np.inf, dtype=np.float64)
    meso_hi = np.full(meso_channels, -np.inf, dtype=np.float64)
    stat_lo: Optional[np.ndarray] = None
    stat_hi: Optional[np.ndarray] = None

    paths = [str(f) for f in files]
    n_ok = 0
    t0 = time.time()

    def _accumulate(res):
        nonlocal meso_lo, meso_hi, stat_lo, stat_hi, n_ok
        if res is None:
            return
        m_lo, m_hi, s_lo, s_hi = res
        # fmin/fmax e non minimum/maximum: ignorano il NaN di un cubo senza
        # valori validi invece di propagarlo a tutti gli altri.
        meso_lo = np.fmin(meso_lo, m_lo)
        meso_hi = np.fmax(meso_hi, m_hi)
        stat_lo = s_lo.copy() if stat_lo is None else np.fmin(stat_lo, s_lo)
        stat_hi = s_hi.copy() if stat_hi is None else np.fmax(stat_hi, s_hi)
        n_ok += 1

    # Riga di avanzamento ogni 5%: una barra con \r nel file di log sarebbe
    # illeggibile, e comunque non si vedrebbe su piu' processi.
    every = max(1, len(paths) // 20)

    if workers and workers > 1:
        with mp.Pool(processes=workers, initializer=_init_scan, initargs=(cfg,)) as pool:
            for i, res in enumerate(pool.imap_unordered(_scan_one, paths, chunksize=4), 1):
                _accumulate(res)
                _log_progress(i, len(paths), t0, every)
    else:
        _init_scan(cfg)
        for i, p in enumerate(paths, 1):
            _accumulate(_scan_one(p))
            _log_progress(i, len(paths), t0, every)

    if n_ok == 0 or stat_lo is None:
        raise RuntimeError("nessun cubo leggibile per il calcolo delle statistiche")

    return {
        "n_cubes": n_ok,
        # Quanti cubi *componevano* lo split, non quanti erano leggibili: e' la
        # chiave con cui ensure_condition_stats capisce se la cache e' stata
        # prodotta su un altro split. n_cubes non basta, cala se un cubo e'
        # illeggibile e farebbe ricalcolare a ogni avvio.
        "n_train_files": len(paths),
        "T": T,
        "steps_per_frame": steps_per_frame,
        "meso_center_crop": meso_center_crop,
        "meso_agg": meso_agg,
        "normalization": "x_norm = 2*(x - lo)/(hi - lo) - 1, clip [-1,1]",
        "meso": {
            "lo": [float(v) for v in meso_lo],
            "hi": [float(v) for v in meso_hi],
        },
        "static": {
            "lo": [float(v) for v in stat_lo],
            "hi": [float(v) for v in stat_hi],
        },
    }


def _log_progress(done: int, total: int, t0: float, every: int) -> None:
    if done % every and done != total:
        return
    elapsed = time.time() - t0
    eta = elapsed / done * (total - done) if done else 0.0
    logger.info(
        f"statistiche condizioni: {done}/{total} cubi "
        f"| {elapsed / 60:.1f} min trascorsi | ETA {eta / 60:.1f} min"
    )


def ensure_condition_stats(
    cache_path: str,
    train_files: Sequence[Path],
    meso_channels: int = 5,
    T: int = 30,
    steps_per_frame: int = 5,
    meso_center_crop: int = 2,
    meso_agg: str = "last",
    workers: int = 0,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Carica cond_stats.json se c'e' (ed e' coerente), altrimenti lo calcola.

    Attenzione al multi-GPU: setup() gira su ogni rank, quindi senza cache tutti
    i processi calcolano le stesse statistiche e scrivono lo stesso file. Il
    risultato e' identico e la scrittura qui e' atomica, quindi non si corrompe
    niente, ma il lavoro viene fatto N volte. Con `--compute-cond-stats` prima
    del training lo si fa una volta sola (e in parallelo).
    """
    cache = Path(cache_path)

    if cache.is_file() and not force:
        payload = json.loads(cache.read_text())
        same = (
            payload.get("T") == T
            and payload.get("steps_per_frame") == steps_per_frame
            and payload.get("meso_center_crop") == meso_center_crop
            and payload.get("meso_agg") == meso_agg
            and len(payload.get("meso", {}).get("lo", [])) == meso_channels
            # Anche lo split conta: sono min/max sul train, e un train diverso
            # puo' avere estremi diversi. Senza questo controllo una cache
            # prodotta su uno split piu' piccolo verrebbe riusata in silenzio, e
            # i valori fuori dal suo range finirebbero clippati a +-1.
            and payload.get("n_train_files") == len(train_files)
        )
        if same:
            logger.info(f"Statistiche condizioni riusate da {cache}")
            return payload
        logger.warning(
            f"{cache} e' stato prodotto con parametri o split diversi "
            f"(T/agg/crop/numero di cubi di train): lo ricalcolo."
        )

    logger.info(
        f"Calcolo min/max di {MESO_DYNAMIC_KEY} e {HIGHRES_STATIC_KEY} su "
        f"{len(train_files)} cubi di train..."
    )
    payload = compute_condition_stats(
        train_files,
        meso_channels=meso_channels,
        T=T,
        steps_per_frame=steps_per_frame,
        meso_center_crop=meso_center_crop,
        meso_agg=meso_agg,
        workers=workers,
    )
    # Scrittura atomica: se piu' rank arrivano qui insieme, ognuno scrive il suo
    # temporaneo e l'ultimo os.replace vince. Senza, si otterrebbe un JSON
    # troncato o con contenuti interlacciati.
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, cache)
    logger.info(f"Statistiche condizioni salvate in {cache}")
    logger.info(f"  meso  lo={_fmt(payload['meso']['lo'])} hi={_fmt(payload['meso']['hi'])}")
    logger.info(f"  static lo={_fmt(payload['static']['lo'])} hi={_fmt(payload['static']['hi'])}")
    return payload


def _fmt(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{v:.4g}" for v in values) + "]"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class S2DiffusionCubeDataset(Dataset):
    """
    Un item = un minicubo, cioe' una sequenza completa di T istanti.

        imgs                     : (T, n_channels, H, W)   in [-1, 1]
        highres_condition_image  : (static_channels, H, W) in [-1, 1]
        meso_condition_image     : (T_meso, meso_channels) in [-1, 1]

    dove T = n_context + n_predict (10 + 20) e T_meso e' T oppure n_predict,
    a seconda di meso_time_span. Il modello usa i primi n_context istanti di
    imgs come passato e gli ultimi n_predict come target.

    With resize, imgs and highres_condition_image are downsampled to
    resize x resize after normalization, with area interpolation: the same
    operation the autoencoder is trained with (resize_images in
    src/model/autoencoder_utils.py), so the frozen VAE sees the same images.
    """

    def __init__(
        self,
        files: Sequence[Path],
        highres_lo: np.ndarray,
        highres_hi: np.ndarray,
        meso_lo: np.ndarray,
        meso_hi: np.ndarray,
        static_lo: np.ndarray,
        static_hi: np.ndarray,
        T: int = 30,
        n_predict: int = 20,
        n_channels: int = 4,
        meso_channels: int = 5,
        steps_per_frame: int = 5,
        meso_center_crop: int = 2,
        meso_agg: str = "last",
        meso_time_span: str = "all",
        do_hflip: bool = False,
        return_paths: bool = False,
        dtype: torch.dtype = torch.float16,
        resize: Optional[int] = None,
    ):
        super().__init__()

        if meso_time_span not in ("all", "future"):
            raise ValueError(f"meso_time_span deve essere 'all' o 'future', non {meso_time_span}")

        self.files = [Path(f) for f in files]
        if not self.files:
            raise RuntimeError("elenco di cubi vuoto")

        self.T = int(T)
        self.n_predict = int(n_predict)
        self.n_context = self.T - self.n_predict
        if self.n_context <= 0:
            raise ValueError(f"n_predict ({n_predict}) deve essere minore di T ({T})")

        self.n_channels = int(n_channels)
        self.meso_channels = int(meso_channels)
        self.steps_per_frame = int(steps_per_frame)
        self.meso_center_crop = int(meso_center_crop)
        self.meso_agg = str(meso_agg)
        self.meso_time_span = meso_time_span

        self.do_hflip = bool(do_hflip)
        self.return_paths = bool(return_paths)
        self.resize = None if resize is None else int(resize)
        if isinstance(dtype, str):
            dtype = DTYPE_MAP.get(dtype, getattr(torch, dtype))
        self.dtype = dtype

        self.highres_lo = highres_lo  # (1, C, 1, 1)
        self.highres_hi = highres_hi
        self.meso_lo = np.asarray(meso_lo, dtype=np.float32).reshape(1, self.meso_channels)
        self.meso_hi = np.asarray(meso_hi, dtype=np.float32).reshape(1, self.meso_channels)
        self.static_lo = np.asarray(static_lo, dtype=np.float32).reshape(-1, 1, 1)
        self.static_hi = np.asarray(static_hi, dtype=np.float32).reshape(-1, 1, 1)

    def __len__(self) -> int:
        return len(self.files)

    def _resize(self,
                x: torch.Tensor,
                path: Path) -> torch.Tensor:
        """(N, C, H, W) -> (N, C, resize, resize) with area interpolation; downsampling only."""
        if tuple(x.shape[-2:]) == (self.resize, self.resize):
            return x
        if self.resize > min(x.shape[-2:]):
            raise ValueError(f"{path.name}: resize={self.resize} exceeds the native "
                             f"resolution {tuple(x.shape[-2:])}, only downsampling is supported")
        return F.interpolate(x.float(),
                             size=(self.resize, self.resize),
                             mode="area")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]

        # --- highresdynamic: 30 frame nello stesso spazio dell'autoencoder ---
        imgs, unfixable = load_highres_dynamic(
            path, n_bands=self.n_channels, n_time=self.T
        )
        imgs = minmax_pm1(imgs, self.highres_lo, self.highres_hi)

        # --- highresstatic: invariante nel tempo, gia' allineata ------------
        static, unfixable_static = load_highres_static(path)
        static = minmax_pm1(static, self.static_lo, self.static_hi)

        # --- mesodynamic: crop 2x2 centrale + allineamento 1:5 --------------
        meso = load_meso_dynamic(
            path,
            n_channels=self.meso_channels,
            n_time=self.T,
            steps_per_frame=self.steps_per_frame,
            center_crop=self.meso_center_crop,
            agg=self.meso_agg,
        )
        meso = minmax_pm1(meso, self.meso_lo, self.meso_hi)
        if self.meso_time_span == "future":
            meso = meso[-self.n_predict :]

        # Rete di sicurezza, dopo la normalizzazione: quello che resta non finito
        # va a 0, cioe' a meta' scala. Stessa convenzione del preprocessing, e
        # per l'interpolazione 3x3 e' un buco che nessun pixel valido raggiunge.
        # Con il filtro sui cubi attivo in preprocessing non dovrebbe succedere,
        # quindi se succede lo si dice.
        if unfixable:
            logger.warning(
                f"{path.name}: {unfixable} valori di {HIGHRES_DYNAMIC_KEY} non "
                f"riempibili (immagini interamente mancanti), messi a meta' scala. "
                f"Rigenera stats.json con preprocessing.py per escludere il cubo."
            )
        if unfixable_static:
            logger.warning(
                f"{path.name}: {unfixable_static} valori di {HIGHRES_STATIC_KEY} "
                f"non riempibili (banda senza nemmeno un pixel valido), messi a "
                f"meta' scala."
            )
        for arr in (imgs, static, meso):
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)

        imgs_t = torch.from_numpy(np.ascontiguousarray(imgs))
        static_t = torch.from_numpy(np.ascontiguousarray(static))
        if self.resize is not None:
            # in float32, before the cast to the (possibly half) output dtype
            imgs_t = self._resize(imgs_t,
                                  path)
            static_t = self._resize(static_t.unsqueeze(0),
                                    path).squeeze(0)
        imgs_t = imgs_t.to(self.dtype)
        static_t = static_t.to(self.dtype)
        meso_t = torch.from_numpy(np.ascontiguousarray(meso)).to(self.dtype)

        # Il flip orizzontale tocca solo le variabili con estensione spaziale:
        # meso e' gia' ridotta a una serie temporale, quindi resta invariata.
        if self.do_hflip and torch.rand(1).item() < 0.5:
            imgs_t = torch.flip(imgs_t, dims=[-1])
            static_t = torch.flip(static_t, dims=[-1])

        H, W = imgs_t.shape[-2], imgs_t.shape[-1]
        out: Dict[str, Any] = {
            "imgs": imgs_t,                              # (T, C, H, W)
            "highres_condition_image": static_t,         # (C_s, H, W) -> crossattn
            "meso_condition_image": meso_t,              # (T_meso, C_m) -> vector
            "original_size_as_tuple": torch.tensor([H, W], dtype=torch.long),
            "crop_coords_top_left": torch.tensor([0, 0], dtype=torch.long),
            "target_size_as_tuple": torch.tensor([H, W], dtype=torch.long),
        }
        if self.return_paths:
            out["cube_path"] = str(path)
        return out


# ---------------------------------------------------------------------------
# Lightning DataModule
# ---------------------------------------------------------------------------
class S2TemporalDataModule(_DataModuleBase):
    """
    DataModule del modello diffusivo. Prende la cartella dei minicubi .npz e le
    statistiche del preprocessing, e produce gli split train e validation
    elencati in dataset.json (the test split is not used during training).

    I percorsi di default sono relativi alla root del progetto (si lancia da
    DF-Diff/.. oppure si passano percorsi assoluti).
    """

    def __init__(
        self,
        data_root: str = "data/raw/train",
        dataset_json: str = "data/preprocessed/n0_s0/dataset.json",
        stats_json: str = "data/preprocessed/n0_s0/stats.json",
        cond_stats_json: str = "data/preprocessed/n0_s0/cond_stats.json",
        stats_time_key: str = "10",

        # sequenza
        T: int = 30,
        noise_image_num: int = 20,
        n_channels: int = 4,
        resize: int = 128,

        # condizioni
        meso_channels: int = 5,
        meso_steps_per_frame: int = 5,
        meso_center_crop: int = 2,
        meso_agg: str = "last",
        meso_time_span: str = "all",

        # dataloader
        batch_size: int = 16,
        num_workers: int = 8,
        shuffle: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        drop_last: bool = True,
        do_hflip: bool = True,

        # varie
        max_bad_frames: Optional[int] = None,
        stats_workers: int = 0,
        dtype: torch.dtype = torch.float16,
        debug_first_batch: bool = False,
    ):
        super().__init__()

        if isinstance(dtype, str):
            dtype = DTYPE_MAP.get(dtype, getattr(torch, dtype))
        self.dtype = dtype
        self.data_root = data_root
        self.dataset_json = dataset_json
        self.stats_json = stats_json
        self.cond_stats_json = cond_stats_json
        self.stats_time_key = str(stats_time_key)

        self.T = int(T)
        self.noise_image_num = int(noise_image_num)
        self.n_channels = int(n_channels)
        self.resize = int(resize)

        self.meso_channels = int(meso_channels)
        self.meso_steps_per_frame = int(meso_steps_per_frame)
        self.meso_center_crop = int(meso_center_crop)
        self.meso_agg = str(meso_agg)
        self.meso_time_span = str(meso_time_span)

        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.shuffle = bool(shuffle)
        self.prefetch_factor = int(prefetch_factor)
        self.persistent_workers = bool(persistent_workers)
        self.pin_memory = bool(pin_memory)
        self.drop_last = bool(drop_last)
        self.do_hflip = bool(do_hflip)

        self.max_bad_frames = max_bad_frames
        self.stats_workers = int(stats_workers)
        self.debug_first_batch = bool(debug_first_batch)

        self.train_dataset: Optional[S2DiffusionCubeDataset] = None
        self.val_dataset: Optional[S2DiffusionCubeDataset] = None

    # -- split ------------------------------------------------------------
    def _filter_bad_cubes(self,
                          files: List[Path],
                          keep_map: Dict[str, List[int]]) -> List[Path]:
        """
        With max_bad_frames, excludes the cubes in which too many of the T
        frames were rejected by the preprocessing (clouds, failed acquisitions).
        With None nothing is excluded: the sparse holes are repaired by
        fill_spatial. The cubes with entirely missing images are not in
        dataset.json at all.
        """
        if self.max_bad_frames is None:
            return files

        out, n_bad_cubes = [], 0
        for f in files:
            kept = keep_map.get(Path(f).name)
            if kept is not None:
                n_bad = self.T - len([u for u in kept if u < self.T])
                if n_bad > self.max_bad_frames:
                    n_bad_cubes += 1
                    continue
            out.append(f)

        if n_bad_cubes:
            logger.info(
                f"Esclusi {n_bad_cubes} cubi con piu' di {self.max_bad_frames} "
                f"istanti scartati sui primi {self.T}"
            )
        return out

    def _build_splits(self) -> Tuple[List[Path], List[Path]]:
        info = load_dataset_info(self.dataset_json)
        if Path(info["raw_dir"]) != Path(self.data_root):
            logger.warning(
                f"{self.dataset_json} was built from {info['raw_dir']}, the cubes are "
                f"read from data_root={self.data_root}"
            )
        checked = info.get("preprocessing", {}).get("check_frames")
        if checked is not None and checked < self.T:
            raise RuntimeError(
                f"{self.dataset_json} checked only the first {checked} frames of each "
                f"cube for missing images, but T={self.T} are read: rerun the "
                f"preprocessing with the current config."
            )
        train = split_files(info,
                            "train",
                            self.data_root)
        val = split_files(info,
                          "val",
                          self.data_root)

        keep_map: Dict[str, List[int]] = {}
        if self.max_bad_frames is not None:
            # stats.json e' grosso (decine di MB per il keep_map): lo si legge
            # solo se serve
            keep_map = load_keep_map(json.loads(Path(self.stats_json).read_text()))
            if not keep_map:
                logger.warning("keep_map assente in stats.json: max_bad_frames ignorato.")

        train = self._filter_bad_cubes(train,
                                       keep_map)
        val = self._filter_bad_cubes(val,
                                     keep_map)

        def n_loc(fs: Sequence[Path]) -> int:
            return len({location_of(f) for f in fs})

        logger.info(
            f"Dataset {info['name']} (id {info['id']}): "
            f"train {len(train)} cubi ({n_loc(train)} location), "
            f"val {len(val)} ({n_loc(val)} location)"
        )
        return train, val

    def setup(self, stage: Optional[str] = None) -> None:
        train_files, val_files = self._build_splits()
        if not train_files:
            raise RuntimeError("split di train vuoto")

        highres_lo, highres_hi = load_highres_stats(
            self.stats_json, time_key=self.stats_time_key, n_bands=self.n_channels
        )

        cond = ensure_condition_stats(
            self.cond_stats_json,
            train_files,
            meso_channels=self.meso_channels,
            T=self.T,
            steps_per_frame=self.meso_steps_per_frame,
            meso_center_crop=self.meso_center_crop,
            meso_agg=self.meso_agg,
            workers=self.stats_workers,
        )
            
   
        common = dict(
            highres_lo=highres_lo,
            highres_hi=highres_hi,
            meso_lo=cond["meso"]["lo"],
            meso_hi=cond["meso"]["hi"],
            static_lo=cond["static"]["lo"],
            static_hi=cond["static"]["hi"],
            T=self.T,
            n_predict=self.noise_image_num,
            n_channels=self.n_channels,
            meso_channels=self.meso_channels,
            steps_per_frame=self.meso_steps_per_frame,
            meso_center_crop=self.meso_center_crop,
            meso_agg=self.meso_agg,
            meso_time_span=self.meso_time_span,
            dtype=self.dtype,
            resize=self.resize,
        )

        self.train_dataset = S2DiffusionCubeDataset(
            train_files, do_hflip=self.do_hflip, **common
        )
        # Niente flip sulla validation: le metriche devono essere confrontabili
        # tra epoche, e un'augmentation casuale le farebbe ballare.
        if val_files:
            # return_paths solo sulla validation: il trainer usa cube_path per
            # dare ai PNG di fine training il nome del .npz di origine. Sui
            # batch di train non serve e resterebbe solo da ignorare.
            self.val_dataset = S2DiffusionCubeDataset(
                val_files, do_hflip=False, return_paths=True, **common
            )
        else:
            logger.warning(
                "split di validation vuoto: il training girera' senza "
                "validazione. Alza preprocessing.val_frac e rifai il preprocessing."
            )

        if self.debug_first_batch:
            self._log_first_item(self.train_dataset)

    def _log_first_item(self, ds: S2DiffusionCubeDataset) -> None:
        item = ds[0]
        logger.info(f"[DEBUG] primo item: {ds.files[0]}")
        for k in ("imgs", "highres_condition_image", "meso_condition_image"):
            v = item[k]
            logger.info(
                f"[DEBUG]  {k}: {tuple(v.shape)} {v.dtype} "
                f"min/max: {float(v.min()):.4f}/{float(v.max()):.4f}"
            )
        logger.info(
            f"[DEBUG]  contesto: {ds.n_context} frame, target: {ds.n_predict} frame"
        )

    # -- dataloaders -------------------------------------------------------
    def _loader(self, dataset, shuffle: bool, drop_last: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, self.shuffle, self.drop_last)

    def val_dataloader(self) -> Optional[DataLoader]:
        """
        Loader di validation, None se lo split e' vuoto. Il
        chiamante deve gestire il caso: validare sul train darebbe numeri privi
        di significato, quindi e' meglio non validare affatto.
        """
        if self.val_dataset is None:
            return None
        return self._loader(self.val_dataset, False, False)


# ---------------------------------------------------------------------------
# CLI: precalcolo statistiche e smoke test
# ---------------------------------------------------------------------------
def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """
    Console + file. Lanciato dal trainer questo non serve: li' la configurazione
    del logging root la fa gia' main(), e i messaggi di questo modulo finiscono
    nello stesso file.
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level,
        handlers=handlers,
        force=True,
    )
    if log_file:
        logging.getLogger(__name__).info(f"log scritto anche in {log_file}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data/raw/train")
    ap.add_argument("--dataset-dir", required=True,
                    help="preprocessed dataset folder (dataset.json, stats.json, cond_stats.json)")
    ap.add_argument("--stats-time-key", default="10")
    ap.add_argument("--T", type=int, default=30)
    ap.add_argument("--noise-image-num", type=int, default=20)
    ap.add_argument("--meso-agg", default="last", choices=["mean", "sum", "last", "first"])
    ap.add_argument("--meso-time-span", default="all", choices=["all", "future"])
    ap.add_argument("--workers", type=int, default=0, help="processi per il calcolo delle statistiche")
    ap.add_argument("--compute-cond-stats", action="store_true",
                    help="calcola (o ricalcola) cond_stats.json e termina")
    ap.add_argument("--force", action="store_true", help="ricalcola anche se la cache esiste")
    ap.add_argument("--smoke-test", action="store_true", help="costruisce un batch e stampa le shape")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--log-file", default="logs/dataloader_s2_dif.log",
                    help="file in cui salvare il log (vuoto = solo console)")
    ap.add_argument(
        "--dtype", 
        default="float16", 
        choices=["float32", "float16", "bfloat16"], 
        help="Tipo di dato dei tensori generati dal DataLoader"
    )
    args = ap.parse_args()

    setup_logging(args.log_file or None)

    dm = S2TemporalDataModule(
        data_root=args.data_root,
        dataset_json=str(Path(args.dataset_dir) / "dataset.json"),
        stats_json=str(Path(args.dataset_dir) / "stats.json"),
        cond_stats_json=str(Path(args.dataset_dir) / "cond_stats.json"),
        stats_time_key=args.stats_time_key,
        T=args.T,
        noise_image_num=args.noise_image_num,
        meso_agg=args.meso_agg,
        meso_time_span=args.meso_time_span,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stats_workers=args.workers,
        persistent_workers=False,
        pin_memory=False,
        drop_last=False,
        debug_first_batch=not args.compute_cond_stats,
        dtype=DTYPE_MAP[args.dtype],
    )

    if args.compute_cond_stats:
        # Lo split va preso dal datamodule, non ricostruito qui: ensure_condition_stats
        # deve vedere gli stessi cubi che vedra' setup() durante il training.
        train_files, _ = dm._build_splits()
        ensure_condition_stats(
            dm.cond_stats_json,
            train_files,
            meso_channels=dm.meso_channels,
            T=args.T,
            steps_per_frame=dm.meso_steps_per_frame,
            meso_center_crop=dm.meso_center_crop,
            meso_agg=args.meso_agg,
            workers=args.workers,
            force=True,
        )
        return

    dm.setup("fit")

    if args.smoke_test:
        batch = next(iter(dm.train_dataloader()))
        for key in ("imgs", "highres_condition_image", "meso_condition_image"):
            logger.info(f"batch[{key}]: {tuple(batch[key].shape)}")


if __name__ == "__main__":
    main()
