#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing EarthNet2021 -> array piatto (N, 4, 128, 128) per il training del VAE 2D.

Input : cartella con minicubi .npz, chiave "highresdynamic" di shape (H, W, C, T)
        = (128, 128, 7, 30). Si tengono le prime 4 bande (blue, green, red, NIR);
        le restanti (cloud mask, scene classification, mask) vengono scartate.
Output: un file .npy per ogni versione temporale richiesta, di shape
        (n_cubi * T, 4, 128, 128), piu' un .json con le statistiche usate.

Normalizzazione: min-max per banda, con estremi derivati dai quantili 5% e 95%
calcolati sull'intero training set e poi allargati al range naturale della
riflettanza [0, 1]:

    lo = min(0, q05)      hi = max(1, q95)
    x_norm = 2 * (x - lo) / (hi - lo) - 1,   poi clip in [-1, 1]

Il clamp fa si' che la scala non si stringa mai sotto [0, 1]: i quantili
possono solo allargarla, quando una banda ha code oltre quel range.

Selezione dei cubi. Un minicubo viene scartato *per intero* se contiene almeno
una immagine interamente mancante, cioe' una slice 2D (banda, istante) senza
nemmeno un valore finito. Il controllo vale sia per highresdynamic (sulle bande
effettivamente usate) sia per mesodynamic (su tutte e 5 le variabili, sui giorni
che coprono la finestra temporale richiesta). Il motivo e' che una immagine
vuota non e' riparabile: fill_spatial propaga dai pixel validi vicini, e se non
ce n'e' nessuno il buco resta. Scartare il solo istante lascerebbe comunque un
cubo con un vuoto in mezzo alla sequenza, inutilizzabile dal modello video, che
la vuole contigua. Con --keep-empty-frame-cubes si torna al comportamento
precedente (si scarta il solo istante).

highresstatic (il DEM) non e' un criterio di scarto: e' invariante nel tempo e i
suoi buchi il dataloader li riempie con la media della banda, non con uno zero
che nella scala originale sarebbe una quota vera. Con --drop-static-nan-frac F
lo si trasforma in un criterio: si scarta il cubo se la frazione di pixel
mancanti del DEM supera F.

Diagnostica. I tre controlli (highresdynamic, mesodynamic, highresstatic) sono
valutati su *ogni* cubo leggibile, prima e indipendentemente dalla decisione di
scarto, e i conteggi finiscono nel log e in stats.json sotto "cube_diag". Serve
perche' altrimenti i numeri di mesodynamic e highresstatic direbbero solo quello
che succede nei cubi sopravvissuti al controllo su highresdynamic: un cubo gia'
bocciato non verrebbe mai esaminato sulle altre variabili, e "0 immagini vuote
in mesodynamic" si leggerebbe come una proprieta' del dataset quando invece e'
una proprieta' del sottoinsieme rimasto. "cube_drops" resta invece
l'attribuzione dello scarto al primo criterio che scatta, e somma al numero di
cubi buttati; le due letture rispondono a domande diverse.

Selezione degli istanti. Nei cubi tenuti, un istante viene scartato se:
  - ha oltre il 50% di valori mancanti, contati su tutte le bande insieme
    (soglia regolabile con --drop-threshold); oppure
  - e' degenere, cioe' ha tutti i valori uguali tra loro, oppure tutti
    maggiori di 1, oppure tutti minori di 0. Sono acquisizioni fallite che
    non contengono NaN e sfuggirebbero al criterio precedente.
Negli istanti tenuti, i mancanti sparsi sono riempiti con la media dei pixel
validi nell'intorno 3x3, iterata finche' le chiazze si chiudono (fill_spatial).
I valori mancanti non contribuiscono mai ai quantili.

Poiche' gli istanti scartati variano da cubo a cubo, il numero di righe
dell'output non e' n_cubi * T: la selezione viene decisa nel primo passaggio e
salvata in stats.json sotto "keep_map".

I quantili sono calcolati in streaming con un istogramma a 40000 bin sul range
[-1, 3] (risoluzione 1e-4, pari alla quantizzazione della riflettanza sorgente),
quindi non serve tenere in RAM i valori.

Configurazione, selezione e split
---------------------------------
All the parameters come from the `preprocessing` section of the config
(src/config.py, registry/configurations/*.yaml). This script is the only place
where cubes are selected (preprocessing.limit, see select_cubes) and split by
location into train / val. The outcome goes into
data/preprocessed/<preprocessing.name>/:

    dataset.json      the cubes of each split + parameters + dataset id
                      (src/dataset/dataset_info.py); written last
    stats.json        normalization, keep_map, dropped cubes, diagnostics
    cond_stats.json   min/max of mesodynamic and highresstatic on train
    {train,val}_T<t>.npy, manifest.json   images for the autoencoder
    config.yaml, preprocessing.log

Both training phases read their cubes from dataset.json and never split again.

Esempi (dalla root, con PYTHONPATH=.:src/model:other)
-----------------------------------------------------
  # the dataset described by the yaml
  python src/dataset/preprocessing.py

  # 200 cubes left after dropping the bad ones, under a name of its own
  python src/dataset/preprocessing.py --set preprocessing.limit=200 \\
      preprocessing.limit_after_drop=true preprocessing.name=smoke200

  # solo statistiche e split, senza scrivere gli array (basta alla fase 2)
  python src/dataset/preprocessing.py --stats-only
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import multiprocessing as mp
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.lib.format import open_memmap
from omegaconf import OmegaConf

from src.dataset.dataloader_s2_dif import ensure_condition_stats
from src.config import load_config, resolve_config_path, save_config, use_project_dir
from src.dataset.dataset_info import compute_dataset_id, dataset_info_path, write_dataset_info


logger = logging.getLogger(__name__)

ARRAY_KEY = "highresdynamic"
MESO_KEY = "mesodynamic"
STATIC_KEY = "highresstatic"
EXPECTED_HW = (128, 128)

# mesodynamic e' giornaliera, highresdynamic ha passo 5 giorni: a ogni istante
# ad alta risoluzione corrispondono 5 passi meteo.
MESO_STEPS_PER_FRAME = 5

# Numero di bin dell'istogramma usato per i quantili in streaming. Il *range*
# e' invece per-banda e stimato dai dati (vedi scan_ranges): le riflettanze
# stanno in [0,1] circa, mentre la cloud mask arriva a 100, e un range unico
# sprecherebbe risoluzione sulle bande che contano.
HIST_BINS = 40_000


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
def list_cubes(root: Path) -> List[Path]:
    """Tutti i .npz sotto root, in ordine deterministico."""
    return sorted(root.rglob("*.npz"))


def location_of(path: Path) -> str:
    """La location e' la cartella che contiene il cubo (es. '29SND')."""
    return path.parent.name


def read_cube(path: Path) -> np.ndarray:
    with np.load(path) as z:
        return z[ARRAY_KEY]


def read_meso(path: Path) -> Optional[np.ndarray]:
    """mesodynamic del cubo, None se la chiave non c'e'."""
    with np.load(path) as z:
        if MESO_KEY not in z.files:
            return None
        return z[MESO_KEY]


def read_static(path: Path) -> Optional[np.ndarray]:
    """highresstatic del cubo, None se la chiave non c'e'."""
    with np.load(path) as z:
        if STATIC_KEY not in z.files:
            return None
        return z[STATIC_KEY]


def empty_frames_per_channel(x: np.ndarray) -> np.ndarray:
    """
    Per ogni canale di x (H, W, C, T), quante immagini sono interamente non
    finite.

    Una slice (c, t) e' vuota se non contiene nemmeno un valore finito: sono
    quelle che fill_spatial non puo' riparare, perche' non ha nessun pixel
    valido da cui propagare. Il dettaglio per canale serve alla diagnostica di
    mesodynamic, dove le 5 variabili vengono da sorgenti diverse e non e' detto
    che manchino insieme.
    """
    has_value = np.isfinite(x).any(axis=(0, 1))  # (C, T)
    return (~has_value).sum(axis=1).astype(np.int64)


def count_empty_frames(x: np.ndarray) -> int:
    """Quante immagini di x (H, W, C, T) sono interamente non finite."""
    return int(empty_frames_per_channel(x).sum())


def check_meso(path: Path, t_max: int) -> Tuple[Optional[np.ndarray], str]:
    """
    Immagini interamente mancanti di mesodynamic, per variabile.

    Si guardano solo i giorni che coprono la finestra richiesta: t_max istanti a
    passo MESO_STEPS_PER_FRAME. Se il cubo ne ha meno, si controlla quello che
    c'e'. Ritorna (conteggi per variabile, "") oppure (None, motivo) se la
    chiave manca, e' malformata o illeggibile.
    """
    try:
        md = read_meso(path)
    except Exception as exc:  # file corrotto
        return None, f"errore di lettura di {MESO_KEY}: {exc}"

    if md is None or md.ndim != 4:
        return None, f"{MESO_KEY} assente o malformato"

    n_days = min(md.shape[3], t_max * MESO_STEPS_PER_FRAME)
    return empty_frames_per_channel(md[:, :, :, :n_days].astype(np.float32)), ""


def check_static(path: Path) -> Tuple[Optional[Tuple[int, int]], str]:
    """
    (pixel mancanti, pixel totali) di highresstatic, oppure (None, motivo).

    highresstatic e' (H, W, C) e non ha asse temporale: la "immagine" e' una
    sola per banda, quindi qui interessa la frazione di pixel mancanti, non il
    conteggio di slice vuote.
    """
    try:
        hs = read_static(path)
    except Exception as exc:  # file corrotto
        return None, f"errore di lettura di {STATIC_KEY}: {exc}"

    if hs is None:
        return None, f"{STATIC_KEY} assente"

    a = np.asarray(hs, dtype=np.float32)
    return (int(a.size - np.count_nonzero(np.isfinite(a))), int(a.size)), ""


def new_cube_diag() -> Dict:
    """
    Contatori della diagnostica per cubo, azzerati. Vengono riempiti su *ogni*
    cubo leggibile, a prescindere da quale criterio poi lo scarta.
    """
    return {
        "n_cubes": 0,
        "read_error": 0,
        "bad_shape": 0,
        "highres_empty_cubes": 0,
        "highres_empty_slices": 0,
        "meso_unreadable": 0,
        "meso_empty_cubes": 0,
        "meso_empty_slices": 0,
        "meso_empty_slices_per_var": {},
        "static_unreadable": 0,
        "static_nan_cubes": 0,
        "static_allnan_cubes": 0,
        "static_nan_pixels": 0,
        "static_pixels": 0,
        "static_max_nan_frac": 0.0,
    }


def merge_cube_diag(dst: Dict, src: Dict) -> None:
    """Somma i contatori di src in dst (max per static_max_nan_frac)."""
    for key, value in src.items():
        if key == "meso_empty_slices_per_var":
            for var, n in value.items():
                dst[key][var] = dst[key].get(var, 0) + n
        elif key == "static_max_nan_frac":
            dst[key] = max(dst[key], value)
        else:
            dst[key] += value


def log_cube_diag(diag: Dict, n_cubes_in: int) -> None:
    """
    Riepilogo della diagnostica per variabile. I conteggi sono su tutti i cubi
    leggibili, non sui soli sopravvissuti, quindi le tre righe si possono
    leggere come proprieta' del dataset e non dell'ordine dei filtri.
    """
    n_read = max(diag["n_cubes"] - diag["read_error"] - diag["bad_shape"], 0)
    frac = lambda n: f"{n / max(n_read, 1):.2%}"  # noqa: E731

    logger.info(f"  DIAGNOSTICA su tutti i {n_read} cubi leggibili "
                f"(su {n_cubes_in} trovati), indipendente dai criteri di scarto:")
    if diag["read_error"] or diag["bad_shape"]:
        logger.info(f"    non esaminabili : {diag['read_error']} illeggibili, "
                    f"{diag['bad_shape']} con shape inattesa")

    logger.info(
        f"    {ARRAY_KEY:14s}: {diag['highres_empty_cubes']} cubi con almeno una "
        f"immagine interamente mancante ({frac(diag['highres_empty_cubes'])}) "
        f"| {diag['highres_empty_slices']} immagini in totale"
    )

    logger.info(
        f"    {MESO_KEY:14s}: {diag['meso_empty_cubes']} cubi con almeno una "
        f"immagine interamente mancante ({frac(diag['meso_empty_cubes'])}) "
        f"| {diag['meso_empty_slices']} immagini in totale"
    )
    per_var = diag["meso_empty_slices_per_var"]
    if per_var:
        detail = ", ".join(f"var {v}: {per_var[v]}" for v in sorted(per_var))
        logger.info(f"                     immagini vuote per variabile: {detail}")
    if diag["meso_unreadable"]:
        logger.info(f"                     {diag['meso_unreadable']} cubi senza "
                    f"{MESO_KEY} o con la chiave malformata")

    n_px = max(diag["static_pixels"], 1)
    logger.info(
        f"    {STATIC_KEY:14s}: {diag['static_nan_cubes']} cubi con almeno un "
        f"valore mancante ({frac(diag['static_nan_cubes'])}) "
        f"| {diag['static_allnan_cubes']} interamente mancanti "
        f"| {diag['static_nan_pixels']} pixel su {diag['static_pixels']} "
        f"({diag['static_nan_pixels'] / n_px:.4%}) "
        f"| peggior cubo: {diag['static_max_nan_frac']:.2%} di mancanti"
    )
    if diag["static_unreadable"]:
        logger.info(f"                     {diag['static_unreadable']} cubi senza "
                    f"{STATIC_KEY} o con la chiave malformata")


def split_by_location(files: Sequence[Path],
                      val_frac: float,
                      seed: int) -> Tuple[List[Path], List[Path]]:
    """
    Split train / val per *location*, non per cubo: cubi della stessa
    tile sono spazialmente sovrapposti, splittarli a caso creerebbe leakage.

    The locations are shuffled once with the seed: the first ones go to val,
    the rest to train. A fraction > 0 always gets at least one location.
    """
    locations = sorted({location_of(f) for f in files})
    rng = np.random.default_rng(seed)
    order = [locations[i] for i in rng.permutation(len(locations))]

    n_val = max(1, int(round(val_frac * len(locations)))) if val_frac > 0 else 0
    if n_val >= len(locations):
        raise SystemExit(
            f"val_frac={val_frac} takes {n_val} of the {len(locations)} locations "
            f"available: none left for train. "
            f"Raise preprocessing.limit or lower the fraction."
        )
    val_locs = set(order[:n_val])

    train = [f for f in files if location_of(f) not in val_locs]
    val = [f for f in files if location_of(f) in val_locs]
    return train, val


def sample_evenly(files: Sequence[Path],
                  n: int) -> List[Path]:
    """
    n files evenly spaced over the sorted list, not the first n: the files are
    sorted by location, so a truncation would give a single tile.
    """
    if n >= len(files):
        return list(files)
    idx = np.linspace(0, len(files) - 1, n).round().astype(int)
    return [files[i] for i in sorted(set(idx.tolist()))]


def chunked(seq: Sequence, size: int) -> List[List]:
    return [list(seq[i : i + size]) for i in range(0, len(seq), size)]


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PiB"


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """Console piu' file: il riepilogo di una run va conservato."""
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level,
        handlers=handlers,
        force=True,
    )
    if log_file:
        logger.info(f"log scritto anche in {log_file}")


def log_progress(done: int, total: int, t0: float, label: str, every: int = 0) -> None:
    """
    Avanzamento come righe di log invece che come barra con \\r: una barra in un
    file di log e' illeggibile. Di default si logga ogni 5%.
    """
    every = every or max(1, total // 20)
    if done % every and done != total:
        return
    elapsed = time.time() - t0
    eta = elapsed / done * (total - done) if done else 0.0
    logger.info(
        f"  {label}: {done}/{total} chunk "
        f"| {elapsed / 60:.1f} min trascorsi | ETA {eta / 60:.1f} min"
    )


# ---------------------------------------------------------------------------
# Pass 0 -- range per banda (su un campione di cubi)
# ---------------------------------------------------------------------------
def range_chunk(task: Tuple[List[str], int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Min/max per banda sui valori finiti di un gruppo di cubi."""
    paths, n_bands, t_max = task

    lo = np.full(n_bands, np.inf, dtype=np.float64)
    hi = np.full(n_bands, -np.inf, dtype=np.float64)

    for p in paths:
        try:
            hd = read_cube(Path(p))
        except Exception:
            continue
        if hd.ndim != 4 or hd.shape[2] < n_bands or hd.shape[3] < t_max:
            continue

        x = hd[:, :, :n_bands, :t_max].astype(np.float32)
        for b in range(n_bands):
            v = x[:, :, b, :].ravel()
            v = v[np.isfinite(v)]
            if v.size:
                lo[b] = min(lo[b], float(v.min()))
                hi[b] = max(hi[b], float(v.max()))

    return lo, hi


def scan_ranges(
    files: Sequence[Path],
    n_bands: int,
    t_max: int,
    n_sample: int,
    workers: int,
    chunk: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stima il range per banda su un campione uniforme di cubi e lo allarga del
    10% per lato. Eventuali valori fuori range nel passaggio completo finiscono
    nei contatori di under/overflow e restano quindi contabilizzati.
    """
    if n_sample >= len(files):
        sample = list(files)
    else:
        idx = np.linspace(0, len(files) - 1, n_sample).round().astype(int)
        sample = [files[i] for i in sorted(set(idx.tolist()))]

    tasks = [(c, n_bands, t_max) for c in chunked([str(f) for f in sample], chunk)]

    lo = np.full(n_bands, np.inf, dtype=np.float64)
    hi = np.full(n_bands, -np.inf, dtype=np.float64)

    t0 = time.time()
    with mp.Pool(processes=workers) as pool:
        for i, (l, h) in enumerate(pool.imap_unordered(range_chunk, tasks), start=1):
            lo = np.minimum(lo, l)
            hi = np.maximum(hi, h)
            log_progress(i, len(tasks), t0, "range")

    if not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise RuntimeError("nessun valore finito nel campione: dati corrotti?")

    span = np.maximum(hi - lo, 1e-6)
    return lo - 0.1 * span, hi + 0.1 * span


# ---------------------------------------------------------------------------
# Pass 1 -- istogrammi per i quantili
# ---------------------------------------------------------------------------
def hist_chunk(
    task: Tuple[
        List[str], List[int], int, bool, np.ndarray, np.ndarray, float, bool,
        Optional[float], int,
    ]
) -> Dict:
    """
    Worker del primo passaggio. Per ogni cubo raccoglie la diagnostica delle tre
    variabili, decide se tenerlo (vedi drop_cube_on_empty) e quali istanti
    tenere (si scartano quelli con piu' di drop_thr di valori mancanti, contati
    su tutte le bande insieme); sui soli istanti tenuti accumula per banda un
    istogramma dei valori validi.

    accumulate=False: il cubo viene validato e si calcola comunque quali
    istanti tenere, ma non contribuisce ai quantili (serve per i file di val).

    The checks cover the first max(times, check_frames) frames: the .npy files
    use only the first max(times), but the diffusion model reads check_frames
    frames of every cube, and an empty image there would reach it unchecked.
    """
    (paths, times, n_bands, accumulate, edge_lo, edge_hi, drop_thr,
     drop_cube_on_empty, static_nan_thr, check_frames) = task
    t_max = max(max(times), check_frames)

    hists = {t: np.zeros((n_bands, HIST_BINS), dtype=np.int64) for t in times}
    n_below = {t: np.zeros(n_bands, dtype=np.int64) for t in times}
    n_above = {t: np.zeros(n_bands, dtype=np.int64) for t in times}
    n_valid = {t: np.zeros(n_bands, dtype=np.int64) for t in times}
    n_nonfinite = {t: np.zeros(n_bands, dtype=np.int64) for t in times}

    ok: List[str] = []
    skipped: List[Tuple[str, str]] = []
    keep_map: Dict[str, List[int]] = {}
    n_drop_nan = n_drop_const = n_drop_over = n_drop_under = 0
    cube_drops = {
        "empty_highres": 0, "empty_meso": 0, "no_meso": 0, "nan_static": 0,
    }
    diag = new_cube_diag()

    for p in paths:
        diag["n_cubes"] += 1
        try:
            hd = read_cube(Path(p))
        except Exception as exc:  # file corrotto o chiave mancante
            skipped.append((p, f"errore di lettura: {exc}"))
            diag["read_error"] += 1
            continue

        if (
            hd.ndim != 4
            or hd.shape[:2] != EXPECTED_HW
            or hd.shape[2] < n_bands
            or hd.shape[3] < t_max
        ):
            skipped.append((p, f"shape inattesa {hd.shape}"))
            diag["bad_shape"] += 1
            continue

        # (H, W, n_bands, t_max) in float32: i confronti su float16 saturano.
        x = hd[:, :, :n_bands, :t_max].astype(np.float32)

        # --- diagnostica ----------------------------------------------------
        # Tutte e tre le variabili sono esaminate su OGNI cubo leggibile, anche
        # su quelli che stanno per essere buttati: altrimenti i numeri di
        # mesodynamic e highresstatic descriverebbero solo i cubi sopravvissuti
        # al controllo su highresdynamic, che e' il primo a scattare.
        n_empty_hr = count_empty_frames(x)
        if n_empty_hr:
            diag["highres_empty_cubes"] += 1
            diag["highres_empty_slices"] += n_empty_hr

        meso_empty, meso_reason = check_meso(Path(p), t_max)
        if meso_empty is None:
            diag["meso_unreadable"] += 1
            n_empty_meso = 0
        else:
            n_empty_meso = int(meso_empty.sum())
            if n_empty_meso:
                diag["meso_empty_cubes"] += 1
                diag["meso_empty_slices"] += n_empty_meso
                for v, n in enumerate(meso_empty):
                    if n:
                        d = diag["meso_empty_slices_per_var"]
                        d[v] = d.get(v, 0) + int(n)

        static_counts, static_reason = check_static(Path(p))
        if static_counts is None:
            diag["static_unreadable"] += 1
            static_nan_frac = 0.0
        else:
            n_nan_static, n_px_static = static_counts
            diag["static_nan_pixels"] += n_nan_static
            diag["static_pixels"] += n_px_static
            static_nan_frac = n_nan_static / max(n_px_static, 1)
            if n_nan_static:
                diag["static_nan_cubes"] += 1
                diag["static_max_nan_frac"] = max(
                    diag["static_max_nan_frac"], static_nan_frac
                )
                if n_nan_static == n_px_static:
                    diag["static_allnan_cubes"] += 1

        # --- decisione di scarto --------------------------------------------
        # Una immagine interamente mancante non e' riparabile da fill_spatial,
        # e scartare il solo istante lascerebbe un buco in mezzo alla sequenza:
        # si butta via tutto il cubo. Il controllo copre anche mesodynamic, che
        # il modello video usa come condizionamento.
        #
        # Il motivo salvato in 'skipped' elenca *tutti* i criteri falliti, cosi'
        # da un cubo scartato si vede l'intero quadro; cube_drops attribuisce
        # invece lo scarto al primo, per avere contatori che sommano al totale.
        reasons: List[Tuple[str, str]] = []
        if drop_cube_on_empty:
            if n_empty_hr:
                reasons.append((
                    "empty_highres",
                    f"{n_empty_hr} immagini di {ARRAY_KEY} interamente mancanti",
                ))
            if meso_empty is None:
                reasons.append(("no_meso", meso_reason))
            elif n_empty_meso:
                reasons.append((
                    "empty_meso",
                    f"{n_empty_meso} immagini di {MESO_KEY} interamente mancanti",
                ))

        # highresstatic non e' un criterio di scarto per default: e' il DEM, non
        # ha asse temporale e i suoi buchi sono riempibili con la media della
        # banda. Diventa un criterio solo se si passa --drop-static-nan-frac.
        if static_nan_thr is not None:
            if static_counts is None:
                reasons.append(("nan_static", static_reason))
            elif static_nan_frac > static_nan_thr:
                reasons.append((
                    "nan_static",
                    f"{STATIC_KEY} ha {static_nan_frac:.1%} di valori mancanti",
                ))

        if reasons:
            cube_drops[reasons[0][0]] += 1
            skipped.append((p, " ; ".join(why for _, why in reasons)))
            continue

        # Frazione di valori mancanti per istante, su tutte le bande insieme.
        frac_nan = (~np.isfinite(x)).mean(axis=(0, 1, 2))
        keep = np.where(frac_nan <= drop_thr)[0]
        n_drop_nan += t_max - keep.size

        # Scarto degli istanti degeneri: acquisizioni fallite che non contengono
        # NaN e sfuggirebbero quindi al criterio precedente. Valutato sui valori
        # finiti dell'istante, su tutte le bande insieme.
        if keep.size:
            xf = x[:, :, :, keep]
            xf = np.where(np.isfinite(xf), xf, np.nan)
            mn = np.nanmin(xf, axis=(0, 1, 2))
            mx = np.nanmax(xf, axis=(0, 1, 2))

            const = mn == mx           # tutti i valori uguali tra loro
            over = ~const & (mn > 1.0)  # tutti sopra 1
            under = ~const & (mx < 0.0)  # tutti sotto 0

            n_drop_const += int(const.sum())
            n_drop_over += int(over.sum())
            n_drop_under += int(under.sum())

            keep = keep[~(const | over | under)]

        ok.append(p)
        keep_map[p] = [int(u) for u in keep]

        if not accumulate or keep.size == 0:
            continue

        for t in times:
            sel = keep[keep < t]
            if sel.size == 0:
                continue
            xt = x[:, :, :, sel]

            for b in range(n_bands):
                v = xt[:, :, b, :].ravel()

                n_fin = int(np.count_nonzero(np.isfinite(v)))
                n_nonfinite[t][b] += v.size - n_fin
                n_valid[t][b] += n_fin

                # I NaN cadono fuori range e np.histogram li ignora; i confronti
                # con NaN sono False, quindi i contatori restano corretti.
                n_below[t][b] += int(np.count_nonzero(v < edge_lo[b]))
                n_above[t][b] += int(np.count_nonzero(v > edge_hi[b]))

                counts, _ = np.histogram(
                    v, bins=HIST_BINS, range=(float(edge_lo[b]), float(edge_hi[b]))
                )
                hists[t][b] += counts

    return {
        "hists": hists,
        "n_below": n_below,
        "n_above": n_above,
        "n_valid": n_valid,
        "n_nonfinite": n_nonfinite,
        "ok": ok,
        "skipped": skipped,
        "keep_map": keep_map,
        "drops": {
            "nan": n_drop_nan,
            "const": n_drop_const,
            "over1": n_drop_over,
            "under0": n_drop_under,
        },
        "cube_drops": cube_drops,
        "cube_diag": diag,
    }


def quantile_from_hist(
    counts: np.ndarray,
    n_below: int,
    n_above: int,
    q: float,
    edge_lo: float,
    edge_hi: float,
) -> float:
    """
    Quantile q (in [0,1]) da un istogramma uniforme su [edge_lo, edge_hi],
    con interpolazione lineare dentro il bin. n_below/n_above sono le masse
    cadute fuori dal range dell'istogramma.
    """
    total = int(counts.sum()) + int(n_below) + int(n_above)
    if total == 0:
        raise ValueError("nessun valore valido per il calcolo del quantile")

    target = q * total

    if target <= n_below:
        raise ValueError(
            f"quantile {q} sotto il range dell'istogramma ({edge_lo:.4g}): "
            f"aumenta --range-sample"
        )
    if target > total - n_above:
        raise ValueError(
            f"quantile {q} sopra il range dell'istogramma ({edge_hi:.4g}): "
            f"aumenta --range-sample"
        )

    csum = np.cumsum(counts)
    i = int(np.searchsorted(csum + n_below, target, side="left"))
    i = min(i, HIST_BINS - 1)

    prev = n_below + (int(csum[i - 1]) if i > 0 else 0)
    frac = (target - prev) / counts[i] if counts[i] > 0 else 0.0

    return edge_lo + (i + frac) * (edge_hi - edge_lo) / HIST_BINS


# ---------------------------------------------------------------------------
# Riempimento dei valori mancanti
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


def fill_spatial(x: np.ndarray, max_iter: int = 256) -> Dict[str, int]:
    """
    Riempie in place i valori non finiti di x (T, B, H, W): ogni pixel mancante
    prende la media dei pixel validi nel suo intorno 3x3. L'operazione e'
    iterata, cosi' le chiazze si chiudono progressivamente dal bordo verso il
    centro.

    Si applica ai soli istanti gia' selezionati, che per costruzione hanno al
    piu' drop_threshold di valori mancanti: c'e' quindi sempre almeno un pixel
    valido da cui propagare. Gli istanti troppo bucati sono stati scartati a
    monte, non riempiti.

    Ritorna il numero di slice (t, b) toccate e di valori riempiti.
    """
    stats = {"slices": 0, "pixels": 0, "unfixable": 0}

    k = x.reshape(-1, x.shape[2], x.shape[3])  # vista (T*B, H, W)
    bad = ~np.isfinite(k)
    idx = np.where(bad.reshape(k.shape[0], -1).any(axis=1))[0]
    if idx.size == 0:
        return stats

    sub = np.array(k[idx], dtype=np.float32)  # copia delle sole slice bucate
    m = ~np.isfinite(sub)

    stats["slices"] = int(idx.size)
    stats["pixels"] = int(m.sum())

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

    stats["unfixable"] = int(m.sum())
    vals[m] = np.nan  # buchi non chiusi: li gestisce il chiamante
    k[idx] = vals

    return stats


# ---------------------------------------------------------------------------
# Pass 2 -- scrittura degli array normalizzati
# ---------------------------------------------------------------------------
_W: Dict = {}


def init_writer(
    out_path: str,
    n_time: int,
    n_bands: int,
    lo: List[float],
    hi: List[float],
    dtype: str,
) -> None:
    """Inizializzatore del pool: apre il memmap una volta per worker."""
    _W["out"] = np.load(out_path, mmap_mode="r+")
    _W["T"] = n_time
    _W["n_bands"] = n_bands
    _W["lo"] = np.asarray(lo, dtype=np.float32).reshape(1, n_bands, 1, 1)
    _W["scale"] = (
        np.asarray(hi, dtype=np.float32) - np.asarray(lo, dtype=np.float32)
    ).reshape(1, n_bands, 1, 1)
    _W["dtype"] = np.dtype(dtype)


def write_chunk(
    task: Tuple[int, List[Tuple[str, List[int]]]]
) -> Tuple[int, int, Dict[str, int]]:
    """
    Scrive i cubi del chunk a partire dalla riga row0. Ogni cubo contribuisce
    con i soli istanti selezionati nel primo passaggio, quindi il numero di
    righe per cubo e' variabile.
    """
    row0, items = task
    out = _W["out"]
    n_bands = _W["n_bands"]

    n_clipped = 0
    n_written = 0
    fill = {"slices": 0, "pixels": 0, "unfixable": 0}

    r = row0
    for p, frames in items:
        if not frames:
            continue

        hd = read_cube(Path(p))

        x = hd[:, :, :n_bands, frames].astype(np.float32)  # (H, W, B, k)
        x = np.ascontiguousarray(np.transpose(x, (3, 2, 0, 1)))  # (k, B, H, W)

        for key, v in fill_spatial(x).items():
            fill[key] += v

        x = 2.0 * (x - _W["lo"]) / _W["scale"] - 1.0

        n_clipped += int(np.count_nonzero((x < -1.0) | (x > 1.0)))
        np.clip(x, -1.0, 1.0, out=x)

        # Rete di sicurezza: dopo lo scarto e il riempimento non dovrebbe
        # restare nulla di non finito.
        np.nan_to_num(x, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)

        out[r : r + len(frames)] = x.astype(_W["dtype"])
        r += len(frames)
        n_written += len(frames)

    out.flush()
    return n_written, n_clipped, fill


# ---------------------------------------------------------------------------
# Orchestrazione
# ---------------------------------------------------------------------------
def compute_stats(
    files: Sequence[Path],
    val_files: Sequence[Path],
    times: List[int],
    n_bands: int,
    edge_lo: np.ndarray,
    edge_hi: np.ndarray,
    drop_thr: float,
    workers: int,
    chunk: int,
    drop_cube_on_empty: bool = True,
    static_nan_thr: Optional[float] = None,
    check_frames: int = 0,
) -> Tuple[
    Dict, List[Path], List[Path], List[Tuple[str, str]],
    Dict[str, List[int]], Dict[str, int], Dict[str, int], Dict,
]:
    """
    Primo passaggio: scarto dei cubi con immagini vuote, selezione degli istanti
    da tenere su tutti i file, e quantili sui soli file di train (calcolati sui
    soli istanti tenuti). Raccoglie anche la diagnostica per variabile su tutti
    i cubi, indipendente dall'ordine in cui scattano i criteri di scarto.
    """
    tasks = [
        (c, times, n_bands, True, edge_lo, edge_hi, drop_thr, drop_cube_on_empty,
         static_nan_thr, check_frames)
        for c in chunked([str(f) for f in files], chunk)
    ]
    tasks += [
        (c, times, n_bands, False, edge_lo, edge_hi, drop_thr, drop_cube_on_empty,
         static_nan_thr, check_frames)
        for c in chunked([str(f) for f in val_files], chunk)
    ]

    hists = {t: np.zeros((n_bands, HIST_BINS), dtype=np.int64) for t in times}
    n_below = {t: np.zeros(n_bands, dtype=np.int64) for t in times}
    n_above = {t: np.zeros(n_bands, dtype=np.int64) for t in times}
    n_valid = {t: np.zeros(n_bands, dtype=np.int64) for t in times}
    n_nonfinite = {t: np.zeros(n_bands, dtype=np.int64) for t in times}

    ok: List[str] = []
    skipped: List[Tuple[str, str]] = []
    keep_map: Dict[str, List[int]] = {}
    drops = {"nan": 0, "const": 0, "over1": 0, "under0": 0}
    cube_drops = {
        "empty_highres": 0, "empty_meso": 0, "no_meso": 0, "nan_static": 0,
    }
    cube_diag = new_cube_diag()

    t0 = time.time()
    with mp.Pool(processes=workers) as pool:
        for i, res in enumerate(pool.imap_unordered(hist_chunk, tasks), start=1):
            for t in times:
                hists[t] += res["hists"][t]
                n_below[t] += res["n_below"][t]
                n_above[t] += res["n_above"][t]
                n_valid[t] += res["n_valid"][t]
                n_nonfinite[t] += res["n_nonfinite"][t]
            ok.extend(res["ok"])
            skipped.extend(res["skipped"])
            keep_map.update(res["keep_map"])
            for key in drops:
                drops[key] += res["drops"][key]
            for key in cube_drops:
                cube_drops[key] += res["cube_drops"][key]
            merge_cube_diag(cube_diag, res["cube_diag"])
            log_progress(i, len(tasks), t0, "scansione")

    n_cubes_in = len(files) + len(val_files)
    log_cube_diag(cube_diag, n_cubes_in)

    logger.info(f"  cubi esaminati: {n_cubes_in}")
    logger.info("    scarti attribuiti al primo criterio che scatta:")
    if drop_cube_on_empty:
        logger.info(
            f"    scartati, immagini vuote in {ARRAY_KEY} : {cube_drops['empty_highres']} "
            f"({cube_drops['empty_highres'] / max(n_cubes_in, 1):.2%})"
        )
        logger.info(
            f"    scartati, immagini vuote in {MESO_KEY}   : {cube_drops['empty_meso']} "
            f"({cube_drops['empty_meso'] / max(n_cubes_in, 1):.2%})"
        )
        if cube_drops["no_meso"]:
            logger.info(
                f"    scartati, {MESO_KEY} assente o illeggibile: {cube_drops['no_meso']}"
            )
    if static_nan_thr is not None:
        logger.info(
            f"    scartati, {STATIC_KEY} con > {static_nan_thr:.0%} di mancanti: "
            f"{cube_drops['nan_static']} "
            f"({cube_drops['nan_static'] / max(n_cubes_in, 1):.2%})"
        )
    logger.info(
        f"    TENUTI                                  : {len(ok)} "
        f"({len(ok) / max(n_cubes_in, 1):.2%})"
    )

    t_max = max(max(times), check_frames)
    n_total = max(len(ok) * t_max, 1)
    n_dropped = sum(drops.values())
    logger.info(f"  istanti esaminati (nei cubi tenuti): {len(ok) * t_max}")
    logger.info(f"    scartati, >{drop_thr:.0%} mancanti : {drops['nan']} ({drops['nan'] / n_total:.2%})")
    logger.info(f"    scartati, tutti uguali      : {drops['const']} ({drops['const'] / n_total:.2%})")
    logger.info(f"    scartati, tutti > 1         : {drops['over1']} ({drops['over1'] / n_total:.2%})")
    logger.info(f"    scartati, tutti < 0         : {drops['under0']} ({drops['under0'] / n_total:.2%})")
    logger.info(f"    TENUTI                      : {n_total - n_dropped} "
                f"({(n_total - n_dropped) / n_total:.2%})")

    ok_set = set(ok)
    train_ok = [f for f in files if str(f) in ok_set]
    val_ok = [f for f in val_files if str(f) in ok_set]

    stats: Dict = {}
    for t in times:
        q05, q95 = [], []
        for b in range(n_bands):
            args_b = (hists[t][b], n_below[t][b], n_above[t][b])
            edges_b = (float(edge_lo[b]), float(edge_hi[b]))
            q05.append(quantile_from_hist(*args_b, 0.05, *edges_b))
            q95.append(quantile_from_hist(*args_b, 0.95, *edges_b))

        # Il clamp al range naturale della riflettanza: i quantili possono solo
        # allargare la scala, mai stringerla sotto [0, 1].
        lo = [min(0.0, float(v)) for v in q05]
        hi = [max(1.0, float(v)) for v in q95]

        stats[t] = {
            "q05": [float(v) for v in q05],
            "q95": [float(v) for v in q95],
            "lo": lo,
            "hi": hi,
            "n_valid_px": [int(v) for v in n_valid[t]],
            "n_nonfinite_px": [int(v) for v in n_nonfinite[t]],
            "frac_nonfinite": [
                float(nf / (nf + nv)) if (nf + nv) else 0.0
                for nf, nv in zip(n_nonfinite[t], n_valid[t])
            ],
            "n_below_hist_range": [int(v) for v in n_below[t]],
            "n_above_hist_range": [int(v) for v in n_above[t]],
            "hist_edge_lo": [float(v) for v in edge_lo],
            "hist_edge_hi": [float(v) for v in edge_hi],
        }

    return stats, train_ok, val_ok, skipped, keep_map, drops, cube_drops, cube_diag


def write_split(
    name: str,
    files: Sequence[Path],
    out_dir: Path,
    n_time: int,
    n_bands: int,
    lo: List[float],
    hi: List[float],
    dtype: str,
    keep_map: Dict[str, List[int]],
    workers: int,
    chunk: int,
) -> Dict:
    # Gli istanti tenuti sono quelli selezionati nel primo passaggio, ristretti
    # alla finestra [0, n_time). Il numero di righe per cubo e' variabile.
    items: List[Tuple[str, List[int]]] = []
    for f in files:
        frames = [u for u in keep_map.get(str(f), []) if u < n_time]
        if frames:
            items.append((str(f), frames))

    n_rows = sum(len(fr) for _, fr in items)
    n_full = len(files) * n_time
    shape = (n_rows, n_bands, *EXPECTED_HW)
    out_path = out_dir / f"{name}.npy"

    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    logger.info(
        f"[{name}] {len(items)} cubi -> shape {shape} ({human_bytes(nbytes)})"
        f"  [scartate {n_full - n_rows} righe su {n_full}, "
        f"{(n_full - n_rows) / max(n_full, 1):.2%}]"
    )

    # Alloca il file (header .npy corretto) senza materializzare i dati in RAM.
    arr = open_memmap(out_path, mode="w+", dtype=np.dtype(dtype), shape=shape)
    del arr

    tasks = []
    row = 0
    for c in chunked(items, chunk):
        tasks.append((row, c))
        row += sum(len(fr) for _, fr in c)

    total_written = 0
    total_clipped = 0
    fill = {"slices": 0, "pixels": 0, "unfixable": 0}
    t0 = time.time()

    with mp.Pool(
        processes=workers,
        initializer=init_writer,
        initargs=(str(out_path), n_time, n_bands, lo, hi, dtype),
    ) as pool:
        for i, (w, c, f) in enumerate(pool.imap_unordered(write_chunk, tasks), start=1):
            total_written += w
            total_clipped += c
            for key, v in f.items():
                fill[key] += v
            log_progress(i, len(tasks), t0, f"scrittura {name}")

    denom = max(total_written * n_bands * 128 * 128, 1)
    frac_clipped = total_clipped / denom
    logger.info(f"[{name}] righe scritte: {total_written} | saturati: {frac_clipped:.2%}")
    logger.info(
        f"[{name}] riempimento spaziale 3x3: {fill['pixels']} pixel "
        f"su {fill['slices']} slice"
        + (f" | NON riempibili: {fill['unfixable']}" if fill["unfixable"] else "")
    )
    if fill["unfixable"]:
        # Con il filtro sui cubi attivo non dovrebbe piu' succedere: una slice
        # non riempibile e' una immagine senza nemmeno un pixel valido.
        logger.warning(
            f"[{name}] {fill['unfixable']} valori non riempibili azzerati: "
            f"e' stato usato --keep-empty-frame-cubes?"
        )

    if total_written != n_rows:
        raise RuntimeError(
            f"[{name}] righe scritte ({total_written}) != righe allocate ({n_rows})"
        )

    return {
        "path": str(out_path),
        "shape": list(shape),
        "dtype": dtype,
        "n_cubes": len(items),
        "n_time": n_time,
        "n_rows_dropped": int(n_full - n_rows),
        "frac_clipped": float(frac_clipped),
        "spatial_fill": fill,
        "cubes": [{"path": p, "frames": fr} for p, fr in items],
    }


def check_cubes(files: Sequence[Path],
                times: List[int],
                check_frames: int,
                n_bands: int,
                drop_thr: float,
                drop_cube_on_empty: bool,
                static_nan_thr: Optional[float],
                workers: int,
                chunk: int) -> Tuple[set, List[Tuple[str, str]]]:
    """
    Only the drop decision of the first pass, without histograms: which of
    `files` survive the checks. Used to choose the cubes when
    preprocessing.limit counts the cubes left after the drop.
    """
    if not files:
        return set(), []
    dummy = np.zeros(n_bands)
    tasks = [
        (c, times, n_bands, False, dummy, dummy, drop_thr, drop_cube_on_empty,
         static_nan_thr, check_frames)
        for c in chunked([str(f) for f in files], chunk)
    ]
    ok: set = set()
    skipped: List[Tuple[str, str]] = []
    t0 = time.time()
    with mp.Pool(processes=workers) as pool:
        for i, res in enumerate(pool.imap_unordered(hist_chunk, tasks), start=1):
            ok.update(res["ok"])
            skipped.extend(res["skipped"])
            log_progress(i, len(tasks), t0, "selezione")
    return ok, skipped


def select_cubes(files: List[Path],
                 pre,
                 times: List[int],
                 drop_cube_on_empty: bool) -> Tuple[List[Path], Dict, List[Tuple[str, str]]]:
    """
    The cubes the dataset is built from, following preprocessing.limit:

    - limit = 0: all the cubes on disk;
    - limit_after_drop = false: `limit` cubes evenly spaced over the sorted
      list; the checks of the first pass then drop the bad ones, so the
      dataset ends up with fewer than `limit` cubes;
    - limit_after_drop = true: candidates evenly spaced over the list are
      checked, enlarging the sample until `limit` of them survive, and
      `limit` survivors evenly spaced among them are kept.

    Returns (selected cubes, selection summary for dataset.json, cubes dropped
    while choosing -- only in the limit_after_drop mode).
    """
    summary = {"n_cubes_on_disk": len(files),
               "limit": pre.limit,
               "limit_after_drop": pre.limit_after_drop}

    if not pre.limit:
        summary["n_selected"] = len(files)
        return list(files), summary, []

    if not pre.limit_after_drop:
        selected = sample_evenly(files,
                                 pre.limit)
        summary["n_selected"] = len(selected)
        return selected, summary, []

    checked: Dict[str, bool] = {}
    skipped: List[Tuple[str, str]] = []
    n = min(pre.limit, len(files))
    while True:
        candidates = sample_evenly(files,
                                   n)
        todo = [f for f in candidates if str(f) not in checked]
        ok, dropped = check_cubes(todo,
                                  times,
                                  pre.check_frames,
                                  pre.n_bands,
                                  pre.drop_threshold,
                                  drop_cube_on_empty,
                                  pre.drop_static_nan_frac,
                                  pre.workers,
                                  pre.chunk)
        for f in todo:
            checked[str(f)] = str(f) in ok
        skipped.extend(dropped)
        survivors = [f for f in candidates if checked[str(f)]]
        logger.info(f"  selezione: {len(survivors)} cubi validi su {len(candidates)} "
                    f"candidati (obiettivo {pre.limit})")
        if len(survivors) >= pre.limit or len(candidates) >= len(files):
            break
        # enlarge the sample in proportion to the survival rate seen so far
        rate = max(len(survivors) / len(candidates), 0.05)
        n = min(len(files), max(n + 1, int(np.ceil(1.05 * pre.limit / rate))))

    selected = sample_evenly(survivors,
                             pre.limit)
    if len(selected) < pre.limit:
        logger.warning(f"solo {len(selected)} cubi superano i controlli su {len(files)} "
                       f"presenti: il dataset ne avra' meno dei {pre.limit} richiesti")
    # only the dropped candidates that would have been in the sample matter
    skipped = [(p, why) for p, why in skipped if not checked.get(p, True)]
    summary.update({"n_candidates_checked": len(checked),
                    "n_candidates_dropped": sum(1 for v in checked.values() if not v),
                    "n_selected": len(selected)})
    return selected, summary, skipped


def clear_output_dir(out_dir: Path,
                     overwrite: bool) -> None:
    """
    A folder that already holds a dataset (dataset.json is written last, so it
    is there only after a complete run) is never overwritten silently: models
    trained on it would no longer match its content. With --overwrite the
    files a previous run produced are removed first, so no stale .npy of a
    different configuration survives next to the new ones.
    """
    if not (out_dir / "dataset.json").is_file():
        return
    if not overwrite:
        raise SystemExit(
            f"{out_dir} already holds a preprocessed dataset (dataset.json). Change "
            f"preprocessing.name to build a new one next to it, or pass --overwrite "
            f"to replace it (models trained on it will then be refused by phase 2)."
        )
    produced = ["dataset.json", "stats.json", "cond_stats.json", "manifest.json",
                "config.yaml", "preprocessing.log"]
    for path in [out_dir / name for name in produced] + sorted(out_dir.glob("*.npy")):
        if path.is_file():
            path.unlink()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfgdir",
                    default="configuration_0.yaml",
                    help="config to read the `preprocessing` section from: a path or "
                         "the name of a file in registry/configurations/")
    ap.add_argument("--set",
                    nargs="*",
                    default=[],
                    metavar="KEY=VALUE",
                    help="config overrides, e.g. --set preprocessing.limit=200")
    ap.add_argument("--stats-only",
                    action="store_true",
                    help="write dataset.json, stats.json and cond_stats.json but not the "
                         ".npy files (enough for phase 2, not for phase 1)")
    ap.add_argument("--overwrite",
                    action="store_true",
                    help="replace the dataset already in paths.preprocessed_dir")
    ap.add_argument("--force",
                    action="store_true",
                    help="ignora il controllo spazio disco")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    # the yaml is resolved from the launch folder, everything else from the
    # project root (see src/config.py)
    config_path = resolve_config_path(args.cfgdir).resolve()
    use_project_dir()
    cfg = load_config(config_path,
                      overrides=args.set)
    pre = cfg.preprocessing
    data = cfg.data.params

    out_dir = Path(cfg.paths.preprocessed_dir)
    raw_dir = Path(pre.raw_dir)
    clear_output_dir(out_dir,
                     args.overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(out_dir / "preprocessing.log"))
    save_config(cfg,
                out_dir / "config.yaml",
                resolve=True)

    times = sorted(set(pre.times))
    drop_cube_on_empty = not pre.keep_empty_frame_cubes

    files = list_cubes(raw_dir)
    if not files:
        raise SystemExit(f"Nessun .npz trovato sotto {raw_dir}")

    logger.info(f"Dataset      : {pre.name} -> {out_dir}")
    logger.info(f"Cubi su disco: {len(files)} in {raw_dir}")
    logger.info(
        "Limite       : "
        + ("nessuno, tutti i cubi" if not pre.limit else
           f"{pre.limit} cubi " + ("DOPO lo scarto" if pre.limit_after_drop else
                                   "campionati, poi scartati quelli non validi"))
    )
    selected, selection, selection_skipped = select_cubes(files,
                                                          pre,
                                                          times,
                                                          drop_cube_on_empty)
    train_files, val_files = split_by_location(selected,
                                               pre.val_frac,
                                               pre.seed)

    logger.info(f"Cubi scelti  : {len(selected)}")
    for split, split_list in (("train", train_files), ("val", val_files)):
        logger.info(f"  {split:5s}      : {len(split_list)} cubi, "
                    f"{len({location_of(f) for f in split_list})} location")
    logger.info(f"Versioni T   : {times} (controlli sui primi "
                f"{max(max(times), pre.check_frames)} istanti)")
    logger.info(f"Bande        : prime {pre.n_bands}")
    logger.info(f"Output dtype : {pre.dtype}")
    logger.info(f"Scarto se    : > {pre.drop_threshold:.0%} di valori mancanti nell'istante")
    logger.info(
        f"Cubo intero  : {'SCARTATO' if drop_cube_on_empty else 'tenuto'} se contiene una "
        f"immagine interamente mancante ({ARRAY_KEY} o {MESO_KEY})"
    )
    logger.info(
        f"{STATIC_KEY:13s}: "
        + (f"cubo SCARTATO se ha > {pre.drop_static_nan_frac:.0%} di valori mancanti"
           if pre.drop_static_nan_frac is not None
           else "solo diagnosticata, non e' un criterio di scarto "
                "(preprocessing.drop_static_nan_frac per renderla tale)")
    )
    logger.info(f"Workers      : {pre.workers}")

    # --- pass 0 + 1: range e quantili ---------------------------------------
    logger.info(f"[1/4] Range per banda su {min(pre.range_sample, len(train_files))} "
                f"cubi campionati...")
    edge_lo, edge_hi = scan_ranges(
        train_files, pre.n_bands, max(times),
        pre.range_sample, pre.workers, pre.chunk,
    )
    for b in range(pre.n_bands):
        logger.info(f"    banda {b}: istogramma su [{edge_lo[b]:.4g}, {edge_hi[b]:.4g}]")

    logger.info("[2/4] Scarto cubi, selezione istanti e quantili 5%/95% per banda...")
    (stats, train_ok, held_ok, skipped, keep_map, drops, cube_drops,
     cube_diag) = compute_stats(
        train_files, val_files, times, pre.n_bands,
        edge_lo, edge_hi, pre.drop_threshold, pre.workers, pre.chunk,
        drop_cube_on_empty=drop_cube_on_empty,
        static_nan_thr=pre.drop_static_nan_frac,
        check_frames=pre.check_frames,
    )
    held = {str(f) for f in held_ok}
    val_ok = [f for f in val_files if str(f) in held]
    # in the limit_after_drop mode the bad cubes were already dropped while
    # choosing the sample: they are reported together with the others
    skipped = selection_skipped + skipped

    if not train_ok:
        raise SystemExit(
            "Nessun cubo di train sopravvissuto ai controlli: rilancia con "
            "preprocessing.keep_empty_frame_cubes=true per vedere quanti ne toglie il filtro."
        )
    if not val_ok:
        raise SystemExit(
            "Nessun cubo di validation sopravvissuto ai controlli: alza "
            "preprocessing.limit o preprocessing.val_frac."
        )

    if skipped:
        logger.warning(f"{len(skipped)} cubi scartati. Primi 5:")
        for p, why in skipped[:5]:
            logger.warning(f"  - {p}: {why}")

    for t in times:
        s = stats[t]
        logger.info(f"  T={t}")
        for b in range(pre.n_bands):
            logger.info(
                f"    banda {b}: q05={s['q05'][b]:.5f} q95={s['q95'][b]:.5f}"
                f"  ->  lo={s['lo'][b]:.5f} hi={s['hi'][b]:.5f}"
                f"  (NaN={s['frac_nonfinite'][b]:.2%})"
            )

    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps({
        "dataset": pre.name,
        "data_root": str(raw_dir),
        "n_bands": pre.n_bands,
        "times": times,
        "check_frames": pre.check_frames,
        "val_frac": pre.val_frac,
        "seed": pre.seed,
        "hist_bins": HIST_BINS,
        "normalization": (
            "lo = min(0, q05); hi = max(1, q95); "
            "x_norm = 2*(x - lo)/(hi - lo) - 1, clip [-1,1]"
        ),
        "drop_threshold": pre.drop_threshold,
        "drop_cube_on_empty_frame": drop_cube_on_empty,
        "drop_static_nan_frac": pre.drop_static_nan_frac,
        "missing_policy": (
            "CUBO scartato per intero se contiene almeno una immagine (slice 2D "
            f"banda x istante) interamente mancante, in {ARRAY_KEY} o in {MESO_KEY}. "
            "Nei cubi tenuti, istante scartato se: frazione mancanti > "
            "drop_threshold, oppure tutti i valori uguali tra loro, oppure tutti "
            "> 1, oppure tutti < 0. Altrimenti i mancanti sparsi sono riempiti "
            "con la media dell'intorno 3x3. "
            + (f"{STATIC_KEY} scarta il cubo sopra "
               f"{pre.drop_static_nan_frac:.0%} di mancanti."
               if pre.drop_static_nan_frac is not None
               else f"{STATIC_KEY} e' solo diagnosticata, non scarta.")
        ) if drop_cube_on_empty else (
            "istante scartato se: frazione mancanti > drop_threshold, oppure "
            "tutti i valori uguali tra loro, oppure tutti > 1, oppure tutti < 0. "
            "Altrimenti i mancanti sparsi sono riempiti con la media "
            "dell'intorno 3x3."
        ),
        "drops": drops,
        # cube_drops: attribuzione al primo criterio che scatta, somma al numero
        # di cubi buttati. cube_diag: conteggi per variabile su tutti i cubi
        # leggibili, quindi indipendenti dall'ordine dei criteri.
        "cube_drops": cube_drops,
        "cube_diag": cube_diag,
        "stats": {str(t): stats[t] for t in times},
        "skipped": [{"path": p, "reason": r} for p, r in skipped],
        "keep_map": keep_map,
    }, indent=2))
    logger.info(f"Statistiche salvate in {stats_path}")

    # --- pass 2: statistiche delle condizioni -------------------------------
    # min/max of mesodynamic and highresstatic on the train split, with the
    # same function and parameters the phase-2 dataloader uses: it finds them
    # ready and consistent instead of computing them at its first start.
    logger.info("[3/4] Statistiche di mesodynamic e highresstatic sul train...")
    ensure_condition_stats(str(out_dir / "cond_stats.json"),
                           train_ok,
                           meso_channels=data.meso_channels,
                           T=data.T,
                           steps_per_frame=data.meso_steps_per_frame,
                           meso_center_crop=data.meso_center_crop,
                           meso_agg=data.meso_agg,
                           workers=pre.workers,
                           force=True)

    # --- spazio disco, ora che si conosce il numero reale di righe ----------
    splits = {"train": train_ok, "val": val_ok}
    itemsize = np.dtype(pre.dtype).itemsize
    row_bytes = pre.n_bands * 128 * 128 * itemsize
    need = 0
    for t in times:
        for f in train_ok + val_ok:
            need += len([u for u in keep_map.get(str(f), []) if u < t]) * row_bytes
    free = shutil.disk_usage(out_dir).free
    logger.info(f"Spazio richiesto: {human_bytes(need)} | disponibile: {human_bytes(free)}")
    if not args.stats_only and need > free and not args.force:
        raise SystemExit("Spazio su disco insufficiente. Usa --force per procedere comunque.")

    # --- pass 3: scrittura --------------------------------------------------
    manifest: Dict = {}
    if args.stats_only:
        logger.info("[4/4] --stats-only: niente .npy (servono solo alla fase 1)")
    else:
        logger.info("[4/4] Scrittura degli array normalizzati...")
        for t in times:
            lo = stats[t]["lo"]
            hi = stats[t]["hi"]
            for split, split_list in splits.items():
                if not split_list:
                    continue
                tag = f"{split}_T{t}"
                manifest[tag] = write_split(
                    tag, split_list, out_dir, t, pre.n_bands,
                    lo, hi, pre.dtype, keep_map, pre.workers, pre.chunk,
                )
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        logger.info(f"Manifest salvato in {manifest_path}")

    # --- dataset.json, per ultimo: c'e' solo se il run e' arrivato in fondo ---
    split_lists = {split: [f.relative_to(raw_dir).as_posix() for f in split_list]
                   for split, split_list in splits.items()}
    info = {
        "name": pre.name,
        "id": compute_dataset_id(split_lists,
                                 {str(t): stats[t] for t in times}),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "raw_dir": str(raw_dir),
        "preprocessing": OmegaConf.to_container(pre, resolve=True),
        "selection": selection,
        "composition": {split: {"n_cubes": len(split_list),
                                "n_locations": len({location_of(f) for f in split_list})}
                        for split, split_list in splits.items()},
        "n_cubes_dropped": len(skipped),
        "files": {"stats": "stats.json",
                  "cond_stats": "cond_stats.json",
                  "manifest": "manifest.json" if manifest else None,
                  "npy": sorted(f"{tag}.npy" for tag in manifest)},
        "splits": split_lists,
    }
    write_dataset_info(dataset_info_path(out_dir),
                       info)
    logger.info(f"Dataset {pre.name} (id {info['id']}) descritto in {dataset_info_path(out_dir)}")
    logger.info("Fatto.")


if __name__ == "__main__":
    main()
