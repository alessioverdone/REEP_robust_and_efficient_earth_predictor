#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metriche di valutazione di un DiT gia' addestrato su uno o piu' split: ENS
(EarthNetScore, con MAD, OLS, EMD, SSIM), SSIM grezza, P-MAE e N-MAE.

    # validation e training
    python src/scripts/evaluate_ens.py --model [date] --split val train

    # split di test EarthNet, K=5 campioni per cubo
    python src/scripts/evaluate_ens.py --model [date]
        --split iid ood extreme seasonal --num-samples 5

    # giro di prova su 50 cubi presi a caso
    python src/scripts/evaluate_ens.py --model [date] --split iid --subset 50

Argomenti principali
--------------------
--model   nome del run addestrato = cartella con il timestamp sotto
          registry/outputs/diffusion/ (anche un pezzo del nome se e' unico,
          'latest' o il percorso della cartella). Stessa risoluzione di
          visualize_predictions.py.
--split   uno o piu' fra
              train, val, test     split di dataset.json (cubi da data/raw/train)
              iid, ood, extreme    split di test EarthNet: data/raw/<split>_test_split/
                                   {context,target}
              seasonal             come sopra, ma con finestre piu' lunghe
          (anche i nomi completi: iid_test_split, ...)

Finestre temporali
------------------
    train / val / test / iid / ood / extreme   10 frame di contesto + 20 target (T = 30)
    seasonal                                    20 frame di contesto + 40 target (T = 60)

Cosa calcola
------------
    ENS     EarthNetScore, media armonica dei quattro sottopunteggi
      MAD       valore        deviazione assoluta mediana sulle 4 bande
      OLS       trend         pendenza OLS delle serie NDVI per pixel
      EMD       distribuzione earth mover distance fra le distribuzioni NDVI
      SSIM      percettiva    similarita' strutturale sulle 4 bande
    SSIM_RAW  la stessa SSIM prima dell'elevazione a potenza (fuori dall'ENS)
    P-MAE   errore assoluto medio in PIXEL, sulle 4 bande di riflettanza
    N-MAE   errore assoluto medio sull'NDVI

Tutti i numeri sono calcolati sui soli frame PREDETTI (20, o 40 per seasonal) e nello
spazio della riflettanza [0, 1], non in quello normalizzato [-1, 1] del training: e'
li' che l'EarthNetScore e' definito.

OLS e' la chiamata diretta a earthnet.parallel_score.CubeCalculator (lo scoring
ufficiale). MAD ed EMD sono riscritture vettorizzate che danno lo stesso numero del
toolkit (--scoring toolkit torna alle funzioni originali, per verificarlo). SSIM e'
riscritta perche' quella del toolkit chiama skimage.metrics.structural_similarity
senza data_range, che da scikit-image 0.23 solleva ValueError sulle immagini float:
stessa funzione (soglia del 70% di pixel validi per frame, pixel mascherati sostituiti
dalla predizione, esponente 10.31885115), con data_range esplicito. Il default 2.0 e'
quello con cui sono calibrati l'esponente e i punteggi pubblicati.

Nuvole
------
Come nello scoring ufficiale i pixel nuvolosi sono esclusi dal confronto, ma il
modello non vede la maschera: riceve gli stessi input del training. La maschera si
legge dai file .npz originali (il dataloader attuale scarta le bande oltre la quarta):

Output
------
registry/outputs/diffusion/<run>/results/statistics/
    metrics_results_<split>_K_<K>.json   aggregati + metadati (checkpoint, finestre,
                                         maschera, tempi) + una voce per cubo
    metrics_results_<split>_K_<K>.csv    una riga per cubo, per gli istogrammi
Ogni esecuzione riscrive i file del proprio split e K, senza toccare gli altri.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# visualize_predictions.py sta nella stessa cartella: lo si importa da li' anche
# quando lo script e' lanciato da un'altra cartella o da un IDE. Il suo import mette
# anche la root del progetto e src/model su sys.path, e serve prima di ogni
# import `src.` o `sgm.`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import visualize_predictions as viz  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from accelerate.utils import set_seed  # noqa: E402
from earthnet.parallel_score import CubeCalculator  # noqa: E402
from skimage.metrics import structural_similarity  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from src.config import PathsCfg, project_path, use_project_dir  # noqa: E402
from src.dataset.dataset_info import load_dataset_info  # noqa: E402

log = logging.getLogger("evaluate_ens")

# Sottocartella dei risultati del run in cui si scrivono i file.
STATISTICS_DIRNAME = "statistics"

# --- costanti dello scoring EarthNet, da earthnet-toolkit 0.3.11 ------------
SSIM_SCALING = 10.31885115          # porta un SSIM di 0.8 a un ssim-score di 0.1
SSIM_MIN_VALID_FRACTION = 0.7       # un frame entra se e' valido per almeno il 70%
SSIM_DATA_RANGE = 2.0               # il default di scikit-image per i float ai tempi

# Esponenti di CubeCalculator.MAD e CubeCalculator.EMD, copiati dal toolkit: cambiarli
# sposterebbe la scala rispetto ai punteggi pubblicati.
MAD_SCALING = 0.06649346971087526
EMD_SCALING = 0.10082047548620601

# Maschera nuvole: probabilita' di nuvola (canale 4 dei cubi a 7 canali) sotto cui un
# pixel e' valido, e fondo scala del canale (EarthNet la da' in percentuale, 0-100).
CLOUD_PROB_THRESHOLD = 0.20
CLOUD_PROB_SCALE = 100.0
CLOUD_PROB_CHANNEL = 4
N_CHANNELS_WITH_CLOUD_PROB = 7

# Implementazioni di MAD ed EMD: 'fast' sono le riscritture vettorizzate di questo
# file, 'toolkit' le funzioni originali di earthnet. Stesso numero.
SCORING_MODES = ("fast", "toolkit")

# Indici delle bande dentro highresdynamic, nell'ordine del dataloader
# (blue, green, red, nir).
BAND_RED = 2
BAND_NIR = 3


# ===========================================================================
# 1. Cubi dello split, finestre e maschera nuvole
# ===========================================================================
# cubo letto dal dataset (percorso .npz) -> (file da cui leggere la maschera, primo
# istante dei frame target dentro quel file)
MaskSources = Dict[str, Tuple[Path, int]]


def select_cubes(pool: Sequence[Path], args) -> List[Path]:
    """
    Tutti i cubi dello split (ordinati), N a caso con --subset (stesso seed = stessi
    cubi su modelli diversi) o i primi N con --max-cubes.
    """
    ordered = sorted(pool)
    if args.subset:
        return viz.select_samples(ordered, args.subset, args.subset_seed)
    if args.max_cubes:
        return ordered[: args.max_cubes]
    return ordered


def merge_pairs(pairs: Sequence[Tuple[Path, Path, Path]], n_context: int, n_target: int,
                n_bands: int, steps_per_frame: int, workers: int) -> None:
    """Fonde context+target dei cubi scelti, in parallelo (la fusione e' gia' in cache se valida)."""
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(viz.merge_pair, ctx, tgt, merged, n_context, n_target,
                               n_bands, steps_per_frame)
                   for merged, ctx, tgt in pairs]
        for fut in futures:
            fut.result()
            done += 1
            if done % 250 == 0 or done == len(futures):
                rate = (time.time() - t0) / done
                log.info(f"[merge] {done}/{len(futures)} cubi ({rate:.2f} s/cubo, "
                         f"~{rate * (len(futures) - done) / 60:.1f} min alla fine)")


def build_split(spec: viz.SplitSpec, cfg, args, n_context: int,
                n_target: int) -> Tuple[List[Path], MaskSources, str]:
    """
    (cubi scelti, sorgenti della maschera, descrizione) di uno split, con i cubi
    gia' pronti da leggere: per gli split EarthNet contesto e target sono fusi in
    un .npz per cubo (solo quelli scelti).
    """
    params = cfg.data.params
    if not spec.earthnet:
        pool, desc, prepare = viz.build_pool(spec, cfg, args, n_context, n_target)
        selected = prepare(select_cubes(pool, args))
        return selected, {str(f): (f, n_context) for f in selected}, desc

    test_dir = (Path(args.test_root) / f"{spec.name}_test_split" if args.test_root
                else project_path(cfg.paths.raw_dir).parent / f"{spec.name}_test_split")
    merged_root = (Path(args.merged_root) if args.merged_root
                   else project_path(cfg.paths.preprocessed_dir) / "merged_test")
    pairs = viz.list_pairs(test_dir, merged_root / spec.name)
    by_merged = {merged: (ctx, tgt) for merged, ctx, tgt in pairs}

    selected = select_cubes(sorted(by_merged), args)
    merge_pairs([(m, *by_merged[m]) for m in selected], n_context, n_target,
                int(params.n_channels), int(params.meso_steps_per_frame), args.merge_workers)
    # i frame target stanno interi nel file di target
    sources = {str(m): (by_merged[m][1], 0) for m in selected}
    desc = f"test {spec.name} ({test_dir.as_posix()}, {len(pairs)} coppie context/target)"
    return selected, sources, desc


def valid_mask(hd: np.ndarray, threshold: float, prob_scale: float) -> np.ndarray:
    """
    highresdynamic (H, W, C, T) di un file originale -> (T, H, W) bool, True = pixel
    da usare. Dispatch sul numero di canali, come i due formati di EarthNet:
    7 canali = cloud probability al canale 4, 5 canali = maschera binaria in fondo.
    I valori non finiti contano come pixel da scartare.
    """
    if hd.shape[2] >= N_CHANNELS_WITH_CLOUD_PROB:
        valid = (hd[:, :, CLOUD_PROB_CHANNEL, :] / prob_scale) < threshold
    else:
        valid = hd[:, :, -1, :] == 0
    return np.transpose(valid, (2, 0, 1))


class MaskedCubeDataset(Dataset):
    """
    Il dataset di valutazione piu' la maschera nuvole dei frame target,
    "cloud_mask" (n_target, 1, H, W), 1 = pixel valido, alla stessa risoluzione
    delle immagini: con data.params.resize < 128 si usa l'interpolazione area come
    per le immagini e un pixel e' valido solo se lo sono tutti i suoi sotto-pixel.
    """

    def __init__(self, inner: Dataset, sources: MaskSources, n_target: int,
                 threshold: float, prob_scale: float, resize: Optional[int]):
        self.inner = inner
        self.sources = sources
        self.n_target = n_target
        self.threshold = threshold
        self.prob_scale = prob_scale
        self.resize = resize

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> Dict:
        item = self.inner[idx]
        path, offset = self.sources[item["cube_path"]]
        with np.load(path) as z:
            hd = z["highresdynamic"][:, :, :, offset: offset + self.n_target]
        if hd.shape[3] != self.n_target:
            raise ValueError(f"{path}: {hd.shape[3]} frame target dal frame {offset}, "
                             f"ne servono {self.n_target}")
        mask = torch.from_numpy(valid_mask(hd, self.threshold, self.prob_scale)
                                .astype(np.float32)).unsqueeze(1)          # (n, 1, H, W)
        if self.resize is not None and tuple(mask.shape[-2:]) != (self.resize, self.resize):
            mask = (F.interpolate(mask, size=(self.resize, self.resize), mode="area")
                    >= 0.999).float()
        item["cloud_mask"] = mask
        return item


def build_loader(cfg, files: List[Path], sources: MaskSources, args, n_context: int,
                 n_target: int, trained_T: int) -> Tuple[Dataset, DataLoader]:
    """
    (dataset, loader) sui cubi scelti: le statistiche di normalizzazione sono quelle
    del training, senza flip, senza shuffle e senza drop_last (nessun cubo deve
    sparire e l'ordine deve essere lo stesso a ogni esecuzione).
    """
    params = cfg.data.params
    dataset = viz.build_eval_dataset(cfg, files, n_context, n_target)
    if args.cloud_mask:
        dataset = MaskedCubeDataset(dataset, sources, n_target, args.cloud_prob_threshold,
                                    args.cloud_prob_scale, params.resize)

    if args.batch_size:
        batch_size = int(args.batch_size)
    else:
        # il batch del config e' tarato sulla T del training: con sequenze piu' lunghe
        # (seasonal) la memoria per cubo cresce in proporzione, e il batch cala
        batch_size = max(1, min(int(params.batch_size),
                                int(params.batch_size) * trained_T // (n_context + n_target)))
    num_workers = int(params.num_workers if args.num_workers is None else args.num_workers)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                        drop_last=False)
    return dataset, loader


# ===========================================================================
# 2. Inferenza: K predizioni in pixel
# ===========================================================================
def predict_pixels(batch: Dict, bundle: viz.DitBundle, device: torch.device, n_target: int,
                   n_samples: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    (K predizioni, target), ognuno (B, n_target, C, H, W) in [-1, 1] su CPU.

    Ogni predizione e' una passata completa di log_videos_local_and_wandb, la
    funzione di validazione del trainer (encode col VAE, rumore iniziale miscelato al
    contesto, EulerEDMSampler con CFG, decode): la procedura e' esattamente quella
    del training. Quella funzione decodifica un solo campione per chiamata, e per
    avere K campioni in pixel la si chiama K volte (K nuovi rumori iniziali).
    """
    from src.model import diffusion_utils as du

    cfg = bundle.cfg
    preds: List[torch.Tensor] = []
    target = None
    for _ in range(n_samples):
        with torch.no_grad(), viz.autocast_context(device, bundle.mixed_precision):
            _, _, forecast = du.log_videos_local_and_wandb(
                batch=batch,
                model=bundle.model,
                conditioner=bundle.conditioner,
                first_stage_model=bundle.first_stage_model,
                sampler=bundle.sampler,
                denoiser=bundle.denoiser,
                input_key=cfg.diffusion.input_key,
                config_diffusion=cfg.diffusion,
                noise_image_num=n_target,
                # argomenti obbligatori della firma, non usati nel corpo
                metrics_rgbn=None,
                metrics_ndvi=None,
                metrics_arvi=None,
                past_weigth=cfg.diffusion.loss_fn_config.params.past_weight,
                val_num_samples=1,
                save_dir=None,
                save_counter=None,
                save_limit=0,
            )
        # .cpu() su CUDA sincronizza: il tempo misurato fuori e' quello vero
        preds.append(forecast["pred"].detach().float().cpu())
        target = forecast["target"].detach().float().cpu()
    return preds, target


# ===========================================================================
# 3. Da tensore del trainer a cubo EarthNet
# ===========================================================================
def to_cube(x: torch.Tensor, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """
    (T, C, H, W) normalizzato in [-1, 1] -> (H, W, C, T) in riflettanza [0, 1].

    Prima l'inversa esatta della normalizzazione del dataloader (denormalize), poi il
    clipping in [0, 1], applicato da CubeCalculator.load_file() a predizioni e target.
    """
    arr = viz.denormalize(x, lo, hi)                 # (T, C, H, W), riflettanza
    arr = np.transpose(arr, (2, 3, 1, 0))            # (H, W, C, T)
    np.nan_to_num(arr, copy=False, nan=0.0)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def to_mask(m: Optional[torch.Tensor], shape: Tuple[int, int, int, int]) -> np.ndarray:
    """
    (T, 1, H, W) -> (H, W, C, T) float32, 1 = pixel valido, ripetuta sulle bande.
    Con m=None (--no-cloud-mask) la maschera e' tutta a 1.
    """
    h, w, c, t = shape
    if m is None:
        return np.ones((h, w, c, t), dtype=np.float32)
    arr = np.transpose(m.numpy().astype(np.float32), (2, 3, 1, 0))   # (H, W, 1, T)
    return np.repeat(arr, c, axis=2)


def ndvi(cube: np.ndarray) -> np.ndarray:
    """(H, W, C, T) -> NDVI (H, W, 1, T), con la stessa epsilon del toolkit."""
    red = cube[:, :, BAND_RED, :]
    nir = cube[:, :, BAND_NIR, :]
    return ((nir - red) / (nir + red + 1e-6))[:, :, np.newaxis, :]


# ===========================================================================
# 4. Sottopunteggi e MAE di un cubo
# ===========================================================================
def mad_score(preds: np.ndarray, targs: np.ndarray,
              masks: np.ndarray) -> Optional[float]:
    """
    mad-score di un cubo, (H, W, C, T) -> [0, 1], 1 = migliore. Stesso numero di
    CubeCalculator.MAD (differenza esatta 0), ~5x piu' veloce: niente debug_info, e
    siccome x**s e' monotona per s > 0 la mediana delle distanze elevate e' la
    mediana elevata, quindi basta np.partition per le statistiche d'ordine centrali
    (una se n e' dispari, due mediate DOPO l'elevazione a potenza se e' pari).
    """
    dists = np.abs(preds - targs)[masks == 1]
    n = dists.size
    if n == 0:
        return None

    k = n // 2
    if n % 2:
        median = float(np.partition(dists, k)[k]) ** MAD_SCALING
    else:
        two = np.partition(dists, [k - 1, k])[k - 1:k + 1].astype(np.float64)
        median = float((two ** MAD_SCALING).mean())

    return float(max(0.0, min(1.0, 1.0 - median)))


def emd_score(preds: np.ndarray, targs: np.ndarray,
              masks: np.ndarray) -> Optional[float]:
    """
    emd-score di un cubo NDVI, (H, W, 1, T) -> [0, 1], 1 = migliore. Stessa formula e
    stesso numero di CubeCalculator.EMD, ~12x piu' veloce.

    Il W1 fra due distribuzioni empiriche 1-D e' l'integrale di |CDF_pred - CDF_targ|,
    una somma finita sui salti delle CDF: i 2T valori di ogni pixel si ordinano una
    volta sola (un argsort su (H*W, 2T)) portandosi dietro due indicatori interi, i
    cumsum sono i conteggi delle CDF e W1 e' il prodotto scalare fra il loro scarto e
    i delta fra valori consecutivi. Gli istanti mascherati del target vanno sul
    massimo della riga (non a +inf, che darebbe delta infiniti per 0). Come nel
    toolkit, i pixel con meno di due istanti validi nel target restano fuori dalla
    media. None se non ne resta nessuno.
    """
    n_t = preds.shape[-1]
    pred = np.reshape(preds, (-1, n_t)).astype(np.float64)
    targ = np.reshape(targs, (-1, n_t)).astype(np.float64)
    valid = np.reshape(masks, (-1, n_t)) == 1

    counts = valid.sum(1)
    usable = counts > 1
    if not usable.any():
        return None
    pred, targ, valid = pred[usable], targ[usable], valid[usable]
    counts = counts[usable].astype(np.float64)[:, np.newaxis]

    top = np.maximum(pred.max(1), np.where(valid, targ, -np.inf).max(1))
    targ = np.where(valid, targ, top[:, np.newaxis])

    zeros = np.zeros(pred.shape, dtype=np.int8)
    ones = np.ones(pred.shape, dtype=np.int8)
    values = np.concatenate([pred, targ], axis=1)
    from_pred = np.concatenate([ones, zeros], axis=1)
    from_targ = np.concatenate([zeros, valid.astype(np.int8)], axis=1)

    order = np.argsort(values, axis=1)
    values = np.take_along_axis(values, order, axis=1)

    def cdf(flags: np.ndarray, size) -> np.ndarray:
        counted = np.cumsum(np.take_along_axis(flags, order, axis=1),
                            axis=1, dtype=np.int32)[:, :-1]
        return counted / size

    gap = np.abs(cdf(from_pred, float(n_t)) - cdf(from_targ, counts))
    dists = np.sum(gap * np.diff(values, axis=1), axis=1)

    return float(max(0.0, min(1.0, 1.0 - (dists ** EMD_SCALING).mean())))


def ssim_scores(preds: np.ndarray, targs: np.ndarray, masks: np.ndarray,
                data_range: float) -> Tuple[Optional[float], Optional[float]]:
    """
    (ssim grezza, ssim-score) di un cubo, (H, W, C, T) -> due valori in [0, 1].

    Riscrittura di CubeCalculator.SSIM, identica tranne il data_range esplicito e il
    fatto che anche il valore pre-esponente viene restituito. Invariati: pixel
    mascherati del target sostituiti con la predizione (la finestra 7x7 resta piena),
    un frame entra nella media solo se e' valido per piu' del 70%, la media e'
    elevata a SSIM_SCALING. (None, None) se nessun frame supera la soglia.
    """
    ssim_targs = np.where(masks, targs, preds)
    new_shape = (-1, preds.shape[0], preds.shape[1])

    def flatten(a: np.ndarray) -> np.ndarray:
        """(H, W, C, T) -> (H, W, C*T): ogni banda di ogni frame e' un'immagine."""
        return np.transpose(np.reshape(np.transpose(a, (3, 2, 0, 1)), new_shape), (1, 2, 0))

    ssim_targs = flatten(ssim_targs)
    ssim_preds = flatten(preds)
    ssim_masks = flatten(masks)

    running, counts = 0.0, 0
    for i in range(ssim_targs.shape[-1]):
        frame_mask = ssim_masks[:, :, i]
        if frame_mask.sum() > SSIM_MIN_VALID_FRACTION * frame_mask.size:
            running += structural_similarity(
                ssim_targs[:, :, i], ssim_preds[:, :, i], data_range=data_range
            )
            counts += 1

    if counts == 0:
        return None, None
    raw = max(0.0, running / counts)
    return float(raw), float(raw ** SSIM_SCALING)


def cube_subscores(preds: np.ndarray, targs: np.ndarray, masks: np.ndarray,
                   ssim_data_range: float,
                   scoring: str = "fast") -> Dict[str, Optional[float]]:
    """
    I quattro sottopunteggi ENS di un cubo, piu' la SSIM grezza. Tutti in [0, 1],
    1 = migliore. OLS e' sempre la chiamata al toolkit (gia' vettorizzata sui pixel);
    riceve copie perche' azzera in place le serie con meno di due istanti validi.
    """
    ndvi_preds, ndvi_targs = ndvi(preds), ndvi(targs)
    ndvi_masks = masks[:, :, :1, :]

    ols, _ = CubeCalculator.OLS(ndvi_preds.copy(), ndvi_targs.copy(), ndvi_masks.copy())

    if scoring == "toolkit":
        mad, _ = CubeCalculator.MAD(preds.copy(), targs.copy(), masks.copy())
        emd, _ = CubeCalculator.EMD(ndvi_preds.copy(), ndvi_targs.copy(),
                                    ndvi_masks.copy())
    else:
        mad = mad_score(preds, targs, masks)
        emd = emd_score(ndvi_preds, ndvi_targs, ndvi_masks)

    ssim_raw, ssim = ssim_scores(preds, targs, masks, ssim_data_range)

    return {"MAD": mad, "OLS": ols, "EMD": emd,
            "SSIM": ssim, "SSIM_RAW": ssim_raw}


def cube_errors(preds: np.ndarray, targs: np.ndarray,
                masks: np.ndarray) -> Dict[str, float]:
    """
    Somme e conteggi di P-MAE e N-MAE per un cubo. Somme e non medie perche'
    l'aggregato e' pesato sui pixel validi: la media delle medie darebbe lo stesso
    peso a un cubo quasi tutto nuvoloso e a uno sgombro.
    """
    valid = masks == 1
    p_abs = np.abs(preds - targs)[valid]

    ndvi_valid = masks[:, :, 0, :] == 1
    n_abs = np.abs(ndvi(preds) - ndvi(targs))[:, :, 0, :][ndvi_valid]

    return {
        "p_sum": float(p_abs.sum()), "p_count": int(p_abs.size),
        "n_sum": float(n_abs.sum()), "n_count": int(n_abs.size),
    }


def harmonic_mean(values: Sequence[Optional[float]]) -> Optional[float]:
    """
    Media armonica dei sottopunteggi, come EarthNetScore.__harmonic_mean. Il filtro
    e' quello del toolkit: scarta i None e, per effetto di filter(None, ...), anche
    gli zeri esatti; lo si mantiene per non spostare la scala dei punteggi pubblicati.
    """
    vals = [v for v in values if v]
    if not vals:
        return None
    return min(1.0, len(vals) / sum(1.0 / (v + 1e-8) for v in vals))


def score_cube(pred_cubes: Sequence[np.ndarray], targs: np.ndarray,
               masks: np.ndarray, ssim_data_range: float,
               scoring: str = "fast") -> Dict[str, object]:
    """
    Riga di risultati di un cubo: sottopunteggi, ENS, somme di P-MAE e N-MAE. Con piu'
    campioni si tiene quello con l'ENS piu' alto (best-of-K di
    EarthNetScore.__get_best_sample); SSIM grezza e MAE sono quelle dello stesso
    campione. Un cubo interamente nuvoloso non ha pixel su cui confrontare: i suoi
    sottopunteggi restano None ed esce dalle medie.
    """
    if masks.sum() == 0:
        return {"MAD": None, "OLS": None, "EMD": None, "SSIM": None,
                "SSIM_RAW": None, "ENS": None,
                "p_sum": 0.0, "p_count": 0, "n_sum": 0.0, "n_count": 0,
                "valid_fraction": 0.0, "sample": -1}

    best: Optional[Dict[str, object]] = None
    for k, preds in enumerate(pred_cubes):
        sub = cube_subscores(preds, targs, masks, ssim_data_range, scoring)
        ens = harmonic_mean([sub["MAD"], sub["OLS"], sub["EMD"], sub["SSIM"]])
        better = best is None or (
            ens is not None and (best["ENS"] is None or ens > best["ENS"])
        )
        if better:
            best = {**sub, "ENS": ens, "sample": k,
                    **cube_errors(preds, targs, masks)}

    best["valid_fraction"] = float(masks.mean())
    return best


# ===========================================================================
# 5. Aggregazione, salvataggio, report
# ===========================================================================
def aggregate(rows: Sequence[Dict[str, object]]) -> Dict[str, Optional[float]]:
    """
    Dai risultati per cubo ai numeri finali. I quattro sottopunteggi si mediano sui
    cubi e l'ENS e' la media armonica delle quattro medie (non la media degli ENS per
    cubo: e' l'ordine di EarthNetScore.summarize()). SSIM_RAW e' mediata allo stesso
    modo ma resta fuori dall'ENS. P-MAE e N-MAE sono medie pesate sui pixel validi.
    """
    def mean_of(key: str) -> Optional[float]:
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    mad, ols, emd, ssim = (mean_of(k) for k in ("MAD", "OLS", "EMD", "SSIM"))
    ssim_raw = mean_of("SSIM_RAW")

    p_sum = sum(float(r["p_sum"]) for r in rows)
    p_count = sum(int(r["p_count"]) for r in rows)
    n_sum = sum(float(r["n_sum"]) for r in rows)
    n_count = sum(int(r["n_count"]) for r in rows)

    return {
        "ENS": harmonic_mean([mad, ols, emd, ssim]),
        "MAD": mad, "OLS": ols, "EMD": emd, "SSIM": ssim, "SSIM_RAW": ssim_raw,
        "P_MAE": p_sum / p_count if p_count else None,
        "N_MAE": n_sum / n_count if n_count else None,
        "valid_fraction": (float(np.mean([r["valid_fraction"] for r in rows]))
                           if rows else None),
    }


def save_results(out_dir: Path, split: str, n_samples: int,
                 payload: Dict[str, object]) -> Tuple[Path, Path]:
    """
    metrics_results_<split>_K_<K>.json (aggregati + metadati + per cubo) e .csv (una
    riga per cubo) in <run>/results/statistics/. Riscritti a ogni esecuzione.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"metrics_results_{split}_K_{n_samples}"

    json_path = out_dir / f"{stem}.json"
    viz.write_json(str(json_path), payload)

    csv_path = out_dir / f"{stem}.csv"
    fields = ["cube", "MAD", "OLS", "EMD", "SSIM", "SSIM_RAW", "ENS",
              "P_MAE", "N_MAE", "valid_fraction", "sample"]
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in payload["per_cube"]:
            writer.writerow({k: row.get(k) for k in fields})

    return json_path, csv_path


def _fmt(value: Optional[float]) -> str:
    return "n/d" if value is None else f"{value:.4f}"


def print_report(payload: Dict[str, object]) -> None:
    agg, run, timing = payload["aggregate"], payload["run"], payload["timing"]

    print()
    print("=" * 62)
    print(f"  {run['model']}  |  split {run['split'].upper()}")
    print(f"  checkpoint {run['checkpoint']} (step {run['step']})")
    print(f"  {run['n_cubes']} cubi, {run['n_context']} frame di contesto + "
          f"{run['n_target']} predetti, K={run['n_samples']}")
    print("=" * 62)
    print()
    print(f"  ENS   EarthNetScore            {_fmt(agg['ENS'])}")
    print(f"          MAD   valore           {_fmt(agg['MAD'])}")
    print(f"          OLS   trend            {_fmt(agg['OLS'])}")
    print(f"          EMD   distribuzione    {_fmt(agg['EMD'])}")
    print(f"          SSIM  percettiva       {_fmt(agg['SSIM'])}")
    print()
    print(f"  SSIM  grezza, fuori dall'ENS   {_fmt(agg['SSIM_RAW'])}")
    print(f"  P-MAE pixel, 4 bande           {_fmt(agg['P_MAE'])}")
    print(f"  N-MAE NDVI                     {_fmt(agg['N_MAE'])}")
    print()
    if agg["valid_fraction"] is not None:
        print(f"  pixel non nuvolosi             {agg['valid_fraction']:.1%}")
    if run["n_cubes_skipped"]:
        print(f"  cubi esclusi (tutti nuvolosi)  {run['n_cubes_skipped']}")
    print(f"  tempo: {timing['total_s']:.0f} s  "
          f"({timing['inference_s']:.0f} s inferenza + "
          f"{timing['scoring_s']:.0f} s scoring [{timing['scoring']}] + "
          f"{timing['data_s']:.0f} s dati) = {timing['s_per_cube']:.2f} s/cubo")
    print()


def print_comparison(payloads: Sequence[Dict[str, object]]) -> None:
    """Tabella di confronto fra split (stesso modello), nell'ordine in cui sono stati chiesti."""
    width = max(len(str(p["run"]["split"])) for p in payloads)
    print()
    print(f"{'split':{width}s} {'cubi':>5s} {'K':>2s} {'px.val':>7s} "
          f"{'ENS':>8s} {'MAD':>8s} {'OLS':>8s} {'EMD':>8s} {'SSIM':>8s} "
          f"{'SSIMraw':>8s} {'P-MAE':>8s} {'N-MAE':>8s}")
    for p in payloads:
        agg, run = p["aggregate"], p["run"]
        valid = agg["valid_fraction"]
        print(f"{str(run['split']):{width}s} {run['n_cubes']:5d} {run['n_samples']:2d} "
              f"{(f'{valid:.1%}' if valid is not None else 'n/d'):>7s} "
              f"{_fmt(agg['ENS']):>8s} {_fmt(agg['MAD']):>8s} {_fmt(agg['OLS']):>8s} "
              f"{_fmt(agg['EMD']):>8s} {_fmt(agg['SSIM']):>8s} "
              f"{_fmt(agg['SSIM_RAW']):>8s} {_fmt(agg['P_MAE']):>8s} {_fmt(agg['N_MAE']):>8s}")
    print()


# ===========================================================================
# 6. Valutazione di uno split
# ===========================================================================
def checkpoint_step(ckpt: Path) -> Optional[int]:
    """Global step del checkpoint: dal nome (checkpoint-<step>) o da best_metric.json (best)."""
    if ckpt.name.startswith(("checkpoint-", "save-checkpoint-")):
        return int(ckpt.name.split("-")[-1])
    try:
        return int(json.loads((ckpt / "best_metric.json").read_text(encoding="utf-8"))["global_step"])
    except (OSError, ValueError, KeyError):
        return None


def evaluate_split(spec: viz.SplitSpec, args, cfg, run_dir: Path, ckpt: Path,
                   device: torch.device, state: Dict) -> Dict[str, object]:
    """Carica il modello (se servono altre finestre), gira sullo split, aggrega e salva."""
    n_context, n_target = viz.split_windows(spec, cfg)
    split_cfg = viz.with_windows(cfg, n_context, n_target)
    if split_cfg is not cfg:
        log.warning(
            f"[{spec.name}] finestre {n_context}+{n_target} (T={n_context + n_target}) diverse da "
            f"quelle del training (T={cfg.data.params.T}, {cfg.data.params.noise_image_num} "
            f"target): i pesi appresi sono gli stessi, ma e' un'estrapolazione in lunghezza e "
            f"le predizioni potrebbero degradare.")

    # il DiT dipende dalla lunghezza della sequenza: lo si (ri)costruisce quando le
    # finestre dello split cambiano
    if state.get("bundle") is None or state["windows"] != (n_context, n_target):
        state["bundle"] = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
        state["bundle"] = viz.load_dit(split_cfg, ckpt, device)
        state["windows"] = (n_context, n_target)
    bundle = state["bundle"]

    t_data = time.time()
    files, sources, desc = build_split(spec, split_cfg, args, n_context, n_target)
    dataset, loader = build_loader(split_cfg, files, sources, args, n_context, n_target,
                                   int(cfg.data.params.T))
    n_cubes = len(files)
    log.info(f"[{spec.name}] {desc}: {n_cubes} cubi valutati, sequenza "
             f"{n_context + n_target} = {n_context} contesto + {n_target} target, "
             f"batch {loader.batch_size}, maschera nuvole {'ON' if args.cloud_mask else 'OFF'}")

    from src.model import diffusion_utils as du
    first = dataset[0]
    du.check_data_model_alignment(first, split_cfg.diffusion, bundle.first_stage_info)
    t_data = time.time() - t_data

    p = split_cfg.data.params
    lo, hi = (a.reshape(-1).astype(np.float32)
              for a in viz.load_highres_stats(p.stats_json, p.stats_time_key, p.n_channels))

    # Il sampling parte da un randn e OLS aggiunge un rumore di regolarizzazione con il
    # generatore legacy di numpy: senza i due seed, due esecuzioni sullo stesso
    # checkpoint darebbero numeri leggermente diversi.
    set_seed(args.seed)
    np.random.seed(args.seed)

    rows: List[Dict[str, object]] = []
    t0 = time.time()
    t_infer = t_score = 0.0

    for batch in loader:
        t_batch = time.time()
        cube_paths = batch["cube_path"]
        cloud = batch.pop("cloud_mask", None)
        batch = viz.move_batch(batch, device, bundle.weight_dtype)
        preds, target = predict_pixels(batch, bundle, device, n_target, args.num_samples)
        if target.shape[1] != n_target or preds[0].shape != target.shape:
            raise SystemExit(f"[{spec.name}] attesi {n_target} frame target, ottenuti "
                             f"{target.shape[1]} (predetti {tuple(preds[0].shape)})")
        t_infer += time.time() - t_batch

        t_batch = time.time()
        for b in range(target.shape[0]):
            targ_cube = to_cube(target[b], lo, hi)
            mask_cube = to_mask(None if cloud is None else cloud[b], targ_cube.shape)
            pred_cubes = [to_cube(q[b], lo, hi) for q in preds]

            row = score_cube(pred_cubes, targ_cube, mask_cube, args.ssim_data_range, args.scoring)
            cube = Path(str(cube_paths[b]))
            row["cube"] = f"{cube.parent.name}/{cube.name}"
            row["P_MAE"] = row["p_sum"] / row["p_count"] if row["p_count"] else None
            row["N_MAE"] = row["n_sum"] / row["n_count"] if row["n_count"] else None
            rows.append(row)
        t_score += time.time() - t_batch

        done = len(rows)
        if done % max(loader.batch_size * 5, 1) == 0 or done == n_cubes:
            rate = (time.time() - t0) / max(done, 1)
            log.info(f"[{spec.name}] {done}/{n_cubes} cubi ({rate:.1f} s/cubo, "
                     f"~{rate * (n_cubes - done) / 60:.1f} min alla fine)")

    if not rows:
        raise SystemExit(f"[{spec.name}] nessun cubo valutato.")

    elapsed = time.time() - t0
    payload = {
        "run": {
            "model": run_dir.name,
            "run_dir": run_dir.as_posix(),
            "checkpoint": ckpt.name,
            "checkpoint_path": ckpt.as_posix(),
            "step": checkpoint_step(ckpt),
            "split": spec.name,
            "sample_pool": desc,
            "selection": ("subset" if args.subset else "max_cubes" if args.max_cubes else "all"),
            "n_context": n_context,
            "n_target": n_target,
            "trained_windows": [int(cfg.data.params.T) - int(cfg.data.params.noise_image_num),
                                int(cfg.data.params.noise_image_num)],
            "n_cubes": len(rows),
            "n_cubes_skipped": sum(1 for r in rows if r["ENS"] is None),
            "n_samples": int(args.num_samples),
            "seed": args.seed,
            "device": str(device),
            "stats_json": str(p.stats_json),
            "dataset_json": str(p.dataset_json),
        },
        "timing": {
            "total_s": round(elapsed, 1),
            "inference_s": round(t_infer, 1),
            "scoring_s": round(t_score, 1),
            # Quel che avanza e' l'attesa dei worker del DataLoader (disco).
            "data_s": round(max(elapsed - t_infer - t_score, 0.0) + t_data, 1),
            "s_per_cube": round(elapsed / max(len(rows), 1), 3),
            "scoring": args.scoring,
        },
        "scoring": {
            "space": "riflettanza [0, 1], clip come CubeCalculator.load_file",
            "mode": args.scoring,
            "cloud_mask": bool(args.cloud_mask),
            "cloud_prob_threshold": float(args.cloud_prob_threshold),
            "cloud_prob_scale": float(args.cloud_prob_scale),
            "ssim_data_range": float(args.ssim_data_range),
            "ssim_scaling": SSIM_SCALING,
            "earthnet_toolkit": (
                "earthnet.parallel_score.CubeCalculator (MAD, OLS, EMD)"
                if args.scoring == "toolkit"
                else "earthnet.parallel_score.CubeCalculator (OLS); "
                     "MAD ed EMD vettorizzate qui, stesso risultato"
            ),
        },
        "aggregate": aggregate(rows),
        "per_cube": rows,
    }

    json_path, csv_path = save_results(run_dir / "results" / STATISTICS_DIRNAME,
                                       spec.name, int(args.num_samples), payload)
    print_report(payload)
    log.info(f"[out] {json_path}")
    log.info(f"[out] {csv_path}")
    return payload


# ===========================================================================
# 7. CLI
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="ENS (MAD, OLS, EMD, SSIM), SSIM grezza, P-MAE e N-MAE di un DiT "
                    "gia' addestrato, su uno o piu' split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", type=str,
                   help="nome del run addestrato (cartella con il timestamp sotto "
                        "registry/outputs/diffusion/); anche un pezzo del nome se e' unico, "
                        "'latest' o il percorso della cartella")
    p.add_argument("--split", type=viz.split_arg, nargs="+", metavar="SPLIT",
                   help=f"uno o piu' fra: {', '.join(viz.SPLITS)} (anche iid_test_split, ...). "
                        f"seasonal usa 20 frame di contesto e 40 di target")
    p.add_argument("--list-runs", action="store_true",
                   help="stampa i run di diffusion disponibili ed esce")
    p.add_argument("--checkpoint", type=str, default="best",
                   help="'best' (checkpoints/best), 'latest' (checkpoint-<step> piu' alto) "
                        "o il nome di una sottocartella di checkpoints/")
    p.add_argument("--num-samples", type=int, default=1,
                   help="K predizioni per cubo, si tiene la migliore (best-of-K dell'ENS)")
    p.add_argument("--subset", type=int, default=0,
                   help="valuta N cubi a caso dello split (0 = tutti), scelti con --subset-seed")
    p.add_argument("--subset-seed", type=int, default=0, help="seed di --subset")
    p.add_argument("--max-cubes", type=int, default=0,
                   help="valuta solo i primi N cubi dello split in ordine (0 = tutti): "
                        "per una prova, non per un numero da riportare")
    p.add_argument("--batch-size", type=int, default=None,
                   help="cubi per batch (default: il batch_size del config del run, "
                        "ridotto in proporzione alla lunghezza della sequenza se piu' lunga)")
    p.add_argument("--num-workers", type=int, default=None,
                   help="worker del DataLoader (default: data.params.num_workers del config)")
    p.add_argument("--merge-workers", type=int, default=8,
                   help="thread per la fusione context+target degli split EarthNet")
    p.add_argument("--scoring", choices=SCORING_MODES, default="fast",
                   help="implementazione di MAD ed EMD: 'fast' (vettorizzate qui) o "
                        "'toolkit' (originali di earthnet). Stesso numero; la seconda "
                        "serve a verificarlo")
    p.add_argument("--no-cloud-mask", dest="cloud_mask", action="store_false",
                   help="misura su TUTTI i pixel, nuvole comprese. Sconsigliato: sotto le "
                        "nuvole il target non e' la superficie")
    p.add_argument("--cloud-prob-threshold", type=float, default=CLOUD_PROB_THRESHOLD,
                   help="cubi a 7 canali: un pixel e' valido se cloud probability < soglia")
    p.add_argument("--cloud-prob-scale", type=float, default=CLOUD_PROB_SCALE,
                   help="fondo scala del canale cloud probability (100 = percentuale)")
    p.add_argument("--ssim-data-range", type=float, default=SSIM_DATA_RANGE,
                   help="data_range della SSIM. 2.0 e' quello con cui e' calibrato "
                        "l'esponente del toolkit; 1.0 e' l'ampiezza reale della "
                        "riflettanza ma rende i punteggi non confrontabili con la letteratura")
    p.add_argument("--seed", type=int, default=42,
                   help="seed del sampling e del rumore di OLS")
    p.add_argument("--device", type=str, default=None, help="cuda, cuda:0, cpu (default: auto)")
    p.add_argument("--outputs-dir", type=str, default=PathsCfg.outputs_dir,
                   help="radice dei run (relativa alla root del progetto)")
    p.add_argument("--test-root", type=str, default=None,
                   help="cartella che contiene <split>_test_split/{context,target} "
                        "(default: la cartella che contiene paths.raw_dir, cioe' data/raw)")
    p.add_argument("--merged-root", type=str, default=None,
                   help="dove si salvano i cubi test fusi context+target "
                        "(default: <preprocessed_dir>/merged_test)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=getattr(logging, args.log_level.upper(), logging.INFO))
    use_project_dir()
    family_dir = project_path(args.outputs_dir) / viz.FAMILY_DIRS["dit"]

    if args.list_runs:
        for run in viz.list_runs(family_dir):
            print(run.relative_to(project_path(args.outputs_dir)).as_posix())
        return
    if not args.model:
        raise SystemExit("--model e' obbligatorio (oppure usa --list-runs)")
    if not args.split:
        raise SystemExit("--split e' obbligatorio: " + ", ".join(viz.SPLITS))
    if args.num_samples < 1:
        raise SystemExit("--num-samples deve essere >= 1")
    if args.subset and args.max_cubes:
        raise SystemExit("--subset e --max-cubes sono alternativi")

    run_dir = viz.resolve_run(args.model, family_dir)
    cfg = viz.load_run_config(run_dir)
    ckpt = viz.resolve_dit_checkpoint(run_dir, args.checkpoint)
    device = viz.resolve_device(args.device)

    # Tutto cio' che serve ai dati deve esistere PRIMA di caricare il VAE e il DiT
    # (qualche minuto e diversi GB): un percorso sbagliato deve fermare lo script subito.
    params = cfg.data.params
    missing = [f"data.params.{key} -> {params[key]}" for key in ("stats_json", "cond_stats_json")
               if not Path(str(params[key])).is_file()]
    if any(not s.earthnet for s in args.split) and not Path(str(params.dataset_json)).is_file():
        missing.append(f"data.params.dataset_json -> {params.dataset_json}")
    if missing:
        raise SystemExit(f"{run_dir.name}: il config del run punta a file che non ci sono "
                         f"(percorsi relativi a {Path.cwd()}):\n  " + "\n  ".join(missing))

    print("=" * 78)
    print(f"modello     : dit | run {run_dir.name}")
    print(f"checkpoint  : {ckpt.as_posix()}")
    print(f"device      : {device}")
    print(f"split       : {', '.join(s.name for s in args.split)} | K={args.num_samples} | "
          f"seed {args.seed}")
    print(f"output      : {(run_dir / 'results' / STATISTICS_DIRNAME).as_posix()}")
    print("=" * 78)

    # gli split non cambiano il modello se hanno le stesse finestre: il caricamento
    # e' condiviso (vedi evaluate_split)
    state: Dict = {"bundle": None, "windows": None}
    payloads: List[Dict[str, object]] = []
    failed: List[Tuple[str, str]] = []
    for spec in args.split:
        try:
            payloads.append(evaluate_split(spec, args, cfg, run_dir, ckpt, device, state))
        except SystemExit as err:
            failed.append((spec.name, str(err)))
            log.error(f"[{spec.name}] saltato -- {err}")
        except Exception as err:  # noqa: BLE001 - uno split rotto non butta via gli altri
            failed.append((spec.name, f"{type(err).__name__}: {err}"))
            log.exception(f"[{spec.name}] saltato")

    if len(payloads) > 1:
        print_comparison(payloads)
    if failed:
        print(f"{len(failed)} split NON valutati:")
        for name, reason in failed:
            print(f"  {name}: {reason.splitlines()[0]}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
