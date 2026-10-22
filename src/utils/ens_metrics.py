"""
Metriche ENS (EarthNetScore) di un cubo e loro aggregazione.

Condiviso da:
    - src/model/diffusion_utils.py   train_loop: ENS in validation e su un campione di train
    - src/scripts/evaluate_ens.py    valutazione del DiT su uno o piu' split
    - src/scripts/evaluate_ae_ens.py valutazione dell'autoencoder

ENS = media armonica di quattro sottopunteggi (MAD, OLS, EMD, SSIM), piu' SSIM_RAW
(la SSIM prima dell'esponente, fuori dall'ENS), P-MAE e N-MAE. Tutto e' calcolato
sui soli frame PREDETTI e nello spazio della riflettanza [0, 1], non in quello
normalizzato [-1, 1] del training: e' li' che l'EarthNetScore e' definito.

Per cubo: score_cube() (con K predizioni tiene quella con l'ENS piu' alto, il
best-of-K di EarthNet). Su un batch: score_batch(). Sull'intero split:
aggregate(), che media i sottopunteggi sui cubi e poi fa la media armonica.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# earthnet (CubeCalculator) and scikit-image (structural_similarity) are imported where
# they are used, so that importing this module costs nothing when the ENS is off.

# Sottocartella dei risultati del run in cui si scrivono i file di evaluate_*.
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

# cubo letto dal dataset (percorso .npz) -> (file da cui leggere la maschera, primo
# istante dei frame target dentro quel file)
MaskSources = Dict[str, Tuple[Path, int]]

# ===========================================================================
# 1. Maschera nuvole
# ===========================================================================

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


def cloud_prob_cube(hd: np.ndarray, prob_scale: float) -> np.ndarray:
    """
    highresdynamic (H, W, C, T) di un file originale -> (T, H, W) float32 in [0, 1],
    1 = nuvola. Stessi due formati di valid_mask: 7 canali = cloud probability al canale 4
    (divisa per prob_scale), 5 canali = la maschera binaria in fondo (0 o 1). I valori non
    finiti contano come nuvola, come in valid_mask.
    """
    if hd.shape[2] >= N_CHANNELS_WITH_CLOUD_PROB:
        prob = hd[:, :, CLOUD_PROB_CHANNEL, :] / prob_scale
    else:
        prob = hd[:, :, -1, :]
    prob = np.nan_to_num(np.asarray(prob, dtype=np.float32), nan=1.0, posinf=1.0, neginf=0.0)
    return np.transpose(np.clip(prob, 0.0, 1.0), (2, 0, 1))


def load_cloud_frames(path, frames: Sequence[int], threshold: float, prob_scale: float,
                      resize: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    (cloud probability, pixel validi) di alcuni istanti di un cubo originale, per le figure:
    due array (n, h, w), float32 in [0, 1] e bool (True = pixel valido, la stessa regola
    dello scoring ENS). Il modello non li vede.

    Con resize minore della risoluzione nativa la probabilita' e' la media dei sotto-pixel
    (interpolazione area, come le immagini), mentre un pixel e' valido solo se lo sono tutti
    i suoi sotto-pixel: e' la regola di MaskedCubeDataset, cosi' i pixel grigi delle figure
    sono esattamente quelli esclusi dalle metriche.
    """
    with np.load(path) as z:
        hd = z["highresdynamic"][:, :, :, list(frames)]
    prob = torch.from_numpy(cloud_prob_cube(hd, prob_scale)).unsqueeze(1)           # (n, 1, H, W)
    valid = torch.from_numpy(valid_mask(hd, threshold, prob_scale).astype(np.float32)).unsqueeze(1)
    if resize is not None and tuple(prob.shape[-2:]) != (resize, resize):
        prob = F.interpolate(prob, size=(resize, resize), mode="area")
        valid = (F.interpolate(valid, size=(resize, resize), mode="area") >= 0.999).float()
    return prob[:, 0].numpy(), valid[:, 0].numpy() > 0.5


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
        if item.pop("hflip", False):                # the images were flipped: so is the mask
            mask = torch.flip(mask, dims=[-1])
        item["cloud_mask"] = mask
        return item


# ===========================================================================
# 2. Da tensore del trainer a cubo EarthNet
# ===========================================================================
def denormalize_minmax(x: torch.Tensor, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """
    (T, C, H, W) in [-1, 1] -> riflettanza, su CPU in float32. Inversa esatta di
    minmax_pm1() del dataloader:
    x_norm = 2*(x - lo)/(hi - lo) - 1  ->  x = (x_norm + 1)/2 * (hi - lo) + lo.
    Il clipping fatto in normalizzazione non e' invertibile: i valori fuori range
    erano gia' saturati a +-1 prima del training.
    """
    arr = x.detach().float().cpu().numpy()              # (T, C, H, W)
    lo_b = np.asarray(lo).reshape(1, -1, 1, 1).astype(np.float32)
    hi_b = np.asarray(hi).reshape(1, -1, 1, 1).astype(np.float32)
    return (arr + 1.0) / 2.0 * (hi_b - lo_b) + lo_b


def to_cube(x: torch.Tensor, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """
    (T, C, H, W) normalizzato in [-1, 1] -> (H, W, C, T) in riflettanza [0, 1].

    Prima l'inversa esatta della normalizzazione del dataloader (denormalize), poi il
    clipping in [0, 1], applicato da CubeCalculator.load_file() a predizioni e target.
    """
    arr = denormalize_minmax(x, lo, hi)              # (T, C, H, W), riflettanza
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
    arr = np.transpose(m.detach().cpu().numpy().astype(np.float32), (2, 3, 1, 0))   # (H, W, 1, T)
    return np.repeat(arr, c, axis=2)


def ndvi(cube: np.ndarray) -> np.ndarray:
    """(H, W, C, T) -> NDVI (H, W, 1, T), con la stessa epsilon del toolkit."""
    red = cube[:, :, BAND_RED, :]
    nir = cube[:, :, BAND_NIR, :]
    return ((nir - red) / (nir + red + 1e-6))[:, :, np.newaxis, :]


# ===========================================================================
# 3. Sottopunteggi e MAE di un cubo
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
    from skimage.metrics import structural_similarity

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
    from earthnet.parallel_score import CubeCalculator

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


def score_batch(pred_samples: Sequence[torch.Tensor], target: torch.Tensor,
                cloud: Optional[torch.Tensor], cube_paths: Sequence[str],
                lo: np.ndarray, hi: np.ndarray, ssim_data_range: float = SSIM_DATA_RANGE,
                scoring: str = "fast") -> List[Dict[str, object]]:
    """
    Una riga di risultati per ogni cubo di un batch.

    pred_samples   K tensori (B, T, C, H, W) in [-1, 1], le K predizioni di ogni cubo:
                   di ogni cubo si tiene quella con l'ENS piu' alto (best-of-K)
    target         (B, T, C, H, W) in [-1, 1]
    cloud          "cloud_mask" del batch (B, T, 1, H, W), 1 = pixel valido; None =
                   nessun pixel escluso
    cube_paths     percorso del .npz di ogni cubo, per il nome della riga
    lo, hi         estremi per banda della normalizzazione min-max del dataloader
    """
    rows = []
    for b in range(target.shape[0]):
        targ_cube = to_cube(target[b], lo, hi)
        mask_cube = to_mask(None if cloud is None else cloud[b], targ_cube.shape)
        pred_cubes = [to_cube(q[b], lo, hi) for q in pred_samples]

        row = score_cube(pred_cubes, targ_cube, mask_cube, ssim_data_range, scoring)
        cube = Path(str(cube_paths[b]))
        row["cube"] = f"{cube.parent.name}/{cube.name}"
        row["P_MAE"] = row["p_sum"] / row["p_count"] if row["p_count"] else None
        row["N_MAE"] = row["n_sum"] / row["n_count"] if row["n_count"] else None
        rows.append(row)
    return rows


def unique_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """
    Le righe senza duplicati (per "cube"), nell'ordine di arrivo. In multi-GPU
    l'ultimo batch puo' essere completato ripetendo dei cubi: contarli due volte
    sposterebbe le medie.
    """
    seen, out = set(), []
    for row in rows:
        if row["cube"] in seen:
            continue
        seen.add(row["cube"])
        out.append(row)
    return out


# ===========================================================================
# 4. Aggregazione
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
