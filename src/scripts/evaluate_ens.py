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

    # extreme e seasonal con le finestre del training, in modo autoregressivo
    python src/scripts/evaluate_ens.py --model [date] --split extreme seasonal --autoregressive

Argomenti principali
--------------------
--model   nome del run addestrato = cartella con il timestamp sotto
          registry/outputs/diffusion/ (anche un pezzo del nome se e' unico,
          'latest' o il percorso della cartella). Stessa risoluzione di
          inference_visualization.py.
--split   uno o piu' fra
              train, val           split di dataset.json (cubi da data/raw/train)
              iid, ood, extreme    split di test EarthNet: data/raw/<split>_test_split/
                                   {context,target}
              seasonal             come sopra, ma con finestre piu' lunghe
          (anche i nomi completi: iid_test_split, ...)

Finestre temporali
------------------
    train / val / iid / ood   10 frame di contesto + 20 target (T = 30)
    extreme                   20 frame di contesto + 40 target (T = 60)
    seasonal                  70 frame di contesto + 140 target (T = 210)
Sono le finestre dei file EarthNet2021 (un cubo con un numero di frame diverso ferma la
valutazione dello split). Con extreme e seasonal ci sono due modi di lavorare:

    default           il DiT lavora su sequenze piu' lunghe di quelle del training
                      (T = 30): si ricostruisce con la lunghezza dello split e il batch si
                      riduce in proporzione (--batch-size per fissarlo)
    --autoregressive  il DiT resta quello del training (10 frame di contesto + 20 predetti)
                      e la finestra target si copre a blocchi di 20 frame: il primo blocco
                      parte dagli ultimi 10 frame osservati, ogni blocco successivo dagli
                      ultimi 10 frame predetti dal precedente. Il meteo di ogni blocco e'
                      quello reale dei suoi 30 frame. Con K > 1 si fanno K catene
                      indipendenti. Vale per gli split con finestre diverse da quelle del
                      training (extreme, seasonal); per gli altri non cambia nulla.

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

Cubi dei test
-------------
Contesto e target di un cubo EarthNet stanno in due file. Non si salva nessuna copia
fusa: il dataset (src/dataset/test_cubes.py) legge i due file e li unisce in memoria nel
worker del DataLoader, un cubo alla volta.

Nuvole
------
Come nello scoring ufficiale i pixel nuvolosi sono esclusi dal confronto, ma il
modello non vede la maschera: riceve gli stessi input del training. La maschera si
legge dai file .npz originali (il dataloader attuale scarta le bande oltre la quarta):

Figure per banda
----------------
--export-cubes N salva, per N cubi di ogni split (sempre gli stessi: seed del run, nomi
ordinati), una cartella per cubo con un PNG per banda: ground truth, predizione (la migliore
delle K), cloud probability ed errore |pred - target| / 2, una colonna per istante predetto.
I pixel non validi sono grigi nella riga dell'errore.

Output
------
registry/outputs/diffusion/<run>/results/forecasts/<split>/<cubo>/<banda>.png
registry/outputs/diffusion/<run>/results/statistics/
    metrics_results_<split>_K_<K>.json   aggregati + metadati (checkpoint, finestre,
                                         maschera, tempi) + una voce per cubo
    metrics_results_<split>_K_<K>.csv    una riga per cubo, per gli istogrammi
Ogni esecuzione riscrive i file del proprio split e K, senza toccare gli altri. Una
valutazione parziale (--subset N / --max-cubes N) aggiunge il suffisso _subsetN / _firstN
ai nomi, cosi' un giro di prova non sovrascrive i file dello split intero. Con
--autoregressive, per gli split che la usano (extreme, seasonal), <split> diventa
<split>_autoreg ovunque: metrics_results_seasonal_autoreg_K_5.json, forecasts/seasonal_autoreg/,
e le figure hanno "autoreg" nel titolo.

Velocita'
---------
L'inferenza e' il ~90% del tempo. Il campionamento (diffusion_utils.sample_forecast) codifica
solo il contesto, una volta per cubo, e decodifica solo i frame predetti; lo scoring gira su
processi separati (--scoring-workers) mentre la GPU lavora sul batch successivo. A fine split
si stampa il picco di VRAM: sopra ~90% della memoria totale l'inferenza rallenta (usa
--batch-size piu' piccolo).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing
import os
import sys
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Deque, Dict, List, NamedTuple, Optional, Sequence, Tuple

# inference_visualization.py sta nella stessa cartella: lo si importa da li' anche
# quando lo script e' lanciato da un'altra cartella o da un IDE. Il suo import mette
# anche la root del progetto e src/model su sys.path, e serve prima di ogni
# import `src.` o `sgm.`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import inference_visualization as viz  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from accelerate.utils import set_seed  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from src.config import PathsCfg, project_path, use_project_dir  # noqa: E402
# lo scoring ENS sta in src/utils/ens_metrics.py, condiviso con il training loop
# e con evaluate_ae_ens.py
from src.utils.example_export import pick_cubes, save_cube_forecast  # noqa: E402
from src.utils.ens_metrics import (  # noqa: E402
    CLOUD_PROB_SCALE, CLOUD_PROB_THRESHOLD, SCORING_MODES, SSIM_DATA_RANGE, SSIM_SCALING,
    STATISTICS_DIRNAME, MaskedCubeDataset, MaskSources, aggregate, init_scoring_worker,
    score_batch_task,
)

log = logging.getLogger("evaluate_ens")


# ===========================================================================
# 1. Cubi dello split, finestre e maschera nuvole
# ===========================================================================

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


class SplitCubes(NamedTuple):
    """I cubi scelti di uno split, con quello che serve per leggerli e per le figure."""
    files: List[Path]
    sources: MaskSources
    desc: str
    # solo per gli split EarthNet: percorso virtuale del cubo -> (file di contesto, file di
    # target), che il dataset fonde in memoria; None per train e val (un .npz per cubo)
    pairs: Optional[Dict[str, Tuple[Path, Path]]]


def build_split(spec: viz.SplitSpec, cfg, args, n_context: int, n_target: int) -> SplitCubes:
    """
    I cubi scelti di uno split, la sorgente della maschera nuvole di ognuno e la
    descrizione. Per gli split EarthNet contesto e target restano nei loro due file
    originali: non si scrive nessuna copia fusa.
    """
    pool, desc, pairs = viz.build_pool(spec, cfg, args)
    selected = select_cubes(pool, args)
    if pairs is None:
        viz.check_cubes_exist(selected)
        return SplitCubes(selected, {str(f): (f, n_context) for f in selected}, desc, None)

    chosen = {str(f): pairs[str(f)] for f in selected}
    # i frame target stanno interi nel file di target
    return SplitCubes(selected, {key: (tgt, 0) for key, (_, tgt) in chosen.items()}, desc, chosen)


def build_loader(cfg, split: SplitCubes, args, n_context: int, n_target: int,
                 reference_T: int) -> Tuple[Dataset, DataLoader]:
    """
    (dataset, loader) sui cubi scelti: le statistiche di normalizzazione sono quelle
    del training, senza flip, senza shuffle e senza drop_last (nessun cubo deve
    sparire e l'ordine deve essere lo stesso a ogni esecuzione).

    reference_T e' la lunghezza di sequenza per cui e' tarato il batch del config: con
    sequenze piu' lunghe il batch si riduce in proporzione (con T uguale resta quello del
    config).
    """
    params = cfg.data.params
    cloud = (args.cloud_prob_threshold, args.cloud_prob_scale) if args.cloud_mask else None
    dataset = viz.build_eval_dataset(cfg, split.files, n_context, n_target,
                                     pairs=split.pairs, cloud=cloud)
    if args.cloud_mask and split.pairs is None:
        dataset = MaskedCubeDataset(dataset, split.sources, n_target, args.cloud_prob_threshold,
                                    args.cloud_prob_scale, params.resize)

    if args.batch_size:
        batch_size = int(args.batch_size)
    else:
        # il batch del config e' tarato sulla T del training: con sequenze piu' lunghe
        # (seasonal) la memoria per cubo cresce in proporzione, e il batch cala
        batch_size = max(1, min(int(params.batch_size),
                                int(params.batch_size) * reference_T // (n_context + n_target)))
    num_workers = int(params.num_workers if args.num_workers is None else args.num_workers)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                        drop_last=False,
                        prefetch_factor=4 if num_workers > 0 else None)
    return dataset, loader


@dataclass(frozen=True)
class WindowPlan:
    """
    Copertura della finestra target a blocchi, con le finestre del training
    (window_context frame di contesto + window_predict predetti per volta): il blocco k
    legge i frame [start(k), start(k) + window_context + window_predict) della sequenza
    dello split.
    """
    n_context: int          # frame osservati dello split prima del primo target
    n_target: int           # frame target dello split
    window_context: int
    window_predict: int

    @property
    def n_windows(self) -> int:
        return self.n_target // self.window_predict

    @property
    def window_length(self) -> int:
        return self.window_context + self.window_predict

    def start(self, k: int) -> int:
        """Primo frame di contesto del blocco k: gli ultimi osservati, poi avanti di window_predict."""
        return self.n_context - self.window_context + k * self.window_predict


def autoregressive_plan(n_context: int, n_target: int, window_context: int,
                        window_predict: int) -> WindowPlan:
    if n_context < window_context:
        raise SystemExit(f"servono almeno {window_context} frame di contesto, lo split ne ha {n_context}")
    if n_target % window_predict:
        raise SystemExit(f"{n_target} frame target non sono multipli dei {window_predict} frame "
                         f"che il modello predice per volta")
    return WindowPlan(n_context, n_target, window_context, window_predict)


def trained_windows(cfg) -> Tuple[int, int]:
    """(contesto, predetti) con cui e' stato addestrato il DiT."""
    n_target = int(cfg.data.params.noise_image_num)
    return int(cfg.data.params.T) - n_target, n_target


def window_item(item: Dict, plan: WindowPlan, k: int = 0) -> Dict:
    """L'item di un cubo (sequenza intera) ristretto ai frame del blocco k, come li vede il DiT."""
    start = plan.start(k)
    stop = start + plan.window_length
    return {**item, "imgs": item["imgs"][start:stop],
            "meso_condition_image": item["meso_condition_image"][start:stop]}


# ===========================================================================
# 2. Inferenza: K predizioni in pixel
# ===========================================================================
def predict_pixels(batch: Dict, bundle: viz.DitBundle, device: torch.device, n_target: int,
                   n_samples: int, plan: Optional[WindowPlan] = None
                   ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    (K predizioni, target), ognuno (B, n_target, C, H, W) in [-1, 1] su CPU.

    Ogni predizione ha la procedura di validazione del trainer (encode col VAE del
    contesto, rumore iniziale miscelato al contesto, EulerEDMSampler con CFG, decode):
    diffusion_utils.sample_forecast, che codifica il contesto una volta sola per i K
    campioni (K nuovi rumori iniziali). Con `plan` il batch porta la sequenza intera dello
    split e la finestra target si copre a blocchi (predict_autoregressive).
    """
    from src.model import diffusion_utils as du

    cfg = bundle.cfg
    imgs = batch[cfg.diffusion.input_key]
    with torch.no_grad(), viz.autocast_context(device, bundle.mixed_precision):
        if plan is None:
            forecasts = du.sample_forecast(
                batch, imgs[:, :-n_target], bundle.model, bundle.conditioner,
                bundle.first_stage_model, bundle.sampler, bundle.denoiser, cfg.diffusion,
                n_target, cfg.diffusion.loss_fn_config.params.past_weight, n_samples)
            preds = [f.detach().float().cpu() for f in forecasts]
        else:
            preds = predict_autoregressive(batch, bundle, plan, n_samples)
    # .cpu() su CUDA sincronizza: il tempo misurato fuori e' quello vero
    return preds, imgs[:, -n_target:].detach().float().cpu()


def predict_autoregressive(batch: Dict, bundle: viz.DitBundle, plan: WindowPlan,
                           n_samples: int) -> List[torch.Tensor]:
    """
    K catene autoregressive, ognuna (B, n_target, C, H, W) in [-1, 1] su CPU.

    Blocco 0: contesto = gli ultimi window_context frame osservati. Blocco k > 0: contesto
    = gli ultimi window_context frame predetti dal blocco k - 1 (in pixel, gia' clampati:
    il VAE li ricodifica). Il meteo del blocco e' quello reale dei suoi frame. Ogni
    blocco predice window_predict frame; i blocchi si concatenano.
    """
    from src.model import diffusion_utils as du

    cfg = bundle.cfg
    imgs = batch[cfg.diffusion.input_key]
    meso = batch["meso_condition_image"]
    first = plan.start(0)
    contexts = [imgs[:, first:first + plan.window_context] for _ in range(n_samples)]
    chains: List[List[torch.Tensor]] = [[] for _ in range(n_samples)]
    for k in range(plan.n_windows):
        start = plan.start(k)
        window = {**batch, "meso_condition_image": meso[:, start:start + plan.window_length]}
        for i in range(n_samples):
            forecast = du.sample_forecast(
                window, contexts[i], bundle.model, bundle.conditioner, bundle.first_stage_model,
                bundle.sampler, bundle.denoiser, cfg.diffusion, plan.window_predict,
                cfg.diffusion.loss_fn_config.params.past_weight, 1)[0]
            chains[i].append(forecast.detach().float().cpu())
            contexts[i] = forecast[:, -plan.window_context:]
    return [torch.cat(chain, dim=1) for chain in chains]


# ===========================================================================
# 5. Aggregazione, salvataggio, report
# ===========================================================================
def selection_suffix(args) -> str:
    """
    '' per lo split intero; '_subset<N>' / '_first<N>' per una valutazione parziale
    (--subset / --max-cubes): i suoi file non devono sovrascrivere quelli dello split
    intero, che costano ore, e un giro di prova e' proprio il caso in cui capita.
    """
    if args.subset:
        return f"_subset{args.subset}" + (f"_seed{args.subset_seed}" if args.subset_seed else "")
    if args.max_cubes:
        return f"_first{args.max_cubes}"
    return ""


def save_results(out_dir: Path, split: str, n_samples: int,
                 payload: Dict[str, object], suffix: str = "") -> Tuple[Path, Path]:
    """
    metrics_results_<split>_K_<K>[<suffix>].json (aggregati + metadati + per cubo) e
    .csv (una riga per cubo) in <run>/results/statistics/. Riscritti a ogni esecuzione.
    `split` e' il nome dello split, con "_autoreg" se la finestra target e' coperta a blocchi.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"metrics_results_{split}_K_{n_samples}{suffix}"

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
    if run.get("autoregressive"):
        print(f"  autoregressivo: {run['n_windows']} blocchi da "
              f"{run['window'][0]}+{run['window'][1]} frame")
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
          f"{timing['scoring_wait_s']:.0f} s attesa scoring + "
          f"{timing['data_s']:.0f} s dati) = {timing['s_per_cube']:.2f} s/cubo")
    print(f"  scoring [{timing['scoring']}]: {timing['scoring_s']:.0f} s di CPU su "
          f"{timing['scoring_workers']} processi")
    memory = payload.get("memory")
    if memory:
        print(f"  VRAM: picco {memory['peak_reserved_gb']:.1f} GB su {memory['total_gb']:.1f} GB "
              f"({memory['peak_reserved_fraction']:.0%}), batch {run['batch_size']}")
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


def prepare_bundle(spec: viz.SplitSpec, args, cfg, ckpt: Path, device: torch.device,
                   state: Dict):
    """
    (contesto, target, config, modello, piano autoregressivo): lo (ri)carica se serve.

    Di default, se le finestre dello split non sono quelle del training, il DiT si ricostruisce
    con la lunghezza dello split (config con le finestre dello split). Con --autoregressive e
    finestre diverse dal training il modello resta quello del training: il piano (WindowPlan)
    copre la finestra target a blocchi.
    """
    n_context, n_target = viz.split_windows(spec, cfg)
    plan: Optional[WindowPlan] = None
    if args.autoregressive and (n_context, n_target) != trained_windows(cfg):
        plan = autoregressive_plan(n_context, n_target, *trained_windows(cfg))
        split_cfg = cfg
        windows = trained_windows(cfg)
        log.info(f"[{spec.name}] autoregressivo: {plan.n_windows} blocchi da "
                 f"{plan.window_context}+{plan.window_predict} frame sulla sequenza "
                 f"{n_context}+{n_target}")
    else:
        split_cfg = viz.with_windows(cfg, n_context, n_target)
        windows = (n_context, n_target)
        if split_cfg is not cfg:
            log.warning(
                f"[{spec.name}] finestre {n_context}+{n_target} (T={n_context + n_target}) diverse da "
                f"quelle del training (T={cfg.data.params.T}, {cfg.data.params.noise_image_num} "
                f"target): i pesi appresi sono gli stessi, ma e' un'estrapolazione in lunghezza e "
                f"le predizioni potrebbero degradare (--autoregressive per usare le finestre "
                f"del training).")

    # il DiT dipende dalla lunghezza della sequenza: lo si (ri)costruisce quando le
    # finestre del modello cambiano
    if state.get("bundle") is None or state["windows"] != windows:
        state["bundle"] = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
        state["bundle"] = viz.load_dit(split_cfg, ckpt, device)
        state["windows"] = windows
    return n_context, n_target, split_cfg, state["bundle"], plan


def split_tag(spec: viz.SplitSpec, plan: Optional[WindowPlan]) -> str:
    """Nome dello split nei file e nei report: '<split>_autoreg' se la finestra e' coperta a blocchi."""
    return f"{spec.name}_autoreg" if plan is not None else spec.name


def export_split(spec: viz.SplitSpec, args, cfg, run_dir: Path, ckpt: Path,
                 device: torch.device, state: Dict) -> Dict[str, object]:
    """
    --export-only: solo le figure per banda di args.export_cubes cubi dello split (gli stessi
    cubi del training e di una valutazione normale), con una predizione per cubo. Niente
    scoring, niente file in statistics/.
    """
    n_context, n_target, split_cfg, bundle, plan = prepare_bundle(spec, args, cfg, ckpt, device,
                                                                   state)
    tag = split_tag(spec, plan)
    split = build_split(spec, split_cfg, args, n_context, n_target)
    chosen = [Path(f) for f in pick_cubes(split.files, args.export_cubes, int(cfg.run.seed))]
    chosen_split = SplitCubes(chosen, {str(f): split.sources[str(f)] for f in chosen}, split.desc,
                              None if split.pairs is None else
                              {str(f): split.pairs[str(f)] for f in chosen})
    _, loader = build_loader(split_cfg, chosen_split, args, n_context, n_target,
                             n_context + n_target if plan else int(cfg.data.params.T))
    export_dir = run_dir / "results" / "forecasts" / tag
    cloud_cfg = SimpleNamespace(cloud_prob_threshold=args.cloud_prob_threshold,
                                cloud_prob_scale=args.cloud_prob_scale)
    log.info(f"[{tag}] per-band figures of {len(chosen)} cubes in {export_dir}")

    set_seed(args.seed)
    for batch in loader:
        cube_paths = batch["cube_path"]
        batch.pop("cloud_mask", None)
        batch = viz.move_batch(batch, device, bundle.weight_dtype)
        preds, target = predict_pixels(batch, bundle, device, n_target, 1, plan)
        for b, path in enumerate(cube_paths):
            save_cube_forecast([export_dir], path, target[b], preds[0][b], split.sources[str(path)],
                               cloud_cfg, split_cfg.data.params.resize,
                               masked=bool(args.cloud_mask),
                               title=f"{tag} | {ckpt.name}")
    return {"split": tag, "n_cubes": len(chosen), "dir": export_dir.as_posix()}


class SerialExecutor:
    """Stessa interfaccia di ProcessPoolExecutor, esegue subito nel processo (--scoring-workers 0)."""

    def submit(self, fn, *fn_args) -> Future:
        future: Future = Future()
        try:
            future.set_result(fn(*fn_args))
        except BaseException as error:  # noqa: BLE001 - lo rilancia result()
            future.set_exception(error)
        return future

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def make_scoring_pool(workers: int):
    """
    Il pool su cui gira lo scoring mentre la GPU lavora sul batch dopo. Processi in 'spawn'
    (come i worker del DataLoader su Windows, e mai un fork di un processo con CUDA): ognuno
    importa ens_metrics e basta.
    """
    if workers <= 0:
        return SerialExecutor()
    return ProcessPoolExecutor(max_workers=workers,
                               mp_context=multiprocessing.get_context("spawn"),
                               initializer=init_scoring_worker)


def vram_report(device: torch.device) -> Optional[Dict[str, float]]:
    """Picco di memoria GPU dal reset (GB) e frazione della memoria totale; None senza CUDA."""
    if device.type != "cuda":
        return None
    total = torch.cuda.get_device_properties(device).total_memory
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    return {"total_gb": round(total / 2 ** 30, 2),
            "peak_allocated_gb": round(allocated / 2 ** 30, 2),
            "peak_reserved_gb": round(reserved / 2 ** 30, 2),
            "peak_reserved_fraction": round(reserved / total, 3)}


# sopra questa frazione della memoria della GPU il driver comincia a spostare dati nella
# memoria condivisa (Windows) o l'allocatore a liberare e riallocare: l'inferenza rallenta
VRAM_WARNING_FRACTION = 0.90


def evaluate_split(spec: viz.SplitSpec, args, cfg, run_dir: Path, ckpt: Path,
                   device: torch.device, state: Dict) -> Dict[str, object]:
    """Carica il modello (se servono altre finestre), gira sullo split, aggrega e salva."""
    n_context, n_target, split_cfg, bundle, plan = prepare_bundle(spec, args, cfg, ckpt, device,
                                                                   state)
    tag = split_tag(spec, plan)

    t_data = time.time()
    split = build_split(spec, split_cfg, args, n_context, n_target)
    # il batch del config e' tarato sulle finestre del training: nel modo autoregressivo il
    # modello vede sempre quelle, quindi il batch non si riduce
    dataset, loader = build_loader(split_cfg, split, args, n_context, n_target,
                                   n_context + n_target if plan else int(cfg.data.params.T))
    files, sources, desc = split.files, split.sources, split.desc
    n_cubes = len(files)
    log.info(f"[{tag}] {desc}: {n_cubes} cubi valutati, sequenza "
             f"{n_context + n_target} = {n_context} contesto + {n_target} target, "
             f"batch {loader.batch_size}, maschera nuvole {'ON' if args.cloud_mask else 'OFF'}")

    from src.model import diffusion_utils as du
    first = dataset[0]
    du.check_data_model_alignment(window_item(first, plan) if plan else first,
                                  split_cfg.diffusion, bundle.first_stage_info)
    t_data = time.time() - t_data

    p = split_cfg.data.params
    lo, hi = (a.reshape(-1).astype(np.float32)
              for a in viz.load_highres_stats(p.stats_json, p.stats_time_key, p.n_channels))

    # Il sampling parte da un randn; il rumore di regolarizzazione di OLS riparte per ogni
    # cubo da un seed che dipende dal cubo (score_batch): senza i seed, due esecuzioni sullo
    # stesso checkpoint darebbero numeri leggermente diversi.
    set_seed(args.seed)
    np.random.seed(args.seed)

    # --export-cubes: the same cubes every time (fixed seed, sorted files), also the ones the
    # training exported for val
    export_cubes = set(pick_cubes(files, args.export_cubes, int(cfg.run.seed)))
    export_dir = run_dir / "results" / "forecasts" / tag
    cloud_cfg = SimpleNamespace(cloud_prob_threshold=args.cloud_prob_threshold,
                                cloud_prob_scale=args.cloud_prob_scale)
    if export_cubes:
        log.info(f"[{tag}] per-band figures of {len(export_cubes)} cubes in {export_dir}")

    rows: List[Dict[str, object]] = []
    t0 = time.time()
    t_infer = t_wait = t_score_cpu = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Lo scoring (CPU) gira su processi separati mentre la GPU predice il batch dopo: il
    # processo principale aspetta solo quando troppi batch sono in coda (la memoria dei
    # tensori in attesa resta limitata) e alla fine.
    workers = min(max(0, int(args.scoring_workers)), max(1, (os.cpu_count() or 2) - 1))
    max_pending = 2 * max(workers, 1)
    pending: Deque[Future] = deque()

    def finish_oldest() -> None:
        nonlocal t_wait, t_score_cpu
        start = time.time()
        batch_rows, cpu_seconds = pending.popleft().result()
        t_wait += time.time() - start
        t_score_cpu += cpu_seconds
        rows.extend(batch_rows)

    n_predicted = 0
    with make_scoring_pool(workers) as pool:
        for i_batch, batch in enumerate(loader, 1):
            t_batch = time.time()
            cube_paths = batch["cube_path"]
            cloud = batch.pop("cloud_mask", None)
            batch = viz.move_batch(batch, device, bundle.weight_dtype)
            preds, target = predict_pixels(batch, bundle, device, n_target, args.num_samples, plan)
            if target.shape[1] != n_target or preds[0].shape != target.shape:
                raise SystemExit(f"[{tag}] attesi {n_target} frame target, ottenuti "
                                 f"{target.shape[1]} (predetti {tuple(preds[0].shape)})")
            t_infer += time.time() - t_batch
            n_predicted += len(cube_paths)

            t_batch = time.time()
            future = pool.submit(score_batch_task, preds, target, cloud, [str(c) for c in cube_paths],
                                 lo, hi, args.ssim_data_range, args.scoring, args.seed)
            pending.append(future)
            t_wait += time.time() - t_batch

            # per-band figures of the chosen cubes, with the best of the K predictions (the one
            # the ENS kept): it needs the scores of this batch. Not counted in the scoring time.
            chosen = [b for b, path in enumerate(cube_paths) if str(path) in export_cubes]
            if chosen:
                batch_rows = future.result()[0]
                for b in chosen:
                    save_cube_forecast([export_dir], cube_paths[b], target[b],
                                       preds[batch_rows[b].get("sample") or 0][b],
                                       sources[str(cube_paths[b])], cloud_cfg, p.resize,
                                       masked=bool(args.cloud_mask),
                                       title=f"{tag} | {ckpt.name} (best of K={args.num_samples})")

            while len(pending) > max_pending:
                finish_oldest()

            if i_batch % 5 == 0 or n_predicted == n_cubes:
                rate = (time.time() - t0) / max(n_predicted, 1)
                log.info(f"[{tag}] {n_predicted}/{n_cubes} cubi ({rate:.1f} s/cubo, "
                         f"~{rate * (n_cubes - n_predicted) / 60:.1f} min alla fine)")
        while pending:
            finish_oldest()

    if not rows:
        raise SystemExit(f"[{tag}] nessun cubo valutato.")

    vram = vram_report(device)
    if vram is not None:
        log.info(f"[{tag}] VRAM: picco {vram['peak_allocated_gb']:.1f} GB usati, "
                 f"{vram['peak_reserved_gb']:.1f} GB riservati su {vram['total_gb']:.1f} GB "
                 f"({vram['peak_reserved_fraction']:.0%}), batch {loader.batch_size}")
        if vram["peak_reserved_fraction"] > VRAM_WARNING_FRACTION:
            log.warning(f"[{tag}] la VRAM e' al {vram['peak_reserved_fraction']:.0%} "
                        f"(batch {loader.batch_size}): sopra ~{VRAM_WARNING_FRACTION:.0%} "
                        f"l'inferenza tende a rallentare. Prova un --batch-size piu' piccolo.")

    elapsed = time.time() - t0
    payload = {
        "run": {
            "model": run_dir.name,
            "run_dir": run_dir.as_posix(),
            "checkpoint": ckpt.name,
            "checkpoint_path": ckpt.as_posix(),
            "step": checkpoint_step(ckpt),
            "split": tag,
            "base_split": spec.name,
            "autoregressive": plan is not None,
            "window": ([plan.window_context, plan.window_predict] if plan else None),
            "n_windows": (plan.n_windows if plan else None),
            "sample_pool": desc,
            "selection": ("subset" if args.subset else "max_cubes" if args.max_cubes else "all"),
            "n_context": n_context,
            "n_target": n_target,
            "trained_windows": [int(cfg.data.params.T) - int(cfg.data.params.noise_image_num),
                                int(cfg.data.params.noise_image_num)],
            "n_cubes": len(rows),
            "n_cubes_skipped": sum(1 for r in rows if r["ENS"] is None),
            "n_samples": int(args.num_samples),
            "batch_size": loader.batch_size,
            "seed": args.seed,
            "device": str(device),
            "stats_json": str(p.stats_json),
            "dataset_json": str(p.dataset_json),
        },
        "timing": {
            "total_s": round(elapsed, 1),
            "inference_s": round(t_infer, 1),
            # secondi di CPU dello scoring (somma sui processi) e secondi in cui il processo
            # principale ha dovuto aspettarlo: la differenza e' lo scoring nascosto dietro la GPU
            "scoring_s": round(t_score_cpu, 1),
            "scoring_wait_s": round(t_wait, 1),
            "scoring_workers": workers,
            # Quel che avanza e' l'attesa dei worker del DataLoader (disco).
            "data_s": round(max(elapsed - t_infer - t_wait, 0.0) + t_data, 1),
            "s_per_cube": round(elapsed / max(len(rows), 1), 3),
            "scoring": args.scoring,
        },
        "memory": vram,
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
                                       tag, int(args.num_samples), payload,
                                       selection_suffix(args))
    print_report(payload)
    log.info(f"[out] {json_path}")
    log.info(f"[out] {csv_path}")
    return payload


# ===========================================================================
# 7. CLI
# ===========================================================================
def parse_args(argv: Optional[Sequence[str]] = None):
    """argv=None legge sys.argv; il training passa la lista, cosi' i default sono questi."""
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
                        f"extreme usa 20 frame di contesto e 40 di target, seasonal 70 e 140")
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
                        "ridotto in proporzione alla lunghezza della sequenza se piu' lunga). "
                        "A fine split si stampa il picco di VRAM: oltre ~90% rallenta")
    p.add_argument("--num-workers", type=int, default=None,
                   help="worker del DataLoader (default: data.params.num_workers del config)")
    p.add_argument("--scoring-workers", type=int, default=4,
                   help="processi che calcolano le metriche mentre la GPU predice il batch "
                        "dopo (0 = nello stesso processo, in serie)")
    p.add_argument("--autoregressive", action="store_true",
                   help="extreme e seasonal con le finestre del training (10 frame di contesto "
                        "+ 20 predetti), coprendo la finestra target a blocchi: ogni blocco "
                        "parte dagli ultimi 10 frame predetti dal precedente. I file hanno "
                        "'autoreg' nel nome. Senza, il DiT si ricostruisce con la lunghezza "
                        "dello split")
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
    p.add_argument("--export-cubes", type=int, default=0,
                   help="per-band figures (ground truth, prediction, cloud probability, error) "
                        "of N cubes of each split, in <run>/results/forecasts/<split>/<cube>/; "
                        "the same cubes at every run (0 = none)")
    p.add_argument("--export-only", action="store_true",
                   help="solo le figure di --export-cubes cubi per split, con una predizione "
                        "per cubo: niente metriche e niente file in statistics/")
    p.add_argument("--seed", type=int, default=42,
                   help="seed del sampling e del rumore di OLS")
    p.add_argument("--device", type=str, default=None, help="cuda, cuda:0, cpu (default: auto)")
    p.add_argument("--outputs-dir", type=str, default=PathsCfg.outputs_dir,
                   help="radice dei run (relativa alla root del progetto)")
    p.add_argument("--test-root", type=str, default=None,
                   help="cartella che contiene <split>_test_split/{context,target} "
                        "(default: la cartella che contiene paths.raw_dir, cioe' data/raw)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def evaluate_run(args, raise_on_failure: bool = True) -> List[Dict[str, object]]:
    """
    Valuta il run args.model sugli split args.split e scrive i risultati in
    <run>/results/statistics/. E' il corpo di main(), richiamabile anche dal training
    (src/training/ens_eval.py) a fine loop. Ritorna i payload degli split valutati;
    se qualcuno fallisce li valuta comunque tutti e poi, con raise_on_failure (il
    default, per la riga di comando), solleva SystemExit. Senza, ritorna i payload
    di quelli riusciti: i fallimenti restano nel log e nel riepilogo stampato.
    """
    use_project_dir()
    family_dir = project_path(args.outputs_dir) / viz.FAMILY_DIRS["dit"]

    if not args.model:
        raise SystemExit("--model e' obbligatorio (oppure usa --list-runs)")
    if not args.split:
        raise SystemExit("--split e' obbligatorio: " + ", ".join(viz.SPLITS))
    if args.num_samples < 1:
        raise SystemExit("--num-samples deve essere >= 1")
    if args.subset and args.max_cubes:
        raise SystemExit("--subset e --max-cubes sono alternativi")
    if args.export_only and args.export_cubes < 1:
        raise SystemExit("--export-only richiede --export-cubes >= 1")
    if args.scoring_workers < 0:
        raise SystemExit("--scoring-workers deve essere >= 0")
    if args.batch_size is not None and args.batch_size < 1:
        raise SystemExit("--batch-size deve essere >= 1")

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
    if args.autoregressive:
        print("modo        : autoregressivo (finestre del training) su extreme e seasonal")
    print(f"output      : {(run_dir / 'results' / STATISTICS_DIRNAME).as_posix()}")
    print("=" * 78)

    # gli split non cambiano il modello se hanno le stesse finestre: il caricamento
    # e' condiviso (vedi evaluate_split)
    state: Dict = {"bundle": None, "windows": None}
    payloads: List[Dict[str, object]] = []
    failed: List[Tuple[str, str]] = []
    run_split = export_split if args.export_only else evaluate_split
    for spec in args.split:
        try:
            payloads.append(run_split(spec, args, cfg, run_dir, ckpt, device, state))
        except SystemExit as err:
            failed.append((spec.name, str(err)))
            log.error(f"[{spec.name}] saltato -- {err}")
        except Exception as err:  # noqa: BLE001 - uno split rotto non butta via gli altri
            failed.append((spec.name, f"{type(err).__name__}: {err}"))
            log.exception(f"[{spec.name}] saltato")

    if len(payloads) > 1 and not args.export_only:
        print_comparison(payloads)
    if failed:
        print(f"{len(failed)} split NON valutati:")
        for name, reason in failed:
            print(f"  {name}: {reason.splitlines()[0]}")
        if raise_on_failure:
            raise SystemExit(1)
    return payloads


def main() -> None:
    args = parse_args()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=getattr(logging, args.log_level.upper(), logging.INFO))
    if args.list_runs:
        use_project_dir()
        family_dir = project_path(args.outputs_dir) / viz.FAMILY_DIRS["dit"]
        for run in viz.list_runs(family_dir):
            print(run.relative_to(project_path(args.outputs_dir)).as_posix())
        return
    evaluate_run(args)


if __name__ == "__main__":
    main()
