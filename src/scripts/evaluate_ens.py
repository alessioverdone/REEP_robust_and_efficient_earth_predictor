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
              train, val           split di dataset.json (cubi da data/raw/train)
              iid, ood, extreme    split di test EarthNet: data/raw/<split>_test_split/
                                   {context,target}
              seasonal             come sopra, ma con finestre piu' lunghe
          (anche i nomi completi: iid_test_split, ...)

Finestre temporali
------------------
    train / val / iid / ood / extreme   10 frame di contesto + 20 target (T = 30)
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
import inference_visualization as viz  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from accelerate.utils import set_seed  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from src.config import PathsCfg, project_path, use_project_dir  # noqa: E402
from src.dataset.dataset_info import load_dataset_info  # noqa: E402
# lo scoring ENS sta in src/utils/ens_metrics.py, condiviso con il training loop
# e con evaluate_ae_ens.py
from src.utils.ens_metrics import (  # noqa: E402
    CLOUD_PROB_SCALE, CLOUD_PROB_THRESHOLD, SCORING_MODES, SSIM_DATA_RANGE, SSIM_SCALING,
    STATISTICS_DIRNAME, MaskedCubeDataset, MaskSources, aggregate, score_batch,
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
# 5. Aggregazione, salvataggio, report
# ===========================================================================
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
        rows.extend(score_batch(preds, target, cloud, cube_paths, lo, hi,
                                args.ssim_data_range, args.scoring))
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
    return p.parse_args(argv)


def evaluate_run(args) -> List[Dict[str, object]]:
    """
    Valuta il run args.model sugli split args.split e scrive i risultati in
    <run>/results/statistics/. E' il corpo di main(), richiamabile anche dal training
    (src/training/ens_eval.py) a fine loop. Ritorna i payload degli split valutati;
    se qualcuno fallisce li valuta comunque tutti e poi solleva SystemExit.
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
