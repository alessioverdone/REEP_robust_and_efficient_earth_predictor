#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metriche delle RICOSTRUZIONI DEL SOLO AUTOENCODER su uno o piu' split: MAE, MSE,
ENS (EarthNetScore, con MAD, OLS, EMD, SSIM), SSIM grezza e N-MAE. Niente
diffusione: i frame target vanno nel VAE e tornano indietro.

Unisce i vecchi evaluate_ae_trainval.py ed evaluate_ae_test.py, sulla stessa
struttura di evaluate_ens.py (che valuta il DiT): stessi split, stessa maschera
nuvole, stesso scoring, stessa forma dei file di output, quindi i numeri sono
confrontabili con quelli del DiT.

    # train e validation
    python src/scripts/evaluate_ae_ens.py --model [date] --split train val

    # split di test EarthNet
    python src/scripts/evaluate_ae_ens.py --model [date] --split iid ood extreme seasonal

    # giro di prova su 50 cubi presi a caso
    python src/scripts/evaluate_ae_ens.py --model [date] --split val --subset 50

Argomenti principali
--------------------
--model   nome del run dell'autoencoder = cartella con il timestamp sotto
          registry/outputs/autoencoder/ (anche un pezzo del nome se e' unico,
          'latest' o il percorso della cartella).
--split   uno o piu' fra
              train, val           split di dataset.json (cubi da data/raw/train);
                                   sono gli stessi che il VAE ha visto in training
                                   (train) e usato per la scelta del best (val)
              iid, ood, extreme    split di test EarthNet: data/raw/<split>_test_split/
                                   {context,target}
              seasonal             come sopra, ma con finestre piu' lunghe
          (anche i nomi completi: iid_test_split, ...)
--checkpoint  'best' (best.pt, default), 'last' (last.pt) o il nome di un file .pt
              in checkpoints/

Quali frame
-----------
Si ricostruiscono i frame TARGET dello split, cioe' gli stessi che il DiT deve
predire: 20 per train / val / iid / ood, 40 per extreme, 140 per seasonal. Il VAE
ricostruisce ogni frame da solo, quindi il contesto serve solo a fare uscire i
cubi dal dataloader del DiT (stesse finestre di evaluate_ens.py).

Il VAE e' quello del run indicato, con l'architettura e la risoluzione
(data.params.resize) del suo artifacts/config.yaml; la normalizzazione e' quella
del preprocessing (stats.json), la stessa del training del VAE.

Cosa calcola
------------
    MAE       errore assoluto medio, pixel validi, 4 bande, riflettanza [0, 1]
    MSE       errore quadratico medio, stessi pixel
    ENS       media armonica di MAD, OLS, EMD, SSIM (come evaluate_ens.py)
    SSIM_RAW  SSIM prima dell'esponente (fuori dall'ENS)
    N_MAE     errore assoluto medio sull'NDVI
Scoring, maschera nuvole (soglia 0.20 sulla cloud probability o maschera binaria,
secondo il formato del cubo) e aggregazione sono quelli di evaluate_ens.py.

Campioni (K)
------------
K = 1 (default): ricostruzione deterministica, media della posterior del VAE.
K > 1: K ricostruzioni con latenti campionati dalla posterior; per ogni cubo si
tiene la migliore per ENS (best-of-K, come evaluate_ens.py). K compare nel nome
dei file di output.

Output
------
registry/outputs/autoencoder/<run>/results/statistics/
    metrics_results_<split>_K_<K>.json   aggregati + metadati + una voce per cubo
    metrics_results_<split>_K_<K>.csv    una riga per cubo
Ogni esecuzione riscrive i file del proprio split e K, senza toccare gli altri.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# inference_visualization.py ed evaluate_ens.py stanno nella stessa cartella: li si
# importa da li' anche quando lo script e' lanciato da un'altra cartella o da un
# IDE. L'import di inference_visualization mette anche la root del progetto e
# src/model su sys.path, e serve prima di ogni import `src.` o `sgm.`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import inference_visualization as viz  # noqa: E402
import evaluate_ens as ens  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from accelerate.utils import set_seed  # noqa: E402

from src.config import PathsCfg, project_path, use_project_dir  # noqa: E402
# lo scoring ENS sta in src/utils/ens_metrics.py, condiviso con il training loop
# e con evaluate_ens.py (che qui serve ancora per build_split e build_loader)
from src.utils import ens_metrics as em  # noqa: E402

log = logging.getLogger("evaluate_ae_ens")

# Le sole colonne in piu' rispetto a evaluate_ens.py: MAE e MSE in riflettanza.
CSV_FIELDS = ["cube", "MAE", "MSE", "MAD", "OLS", "EMD", "SSIM", "SSIM_RAW", "ENS",
              "N_MAE", "valid_fraction", "sample"]


# ===========================================================================
# 1. Ricostruzione
# ===========================================================================
def reconstruct(vae: torch.nn.Module, x: torch.Tensor, chunk: int,
                sample_posterior: bool) -> torch.Tensor:
    """
    (B, T, C, H, W) in [-1, 1] -> ricostruzione (B, T, C, H, W) su CPU, in fp32 come
    nel training della fase 1. I frame passano nel VAE a blocchi di `chunk` immagini
    (limita la memoria sulle sequenze lunghe). Con sample_posterior=False si usa la
    media della posterior, come run_inference_ae di inference_visualization.py.
    """
    b, t, c, h, w = x.shape
    flat = x.reshape(b * t, c, h, w)
    outs = []
    for i in range(0, flat.shape[0], chunk):
        dist = vae.encode(flat[i:i + chunk]).latent_dist
        z = dist.sample() if sample_posterior else dist.mean
        outs.append(vae.decode(z).sample.clamp(-1.0, 1.0).detach().float().cpu())
    return torch.cat(outs, dim=0).reshape(b, t, c, h, w)


def squared_error_sum(pred: np.ndarray, targ: np.ndarray, masks: np.ndarray) -> float:
    """Somma degli scarti quadratici sui pixel validi (la maschera e' (H, W, C, T))."""
    return float(((pred - targ) ** 2)[masks == 1].sum())


# ===========================================================================
# 2. Salvataggio e report
# ===========================================================================
def save_results(out_dir: Path, split: str, n_samples: int,
                 payload: Dict[str, object], suffix: str = "") -> Tuple[Path, Path]:
    """
    metrics_results_<split>_K_<K>[<suffix>].json (aggregati + metadati + per cubo) e .csv
    (una riga per cubo) in <run>/results/statistics/. Riscritti a ogni esecuzione; il
    suffisso (ens.selection_suffix) distingue una valutazione parziale dallo split intero.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"metrics_results_{split}_K_{n_samples}{suffix}"

    json_path = out_dir / f"{stem}.json"
    viz.write_json(str(json_path), payload)

    csv_path = out_dir / f"{stem}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in payload["per_cube"]:
            writer.writerow({k: row.get(k) for k in CSV_FIELDS})
    return json_path, csv_path


def print_report(payload: Dict[str, object]) -> None:
    agg, run, timing = payload["aggregate"], payload["run"], payload["timing"]
    fmt = lambda v, d=4: "n/d" if v is None else f"{v:.{d}f}"  # noqa: E731

    print()
    print("=" * 62)
    print(f"  autoencoder {run['model']}  |  split {run['split'].upper()}")
    print(f"  checkpoint {run['checkpoint']} (epoca {run['epoch']})")
    print(f"  {run['n_cubes']} cubi, {run['n_target']} frame target ricostruiti, "
          f"K={run['n_samples']}")
    print("=" * 62)
    print()
    print(f"  MAE   pixel, 4 bande           {fmt(agg['MAE'], 6)}")
    print(f"  MSE   pixel, 4 bande           {fmt(agg['MSE'], 6)}")
    print()
    print(f"  ENS   EarthNetScore            {fmt(agg['ENS'])}")
    print(f"          MAD   valore           {fmt(agg['MAD'])}")
    print(f"          OLS   trend            {fmt(agg['OLS'])}")
    print(f"          EMD   distribuzione    {fmt(agg['EMD'])}")
    print(f"          SSIM  percettiva       {fmt(agg['SSIM'])}")
    print()
    print(f"  SSIM  grezza, fuori dall'ENS   {fmt(agg['SSIM_RAW'])}")
    print(f"  N-MAE NDVI                     {fmt(agg['N_MAE'])}")
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
    fmt = lambda v: "n/d" if v is None else f"{v:.4f}"  # noqa: E731
    print()
    print(f"{'split':{width}s} {'cubi':>5s} {'K':>2s} {'px.val':>7s} "
          f"{'MAE':>8s} {'MSE':>8s} {'ENS':>8s} {'MAD':>8s} {'OLS':>8s} {'EMD':>8s} "
          f"{'SSIM':>8s} {'SSIMraw':>8s}")
    for p in payloads:
        agg, run = p["aggregate"], p["run"]
        valid = agg["valid_fraction"]
        print(f"{str(run['split']):{width}s} {run['n_cubes']:5d} {run['n_samples']:2d} "
              f"{(f'{valid:.1%}' if valid is not None else 'n/d'):>7s} "
              f"{fmt(agg['MAE']):>8s} {fmt(agg['MSE']):>8s} {fmt(agg['ENS']):>8s} "
              f"{fmt(agg['MAD']):>8s} {fmt(agg['OLS']):>8s} {fmt(agg['EMD']):>8s} "
              f"{fmt(agg['SSIM']):>8s} {fmt(agg['SSIM_RAW']):>8s}")
    print()


# ===========================================================================
# 3. Valutazione di uno split
# ===========================================================================
def evaluate_split(spec: viz.SplitSpec, args, cfg, run_dir: Path, ckpt: Path, ae_info: Dict,
                   vae: torch.nn.Module, device: torch.device) -> Dict[str, object]:
    """Ricostruisce i cubi dello split, calcola le metriche, aggrega e salva."""
    n_context, n_target = viz.split_windows(spec, cfg)

    t_data = time.time()
    files, sources, desc = ens.build_split(spec, cfg, args, n_context, n_target)
    dataset, loader = ens.build_loader(cfg, files, sources, args, n_context, n_target,
                                       int(cfg.data.params.T))
    n_cubes = len(files)
    log.info(f"[{spec.name}] {desc}: {n_cubes} cubi valutati, {n_target} frame target "
             f"(sequenza {n_context + n_target} = {n_context} contesto + {n_target} target), "
             f"batch {loader.batch_size}, maschera nuvole {'ON' if args.cloud_mask else 'OFF'}")
    t_data = time.time() - t_data

    p = cfg.data.params
    lo, hi = (a.reshape(-1).astype(np.float32)
              for a in viz.load_highres_stats(p.stats_json, p.stats_time_key, p.n_channels))
    chunk = int(cfg.autoencoder.batch_size)
    sample_posterior = args.num_samples > 1

    # Il rumore di regolarizzazione di OLS usa il generatore legacy di numpy, il
    # campionamento della posterior (K > 1) quello di torch: senza i due seed due
    # esecuzioni darebbero numeri leggermente diversi.
    set_seed(args.seed)
    np.random.seed(args.seed)

    rows: List[Dict[str, object]] = []
    t0 = time.time()
    t_infer = t_score = 0.0

    for batch in loader:
        t_batch = time.time()
        cube_paths = batch["cube_path"]
        cloud = batch.pop("cloud_mask", None)
        # i frame target sono gli ultimi n_target dell'asse del tempo
        target = batch["imgs"][:, -n_target:].to(device=device, dtype=torch.float32)
        if target.shape[1] != n_target or target.shape[2] != int(p.n_channels):
            raise SystemExit(f"[{spec.name}] attesi {n_target} frame x {p.n_channels} bande, "
                             f"ottenuti {tuple(target.shape)}")
        with torch.no_grad():
            recons = [reconstruct(vae, target, chunk, sample_posterior)
                      for _ in range(args.num_samples)]
        target = target.cpu()
        t_infer += time.time() - t_batch

        t_batch = time.time()
        for b in range(target.shape[0]):
            targ_cube = em.to_cube(target[b], lo, hi)
            mask_cube = em.to_mask(None if cloud is None else cloud[b], targ_cube.shape)
            pred_cubes = [em.to_cube(r[b], lo, hi) for r in recons]

            row = em.score_cube(pred_cubes, targ_cube, mask_cube, args.ssim_data_range,
                                 args.scoring)
            # MAE e MSE dello stesso campione scelto per l'ENS
            sq = (squared_error_sum(pred_cubes[row["sample"]], targ_cube, mask_cube)
                  if row["sample"] >= 0 else 0.0)
            cube = Path(str(cube_paths[b]))
            row["cube"] = f"{cube.parent.name}/{cube.name}"
            row["mse_sum"] = sq
            row["MAE"] = row["p_sum"] / row["p_count"] if row["p_count"] else None
            row["MSE"] = sq / row["p_count"] if row["p_count"] else None
            row["N_MAE"] = row["n_sum"] / row["n_count"] if row["n_count"] else None
            rows.append(row)
        t_score += time.time() - t_batch

        done = len(rows)
        if done % max(loader.batch_size * 5, 1) == 0 or done == n_cubes:
            rate = (time.time() - t0) / max(done, 1)
            log.info(f"[{spec.name}] {done}/{n_cubes} cubi ({rate:.2f} s/cubo, "
                     f"~{rate * (n_cubes - done) / 60:.1f} min alla fine)")

    if not rows:
        raise SystemExit(f"[{spec.name}] nessun cubo valutato.")

    agg = em.aggregate(rows)
    p_count = sum(int(r["p_count"]) for r in rows)
    agg["MAE"] = agg.pop("P_MAE")
    agg["MSE"] = sum(float(r["mse_sum"]) for r in rows) / p_count if p_count else None

    elapsed = time.time() - t0
    epoch = ae_info.get("epoch")
    payload = {
        "run": {
            "model": run_dir.name,
            "type": "ae",
            "run_dir": run_dir.as_posix(),
            "checkpoint": ckpt.name,
            "checkpoint_path": ckpt.as_posix(),
            "epoch": None if epoch is None else int(epoch) + 1,
            "best_metric": ae_info.get("best_metric"),
            "save_best_by": ae_info.get("save_best_by"),
            "split": spec.name,
            "sample_pool": desc,
            "selection": ("subset" if args.subset else "max_cubes" if args.max_cubes else "all"),
            "frames_evaluated": "target",
            "n_context": n_context,
            "n_target": n_target,
            "n_cubes": len(rows),
            "n_cubes_skipped": sum(1 for r in rows if r["ENS"] is None),
            "n_samples": int(args.num_samples),
            "latent": "posterior sample" if sample_posterior else "posterior mean",
            "seed": args.seed,
            "device": str(device),
            "resize": int(p.resize),
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
            "ssim_scaling": em.SSIM_SCALING,
        },
        "aggregate": agg,
        "per_cube": rows,
    }

    json_path, csv_path = save_results(run_dir / "results" / em.STATISTICS_DIRNAME,
                                       spec.name, int(args.num_samples), payload,
                                       ens.selection_suffix(args))
    print_report(payload)
    log.info(f"[out] {json_path}")
    log.info(f"[out] {csv_path}")
    return payload


# ===========================================================================
# 4. CLI
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="MAE, MSE, ENS (MAD, OLS, EMD, SSIM), SSIM grezza e N-MAE delle "
                    "ricostruzioni di un autoencoder gia' addestrato, su uno o piu' split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", type=str,
                   help="nome del run dell'autoencoder (cartella con il timestamp sotto "
                        "registry/outputs/autoencoder/); anche un pezzo del nome se e' unico, "
                        "'latest' o il percorso della cartella")
    p.add_argument("--split", type=viz.split_arg, nargs="+", metavar="SPLIT",
                   help=f"uno o piu' fra: {', '.join(viz.SPLITS)} (anche iid_test_split, ...). "
                        f"extreme usa 20 frame di contesto e 40 di target, seasonal 70 e 140")
    p.add_argument("--list-runs", action="store_true",
                   help="stampa i run di autoencoder disponibili ed esce")
    p.add_argument("--checkpoint", type=str, default="best",
                   help="'best' (best.pt), 'last' (last.pt) o il nome di un file .pt "
                        "in checkpoints/")
    p.add_argument("--num-samples", type=int, default=1,
                   help="K ricostruzioni per cubo. 1 = media della posterior (deterministica); "
                        "K > 1 = latenti campionati, si tiene la migliore (best-of-K dell'ENS)")
    p.add_argument("--subset", type=int, default=0,
                   help="valuta N cubi a caso dello split (0 = tutti), scelti con --subset-seed")
    p.add_argument("--subset-seed", type=int, default=0, help="seed di --subset")
    p.add_argument("--max-cubes", type=int, default=0,
                   help="valuta solo i primi N cubi dello split in ordine (0 = tutti): "
                        "per una prova, non per un numero da riportare")
    p.add_argument("--batch-size", type=int, default=None,
                   help="cubi per batch (default: data.params.batch_size del config del run, "
                        "ridotto in proporzione alla lunghezza della sequenza se piu' lunga)")
    p.add_argument("--num-workers", type=int, default=None,
                   help="worker del DataLoader (default: data.params.num_workers del config)")
    p.add_argument("--merge-workers", type=int, default=8,
                   help="thread per la fusione context+target degli split EarthNet")
    p.add_argument("--scoring", choices=em.SCORING_MODES, default="fast",
                   help="implementazione di MAD ed EMD: 'fast' (vettorizzate) o 'toolkit' "
                        "(originali di earthnet). Stesso numero")
    p.add_argument("--no-cloud-mask", dest="cloud_mask", action="store_false",
                   help="misura su TUTTI i pixel, nuvole comprese. Sconsigliato: sotto le "
                        "nuvole il target non e' la superficie")
    p.add_argument("--cloud-prob-threshold", type=float, default=em.CLOUD_PROB_THRESHOLD,
                   help="cubi a 7 canali: un pixel e' valido se cloud probability < soglia")
    p.add_argument("--cloud-prob-scale", type=float, default=em.CLOUD_PROB_SCALE,
                   help="fondo scala del canale cloud probability (100 = percentuale)")
    p.add_argument("--ssim-data-range", type=float, default=em.SSIM_DATA_RANGE,
                   help="data_range della SSIM. 2.0 e' quello di evaluate_ens.py (numeri "
                        "confrontabili con il DiT); 1.0 e' l'ampiezza reale della riflettanza")
    p.add_argument("--seed", type=int, default=42,
                   help="seed del campionamento della posterior (K > 1) e del rumore di OLS")
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
    family_dir = project_path(args.outputs_dir) / viz.FAMILY_DIRS["ae"]

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
    ckpt = viz.resolve_ae_checkpoint(run_dir, args.checkpoint)
    device = viz.resolve_device(args.device)

    # Tutto cio' che serve ai dati deve esistere PRIMA di caricare il VAE: un percorso
    # sbagliato deve fermare lo script subito.
    params = cfg.data.params
    missing = [f"data.params.{key} -> {params[key]}" for key in ("stats_json", "cond_stats_json")
               if not Path(str(params[key])).is_file()]
    if any(not s.earthnet for s in args.split) and not Path(str(params.dataset_json)).is_file():
        missing.append(f"data.params.dataset_json -> {params.dataset_json}")
    if missing:
        raise SystemExit(f"{run_dir.name}: il config del run punta a file che non ci sono "
                         f"(percorsi relativi a {Path.cwd()}):\n  " + "\n  ".join(missing))

    vae, ae_info = viz.load_ae(cfg, ckpt, device)
    epoch = None if ae_info["epoch"] is None else int(ae_info["epoch"]) + 1

    print("=" * 78)
    print(f"modello     : ae | run {run_dir.name}")
    print(f"checkpoint  : {ckpt.as_posix()}")
    print(f"epoca       : {epoch} | {ae_info['save_best_by']} = {ae_info['best_metric']}")
    print(f"device      : {device}")
    print(f"split       : {', '.join(s.name for s in args.split)} | K={args.num_samples} | "
          f"seed {args.seed}")
    print(f"output      : {(run_dir / 'results' / em.STATISTICS_DIRNAME).as_posix()}")
    print("=" * 78)

    payloads: List[Dict[str, object]] = []
    failed: List[Tuple[str, str]] = []
    for spec in args.split:
        try:
            payloads.append(evaluate_split(spec, args, cfg, run_dir, ckpt, ae_info, vae, device))
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
