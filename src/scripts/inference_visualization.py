#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Confronto visivo fra frame reali e frame predetti/ricostruiti da un modello
gia' addestrato, su uno o piu' split.

    # DiT: 20 frame futuri veri vs 20 predetti, per ogni cubo
    python src/scripts/visualize_predictions.py --type dit --model 2026-10-12_09-30-00 \
        --split val iid seasonal --num-samples 5 --seed 42

    # Autoencoder: 10 frame di contesto veri vs 10 ricostruiti, per ogni cubo
    python src/scripts/visualize_predictions.py --type ae --model 2026-10-10_18-02-11 \
        --split train val ood --num-samples 5

    python src/scripts/visualize_predictions.py --type dit --list-runs

Argomenti principali
--------------------
--type    ae | dit. Quale fase si applica: autoencoder (fase 1) o diffusion (fase 2).
--model   nome del run addestrato = la cartella con il timestamp sotto
          registry/outputs/autoencoder/ o registry/outputs/diffusion/. Si accetta
          anche un pezzo del nome (se e' unico), 'latest' (l'ultimo run) o il
          percorso della cartella del run.
--split   uno o piu' fra
              train, val       split di dataset.json (preprocessing), cubi da data/raw/train
              test             split di test di dataset.json (stesse location del train)
              iid, ood,
              extreme          split di test EarthNet: data/raw/<split>_test_split/{context,target}
              seasonal         come sopra, ma con finestre piu' lunghe
          Si accetta anche il nome completo (iid_test_split, ...).

Finestre temporali
------------------
    train / val / test / iid / ood / extreme   10 frame di contesto + 20 target (T = 30)
    seasonal                                    20 frame di contesto + 40 target (T = 60)

Output
------
registry/outputs/<autoencoder|diffusion>/<run>/results/predictions/<ae|dit>/<split>/
    <cubo>/band_1.png ... band_4.png    una figura per banda: riga sopra reale,
                                        riga sotto predetto/ricostruito
    <cubo>/comparison.pdf               le stesse figure, una pagina per banda
    selected_samples.txt                i cubi scelti (il seed li rende riproducibili)
    run_info.json                       modello, checkpoint, finestre, seed, rmse per cubo
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# --- percorsi ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "src" / "model"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import torch
from accelerate import PartialState
from accelerate.utils import set_seed
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from safetensors.torch import load_file as load_safetensors
from torch.utils.data import DataLoader

from src.config import PathsCfg, load_config, project_path, use_project_dir, validate_config
from src.dataset.dataloader_s2_dif import S2DiffusionCubeDataset, load_highres_stats
from src.dataset.dataset_info import load_dataset_info, split_files
from src.model.autoencoder_utils import get_vae
from src.training.run_outputs import write_json

log = logging.getLogger("visualize_predictions")

# --type -> sottocartella di outputs_dir con i run della fase
FAMILY_DIRS = {"ae": "autoencoder", "dit": "diffusion"}

# Le prime 4 bande di highresdynamic, nell'ordine in cui il dataloader le consegna
BAND_NAMES = ("blue", "green", "red", "nir")


# ===========================================================================
# 1. Split e finestre temporali
# ===========================================================================
@dataclass(frozen=True)
class SplitSpec:
    name: str
    # True = split di test EarthNet, con context/ e target/ in file separati
    earthnet: bool
    # None = dal config del run (data.params.T e noise_image_num)
    n_context: Optional[int] = None
    n_target: Optional[int] = None


SPLITS: Dict[str, SplitSpec] = {
    "train": SplitSpec("train", earthnet=False),
    "val": SplitSpec("val", earthnet=False),
    "test": SplitSpec("test", earthnet=False),
    "iid": SplitSpec("iid", earthnet=True, n_context=10, n_target=20),
    "ood": SplitSpec("ood", earthnet=True, n_context=10, n_target=20),
    "extreme": SplitSpec("extreme", earthnet=True, n_context=10, n_target=20),
    # contesto e target lunghi il doppio
    "seasonal": SplitSpec("seasonal", earthnet=True, n_context=20, n_target=40),
}


def split_arg(value: str) -> SplitSpec:
    """Nome di uno split (anche nella forma iid_test_split) -> SplitSpec."""
    name = value.lower()
    if name.endswith("_test_split"):
        name = name[: -len("_test_split")]
    if name not in SPLITS:
        raise argparse.ArgumentTypeError(
            f"split {value!r} sconosciuto, valori possibili: {', '.join(SPLITS)}")
    return SPLITS[name]


def split_windows(spec: SplitSpec, cfg) -> Tuple[int, int]:
    """(frame di contesto, frame target) dello split."""
    if spec.n_context is not None:
        return spec.n_context, spec.n_target
    n_target = int(cfg.data.params.noise_image_num)
    return int(cfg.data.params.T) - n_target, n_target


# ===========================================================================
# 2. Run, config, checkpoint
# ===========================================================================
def list_runs(family_dir: Path) -> List[Path]:
    """I run di una fase (cartelle con artifacts/config.yaml), dal piu' vecchio."""
    if not family_dir.is_dir():
        return []
    return [d for d in sorted(family_dir.iterdir())
            if (d / "artifacts" / "config.yaml").is_file()]


def resolve_run(model: str, family_dir: Path) -> Path:
    """
    Nome del modello -> cartella del run. Il nome e' quello della cartella con il
    timestamp; si accetta anche una corrispondenza parziale, purche' unica, il
    percorso della cartella e 'latest'.
    """
    as_path = Path(model)
    if as_path.is_dir() and (as_path / "artifacts" / "config.yaml").is_file():
        return as_path.resolve()

    runs = list_runs(family_dir)
    if not runs:
        raise SystemExit(f"Nessun run in {family_dir} (manca artifacts/config.yaml).")
    if model == "latest":
        return runs[-1]

    exact = [d for d in runs if d.name == model]
    if exact:
        return exact[0]
    partial = [d for d in runs if model in d.name]
    if len(partial) == 1:
        log.info(f"run '{model}' risolto in {partial[0].name}")
        return partial[0]
    if len(partial) > 1:
        raise SystemExit(f"'{model}' e' ambiguo, corrisponde a:\n  "
                         + "\n  ".join(d.name for d in partial))
    raise SystemExit(f"Run '{model}' non trovato in {family_dir}. Disponibili:\n  "
                     + "\n  ".join(d.name for d in runs))


def load_run_config(run_dir: Path):
    """
    artifacts/config.yaml del run: il config risolto all'avvio (percorso del VAE,
    scale/shift factor, checkpoint ripreso). E' quello, non un config del repo, la
    sorgente di verita' su normalizzazione, dimensioni e sampler del modello.
    """
    path = run_dir / "artifacts" / "config.yaml"
    try:
        return load_config(path)
    except Exception as err:
        raise SystemExit(f"config del run non caricabile ({path}): {err}") from err


def resolve_dit_checkpoint(run_dir: Path, which: str) -> Path:
    """
    'best'   -> checkpoints/best, lo stato con il latter_rmse piu' basso
    'latest' -> il checkpoint-<step> con lo step piu' alto
    altro    -> il nome di una sottocartella di checkpoints/
    """
    root = run_dir / "checkpoints"
    if not root.is_dir():
        raise SystemExit(f"{run_dir}: manca la cartella checkpoints/")
    if which == "best":
        path = root / "best"
    elif which == "latest":
        steps = [d for d in root.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
        if not steps:
            raise SystemExit(f"{root}: nessun checkpoint-<step> da cui prendere 'latest'")
        path = max(steps, key=lambda d: int(d.name.split("-")[-1]))
    else:
        path = root / which
    if not (path / "model.safetensors").is_file():
        available = sorted(d.name for d in root.iterdir() if d.is_dir())
        raise SystemExit(f"{path}/model.safetensors non trovato. "
                         f"Cartelle in {root}: {', '.join(available) or 'nessuna'}")
    return path


def resolve_ae_checkpoint(run_dir: Path, which: str) -> Path:
    """'best' -> best.pt, 'last'/'latest' -> last.pt, altro -> il nome di un file in checkpoints/."""
    root = run_dir / "checkpoints"
    name = {"best": "best.pt", "last": "last.pt", "latest": "last.pt"}.get(which, which)
    path = root / name
    if not path.is_file():
        available = sorted(p.name for p in root.glob("*.pt")) if root.is_dir() else []
        raise SystemExit(f"{path} non trovato (checkpoint salvati: {', '.join(available) or 'nessuno'}). "
                         f"Con autoencoder.save_checkpoints=false il run non li scrive.")
    return path


# ===========================================================================
# 3. Device e dtype (stessa logica del trainer)
# ===========================================================================
def resolve_device(requested: Optional[str]) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtypes(cfg, device: torch.device) -> Tuple[torch.dtype, torch.dtype, str]:
    """
    (weight_dtype, first_stage_dtype, mixed_precision), come in
    src/training/runtime.py: mixed_precision fissa il dtype del VAE congelato,
    accelerate.weight_dtype='float32' alza a fp32 i soli moduli allenati.
    """
    mixed = str(cfg.accelerate.mixed_precision)
    table = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if mixed not in table:
        raise SystemExit(f"accelerate.mixed_precision={mixed!r} non gestito: {tuple(table)}")
    first_stage_dtype = table[mixed]
    weight_dtype = torch.float32 if cfg.accelerate.weight_dtype == "float32" else first_stage_dtype
    if device.type == "cpu" and torch.float16 in (first_stage_dtype, weight_dtype):
        log.warning("fp16 non e' supportato su CPU: inferenza in fp32.")
        return torch.float32, torch.float32, "no"
    return weight_dtype, first_stage_dtype, mixed


def autocast_context(device: torch.device, mixed_precision: str):
    """
    L'autocast che accelerator.prepare() mette attorno al forward del modello:
    con weight_dtype=float32 i pesi sono in fp32 ma il calcolo resta in bf16, ed
    e' cosi' che il modello e' stato addestrato e validato.
    """
    if mixed_precision == "no":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def load_weights(module: torch.nn.Module, path: Path, label: str) -> None:
    """
    Carica i pesi con la severita' della validazione del trainer: una chiave
    mancante lascerebbe parametri inizializzati a caso e la predizione non
    vorrebbe dire niente.

    Eccezione: i buffer sinusoidali `pe` (posizioni temporali), che dipendono
    dalla lunghezza della sequenza e sono deterministici. Se la lunghezza e' stata
    cambiata (split seasonal) quelli del checkpoint si scartano e restano quelli
    ricostruiti dal modello; qualunque altra differenza di forma e' un errore.
    """
    state = load_safetensors(str(path))
    own = module.state_dict()
    resized: List[str] = []
    for key in list(state):
        if key in own and state[key].shape != own[key].shape:
            if key.rsplit(".", 1)[-1] == "pe":
                resized.append(key)
                del state[key]
            else:
                raise SystemExit(f"{label}: {key} ha shape {tuple(state[key].shape)} nel "
                                 f"checkpoint e {tuple(own[key].shape)} nel modello")
    missing, unexpected = module.load_state_dict(state, strict=False)
    missing = [k for k in missing if k not in resized]
    if missing:
        raise SystemExit(f"{label}: {len(missing)} chiavi mancanti in {path}, es. {missing[:5]}")
    if unexpected:
        log.warning(f"{label}: {len(unexpected)} chiavi inattese ignorate, es. {unexpected[:5]}")
    if resized:
        log.info(f"{label}: {len(resized)} buffer di posizione temporale ricostruiti "
                 f"per la nuova lunghezza della sequenza")


def with_windows(cfg, n_context: int, n_target: int):
    """
    Copia di cfg con le finestre dello split. I parametri del DiT che dipendono
    dalla lunghezza (image_time_length, noise_image_num, ...) sono interpolazioni
    di data.params, quindi seguono da soli; la lunghezza massima della posizione
    temporale va alzata a mano se serve.
    """
    total = n_context + n_target
    if total == cfg.data.params.T and n_target == cfg.data.params.noise_image_num:
        return cfg
    cfg = copy.deepcopy(cfg)
    cfg.data.params.T = total
    cfg.data.params.noise_image_num = n_target
    temporal = cfg.diffusion.network_config.params.temporal_module_kwargs
    temporal.temporal_position_encoding_max_len = max(
        int(temporal.temporal_position_encoding_max_len), total)
    try:
        validate_config(cfg)
    except ValueError as err:
        raise SystemExit(f"finestre {n_context}+{n_target} incompatibili con il config del run: {err}")
    return cfg


# ===========================================================================
# 4. Modelli
# ===========================================================================
@dataclass
class DitBundle:
    cfg: object
    model: torch.nn.Module
    conditioner: torch.nn.Module
    denoiser: torch.nn.Module
    sampler: object
    first_stage_model: torch.nn.Module
    first_stage_info: dict
    weight_dtype: torch.dtype
    mixed_precision: str


def load_dit(cfg, ckpt_dir: Path, device: torch.device) -> DitBundle:
    """
    DiT + wrapper + conditioner + denoiser + sampler + VAE congelato, istanziati
    dal config del run come fa il trainer (instantiate_models), con i pesi del
    checkpoint. denoiser e sampler non stanno nel checkpoint: si ricostruiscono dal
    config, esattamente come a ogni avvio del training.
    """
    # diffusion_utils e' pesante da importare e il suo logger e' quello di
    # accelerate, che vuole uno stato gia' costruito
    PartialState()
    from src.model import diffusion_utils as du
    from src.model.sgm import instantiate_from_config
    from src.model.sgm.util import get_obj_from_str

    weight_dtype, first_stage_dtype, mixed = resolve_dtypes(cfg, device)
    diffusion = cfg.diffusion

    model = instantiate_from_config(diffusion.network_config)
    model = get_obj_from_str(diffusion.network_wrapper)(model, compile_model=diffusion.compile_model)
    load_weights(model, ckpt_dir / "model.safetensors", "model")
    model = model.to(device, dtype=weight_dtype).eval()

    conditioner = instantiate_from_config(diffusion.conditioner_config)
    cond_ckpt = ckpt_dir / "model_1.safetensors"
    if cond_ckpt.is_file():
        load_weights(conditioner, cond_ckpt, "conditioner")
    else:
        log.warning(f"{cond_ckpt} assente: il conditioner resta come costruito dal config.")
    conditioner = conditioner.to(device, dtype=weight_dtype).eval()

    denoiser = instantiate_from_config(diffusion.denoiser_config).to(device, dtype=weight_dtype)
    sampler = instantiate_from_config(diffusion.sampler_config)

    # first stage: percorso, controllo sul dataset e normalizzazione dei latenti
    du.resolve_first_stage_path(cfg)
    first_stage_model, first_stage_info = du.build_first_stage_model(diffusion.first_stage_config)
    du.check_first_stage_dataset(cfg, first_stage_info)
    du.resolve_latent_normalization(cfg, first_stage_info)
    first_stage_model = first_stage_model.to(device, dtype=first_stage_dtype).eval()
    for p in first_stage_model.parameters():
        p.requires_grad = False

    return DitBundle(cfg, model, conditioner, denoiser, sampler, first_stage_model,
                     first_stage_info, weight_dtype, mixed)


def load_ae(cfg, ckpt_path: Path, device: torch.device):
    """L'AutoencoderKL descritto da cfg.autoencoder, con i pesi del checkpoint della fase 1."""
    auto = cfg.autoencoder
    n_bands = int(cfg.data.params.n_channels)
    vae = get_vae(in_channels=n_bands,
                  out_channels=n_bands,
                  latent_channels=auto.latent_channels,
                  block_out_channels=tuple(auto.block_out_channels),
                  layers_per_block=auto.layers_per_block,
                  norm_num_groups=auto.norm_num_groups)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        # i checkpoint della fase 1 portano anche optimizer e metriche; li abbiamo
        # prodotti noi, quindi si ricaricano senza restrizioni
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
             for k, v in state.items()}
    try:
        vae.load_state_dict(state)
    except RuntimeError as err:
        raise SystemExit(f"{ckpt_path} non combacia con l'AutoencoderKL del config: {err}") from err

    trained_at = (ckpt.get("arch") or {}).get("image_size") if isinstance(ckpt, dict) else None
    if trained_at is not None and int(trained_at) != int(cfg.data.params.resize):
        raise SystemExit(f"{ckpt_path} e' stato addestrato a {trained_at}x{trained_at}, "
                         f"ma il config dice data.params.resize={cfg.data.params.resize}")
    info = {"epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
            "best_metric": ckpt.get("best_metric") if isinstance(ckpt, dict) else None,
            "save_best_by": ckpt.get("save_best_by") if isinstance(ckpt, dict) else None}
    return vae.to(device).eval(), info


# ===========================================================================
# 5. Cubi: pool, fusione contesto/target, dataset
# ===========================================================================
def list_pairs(test_dir: Path, merged_dir: Path) -> List[Tuple[Path, Path, Path]]:
    """
    (cubo fuso, file di contesto, file di target) per ogni cubo di uno split di
    test EarthNet. I due file stanno in context/<tile>/ e target/<tile>/ con lo
    stesso nome a meno del prefisso (context_<cubo>.npz, target_<cubo>.npz).
    """
    ctx_root, tgt_root = test_dir / "context", test_dir / "target"
    for root in (ctx_root, tgt_root):
        if not root.is_dir():
            raise SystemExit(f"{root} non trovato: scarica lo split di test in {test_dir} "
                             f"(oppure indica --test-root).")

    def key(path: Path, root: Path) -> Tuple[str, str]:
        rel = path.relative_to(root)
        return rel.parent.as_posix(), rel.name.split("_", 1)[-1]

    targets = {key(p, tgt_root): p for p in tgt_root.rglob("*.npz")}
    pairs, missing = [], 0
    for ctx in sorted(ctx_root.rglob("*.npz")):
        tile, cube = key(ctx, ctx_root)
        tgt = targets.get((tile, cube))
        if tgt is None:
            missing += 1
            continue
        pairs.append((merged_dir / tile / cube, ctx, tgt))
    if missing:
        log.warning(f"{missing} file di contesto senza il corrispondente target in {tgt_root}")
    if not pairs:
        raise SystemExit(f"nessuna coppia context/target in {test_dir}")
    return pairs


def _merged_is_valid(path: Path, sources: Sequence[Path], expected: np.ndarray) -> bool:
    """Cubo fuso gia' presente, piu' recente dei sorgenti e fatto con gli stessi parametri."""
    if not path.is_file():
        return False
    if path.stat().st_mtime < max(s.stat().st_mtime for s in sources):
        return False
    try:
        with np.load(path) as z:
            return bool(np.array_equal(z["merge_info"], expected))
    except Exception:
        return False


def merge_pair(ctx_path: Path, tgt_path: Path, out_path: Path, n_context: int,
               n_target: int, n_bands: int, steps_per_frame: int) -> None:
    """
    Un cubo di test = file di contesto + file di target -> un solo .npz nel
    formato che il dataset del training legge (highresdynamic, highresstatic,
    mesodynamic), con i frame di contesto prima di quelli di target.

    - highresdynamic: i frame si concatenano sull'asse del tempo, tenendo le
      prime n_bands bande (blue, green, red, nir; il resto sono maschere);
    - mesodynamic: i giorni di contesto e quelli di target si concatenano, a meno
      che il contesto non li copra gia' tutti;
    - highresstatic e mesostatic: dal contesto.
    Se qualcosa non torna (numero di frame, giorni di meteo) si ferma qui con le
    shape in chiaro, invece di produrre un cubo sfasato.
    """
    expected = np.array([n_context, n_target, n_bands, steps_per_frame])
    if _merged_is_valid(out_path, (ctx_path, tgt_path), expected):
        return

    with np.load(ctx_path) as ctx, np.load(tgt_path) as tgt:
        hd_c, hd_t = ctx["highresdynamic"], tgt["highresdynamic"]
        if hd_c.shape[3] != n_context or hd_t.shape[3] != n_target:
            raise SystemExit(
                f"{ctx_path.name}: attesi {n_context} frame di contesto e {n_target} di target, "
                f"trovati {hd_c.shape[3]} e {hd_t.shape[3]} (shape {hd_c.shape} / {hd_t.shape})")
        if min(hd_c.shape[2], hd_t.shape[2]) < n_bands:
            raise SystemExit(f"{ctx_path.name}: servono {n_bands} bande, trovate "
                             f"{hd_c.shape[2]} nel contesto e {hd_t.shape[2]} nel target")

        merged = {"highresdynamic": np.concatenate([hd_c[:, :, :n_bands], hd_t[:, :, :n_bands]],
                                                   axis=3)}

        static_src = ctx if "highresstatic" in ctx.files else tgt
        if "highresstatic" not in static_src.files:
            raise SystemExit(f"{ctx_path.name}: highresstatic assente in contesto e target")
        merged["highresstatic"] = static_src["highresstatic"]
        if "mesostatic" in ctx.files:
            merged["mesostatic"] = ctx["mesostatic"]

        need = (n_context + n_target) * steps_per_frame
        md_c = ctx["mesodynamic"]
        md_t = tgt["mesodynamic"] if "mesodynamic" in tgt.files else None
        if md_c.shape[3] >= need:
            merged["mesodynamic"] = md_c
        elif md_t is not None and md_c.shape[3] + md_t.shape[3] == need:
            merged["mesodynamic"] = np.concatenate([md_c, md_t], axis=3)
        else:
            raise SystemExit(
                f"{ctx_path.name}: servono {need} giorni di mesodynamic "
                f"({n_context + n_target} frame x {steps_per_frame}), trovati "
                f"{md_c.shape[3]} nel contesto e "
                f"{'nessuno' if md_t is None else md_t.shape[3]} nel target")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez(tmp, merge_info=expected, **merged)
    os.replace(tmp, out_path)


def select_samples(pool: Sequence[Path], num_samples: int, seed: int) -> List[Path]:
    """
    num_samples cubi estratti dal pool con un seed esplicito.

    L'estrazione avviene sul pool ORDINATO: lo stesso seed da' gli stessi cubi su
    modelli diversi. Stesso generatore usato dal resto del repo.
    """
    ordered = sorted(pool)
    if not ordered:
        raise SystemExit("il pool di campioni e' vuoto")
    if num_samples < 1:
        raise SystemExit("--num-samples deve essere >= 1")
    if num_samples > len(ordered):
        raise SystemExit(f"richiesti {num_samples} campioni ma il pool ne ha {len(ordered)}")
    order = np.random.default_rng(seed).permutation(len(ordered))
    return [ordered[i] for i in sorted(order[:num_samples].tolist())]


def load_condition_stats(params) -> dict:
    """
    cond_stats.json del preprocessing (min/max di mesodynamic e highresstatic sul
    train). Si legge soltanto: ricalcolarlo da qui sovrascriverebbe il file con cui
    il modello e' stato addestrato.
    """
    path = Path(params.cond_stats_json)
    if not path.is_file():
        raise SystemExit(f"{path} non trovato: lancia src/dataset/preprocessing.py")
    stats = json.loads(path.read_text())
    differ = [k for k, want in (("steps_per_frame", params.meso_steps_per_frame),
                                ("meso_center_crop", params.meso_center_crop),
                                ("meso_agg", params.meso_agg))
              if stats.get(k) != want]
    if differ or len(stats["meso"]["lo"]) != params.meso_channels:
        raise SystemExit(f"{path} e' stato calcolato con altri parametri ({differ or 'meso_channels'}): "
                         f"rilancia il training o `dataloader_s2_dif.py --compute-cond-stats`")
    return stats


def build_eval_dataset(cfg, files: Sequence[Path], n_context: int, n_target: int) -> S2DiffusionCubeDataset:
    """
    Stesso dataset del training, con le stesse statistiche di normalizzazione, ma
    senza flip casuale, con il percorso del .npz nel batch e con le finestre
    dello split (T = contesto + target).
    """
    p = cfg.data.params
    highres_lo, highres_hi = load_highres_stats(p.stats_json, time_key=p.stats_time_key,
                                                n_bands=p.n_channels)
    cond = load_condition_stats(p)
    return S2DiffusionCubeDataset(
        files=list(files),
        highres_lo=highres_lo,
        highres_hi=highres_hi,
        meso_lo=cond["meso"]["lo"],
        meso_hi=cond["meso"]["hi"],
        static_lo=cond["static"]["lo"],
        static_hi=cond["static"]["hi"],
        T=n_context + n_target,
        n_predict=n_target,
        n_channels=p.n_channels,
        meso_channels=p.meso_channels,
        steps_per_frame=p.meso_steps_per_frame,
        meso_center_crop=p.meso_center_crop,
        meso_agg=p.meso_agg,
        meso_time_span="all",
        do_hflip=False,
        return_paths=True,
        dtype=p.dtype,
        resize=p.resize,
    )


def build_pool(spec: SplitSpec, cfg, args, n_context: int, n_target: int
               ) -> Tuple[List[Path], str, Callable[[List[Path]], List[Path]]]:
    """
    (elenco dei cubi, descrizione, prepara) dello split. `prepara` riceve i cubi
    scelti e restituisce i .npz da leggere: per gli split EarthNet fonde prima
    contesto e target, e solo per i cubi scelti.
    """
    params = cfg.data.params
    if not spec.earthnet:
        info = load_dataset_info(params.dataset_json)
        try:
            files = split_files(info, spec.name, params.data_root)
        except KeyError as err:
            raise SystemExit(str(err)) from err
        if not files:
            raise SystemExit(f"lo split '{spec.name}' di {params.dataset_json} e' vuoto")

        def prepare(selected: List[Path]) -> List[Path]:
            missing = [f for f in selected if not f.is_file()]
            if missing:
                raise SystemExit(f"cubi non trovati (manca data/raw?): {missing[:3]}")
            return selected

        return (files,
                f"split '{spec.name}' di {Path(params.dataset_json).name} ({len(files)} cubi)",
                prepare)

    test_dir = (Path(args.test_root) / f"{spec.name}_test_split" if args.test_root
                else project_path(cfg.paths.raw_dir).parent / f"{spec.name}_test_split")
    merged_root = (Path(args.merged_root) if args.merged_root
                   else project_path(cfg.paths.preprocessed_dir) / "merged_test")
    pairs = list_pairs(test_dir, merged_root / spec.name)
    by_merged = {merged: (ctx, tgt) for merged, ctx, tgt in pairs}

    def prepare(selected: List[Path]) -> List[Path]:
        for merged in selected:
            ctx, tgt = by_merged[merged]
            merge_pair(ctx, tgt, merged, n_context, n_target,
                       int(params.n_channels), int(params.meso_steps_per_frame))
        return selected

    return (sorted(by_merged),
            f"test {spec.name} ({test_dir.as_posix()}, {len(pairs)} coppie context/target)",
            prepare)


# ===========================================================================
# 6. Inferenza
# ===========================================================================
Captured = List[Tuple[str, torch.Tensor, torch.Tensor]]   # (cubo, reale, predetto) (T, C, H, W) in [-1, 1]


def move_batch(batch: Dict, device: torch.device, dtype: torch.dtype) -> Dict:
    """Batch sul device e nel dtype dei pesi, come nel ciclo di validazione del trainer."""
    return {key: value.to(device=device, dtype=dtype) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def run_inference_dit(loader: DataLoader, bundle: DitBundle, device: torch.device,
                      n_target: int) -> Captured:
    """
    Predizione dei n_target frame futuri con la funzione di validazione del
    trainer, cosi' la procedura (encode col VAE, rumore iniziale miscelato al
    contesto, EulerEDMSampler con CFG, decode) e' esattamente quella del training.
    Restituisce, per ogni cubo, i frame futuri veri e quelli predetti.
    """
    from src.model import diffusion_utils as du

    cfg = bundle.cfg
    captured: Captured = []
    for batch in loader:
        cube_paths = batch.get("cube_path")
        batch = move_batch(batch, device, bundle.weight_dtype)
        with torch.no_grad(), autocast_context(device, bundle.mixed_precision):
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
                # le metriche EarthNet sono argomenti obbligatori della firma ma
                # non vengono usate nel corpo
                metrics_rgbn=None,
                metrics_ndvi=None,
                metrics_arvi=None,
                past_weigth=cfg.diffusion.loss_fn_config.params.past_weight,
                val_num_samples=1,
                save_dir=None,
                save_counter=None,
                save_limit=0,
            )
        target, pred = forecast["target"], forecast["pred"]
        for b in range(pred.shape[0]):
            captured.append((str(cube_paths[b]),
                             target[b].detach().float().cpu(),
                             pred[b].detach().float().cpu()))
    return captured


def run_inference_ae(loader: DataLoader, vae: torch.nn.Module, device: torch.device,
                     n_context: int) -> Captured:
    """
    Ricostruzione dei n_context frame di contesto: encode (media della posterior,
    come evaluate_vae) e decode, un frame alla volta, in fp32 come nel training
    della fase 1. Restituisce, per ogni cubo, i frame veri e quelli ricostruiti.
    """
    captured: Captured = []
    for batch in loader:
        imgs = batch["imgs"][:, :n_context].to(device=device, dtype=torch.float32)  # (B, n, C, H, W)
        b, n, c, h, w = imgs.shape
        with torch.no_grad():
            flat = imgs.reshape(b * n, c, h, w)
            recon = vae.decode(vae.encode(flat).latent_dist.mean).sample
        recon = recon.clamp(-1.0, 1.0).reshape(b, n, c, h, w)
        for i, path in enumerate(batch["cube_path"]):
            captured.append((str(path), imgs[i].cpu(), recon[i].float().cpu()))
    return captured


# ===========================================================================
# 7. Post-processing e controlli
# ===========================================================================
def check_pair(real: torch.Tensor, pred: torch.Tensor, sample: str, label: str,
               n_frames: int, n_bands: int) -> None:
    """
    Controlli espliciti prima di disegnare: numero di frame, numero di bande e
    compatibilita' fra reale e predetto. Niente reshape di comodo: se qualcosa non
    torna il messaggio dice cosa era atteso, cosa e' arrivato e su quale campione.
    """
    expected = (n_frames, n_bands)
    problems = []
    for name, x in (("reale", real), ("predetto", pred)):
        if x.ndim != 4:
            problems.append(f"{name}: attesa (T, C, H, W), ottenuta {tuple(x.shape)}")
        elif tuple(x.shape[:2]) != expected:
            problems.append(f"{name}: attesi {n_frames} frame x {n_bands} bande, "
                            f"ottenuti {x.shape[0]} x {x.shape[1]} (shape {tuple(x.shape)})")
    if not problems and real.shape != pred.shape:
        problems.append(f"shape incompatibili: reale {tuple(real.shape)} vs predetto {tuple(pred.shape)}")
    if problems:
        raise SystemExit(f"[{label}] [campione {sample}] controllo dimensioni fallito:\n  - "
                         + "\n  - ".join(problems))


def denormalize(x: torch.Tensor, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """
    Inversa esatta di minmax_pm1() del dataloader:
    x_norm = 2*(x - lo)/(hi - lo) - 1  ->  x = (x_norm + 1)/2 * (hi - lo) + lo.
    Il clipping fatto in normalizzazione non e' invertibile: i valori fuori range
    erano gia' saturati a +-1 prima del training.
    """
    arr = x.numpy().astype(np.float32)                  # (T, C, H, W)
    lo_b = lo.reshape(1, -1, 1, 1).astype(np.float32)
    hi_b = hi.reshape(1, -1, 1, 1).astype(np.float32)
    return (arr + 1.0) / 2.0 * (hi_b - lo_b) + lo_b


def band_limits(real: np.ndarray, pred: np.ndarray, mode: str,
                lo: np.ndarray, hi: np.ndarray) -> List[Tuple[float, float]]:
    """
    Un solo intervallo (vmin, vmax) per banda, condiviso da tutti i frame e da
    entrambe le righe: normalizzare ogni immagine per conto suo farebbe sparire
    proprio l'errore che si vuole vedere.

    gt       percentili 2-98 del solo reale: il predetto si legge sulla scala del
             riferimento, e quello che esce dai limiti e' visibilmente saturo.
    joint    percentili 2-98 di reale e predetto insieme.
    physical il range pieno [lo, hi] di stats.json, uguale per ogni cubo e modello.
    """
    limits = []
    for c in range(real.shape[1]):
        if mode == "physical":
            vmin, vmax = float(lo[c]), float(hi[c])
        else:
            data = real[:, c] if mode == "gt" else np.concatenate([real[:, c], pred[:, c]])
            vmin, vmax = (float(v) for v in np.percentile(data, (2.0, 98.0)))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = float(np.min(real[:, c])), float(np.max(real[:, c]))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        limits.append((vmin, vmax))
    return limits


# ===========================================================================
# 8. Visualizzazione
# ===========================================================================
@dataclass(frozen=True)
class FigureStyle:
    row_labels: Tuple[str, str]             # riga sopra (reale), riga sotto (predetto)
    frame_title: Callable[[int], str]       # titolo della colonna del frame t (da 0)
    frames_desc: str                        # "20 frame futuri (istanti 11-30 del minicubo)"


def plot_band(real: np.ndarray, pred: np.ndarray, band: int, limits: Tuple[float, float],
              header: str, style: FigureStyle, cols: int, cmap: str) -> Figure:
    """
    Una figura per banda: tutti i frame su blocchi da `cols` colonne e, per ogni
    blocco, due righe, reale sopra e predetto sotto, incolonnati sullo stesso
    frame. Tutte le immagini condividono (vmin, vmax), riassunto dalla colorbar.
    """
    n_frames = real.shape[0]
    n_blocks = int(np.ceil(n_frames / cols))
    vmin, vmax = limits

    fig = Figure(figsize=(1.25 * cols + 1.8, 1.45 * 2 * n_blocks + 1.1))
    axes = fig.subplots(2 * n_blocks, cols, squeeze=False)
    image = None
    for block in range(n_blocks):
        for col in range(cols):
            t = block * cols + col
            for row_off, (data, label) in enumerate(zip((real, pred), style.row_labels)):
                ax = axes[2 * block + row_off][col]
                ax.set_xticks([])
                ax.set_yticks([])
                if t >= n_frames:
                    ax.axis("off")
                    continue
                image = ax.imshow(data[t, band], vmin=vmin, vmax=vmax, cmap=cmap,
                                  interpolation="nearest")
                # il frame va scritto su entrambe le righe: con piu' blocchi
                # incolonnati, un'etichetta sulla sola riga reale lascerebbe le
                # predizioni del blocco sopra vicine ai titoli del blocco sotto
                ax.set_title(style.frame_title(t), fontsize=8, pad=3)
                if col == 0:
                    ax.set_ylabel(label, fontsize=9)

    band_name = BAND_NAMES[band] if band < len(BAND_NAMES) else "?"
    fig.suptitle(f"{header}\nbanda {band + 1} ({band_name})  |  {style.frames_desc}  |  "
                 f"scala comune [{vmin:.4f}, {vmax:.4f}]", fontsize=10)
    fig.tight_layout(rect=(0.0, 0.0, 0.93, 0.93))
    if image is not None:
        cax = fig.add_axes((0.945, 0.08, 0.012, 0.78))
        fig.colorbar(image, cax=cax)
    return fig


def save_sample_figures(real: np.ndarray, pred: np.ndarray, limits, header: str,
                        style: FigureStyle, out_dir: Path, cols: int, cmap: str,
                        dpi: int) -> List[Path]:
    """Un PNG per banda piu' un PDF con una pagina per banda."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    pdf_path = out_dir / "comparison.pdf"
    with PdfPages(pdf_path) as pdf:
        for band in range(real.shape[1]):
            fig = plot_band(real, pred, band, limits[band], header, style, cols, cmap)
            png_path = out_dir / f"band_{band + 1}.png"
            fig.savefig(png_path, dpi=dpi)
            pdf.savefig(fig)
            written.append(png_path)
    written.append(pdf_path)
    return written


def make_style(model_type: str, n_context: int, n_frames: int) -> FigureStyle:
    if model_type == "dit":
        return FigureStyle(
            row_labels=("Ground truth", "Prediction"),
            frame_title=lambda t: f"t+{t + 1}",
            frames_desc=(f"{n_frames} frame futuri (istanti {n_context + 1}-{n_context + n_frames} "
                         f"del minicubo)"))
    return FigureStyle(
        row_labels=("Original", "Reconstruction"),
        frame_title=lambda t: f"t={t + 1}",
        frames_desc=f"{n_frames} frame di contesto (istanti 1-{n_frames} del minicubo)")


# ===========================================================================
# 9. CLI
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Confronto visivo fra frame reali e predetti (dit) o ricostruiti (ae), "
                    "per un modello gia' addestrato.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--type", choices=tuple(FAMILY_DIRS), required=True,
                   help="ae = autoencoder (10 frame di contesto veri vs ricostruiti), "
                        "dit = diffusion (20 frame futuri veri vs predetti)")
    p.add_argument("--model", type=str,
                   help="nome del run addestrato (cartella con il timestamp sotto "
                        "registry/outputs/<autoencoder|diffusion>/); anche un pezzo del "
                        "nome se e' unico, 'latest' o il percorso della cartella")
    p.add_argument("--split", type=split_arg, nargs="+", metavar="SPLIT",
                   help=f"uno o piu' fra: {', '.join(SPLITS)} (anche iid_test_split, ...). "
                        f"seasonal usa 20 frame di contesto e 40 di target")
    p.add_argument("--checkpoint", type=str, default="best",
                   help="dit: 'best' (checkpoints/best), 'latest' (checkpoint-<step> piu' alto) "
                        "o il nome di una sottocartella di checkpoints/. "
                        "ae: 'best' (best.pt), 'last' (last.pt) o il nome di un file .pt")
    p.add_argument("--list-runs", action="store_true",
                   help="stampa i run disponibili per --type ed esce")
    p.add_argument("--num-samples", type=int, default=1, help="quanti minicubi per split")
    p.add_argument("--seed", type=int, default=42,
                   help="seed della selezione e del sampling: stesso seed = stessi cubi")
    p.add_argument("--batch-size", type=int, default=1,
                   help="cubi per batch durante l'inferenza (memoria GPU)")
    p.add_argument("--device", type=str, default=None, help="cuda, cuda:0, cpu (default: auto)")
    p.add_argument("--outputs-dir", type=str, default=PathsCfg.outputs_dir,
                   help="radice dei run (relativa alla root del progetto)")
    p.add_argument("--test-root", type=str, default=None,
                   help="cartella che contiene <split>_test_split/{context,target} "
                        "(default: la cartella che contiene paths.raw_dir, cioe' data/raw)")
    p.add_argument("--merged-root", type=str, default=None,
                   help="dove si salvano i cubi test fusi context+target "
                        "(default: <preprocessed_dir>/merged_test)")
    p.add_argument("--cols", type=int, default=10,
                   help="frame per riga; con 20 o 40 frame si ottengono 2 o 4 blocchi")
    p.add_argument("--scale", choices=("gt", "joint", "physical"), default="gt",
                   help="come si sceglie l'intervallo di intensita' condiviso da reale e predetto")
    p.add_argument("--cmap", type=str, default="viridis", help="colormap di matplotlib")
    p.add_argument("--dpi", type=int, default=120, help="risoluzione dei PNG")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    use_project_dir()
    family_dir = project_path(args.outputs_dir) / FAMILY_DIRS[args.type]

    if args.list_runs:
        for run in list_runs(family_dir):
            print(run.relative_to(project_path(args.outputs_dir)).as_posix())
        return
    if not args.model:
        raise SystemExit("--model e' obbligatorio (oppure usa --list-runs)")
    if not args.split:
        raise SystemExit("--split e' obbligatorio: " + ", ".join(SPLITS))
    if args.cols < 1:
        raise SystemExit("--cols deve essere >= 1")

    # --- run, config, checkpoint -----------------------------------------
    run_dir = resolve_run(args.model, family_dir)
    cfg = load_run_config(run_dir)
    device = resolve_device(args.device)
    if args.type == "dit":
        ckpt = resolve_dit_checkpoint(run_dir, args.checkpoint)
    else:
        ckpt = resolve_ae_checkpoint(run_dir, args.checkpoint)

    print("=" * 78)
    print(f"modello     : {args.type} | run {run_dir.name}")
    print(f"run dir     : {run_dir.as_posix()}")
    print(f"checkpoint  : {ckpt.as_posix()}")
    print(f"device      : {device}")
    print(f"split       : {', '.join(s.name for s in args.split)}")
    print("=" * 78)

    vae, bundle, bundle_windows = None, None, None
    if args.type == "ae":
        vae, ae_info = load_ae(cfg, ckpt, device)
        epoch = None if ae_info["epoch"] is None else int(ae_info["epoch"]) + 1
        print(f"epoca       : {epoch} | {ae_info['save_best_by']} = {ae_info['best_metric']}")

    pred_root = run_dir / "results" / "predictions" / args.type

    for spec in args.split:
        n_context, n_target = split_windows(spec, cfg)
        split_cfg = cfg
        if args.type == "dit":
            # il DiT dipende dalla lunghezza della sequenza: lo si (ri)costruisce
            # quando le finestre dello split cambiano
            split_cfg = with_windows(cfg, n_context, n_target)
            if split_cfg is not cfg:
                log.warning(
                    f"[{spec.name}] finestre {n_context}+{n_target} (T={n_context + n_target}) "
                    f"diverse da quelle del training (T={cfg.data.params.T}, "
                    f"{cfg.data.params.noise_image_num} target): i pesi appresi sono gli stessi, "
                    f"ma e' un'estrapolazione in lunghezza e le predizioni potrebbero degradare.")
            if bundle is None or bundle_windows != (n_context, n_target):
                bundle, bundle_windows = None, None
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                bundle = load_dit(split_cfg, ckpt, device)
                bundle_windows = (n_context, n_target)

        # --- cubi -----------------------------------------------------------
        pool, pool_desc, prepare = build_pool(spec, split_cfg, args, n_context, n_target)
        selected = prepare(select_samples(pool, args.num_samples, args.seed))

        print(f"\n[{spec.name}] pool: {pool_desc}")
        print(f"[{spec.name}] sequenza: {n_context + n_target} istanti = {n_context} contesto + "
              f"{n_target} target | seed {args.seed}")
        for f in selected:
            print(f"  {Path(f).name}")

        out_root = pred_root / spec.name
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "selected_samples.txt").write_text(
            "".join(f"{Path(f).as_posix()}\n" for f in selected), encoding="utf-8")

        dataset = build_eval_dataset(split_cfg, selected, n_context, n_target)
        loader = DataLoader(dataset, batch_size=max(1, args.batch_size), shuffle=False,
                            num_workers=0)

        # --- inferenza ------------------------------------------------------
        set_seed(args.seed)
        if args.type == "dit":
            from src.model import diffusion_utils as du
            du.check_data_model_alignment(dataset[0], split_cfg.diffusion, bundle.first_stage_info)
            captured = run_inference_dit(loader, bundle, device, n_target)
            n_frames = n_target
        else:
            captured = run_inference_ae(loader, vae, device, n_context)
            n_frames = n_context
        if len(captured) != len(selected):
            raise SystemExit(f"[{spec.name}] attese {len(selected)} predizioni, ottenute "
                             f"{len(captured)}: l'inferenza non ha coperto tutti i campioni.")

        # --- denormalizzazione, controlli, figure ---------------------------
        p = split_cfg.data.params
        lo, hi = (a.reshape(-1).astype(np.float32)
                  for a in load_highres_stats(p.stats_json, p.stats_time_key, p.n_channels))
        style = make_style(args.type, n_context, n_frames)

        manifest = []
        for (path, real_t, pred_t), expected_path in zip(captured, selected):
            sample = Path(path).stem
            check_pair(real_t, pred_t, sample, f"{args.type} {run_dir.name} {spec.name}",
                       n_frames, int(p.n_channels))
            real = denormalize(real_t, lo, hi)
            pred = denormalize(pred_t, lo, hi)
            limits = band_limits(real, pred, args.scale, lo, hi)

            header = f"{args.type} {run_dir.name}  |  {spec.name}  |  {sample}"
            written = save_sample_figures(real, pred, limits, header, style, out_root / sample,
                                          args.cols, args.cmap, args.dpi)
            rmse = float(np.sqrt(np.mean((real - pred) ** 2)))
            print(f"[{spec.name}] {sample}: {n_frames} frame x {real.shape[1]} bande per lato, "
                  f"rmse {rmse:.5f} -> {(out_root / sample).as_posix()}")
            manifest.append({
                "sample": sample,
                "npz": Path(expected_path).as_posix(),
                "shape": list(real.shape),
                "rmse_denormalized": rmse,
                "band_limits": [list(lim) for lim in limits],
                "files": [f.as_posix() for f in written],
            })

        write_json(str(out_root / "run_info.json"), {
            "type": args.type,
            "run": run_dir.name,
            "run_dir": run_dir.as_posix(),
            "checkpoint": ckpt.as_posix(),
            "split": spec.name,
            "sample_pool": pool_desc,
            "n_context": n_context,
            "n_target": n_target,
            "frames_shown": n_frames,
            "seed": args.seed,
            "num_samples": args.num_samples,
            "device": str(device),
            "stats_json": str(p.stats_json),
            "highres_lo": lo.tolist(),
            "highres_hi": hi.tolist(),
            "intensity_scale": args.scale,
            "trained_windows": [int(cfg.data.params.T) - int(cfg.data.params.noise_image_num),
                                int(cfg.data.params.noise_image_num)],
            "samples": manifest,
        })

    print("=" * 78)
    print(f"Output in {pred_root.as_posix()}")


if __name__ == "__main__":
    main()
