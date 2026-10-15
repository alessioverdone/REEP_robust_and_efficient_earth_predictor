#REFACTORING: import inutilizzati (torchvision, logging, imageio,
#             torch.nn.functional as F, amp, safe_open)
import csv
import datetime
import functools
import os
import json
import traceback
import torch
import time
import torchvision
import logging
import math
import shutil
import accelerate
import torch
import torch.utils.checkpoint
import imageio
import numpy as np
import torch.nn.functional as F
from PIL import Image

from torch.cuda import amp
from omegaconf import OmegaConf
from accelerate import skip_first_batches
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from tqdm.auto import tqdm
from transformers.utils import ContextManagers
from diffusers.optimization import get_scheduler
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors

from einops import rearrange

from src.config import DiffusionCfg
from src.dataset.dataset_info import dataset_identity, load_dataset_info
from src.model.autoencoder_utils import append_history_row, get_vae
from src.model.sgm import instantiate_from_config
from src.model.sgm.modules.diffusionmodules.metric import EarthnetX_Metric
from src.model.sgm.util import get_obj_from_str, default
from src.training.run_outputs import (count_parameters, environment_info, relative_to_project,
                                      save_run_config, write_json)
from src.training.ens_eval import ENS_KEYS, EnsEvaluator, wrap_with_cloud_mask
from src.training.state import TrainState
from src.utils.plots import plot_diffusion_history, plot_lead_time_rmse, save_forecast_grid

logger = get_logger(__name__, log_level="INFO")

# Ordine delle somme accumulate durante la validazione, prodotte da
# log_videos_local_and_wandb(): errori quadratici e conteggi, non RMSE. Le RMSE
# sull'intero validation set si ricavano da qui con validation_rmse().
# La tupla sta nello schema del config (unica fonte, e resta accanto ai
# parametri della validazione che la producono).
VAL_SUM_KEYS = DiffusionCfg.VAL_SUM_KEYS


def validation_rmse(totals):
    """
    Dalle somme accumulate sul validation set alle quattro RMSE.

    totals ha lo stesso ordine di VAL_SUM_KEYS ed e' gia' sommato su tutti i
    batch e su tutti i processi: la radice si prende qui, una volta sola, sul
    totale. E' il motivo per cui in giro si passano somme e non medie.
    """
    s = dict(zip(VAL_SUM_KEYS, [float(v) for v in totals]))

    def rmse(se, n):
        return float(np.sqrt(se / n)) if n > 0 else float("nan")

    return {
        "past_mean_latent_rmse": rmse(s["past_mean_se"], s["n_latent"]),
        "predict_latent_rmse": rmse(s["predict_se"], s["n_latent"]),
        "fromer_rmse": rmse(s["fromer_se"], s["n_fromer"]),
        "latter_rmse": rmse(s["latter_se"], s["n_latter"]),
        "mean_of_k_latent_rmse": rmse(s["mean_k_se"], s["n_latent"]),
        "best_of_k_latent_rmse": rmse(s["best_k_se"], s["n_latent"]),
    }


# ---------------------------------------------------------------------------
# First stage (autoencoder gia' addestrato da train_autoencoder.py)
# ---------------------------------------------------------------------------
# Iperparametri che definiscono l'architettura dell'AutoencoderKL: sono quelli
# su cui fase 1 e fase 2 devono essere d'accordo.
_VAE_ARCH_KEYS = ("in_channels", "latent_channels", "block_out_channels",
                  "layers_per_block", "norm_num_groups", "image_size")


def load_first_stage_state_dict(path):
    """
    Estrae dal checkpoint del VAE i pesi e, se ci sono, gli iperparametri con
    cui e' stato addestrato.

    save_checkpoint() (fase 1) scrive un dict con 'model_state_dict',
    'optimizer_state_dict', le metriche ed eventualmente l'architettura;
    torch.save(vae.state_dict(), ...) scrive invece i soli pesi. Qui si
    accettano entrambi i formati, e si toglie l'eventuale prefisso 'module.'
    lasciato da DataParallel/DDP.

    Restituisce (state_dict, arch, meta): `arch` sono gli iperparametri
    dichiarati dal checkpoint, vuoto se il file non li porta con se'; `meta`
    holds the dataset the weights were trained on, and the scale_factor,
    shift_factor and latent statistics measured by phase 1 (None when the
    checkpoint predates them).
    """
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # I checkpoint della fase 1 contengono anche stati dell'optimizer e
        # metriche: da torch 2.6 weights_only e' True per default e li rifiuta.
        # Il file lo abbiamo prodotto noi, quindi si ricarica senza restrizioni.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    arch = {}
    meta = {"dataset": None, "scale_factor": None, "shift_factor": None, "latent_stats": None}
    if isinstance(ckpt, dict):
        # l'architettura puo' stare in un sottodizionario o sparsa alla radice
        # (save_checkpoint passa `extra` con update())
        declared = ckpt.get("arch") if isinstance(ckpt.get("arch"), dict) else ckpt
        arch = {k: declared[k] for k in _VAE_ARCH_KEYS if k in declared}
        meta = {"dataset": ckpt.get("dataset"),
                "scale_factor": ckpt.get("scale_factor"),
                "shift_factor": ckpt.get("shift_factor"),
                "latent_stats": ckpt.get("latent_stats")}

        for key in ("model_state_dict", "state_dict", "vae", "module"):
            inner = ckpt.get(key)
            if isinstance(inner, dict):
                ckpt = inner
                break

    if not isinstance(ckpt, dict):
        raise TypeError(f"{path}: atteso uno state_dict, trovato {type(ckpt)}")

    if any(k.startswith("module.") for k in ckpt.keys()):
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}

    return ckpt, arch, meta


def vae_arch_from_state_dict(state_dict):
    """
    Deduce l'architettura dalle forme dei pesi, per i checkpoint che non la
    dichiarano (i primi, salvati come semplice state_dict).

    Le forme parlano da sole: conv_in dice quanti canali ha l'immagine,
    quant_conv quanti ne ha il latente (x2, media e varianza), il numero di
    down_blocks e di resnets per blocco dice il resto.
    """
    arch = {}
    conv_in = state_dict.get("encoder.conv_in.weight")
    if conv_in is not None:
        arch["in_channels"] = int(conv_in.shape[1])

    quant_conv = state_dict.get("quant_conv.weight")
    if quant_conv is not None:
        arch["latent_channels"] = int(quant_conv.shape[0]) // 2

    block_out_channels = []
    i = 0
    while f"encoder.down_blocks.{i}.resnets.0.conv1.weight" in state_dict:
        block_out_channels.append(
            int(state_dict[f"encoder.down_blocks.{i}.resnets.0.conv1.weight"].shape[0]))
        i += 1
    if block_out_channels:
        arch["block_out_channels"] = block_out_channels

    layers = 0
    while f"encoder.down_blocks.0.resnets.{layers}.conv1.weight" in state_dict:
        layers += 1
    if layers:
        arch["layers_per_block"] = layers

    return arch


def check_first_stage_arch(pretrained_path, declared, inferred, expected):
    """
    Il config della fase 2 deve descrivere lo STESSO modello che la fase 1 ha
    addestrato. `declared` e' quello che dice il checkpoint, `inferred` quello
    che si legge dalle forme dei pesi, `expected` quello che dice il config:
    al primo disaccordo si ferma qui, con i numeri in chiaro, invece di
    schiantarsi piu' sotto dentro load_state_dict.
    """
    def uguali(a, b):
        # le liste possono arrivare come ListConfig, tuple o list
        if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            return list(a) == list(b)
        return a == b

    problemi = []
    for key, atteso in expected.items():
        for fonte, trovato in (("dichiarato nel checkpoint", declared.get(key)),
                               ("dedotto dai pesi", inferred.get(key))):
            if trovato is not None and not uguali(trovato, atteso):
                problemi.append(f"{key}: il config dice {atteso}, {fonte} e' {trovato}")

    if problemi:
        raise RuntimeError(
            f"{pretrained_path} non e' il VAE descritto dal config:\n  - "
            + "\n  - ".join(problemi)
            + "\n\nL'architettura della fase 2 viene dalla sezione `autoencoder` del "
              "config: o si allinea quella, o si punta pretrained_path al "
              "checkpoint giusto."
        )


def build_first_stage_model(first_stage_config):
    """
    Costruisce il first stage congelato e ci carica sopra i pesi della fase 1.

    In ingresso arriva il nodo `diffusion.first_stage_config` del config, che e'
    completo anche quando nel yaml si vede solo `type: vae`: i params li mette
    lo schema di src/config.py, e per il VAE sono interpolazioni della sezione
    `autoencoder`, cioe' gli stessi iperparametri con cui la fase 1 lo ha
    addestrato. Passando il nodo (non una copia staccata) le interpolazioni si
    risolvono, perche' conserva il riferimento alla radice del config:

        type: vae
        pretrained_path: checkpoints_s2_T10/vae_s2_T10_best.pth   # ${autoencoder.checkpoints_dir}/${autoencoder.save_path}
        params:
          in_channels: 4        # ${data.params.n_channels}
          latent_channels: 4    # ${autoencoder.latent_channels}
          block_out_channels: [64, 128, 256]   # ${autoencoder.block_out_channels}
          layers_per_block: 2   # ${autoencoder.layers_per_block}
          norm_num_groups: 32   # ${autoencoder.norm_num_groups}

    Il modello lo costruisce get_vae(), la stessa funzione che usa la fase 1:
    le due fasi devono creare lo stesso identico modulo, non due copie che si
    somigliano. I valori vengono poi ricontrollati contro il checkpoint.

    Con 'type: mae' il first stage e' invece un Masked Autoencoder
    (mae/first_stage.py), che espone la stessa interfaccia encode/decode e porta
    i suoi params (non ha una fase 1 in questo repo).
    """
    if str(first_stage_config.get("type", "vae")).lower() == "mae":
        # Import locale: se il pacchetto mae non c'e', il percorso VAE resta
        # comunque funzionante.
        from mae.first_stage import build_mae_first_stage

        return build_mae_first_stage(first_stage_config)

    params = first_stage_config.get("params", None)
    if params is None:
        raise ValueError(
            "diffusion.first_stage_config.params assente: lo schema di "
            "src/config.py lo riempie da solo (VAEParams), quindi se manca il "
            "config e' stato costruito a mano o il tipo del first stage non e' "
            "quello che sembra."
        )
    params = OmegaConf.to_container(params, resolve=True)

    # Niente default silenziosi: un iperparametro mancante qui significa un VAE
    # diverso da quello addestrato, e un load_state_dict che fallisce piu' sotto.
    mancanti = [k for k in ("in_channels", "latent_channels", "block_out_channels",
                            "layers_per_block", "norm_num_groups") if params.get(k) is None]
    if mancanti:
        raise ValueError(
            f"diffusion.first_stage_config.params incompleto, mancano {mancanti}. "
            f"Per il VAE quei valori arrivano dalla sezione `autoencoder` del config."
        )

    in_channels = params["in_channels"]
    out_channels = params.get("out_channels", in_channels)
    latent_channels = params["latent_channels"]
    block_out_channels = tuple(params["block_out_channels"])
    layers_per_block = params["layers_per_block"]
    norm_num_groups = params["norm_num_groups"]

    vae = get_vae(
        in_channels=in_channels,
        out_channels=out_channels,
        latent_channels=latent_channels,
        block_out_channels=block_out_channels,
        layers_per_block=layers_per_block,
        norm_num_groups=norm_num_groups,
    )

    pretrained_path = first_stage_config.get("pretrained_path", None)
    if pretrained_path is None:
        raise ValueError("first_stage_config.pretrained_path non impostato nel config")
    if not os.path.isfile(pretrained_path):
        raise FileNotFoundError(f"pesi del VAE non trovati: {pretrained_path}")

    state_dict, declared_arch, meta = load_first_stage_state_dict(pretrained_path)
    check_first_stage_arch(
        pretrained_path,
        declared=declared_arch,
        inferred=vae_arch_from_state_dict(state_dict),
        expected={"in_channels": in_channels,
                  "latent_channels": latent_channels,
                  "block_out_channels": list(block_out_channels),
                  "layers_per_block": layers_per_block,
                  # the resolution the VAE was trained at must be the one the
                  # dataloader delivers (data.params.resize)
                  "image_size": params.get("image_size")},
    )
    if "image_size" not in declared_arch:
        logger.warning(
            f"{pretrained_path} does not declare the resolution it was trained at "
            f"(checkpoint older than arch.image_size): make sure it matches "
            f"data.params.resize={params.get('image_size')}."
        )

    info = vae.load_state_dict(state_dict, strict=False)
    if info.missing_keys:
        raise RuntimeError(
            f"{pretrained_path} non combacia con l'AutoencoderKL costruito dal config "
            f"(latent_channels={latent_channels}, block_out_channels={block_out_channels}): "
            f"{len(info.missing_keys)} pesi mancanti, es. {info.missing_keys[:5]}"
        )
    if info.unexpected_keys:
        logger.warning(
            f"{pretrained_path}: {len(info.unexpected_keys)} chiavi ignorate, "
            f"es. {info.unexpected_keys[:5]}"
        )

    vae.eval()
    # Fattore di downsampling spaziale: un conv di stride 2 per ogni blocco
    # tranne l'ultimo. Serve al DiT per sapere a che risoluzione arrivano i
    # latenti rispetto alle immagini e alla variabile statica.
    info = {
        "in_channels": in_channels,
        "latent_channels": latent_channels,
        "down_ratio": 2 ** (len(block_out_channels) - 1),
        "pretrained_path": pretrained_path,
        "dataset": meta["dataset"],
        "scale_factor": meta["scale_factor"],
        "shift_factor": meta["shift_factor"],
        "latent_stats": meta["latent_stats"],
    }
    logger.info(
        f"First stage caricato da {pretrained_path} | in_channels={in_channels} "
        f"| latent_channels={latent_channels} | down_ratio={info['down_ratio']}"
    )
    return vae, info


def resolve_first_stage_path(cfg):
    """
    diffusion.first_stage_config.pretrained_path = 'latest' becomes the best.pt
    of the most recent COMPLETED autoencoder run (under autoencoder.outputs_dir)
    trained on the current dataset at data.params.resize. Any other value is a
    path and is left as it is. The config is updated in place, so the path
    actually used ends up in the logs and in the trackers.
    """
    first_stage = cfg.diffusion.first_stage_config
    if first_stage.type != "vae" or first_stage.pretrained_path != "latest":
        return first_stage.pretrained_path

    current = dataset_identity(load_dataset_info(cfg.data.params.dataset_json))
    resize = cfg.data.params.resize
    outputs_dir = cfg.autoencoder.outputs_dir
    runs = sorted(os.listdir(outputs_dir), reverse=True) if os.path.isdir(outputs_dir) else []
    for run in runs:
        best = os.path.join(outputs_dir, run, "checkpoints", "best.pt")
        info_json = os.path.join(outputs_dir, run, "artifacts", "training_info.json")
        if not (os.path.isfile(best) and os.path.isfile(info_json)):
            continue
        with open(info_json) as f:
            info = json.load(f)
        dataset = info.get("dataset", {})
        if (info.get("run", {}).get("status") == "completed"
                and (dataset.get("identity") or {}).get("id") == current["id"]
                and list(dataset.get("training_resolution") or []) == [resize, resize]):
            first_stage.pretrained_path = best
            logger.info(f"first_stage pretrained_path=latest -> {best}")
            return best

    raise FileNotFoundError(
        f"diffusion.first_stage_config.pretrained_path=latest: no completed "
        f"autoencoder run in {outputs_dir} trained on dataset {current['name']} "
        f"(id {current['id']}) at {resize}x{resize}. Train one with "
        f"src/scripts/run_autoencoder_train.py, or set pretrained_path to a checkpoint."
    )


def check_first_stage_dataset(cfg,
                              first_stage_info):
    """
    The frozen VAE must come from the same preprocessed dataset (dataset.json
    id) phase 2 trains on: otherwise its normalization and its splits differ,
    and the VAE may have been trained on cubes phase 2 validates on.
    """
    current = dataset_identity(load_dataset_info(cfg.data.params.dataset_json))
    trained_on = first_stage_info.get("dataset")
    if trained_on is None:
        logger.warning(
            f"{first_stage_info.get('pretrained_path')} does not declare the dataset it "
            f"was trained on (checkpoint older than dataset.json): make sure it is "
            f"{current['name']} (id {current['id']})."
        )
        return
    if trained_on.get("id") != current["id"]:
        raise RuntimeError(
            f"{first_stage_info.get('pretrained_path')} was trained on dataset "
            f"{trained_on.get('name')} (id {trained_on.get('id')}), but phase 2 uses "
            f"{current['name']} (id {current['id']}, {cfg.data.params.dataset_json}). "
            f"Train the autoencoder on this dataset, or point paths/preprocessing.name "
            f"at the dataset the VAE was trained on."
        )
    logger.info(f"First stage trained on the current dataset {current['name']} (id {current['id']})")


def load_autoencoder_run_info(pretrained_path):
    """
    training_info.json of the phase-1 run a best.pt belongs to
    (<run>/checkpoints/best.pt -> <run>/artifacts/training_info.json), or {}
    when the weights do not come from a run folder.
    """
    vae_run = os.path.dirname(os.path.dirname(os.path.abspath(pretrained_path)))
    info_json = os.path.join(vae_run, "artifacts", "training_info.json")
    if not os.path.isfile(info_json):
        return {}
    with open(info_json, encoding="utf-8") as fh:
        return json.load(fh)


def latent_normalization_values(config_diffusion):
    """
    (shift, scale) of the current latent_normalization: two floats, or two
    lists with one value per latent channel ('shift_scale_per_channel').
    The shift is 0 with 'scale'.
    """
    mode = config_diffusion.latent_normalization
    if mode == "shift_scale_per_channel":
        return (list(config_diffusion.channel_shift_factors),
                list(config_diffusion.channel_scale_factors))
    if mode == "shift_scale":
        return config_diffusion.shift_factor, config_diffusion.scale_factor
    return 0.0, config_diffusion.scale_factor


def _broadcastable(value,
                   like):
    """A per-channel list becomes a (C, 1, 1) tensor that broadcasts on (..., C, H, W)."""
    if isinstance(value, list):
        return torch.tensor(value,
                            device=like.device,
                            dtype=like.dtype).view(-1, 1, 1)
    return value


def normalize_latents(z,
                      config_diffusion):
    """VAE latents (..., C, H, W) -> the latents the DiT works on."""
    shift, scale = latent_normalization_values(config_diffusion)
    return (z - _broadcastable(shift, z)) * _broadcastable(scale, z)


def denormalize_latents(z,
                        config_diffusion):
    """Inverse of normalize_latents, before the VAE decoder."""
    shift, scale = latent_normalization_values(config_diffusion)
    return z / _broadcastable(scale, z) + _broadcastable(shift, z)


def latent_normalization_formula(config_diffusion):
    """The encode formula of the current latent_normalization, for logs and artifacts."""
    return {"shift_scale": "z = (z_vae - shift_factor) * scale_factor",
            "shift_scale_per_channel": "z[c] = (z_vae[c] - channel_shift_factors[c]) * channel_scale_factors[c]",
            "scale": "z = z_vae * scale_factor"}[config_diffusion.latent_normalization]


def measured_latent_stats(first_stage_info):
    """
    What phase 1 measured on the latents of this VAE (latent_mean / latent_std,
    per_channel mean / std) and where it was read: best.pt, or for a checkpoint
    written before it stored them, its run's training_info.json.
    """
    if first_stage_info.get("latent_stats"):
        return first_stage_info["latent_stats"], "VAE checkpoint (measured by phase 1)"
    stats = load_autoencoder_run_info(first_stage_info.get("pretrained_path")).get("scale_factor")
    if stats:
        return stats, "VAE run's training_info.json (measured by phase 1)"
    return {}, None


def _format_values(value,
                   fmt):
    if isinstance(value, list):
        return "[" + ", ".join(format(v, fmt) for v in value) + "]"
    return format(value, fmt)


def resolve_latent_normalization(cfg,
                                 first_stage_info):
    """
    Fills the values the current diffusion.latent_normalization needs:
    scale_factor (+ shift_factor with 'shift_scale'), or channel_scale_factors
    and channel_shift_factors with 'shift_scale_per_channel'. null takes what
    phase 1 measured on the VAE (best.pt, or its run's training_info.json for
    a checkpoint written before the shift existed); a value in the config
    wins. The config is updated in place: every encode/decode reads it. Logs
    mode, formula, values with their source and the mean/std the DiT ends up
    seeing; the sources go into first_stage_info, then training_info.json.
    """
    diffusion = cfg.diffusion
    path = first_stage_info.get("pretrained_path")
    mode = diffusion.latent_normalization
    stats, stats_source = measured_latent_stats(first_stage_info)

    if mode == "shift_scale_per_channel":
        per_channel = stats.get("per_channel") or {}
        measured = {"channel_shift_factors": per_channel.get("mean"),
                    "channel_scale_factors": ([1.0 / v for v in per_channel["std"]]
                                              if per_channel.get("std") else None)}
        for key, value in measured.items():
            source_key = key.replace("channel_", "").replace("_factors", "_factor") + "_source"
            if diffusion[key] is not None:
                first_stage_info[source_key] = "config"
                continue
            if value is None:
                raise RuntimeError(
                    f"diffusion.{key} is null, but no per-channel statistics were measured "
                    f"for {path}. Measure them on that VAE run (adds them to its best.pt "
                    f"and training_info.json): python src/scripts/measure_latent_stats.py --run_dir "
                    f"{os.path.dirname(os.path.dirname(os.path.abspath(path)))} -- or set "
                    f"diffusion.{key} by hand, or use latent_normalization=shift_scale"
                )
            diffusion[key] = [float(v) for v in value]
            first_stage_info[source_key] = stats_source
        first_stage_info["channel_factors_measured"] = measured
    else:
        measured_scale = first_stage_info.get("scale_factor")
        if diffusion.scale_factor is None:
            if measured_scale is None:
                raise RuntimeError(
                    f"diffusion.scale_factor is null, but {path} carries no measured scale_factor "
                    f"(older checkpoint, or a run stopped before the end). Set it by hand: "
                    f"--set diffusion.scale_factor=<1/std of the latents>"
                )
            diffusion.scale_factor = float(measured_scale)
            first_stage_info["scale_factor_source"] = "VAE checkpoint (measured by phase 1)"
        else:
            first_stage_info["scale_factor_source"] = "config"

        first_stage_info["shift_factor_source"] = "not used (latent_normalization=scale)"
        if mode == "shift_scale":
            measured_shift = first_stage_info.get("shift_factor")
            shift_source = "VAE checkpoint (measured by phase 1)"
            if measured_shift is None:
                measured_shift, shift_source = stats.get("latent_mean"), stats_source
            if diffusion.shift_factor is None:
                if measured_shift is None:
                    raise RuntimeError(
                        f"diffusion.shift_factor is null, but neither {path} nor its run's "
                        f"training_info.json carries a measured latent mean. Set it by hand "
                        f"(--set diffusion.shift_factor=<mean of the latents>), or use the "
                        f"formula without shift: --set diffusion.latent_normalization=scale"
                    )
                diffusion.shift_factor = float(measured_shift)
                first_stage_info["shift_factor_source"] = shift_source
            else:
                first_stage_info["shift_factor_source"] = "config"
            first_stage_info["shift_factor"] = measured_shift

    shift, scale = latent_normalization_values(diffusion)
    logger.info(f"[latent normalization] {mode}: {latent_normalization_formula(diffusion)}")
    logger.info(
        f"[latent normalization] scale={_format_values(scale, '.6f')} "
        f"({first_stage_info['scale_factor_source']}) | shift={_format_values(shift, '+.6f')} "
        f"({first_stage_info['shift_factor_source']})"
    )
    if stats.get("latent_mean") is not None and not isinstance(scale, list):
        mean, std = stats["latent_mean"], stats["latent_std"]
        logger.info(f"[latent normalization] VAE latents mean={mean:+.4f} std={std:.4f} -> "
                    f"the DiT sees mean={(mean - shift) * scale:+.4f} std={std * scale:.4f}")
    per_channel = stats.get("per_channel") or {}
    if per_channel.get("mean"):
        n = len(per_channel["mean"])
        shifts = shift if isinstance(shift, list) else [shift] * n
        scales = scale if isinstance(scale, list) else [scale] * n
        seen_mean = [(m - sh) * sc for m, sh, sc in zip(per_channel["mean"], shifts, scales)]
        seen_std = [v * sc for v, sc in zip(per_channel["std"], scales)]
        logger.info(f"[latent normalization] per channel, the DiT sees mean="
                    f"{_format_values(seen_mean, '+.3f')} std={_format_values(seen_std, '.3f')}")
    return shift, scale


def check_data_model_alignment(sample, config_diffusion, first_stage_info):
    """
    Controlla che quello che esce dal dataloader sia quello che il DiT si
    aspetta. Sono errori che altrimenti si manifestano come broadcast strani
    dentro l'attention, molto piu' avanti e molto meno leggibili.
    """
    net = config_diffusion.network_config.params
    T = int(net.image_time_length)
    n_predict = int(net.noise_image_num)
    input_size = int(net.input_size)
    vae_down_ratio = int(net.vae_down_ratio)

    vae_in_channels = first_stage_info["in_channels"]
    latent_channels = first_stage_info["latent_channels"]
    down_ratio = first_stage_info["down_ratio"]

    imgs = sample[config_diffusion.input_key]
    static = sample["highres_condition_image"]
    meso = sample["meso_condition_image"]

    problems = []

    if imgs.shape[0] != T:
        problems.append(
            f"imgs ha {imgs.shape[0]} istanti, il DiT ne vuole "
            f"image_time_length={T} ({T - n_predict} di contesto + {n_predict} da predire)"
        )
    if imgs.shape[1] != vae_in_channels:
        problems.append(
            f"imgs ha {imgs.shape[1]} bande, il VAE ne vuole {vae_in_channels}"
        )
    # Il DiT lavora sui latenti, quindi il suo in_channels e' il numero di
    # canali latenti del VAE, non le bande dell'immagine.
    if latent_channels != int(net.in_channels):
        problems.append(
            f"il VAE produce {latent_channels} canali latenti, il DiT vuole "
            f"in_channels={net.in_channels}"
        )

    expected_hw = input_size * vae_down_ratio
    if vae_down_ratio != down_ratio:
        problems.append(
            f"vae_down_ratio={vae_down_ratio} nel config, ma l'autoencoder "
            f"caricato comprime di {down_ratio}x"
        )
    if imgs.shape[-1] != expected_hw or imgs.shape[-2] != expected_hw:
        problems.append(
            f"imgs e' {tuple(imgs.shape[-2:])}, attese {expected_hw}x{expected_hw} "
            f"(input_size {input_size} x vae_down_ratio {vae_down_ratio})"
        )

    if static.shape[0] != int(net.static_channel):
        problems.append(
            f"highres_condition_image ha {static.shape[0]} canali, "
            f"static_channel={net.static_channel}"
        )
    if static.shape[-1] != expected_hw or static.shape[-2] != expected_hw:
        problems.append(
            f"highres_condition_image e' {tuple(static.shape[-2:])}, "
            f"attese {expected_hw}x{expected_hw}"
        )

    if meso.shape[0] != T:
        problems.append(
            f"meso_condition_image ha {meso.shape[0]} istanti, il DiT ne vuole {T} "
            f"(climate_var viene sommato all'embedding di sigma, replicato "
            f"image_time_length volte): usa meso_time_span='all' nel dataloader"
        )
    if meso.shape[1] != int(net.climate_channel):
        problems.append(
            f"meso_condition_image ha {meso.shape[1]} variabili, "
            f"climate_channel={net.climate_channel}"
        )

    if problems:
        raise ValueError(
            "dataloader e network_config non sono allineati:\n  - "
            + "\n  - ".join(problems)
        )

    logger.info(
        f"[DATA] imgs {tuple(imgs.shape)} -> latenti "
        f"({T}, {latent_channels}, {input_size}, {input_size}) | "
        f"static {tuple(static.shape)} | meso {tuple(meso.shape)}"
    )




def _module_dtype(module, default=torch.float32):
    """
    dtype dei parametri di un modulo, per castare ai suoi confini.
    Serve perche' con accelerate.weight_dtype=float32 il VAE resta in bf16
    mentre il resto passa a fp32: senza il cast, conv2d riceverebbe input e
    pesi di dtype diverso e solleverebbe un errore.
    """
    for p in module.parameters():
        return p.dtype
    for b in module.buffers():
        return b.dtype
    return default


def _to_uint8_rgb(frame):
    """
    (C,H,W) in [-1,1] -> (H,W,3) uint8. The bands are blue, green, red, NIR
    (src/dataset/preprocessing.py): RGB is bands 2, 1, 0.
    """
    x = (frame[[2, 1, 0]].float().clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0
    return x.round().to(torch.uint8).cpu().numpy().transpose(1, 2, 0)


def save_prediction_grids(pred, target, cube_paths, save_dir, counter, limit=0):
    """
    Un PNG per cubo di validazione: riga superiore il ground truth, riga
    inferiore la predizione, una colonna per ciascuno dei frame predetti.

    Il nome del file e' lo stem del .npz da cui il cubo viene, per esempio

        29SND_2017-06-10_2017-11-06_2105_2233_2873_3001_32_112_44_124.png

    che si ritrova in data/raw/train/29SND/<stesso nome>.npz -- il primo campo
    del nome e' gia' la tile, quindi la sottocartella si legge dal nome stesso.
    In index.tsv resta comunque la corrispondenza esplicita png -> percorso.

    counter e' un dict condiviso fra i batch ({"n": 0}), limit=0 significa
    nessun tetto.
    """
    os.makedirs(save_dir, exist_ok=True)
    index_path = os.path.join(save_dir, "index.tsv")
    new_index = not os.path.isfile(index_path)

    B, T = pred.shape[0], pred.shape[1]
    written = []
    for b in range(B):
        if limit and counter["n"] >= limit:
            break
        if cube_paths is not None and b < len(cube_paths):
            src = str(cube_paths[b])
            stem = os.path.splitext(os.path.basename(src))[0]
        else:
            # senza cube_path (config vecchi, return_paths spento) resta almeno
            # un nome progressivo, ma il confronto con l'originale va perso
            src = "?"
            stem = f"cube_{counter['n']:05d}"

        rows = [
            np.concatenate([_to_uint8_rgb(seq[t]) for t in range(T)], axis=1)
            for seq in (target[b], pred[b])
        ]
        sep = np.full((4, rows[0].shape[1], 3), 255, dtype=np.uint8)
        Image.fromarray(np.concatenate([rows[0], sep, rows[1]], axis=0)).save(
            os.path.join(save_dir, f"{stem}.png")
        )
        counter["n"] += 1
        written.append((f"{stem}.png", src))

    if written:
        with open(index_path, "a", encoding="utf-8") as fh:
            if new_index:
                fh.write("# png\tcubo_originale\t(riga 1 = ground truth, riga 2 = predizione)\n")
            for name, src in written:
                fh.write(f"{name}\t{src}\n")
    return len(written)


# Prepare log
def log_videos_local_and_wandb(
        batch,
        model,
        conditioner,
        first_stage_model,
        sampler,
        denoiser,
        input_key,
        config_diffusion,
        noise_image_num,
        metrics_rgbn,
        metrics_ndvi,
        metrics_arvi,
        past_weigth,
        val_num_samples=1,
        save_dir=None,
        save_counter=None,
        save_limit=0,
        return_all_samples=False,
):
    """
    Forecast of a batch of validation cubes and the accumulators of the validation
    RMSEs (sums, see VAL_SUM_KEYS). With return_all_samples the K = val_num_samples predictions of every cube
    are decoded to pixels and returned in forecast["pred_samples"] (a list of K
    (B, T, C, H, W) tensors in [-1, 1], the first one is forecast["pred"]): the ENS
    keeps the best of them. Without it only the first one is decoded, the other K-1
    stay in the latent space for the latent RMSEs.
    """
    device = batch[input_key].device
    weight_dtype = batch[input_key].dtype

    def encode_first_stage(first_stage_model,
                           input_tensor):
        # Stessa scelta fatta nel training loop e in evaluate_vae(): si usa la
        # media della posterior, non un campione, cosi' i latenti di riferimento
        # sono deterministici.
        fs_dtype = _module_dtype(first_stage_model)
        with torch.no_grad():
            z = first_stage_model.encode(input_tensor.to(fs_dtype)).latent_dist.mean
            z = normalize_latents(z,
                                  config_diffusion)
        return z.to(input_tensor.dtype)

    def decode_first_stage(first_stage_model,
                           input_tensor):
        fs_dtype = _module_dtype(first_stage_model)
        with torch.no_grad():
            z = denormalize_latents(input_tensor,
                                    config_diffusion)
            out = first_stage_model.decode(z.to(fs_dtype))
            if not torch.is_tensor(out):
                out = out.sample
        return out.to(input_tensor.dtype)

    extra_pixel_samples = []     # the samples 2..K in pixels, only with return_all_samples

    def validate_and_log(
            batch,
            sample=True,
            ucg_keys=None,
            noise_image_num=noise_image_num,
            past_weigth=past_weigth,
    ):
        def sample_func(
                cond,
                uc=None,
                batch_size=16,
                shape=None,
                past_images=None,
                past_mean=None,
                noise_image_num=None,
                past_weigth=past_weigth,
        ):
            # 这里控制randn的时间只有20，past_images是前面的10帧图像，控制形状为B,T,C,H,W。同时注意在后面把randn转成B,T,C,H,W的形状
            randn = torch.randn(batch_size, *shape).to(device, dtype=weight_dtype)

            # 把噪声和过去均值图像加权混合
            past_mean = past_mean.repeat(1, noise_image_num, 1, 1, 1)  # b,t,c,h,w
            past_mean = rearrange(past_mean, "b t c h w -> b (t c) h w").contiguous()  # b,t*c,h,w
            randn = past_weigth * past_mean + (1 - past_weigth) * randn  # b,t*c,h,w

            denoiser_func = lambda input, sigma, c, past_images: denoiser(
                model, past_images, input, sigma, c
            )
            samples = sampler(denoiser_func, randn, cond, uc=uc, past_images=past_images,
                              noise_image_num=noise_image_num,
                              past_mean=past_mean, past_weigth=past_weigth)

            return samples

        # conditioner_input_keys = [e.input_key for e in conditioner.module.embedders]
        conditioner_input_keys = [e.input_key for e in conditioner.embedders]

        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys

        # get input
        x = batch[input_key]

        c, uc = conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(conditioner.embedders) > 0
            else [],
        )

        # log reconstructions
        log = dict()
        N = x.shape[0]  # batchsize
        x = x.to(device, dtype=weight_dtype)[:N]

        log["inputs_sample"] = x[0, -1, ...]  # 取最后一张图像记录

        x_ = x
        B, T, C, H, W = x_.shape
        assert B == N
        x_ = x_.view(B * T, C, H, W)
        z = encode_first_stage(first_stage_model,
                               x_).to(device, dtype=weight_dtype)
        _, C_, H_, W_ = z.shape
        log["reconstructions"] = torch.clamp(
            decode_first_stage(first_stage_model,
                               z).view(B, T, C, H, W),
            -1.0, 1.0)[:, -noise_image_num:, ...]  # 只取出后面预测的重建部分
        log["reconstructions_sample"] = log["reconstructions"][0, -1, ...]  # 取最后一张图像记录
        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k].to(device, dtype=weight_dtype), (c, uc))
            elif isinstance(c[k], list):
                for i in range(len(c[k])):
                    c[k][i] = c[k][i].to(device, dtype=weight_dtype)
                    uc[k][i] = uc[k][i].to(device, dtype=weight_dtype)

        # sample
        past_images = z.view(B, -1, C_, H_, W_)[:, :-noise_image_num, ...]
        future_images = z.view(B, -1, C_, H_, W_)[:, -noise_image_num:, ...]  # b,t,c,h,w
        past_mean = torch.mean(past_images, dim=1, keepdim=True)  # b,1,c,h,w
        if sample:
            # sample 形状为b,t*c,h,w
            samples = sample_func(c, shape=torch.Size([noise_image_num * C_, H_, W_]), uc=uc, batch_size=N,
                                  past_images=past_images, past_mean=past_mean, noise_image_num=noise_image_num)
            # TODO Check denoiser dtype

            samples = rearrange(samples, "b (t c) h w -> (b t) c h w", t=noise_image_num).contiguous().to(device,
                                                                                                          dtype=weight_dtype)  # b,t2,c,h,w
            sample_latent = rearrange(samples, "(b t) c h w -> b t c h w", t=noise_image_num).contiguous()
            samples = decode_first_stage(first_stage_model,
                                         samples).view(B, noise_image_num, C, H, W)
            samples = torch.clamp(samples, -1.0, 1.0)
            log["last_samples"] = samples[0, -1, ...]
            log["first_samples"] = samples[0, 0, ...]

        # Somme degli errori quadratici, non le RMSE: per avere la RMSE sull'intero
        # validation set il chiamante accumula queste somme e prende la radice alla
        # fine. Mediare le RMSE dei singoli batch darebbe un altro numero (la media
        # delle radici e' sempre <= della radice della media) e peserebbe l'ultimo
        # batch, piu' corto, come tutti gli altri. Accumulo in float64: su ~2e9
        # addendi il float32 perde cifre significative.
        # .float() prima della sottrazione: in bf16 il residuo ha ~3 cifre
        # significative, e siccome da questi numeri dipende quale checkpoint
        # viene tenuto come migliore, non conviene lasciare rumore evitabile.
        fut = future_images.float()
        past_mean_se = torch.sum(
            (past_mean.repeat(1, noise_image_num, 1, 1, 1).float() - fut) ** 2,
            dtype=torch.float64,
        )
        """ codice vecchio commentato
        predict_se = torch.sum((sample_latent.float() - fut) ** 2, dtype=torch.float64)
        n_latent = future_images.numel()
        """
        # Valutazione a K campioni (K = val_num_samples). Il primo campione e'
        # quello gia' generato sopra (e resta l'unico decodificato, per le
        # metriche pixel); gli altri K-1 vengono campionati solo in latente.
        # - predict_se: MEDIA delle SE dei singoli campioni (stessa quantita'
        #   di prima, stimata con meno rumore; con K=1 identica al vecchio
        #   codice).
        # - mean_k_se: SE della media dei K campioni, la stima della media
        #   condizionata da confrontare con la baseline past_mean.
        # - best_k_se: SE del campione migliore per cubo (best-of-K).
        first_latent = sample_latent.float()
        per_sample_se = [torch.sum((first_latent - fut) ** 2, dtype=torch.float64)]
        per_cube_se = [torch.sum((first_latent - fut) ** 2, dim=(1, 2, 3, 4),
                                 dtype=torch.float64)]
        latent_sum = first_latent.clone()
        for _ in range(val_num_samples - 1):
            extra = sample_func(
                c, shape=torch.Size([noise_image_num * C_, H_, W_]), uc=uc,
                batch_size=N, past_images=past_images, past_mean=past_mean,
                noise_image_num=noise_image_num,
            )
            extra = rearrange(extra, "b (t c) h w -> b t c h w",
                              t=noise_image_num).contiguous().float()
            per_sample_se.append(torch.sum((extra - fut) ** 2, dtype=torch.float64))
            per_cube_se.append(torch.sum((extra - fut) ** 2, dim=(1, 2, 3, 4),
                                         dtype=torch.float64))
            latent_sum += extra
            if return_all_samples:
                extra_pixel_samples.append(torch.clamp(
                    decode_first_stage(first_stage_model,
                                       rearrange(extra.to(device, dtype=weight_dtype),
                                                 "b t c h w -> (b t) c h w").contiguous()
                                       ).view(B, noise_image_num, C, H, W),
                    -1.0, 1.0))
        predict_se = torch.stack(per_sample_se).mean()
        mean_latent = latent_sum / float(val_num_samples)
        mean_k_se = torch.sum((mean_latent - fut) ** 2, dtype=torch.float64)
        best_k_se = torch.sum(torch.stack(per_cube_se).min(dim=0).values)
        n_latent = future_images.numel()
        return log, samples, (past_mean_se, predict_se, n_latent, mean_k_se, best_k_se)

    model.eval()
    conditioner.eval()
    with torch.no_grad():
        # validate_log是个字典里面有inputs,reconstructions和samples这三个键，分别对应全时序输入、全时序重建、生成的未来序列（时间数量为noise_image_num）
        """ codice vecchio commentato
        validate_log, samples, (past_mean_se, predict_se, n_latent) = validate_and_log(batch)
        """
        validate_log, samples, (past_mean_se, predict_se, n_latent,
                                mean_k_se, best_k_se) = validate_and_log(batch)
        pred = samples  # b,t,c,h,w
        target = batch['imgs'][:, -noise_image_num:, ...]
        # mask = batch['mask'][:,-noise_image_num:,...]
        b, t, c, h, w = pred.shape

        # Immagini di confronto: solo quando il chiamante passa save_dir, cioe'
        # all'ultima epoca. Si salva qui perche' e' l'unico punto in cui pred e
        # target sono gia' decodificati in pixel e ancora appaiati al batch.
        if save_dir is not None:
            save_prediction_grids(
                pred, target, batch.get("cube_path"), save_dir,
                save_counter if save_counter is not None else {"n": 0},
                save_limit,
            )

        # Come sopra: somme, la radice la prende il chiamante sull'intero set.
        # Gli indici :10 / 10: dividono i 20 frame predetti nella prima e nella
        # seconda meta'; i conteggi si leggono da numel(), cosi' restano corretti
        # anche se noise_image_num non fosse 20.
        fromer_se = torch.sum(
            (pred[:, :10, ...].float() - target[:, :10, ...].float()) ** 2, dtype=torch.float64)
        latter_se = torch.sum(
            (pred[:, 10:, ...].float() - target[:, 10:, ...].float()) ** 2, dtype=torch.float64)
        n_fromer = pred[:, :10, ...].numel()
        n_latter = pred[:, 10:, ...].numel()

        # Per lead time, again as sums: the model and the context-mean baseline
        # (the mean of the context frames, in pixels), one value per predicted frame.
        context = batch['imgs'][:, :-noise_image_num, ...]
        context_mean = context.float().mean(dim=1, keepdim=True)
        forecast = {
            "lead_se": torch.sum((pred.float() - target.float()) ** 2,
                                 dim=(0, 2, 3, 4),
                                 dtype=torch.float64),
            "lead_baseline_se": torch.sum((context_mean - target.float()) ** 2,
                                          dim=(0, 2, 3, 4),
                                          dtype=torch.float64),
            "lead_count": float(b * c * h * w),
            # pixels of the batch, for the forecast figures
            "context": context,
            "target": target,
            "pred": pred,
            # the K predictions (see return_all_samples); only the first one without it
            "pred_samples": [pred] + extra_pixel_samples,
        }

        # Ordine fisso, e' la chiave con cui il chiamante somma: vedi VAL_SUM_KEYS.
        """ codice vecchio commentato
        batch_sums = torch.tensor(
            [float(past_mean_se), float(predict_se), float(n_latent),
             float(fromer_se), float(n_fromer), float(latter_se), float(n_latter),
             float(b)],
            device=pred.device, dtype=torch.float64,
        )
        """
        batch_sums = torch.tensor(
            [float(past_mean_se), float(predict_se), float(n_latent),
             float(fromer_se), float(n_fromer), float(latter_se), float(n_latter),
             float(b), float(mean_k_se), float(best_k_se)],
            device=pred.device, dtype=torch.float64,
        )

        # mask_ndvi = mask.clone().squeeze(2)  # b,t,1,h,w
        # mask_origin = mask.repeat(1,1,c,1,1)  # b,t,c,h,w
        # mask_origin = mask.view(b_mo,self.noise_image_num*c_if,h_if,w_if).contiguous()  # b,t*c,h,w
        pred = torch.clamp(pred, -1.0, 1.0)

        def denormalize(array):
            return (array + 1.0) / 2

        pred = denormalize(pred)
        target = denormalize(target)


    model.train()
    conditioner.train()
    # return btach_metric_rgbn, btach_metric_ndvi, batch_upper_limit_rgbn, batch_upper_limit_ndvi, batch_mean_rgbn, batch_mean_ndvi, batch_decode_mean_rgbn, batch_decode_mean_ndvi, validate_log
    return validate_log, batch_sums, forecast


def _fmt_ens(value):
    """An ENS statistic for the log: 6 decimals, or n/a when the cubes gave none."""
    return "n/a" if value is None else f"{value:.6f}"


def count_params(m: torch.nn.Module):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def fmt(n: int) -> str:
    # 123456789 -> "123.46M"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)


def instantiate_optimizer_from_config(params,
                                      lr,
                                      cfg):
    return get_obj_from_str(cfg["target"])(
        params, lr=lr, **cfg.get("params", dict()))

def instantiate_models(cfg,
                       rt):
    config_diffusion = cfg.diffusion
    accelerator = rt.accelerator
    weight_dtype = rt.weight_dtype
    first_stage_dtype = rt.first_stage_dtype


    model = instantiate_from_config(config_diffusion.network_config)
    model = get_obj_from_str(config_diffusion.network_wrapper)(model,
                                                               compile_model=config_diffusion.compile_model)

    # Log model params
    if cfg.run.show_debug_log:
        for name, param in model.named_parameters():
            param.requires_grad = True
            logger.info(f"model trainable: {name}")

    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        conditioner = instantiate_from_config(config_diffusion.conditioner_config)

        # Autoencoder gia' addestrato: architettura e percorso dei pesi stanno
        # nel config (diffusion.first_stage_config).
        resolve_first_stage_path(cfg)
        first_stage_model, first_stage_info = build_first_stage_model(config_diffusion.first_stage_config)
    check_first_stage_dataset(cfg,
                              first_stage_info)
    resolve_latent_normalization(cfg,
                                 first_stage_info)

    # Put model to gpu and cast to correct dtype
    conditioner = conditioner.to(accelerator.device, dtype=weight_dtype)

    # Il VAE e' congelato: nessun optimizer lo aggiorna, quindi l'argomento dei
    # master weights in fp32 non lo riguarda. Resta al dtype di mixed_precision
    # anche con weight_dtype=float32, cosi' l'encode dei 240 frame per step non
    # rallenta. Il cast ai suoi confini lo fanno encode/decode_first_stage.
    first_stage_model = first_stage_model.to(accelerator.device, dtype=first_stage_dtype)
    for param in first_stage_model.parameters():
        param.requires_grad = False
    for name, param in conditioner.named_parameters():
        param.requires_grad = True
        logger.info(f"conditioner trainable: {name}")

    return model, conditioner, first_stage_model, first_stage_info

def deepspeed_zero_init_disabled_context_manager():
    """
    returns either a context list that includes one that will disable zero.Init or an empty context list
    """
    deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
    if deepspeed_plugin is None:
        return []

    return [deepspeed_plugin.zero3_init_context_manager(enable=False)]


def setup_dataloader(cfg,
                     rt,
                     first_stage_info,
                     model,
                     conditioner,
                     first_stage_model):

    config_data = cfg.data
    accelerator = rt.accelerator
    config_diffusion = cfg.diffusion
    config_accelerate = cfg.accelerate

    datamodule = instantiate_from_config(config_data)
    with accelerator.main_process_first():
        datamodule.setup("fit")
    train_dataloader = datamodule.train_dataloader()
    # ens.validation: the validation cubes also carry the cloud mask of their target
    # frames (read from the original .npz), to leave the cloudy pixels out of the ENS.
    # The model never sees it: train_loop takes it out of the batch.
    if (cfg.ens.enabled and cfg.ens.validation and cfg.ens.cloud_mask
            and datamodule.val_dataset is not None):
        datamodule.val_dataset = wrap_with_cloud_mask(datamodule.val_dataset, cfg.ens)
    val_dataloader = datamodule.val_dataloader()
    if val_dataloader is None:
        raise RuntimeError(
            "il datamodule non ha uno split di validation: alza "
            "preprocessing.val_frac e rifai il preprocessing."
        )
    train_len = len(datamodule.train_dataset)

    # Verifica una volta sola che shape delle variabili e iperparametri del DiT
    # combacino, prima di spendere tempo in setup di optimizer e accelerate.
    check_data_model_alignment(datamodule.train_dataset[0],
                               config_diffusion,
                               first_stage_info)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config_accelerate.gradient_accumulation_steps)
    if config_accelerate.max_train_steps is None:  # not assign max steps
        config_accelerate.max_train_steps = config_accelerate.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    tot_m, tr_m, fr_m = count_params(model)
    tot_c, tr_c, fr_c = count_params(conditioner)
    tot_v, tr_v, fr_v = count_params(first_stage_model)

    tot_all = tot_m + tot_c + tot_v
    tr_all = tr_m + tr_c + tr_v
    fr_all = tot_all - tr_all

    logger.info(f"[PARAMS] model:      total={fmt(tot_m)} trainable={fmt(tr_m)} frozen={fmt(fr_m)}")
    logger.info(f"[PARAMS] conditioner: total={fmt(tot_c)} trainable={fmt(tr_c)} frozen={fmt(fr_c)}")
    logger.info(f"[PARAMS] VAE:        total={fmt(tot_v)} trainable={fmt(tr_v)} frozen={fmt(fr_v)}")
    logger.info(f"[PARAMS] ALL:        total={fmt(tot_all)} trainable={fmt(tr_all)} frozen={fmt(fr_all)}")
    return train_dataloader, val_dataloader, train_len, overrode_max_train_steps

def prepare_training(cfg,
                     rt,
                     outputs,
                     model,
                     conditioner,
                     train_dataloader,
                     val_dataloader,
                     overrode_max_train_steps):


    config_diffusion = cfg.diffusion
    accelerator = rt.accelerator
    weight_dtype = rt.weight_dtype
    config_accelerate = cfg.accelerate
    learning_rate = rt.learning_rate


    # Setup optimizer and lr_scheduler
    loss_fn = instantiate_from_config(config_diffusion.loss_fn_config)
    denoiser = instantiate_from_config(config_diffusion.denoiser_config)
    sampler = instantiate_from_config(config_diffusion.sampler_config)

    loss_fn = loss_fn.to(accelerator.device, dtype=weight_dtype)
    params = list(model.parameters())
    for embedder in conditioner.embedders:
        params = params + list(embedder.parameters())
    params = params + list(loss_fn.parameters())
    optimizer_config = default(
        config_accelerate.optimizer, {"target": "torch.optim.AdamW"}
    )
    optimizer = instantiate_optimizer_from_config(params, learning_rate, optimizer_config)
    lr_scheduler = get_scheduler(
        config_accelerate.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config_accelerate.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config_accelerate.max_train_steps * accelerator.num_processes,
    )

    # Prepare Accelerate
    model, conditioner, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(model,
                                                                                                        conditioner,
                                                                                                        optimizer,
                                                                                                        train_dataloader,
                                                                                                        val_dataloader,
                                                                                                        lr_scheduler)
    model = model.to(dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config_accelerate.gradient_accumulation_steps)
    if overrode_max_train_steps:  #
        config_accelerate.max_train_steps = config_accelerate.num_train_epochs * num_update_steps_per_epoch

    # Afterwards we recalculate our number of training epochs
    config_accelerate.num_train_epochs = math.ceil(config_accelerate.max_train_steps / num_update_steps_per_epoch)

    # model, conditioner and the dataloaders are returned too: what
    # accelerator.prepare() gives back (device placement of the batches,
    # mixed-precision autocast, DDP) must replace the caller's originals.
    return (num_update_steps_per_epoch, denoiser, optimizer, lr_scheduler, loss_fn, sampler,
            model, conditioner, train_dataloader, val_dataloader)


def encode_first_stage(input_tensor,
                       first_stage_model,
                       config_diffusion):
    fs_dtype = _module_dtype(first_stage_model)
    with torch.no_grad():
        z = first_stage_model.encode(input_tensor.to(fs_dtype))
        z = z.latent_dist.mean
        z = normalize_latents(z,
                              config_diffusion)
    return z.to(input_tensor.dtype)

def decode_first_stage(input_tensor,
                       first_stage_model,
                       config_diffusion):
    # 我在最后的输出部分计算损失，需要这里记录一些值
    # with torch.no_grad():
    z = denormalize_latents(input_tensor,
                            config_diffusion)
    out = first_stage_model.decode(z.to(_module_dtype(first_stage_model)))
    if isinstance(out, dict):
        out = out.sample
    return out.to(input_tensor.dtype)



# ============================ CHECKPOINT DA CUI RIPARTIRE ============================

def _checkpoint_global_step(path):
    """
    global_step of a saved accelerate state: from the folder name for
    checkpoint-<step> / save-checkpoint-<step>, from best_metric.json for best/.
    """
    name = os.path.basename(os.path.normpath(path))
    if name.startswith(("checkpoint-", "save-checkpoint-")):
        return int(name.split("-")[-1])
    try:
        with open(os.path.join(path, "best_metric.json"), encoding="utf-8") as fh:
            return int(json.load(fh)["global_step"])
    except (OSError, ValueError, KeyError):
        return 0


def _step_checkpoints(ckptdir):
    """The checkpoint-<step> folders of a run, oldest first."""
    if not os.path.isdir(ckptdir):
        return []
    names = [d for d in os.listdir(ckptdir) if d.startswith("checkpoint-")]
    return sorted(names, key=lambda name: int(name.split("-")[-1]))


def resolve_resume_checkpoint(cfg,
                              outputs):
    """
    run.resume_from_checkpoint as the folder of a saved accelerate state, or
    None for a new training.

    'latest' = the newest checkpoint-<step> of the newest diffusion run (under
    diffusion.outputs_dir) with the same project_name, the current run
    excluded. Any other value is the path of a checkpoint folder.
    """
    resume = cfg.run.resume_from_checkpoint
    if resume is None:
        if cfg.run.validate_only:
            raise SystemExit("--validate_only needs --resume_from_checkpoint: the "
                             "checkpoint folder whose weights are validated "
                             "(e.g. registry/outputs/diffusion/<run>/checkpoints/best)")
        return None
    if resume != "latest":
        if not os.path.isdir(resume):
            raise FileNotFoundError(f"run.resume_from_checkpoint={resume}: folder not found")
        return resume

    base = cfg.diffusion.outputs_dir
    runs = sorted(os.listdir(base), reverse=True) if os.path.isdir(base) else []
    for run in runs:
        root = os.path.join(base, run)
        info_json = os.path.join(root, "artifacts", "training_info.json")
        if os.path.abspath(root) == os.path.abspath(outputs.root) or not os.path.isfile(info_json):
            continue
        with open(info_json, encoding="utf-8") as fh:
            info = json.load(fh)
        if info.get("run", {}).get("project_name") != cfg.run.project_name:
            continue
        names = _step_checkpoints(os.path.join(root, "checkpoints"))
        if names:
            path = os.path.join(root, "checkpoints", names[-1])
            logger.info(f"run.resume_from_checkpoint=latest -> {path}")
            return path
    raise FileNotFoundError(
        f"run.resume_from_checkpoint=latest: no diffusion run of project "
        f"{cfg.run.project_name!r} with a checkpoint-<step> folder in {base}. "
        f"Pass the path of a checkpoint folder, or null for a new training."
    )


def _inherit_history(state,
                     resume_from):
    """
    When resuming from a checkpoint of an earlier run, copy that run's
    history.csv and lead_time.csv rows up to the resumed epoch, so the curves
    of the new run start from the beginning of the training, and take over its
    best latter_rmse: without it, the first validation would save as "best" a
    model worse than the one the earlier run already has.
    """
    previous = os.path.dirname(os.path.dirname(os.path.abspath(resume_from)))
    for name, target in (("history.csv", state.history_csv),
                         ("lead_time.csv", state.lead_time_csv)):
        source = os.path.join(previous, "artifacts", name)
        if not os.path.isfile(source):
            logger.info(f"no {name} in {previous}: the curves start from the resumed epoch")
            continue
        with open(source, newline="") as fh:
            rows = [row for row in csv.DictReader(fh) if int(row["epoch"]) <= state.first_epoch]
        for row in rows:
            append_history_row(target,
                               row)
        logger.info(f"{len(rows)} rows of {name} inherited from {source}")

    best_metric_file = os.path.join(previous, "checkpoints", "best", "best_metric.json")
    if not os.path.isfile(best_metric_file):
        return
    with open(best_metric_file, encoding="utf-8") as fh:
        best = json.load(fh)
    if int(best.get("global_step", 0)) > state.global_step:
        # saved after the resumed checkpoint: not part of this run's history
        return
    state.best_latter_rmse = float(best["latter_rmse"])
    state.best_epoch = best.get("epoch")
    state.best_global_step = best.get("global_step")
    state.best_val_metrics = {k: v for k, v in best.items()
                              if k not in ("epoch", "global_step", "n_cubes", "k")}
    state.checkpoints["best"] = {"folder": relative_to_project(os.path.dirname(best_metric_file)),
                                 "inherited": True,
                                 "epoch": state.best_epoch,
                                 "global_step": state.best_global_step,
                                 "latter_rmse": state.best_latter_rmse}
    logger.info(f"latter_rmse migliore ripreso da {best_metric_file}: "
                f"{state.best_latter_rmse:.6f} (epoca {state.best_epoch})")


def _load_weights_only(accelerator,
                       model,
                       conditioner,
                       ckpt_dir):
    """
    --validate_only: SOLO i pesi (model.safetensors / model_1.safetensors),
    senza accelerator.load_state: optimizer, scheduler e stato dei dataloader
    sono irrilevanti per una validazione pura, e best/ (salvato senza i file
    sampler*.bin dei dataloader) fallirebbe il load completo.
    """
    model_sd = load_safetensors(os.path.join(ckpt_dir, "model.safetensors"))
    missing, unexpected = accelerator.unwrap_model(model).load_state_dict(model_sd,
                                                                          strict=False)
    if missing:
        # Chiavi non coperte dal file = parametri che resterebbero
        # inizializzati a caso: la validazione non avrebbe senso.
        raise RuntimeError(
            f"[validate_only] {len(missing)} chiavi del modello mancano in "
            f"{os.path.join(ckpt_dir, 'model.safetensors')}: {missing[:5]} ..."
        )
    if unexpected:
        logger.warning(f"[validate_only] chiavi inattese ignorate nel checkpoint: "
                       f"{unexpected[:5]} ...")

    cond_path = os.path.join(ckpt_dir, "model_1.safetensors")
    if os.path.isfile(cond_path):
        cond_missing, cond_unexpected = accelerator.unwrap_model(conditioner).load_state_dict(
            load_safetensors(cond_path),
            strict=False)
        if cond_missing or cond_unexpected:
            logger.warning(f"[validate_only] conditioner: chiavi mancanti "
                           f"{cond_missing}, inattese {cond_unexpected}")


def configure_train(cfg,
                    rt,
                    outputs,
                    model,
                    conditioner,
                    num_update_steps_per_epoch,
                    denoiser):
    """
    Prepara il training e restituisce il TrainState: riprende il checkpoint (o
    i soli pesi, con --validate_only), stabilisce da quale epoca e step
    ripartire, eredita history e minimo di latter_rmse del run da cui si
    riprende. The run itself always writes into its own
    new folder, also when resuming.
    """
    config_data = cfg.data
    accelerator = rt.accelerator
    config_accelerate = cfg.accelerate
    args = cfg.run

    state = TrainState(history_csv=outputs.history_csv,
                       lead_time_csv=os.path.join(outputs.artifacts, "lead_time.csv"),
                       best_ckpt_path=os.path.join(outputs.checkpoints, "best"),
                       mode="validate_only" if args.validate_only else "train",
                       started_at=datetime.datetime.now().isoformat(timespec="seconds"))

    total_batch_size = config_data.params.batch_size * accelerator.num_processes * config_accelerate.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config_accelerate.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config_data.params.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Learning rate = {rt.learning_rate}")
    logger.info(f"  Gradient Accumulation steps = {config_accelerate.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config_accelerate.max_train_steps}")

    resume_from = resolve_resume_checkpoint(cfg,
                                            outputs)
    if resume_from is not None:
        # the path actually used ends up in config.json and training_info.json
        args.resume_from_checkpoint = resume_from
        state.resumed_from = relative_to_project(resume_from)
        state.global_step = _checkpoint_global_step(resume_from)

    if args.validate_only:
        _load_weights_only(accelerator,
                           model,
                           conditioner,
                           resume_from)
        # the validation is labelled with the epoch the weights come from
        state.first_epoch = max(math.ceil(state.global_step / num_update_steps_per_epoch) - 1, 0)
        logger.info(f"[validate_only] pesi caricati da {resume_from} "
                    f"(global_step {state.global_step}, epoca {state.first_epoch + 1})")
    elif resume_from is not None:
        # global_step, config_accelerate.max_train_steps: no *acc
        # resume_global_step, resume_step: has *acc
        logger.info(f"Resuming from checkpoint {resume_from}")
        accelerator.load_state(resume_from)
        resume_global_step = state.global_step * config_accelerate.gradient_accumulation_steps
        state.first_epoch = state.global_step // num_update_steps_per_epoch
        state.resume_step = resume_global_step % (
                num_update_steps_per_epoch * config_accelerate.gradient_accumulation_steps)
        state.epochs_completed = state.first_epoch
        if accelerator.is_main_process:
            _inherit_history(state,
                             resume_from)

    # Config of the run, saved after everything resolved at start-up (VAE path,
    # scale_factor, checkpoint to resume from), so it says what actually ran.
    if accelerator.is_main_process:
        save_run_config(cfg,
                        outputs)
    logger.info(f"outputs del run in {outputs.root}")

    # Prepare diffusion
    denoiser.to(accelerator.device, dtype=rt.weight_dtype)
    return state


# ============================ DESCRIZIONE DEL RUN ============================

def describe_run(cfg,
                 rt,
                 outputs,
                 state,
                 model,
                 conditioner,
                 first_stage_model,
                 first_stage_info,
                 train_dataloader,
                 val_dataloader):
    """
    The parts of training_info.json fixed for the whole run: environment,
    dataset, frozen first stage, model. Stored in state.static and written
    right away, so the json exists from the first second of the run.
    """
    accelerator = rt.accelerator
    params = cfg.data.params
    net = cfg.diffusion.network_config.params
    dataset = load_dataset_info(params.dataset_json)

    pretrained_path = first_stage_info["pretrained_path"]
    vae_info = load_autoencoder_run_info(pretrained_path)
    measured, measured_source = measured_latent_stats(first_stage_info)
    latent_shift, latent_scale = latent_normalization_values(cfg.diffusion)

    state.static = {
        "environment": {**environment_info(),
                        "num_processes": accelerator.num_processes,
                        "mixed_precision": cfg.accelerate.mixed_precision,
                        "weight_dtype": str(rt.weight_dtype),
                        "first_stage_dtype": str(rt.first_stage_dtype)},
        "dataset": {
            "identity": dataset_identity(dataset),
            "dataset_json": params.dataset_json,
            "selection": dataset.get("selection"),
            "splits": dataset.get("composition"),
            "raw_dir": params.data_root,
            "stats_json": params.stats_json,
            "cond_stats_json": params.cond_stats_json,
            "cubes": {"train": len(train_dataloader.dataset),
                      "val": len(val_dataloader.dataset)},
            "frames": {"total": params.T,
                       "context": params.T - params.noise_image_num,
                       "predicted": params.noise_image_num},
            "n_channels": params.n_channels,
            "resolution": [params.resize, params.resize],
            "batches_per_epoch_per_process": {"train": len(train_dataloader),
                                              "val": len(val_dataloader)},
        },
        "first_stage": {
            "type": cfg.diffusion.first_stage_config.type,
            "pretrained_path": relative_to_project(pretrained_path),
            # the phase-1 run the weights come from, when they are a run's best.pt
            "autoencoder_run": vae_info.get("run", {}).get("run_id"),
            "autoencoder_val_best": vae_info.get("metrics", {}).get("val_best"),
            "autoencoder_test": vae_info.get("metrics", {}).get("test"),
            "trained_on_dataset": first_stage_info.get("dataset"),
            "latent_normalization": {
                "mode": cfg.diffusion.latent_normalization,
                "formula": latent_normalization_formula(cfg.diffusion),
                # floats, or one value per latent channel with shift_scale_per_channel
                "scale": latent_scale,
                "scale_source": first_stage_info.get("scale_factor_source"),
                "shift": latent_shift,
                "shift_source": first_stage_info.get("shift_factor_source"),
                "measured_by_phase_1": {
                    "read_from": measured_source,
                    "scale_factor": first_stage_info.get("scale_factor"),
                    "shift_factor": first_stage_info.get("shift_factor"),
                    "latent_mean": measured.get("latent_mean"),
                    "latent_std": measured.get("latent_std"),
                    "per_channel": measured.get("per_channel"),
                },
            },
            "latent_channels": first_stage_info["latent_channels"],
            "down_ratio": first_stage_info["down_ratio"],
            "parameters": count_parameters(first_stage_model),
        },
        "model": {
            "network": cfg.diffusion.network_config.target,
            "architecture": {"depth": net.depth,
                             "hidden_size": net.hidden_size,
                             "patch_size": net.patch_size,
                             "num_heads": net.num_heads,
                             "input_size": net.input_size,
                             "in_channels": net.in_channels,
                             "image_time_length": net.image_time_length,
                             "noise_image_num": net.noise_image_num},
            "parameters": {"network": count_parameters(model),
                           "conditioner": count_parameters(conditioner)},
            "latent_shape": [params.T, net.in_channels, net.input_size, net.input_size],
            "denoiser_weighting": cfg.diffusion.denoiser_config.params.weighting_config.target,
            "sampler": cfg.diffusion.sampler_config.target,
        },
    }
    logger.info(f"dataset {dataset['name']} (id {dataset['id']}) | first stage "
                f"{relative_to_project(pretrained_path)} | latent normalization "
                f"{cfg.diffusion.latent_normalization} (scale {_format_values(latent_scale, '.6f')}, "
                f"shift {_format_values(latent_shift, '+.6f')})")


def training_info(cfg,
                  rt,
                  outputs,
                  state,
                  lr_scheduler):
    """The full content of training_info.json at this point of the run."""
    acc = cfg.accelerate
    metrics = state.metrics
    gpu_memory = metrics.gpu_memory_gb()
    train_seconds = float(sum(state.epoch_seconds))
    wall_seconds = metrics.wall_time if metrics.start_time else 0.0
    optimizer_params = OmegaConf.to_container(acc.optimizer.params, resolve=True)

    return {
        "run": {
            "phase": "diffusion",
            "mode": state.mode,
            "project_name": cfg.run.project_name,
            "run_id": outputs.timestamp,
            "status": state.status,
            "error": state.error,
            "started_at": state.started_at,
            "ended_at": state.ended_at,
            "last_update": datetime.datetime.now().isoformat(timespec="seconds"),
            "seed": cfg.run.seed,
            **state.static.get("environment", {}),
        },
        "dataset": state.static.get("dataset", {}),
        "first_stage": state.static.get("first_stage", {}),
        "model": state.static.get("model", {}),
        "training": {
            "epochs_planned": acc.num_train_epochs,
            "max_train_steps": acc.max_train_steps,
            "epochs_completed": state.epochs_completed,
            "start_epoch": state.first_epoch,
            "resumed_from": state.resumed_from,
            "global_step": state.global_step,
            "batch_size_per_process": cfg.data.params.batch_size,
            "gradient_accumulation_steps": acc.gradient_accumulation_steps,
            "total_batch_size": (cfg.data.params.batch_size * rt.accelerator.num_processes
                                 * acc.gradient_accumulation_steps),
            "optimizer": {"name": acc.optimizer.target,
                          "lr": acc.learning_rate,
                          "lr_effective": rt.learning_rate,
                          **optimizer_params},
            "lr_scheduler": {"name": acc.lr_scheduler,
                             "warmup_steps": acc.lr_warmup_steps},
            "lr_final": lr_scheduler.get_last_lr()[0],
            "max_grad_norm": acc.max_grad_norm,
            "validation": {"every_epochs": acc.validate_epoch_interval,
                           "samples_per_cube": cfg.run.val_num_samples,
                           "max_batches": acc.validate_batch_num},
        },
        "metrics": {
            "selection_criterion": "val_latter_rmse",
            "train_last": state.train_metrics,
            "val_last": {"epoch": state.val_epoch,
                         "n_cubes": metrics.n_cubes if state.val_metrics else None,
                         **state.val_metrics},
            "val_best": {"epoch": state.best_epoch,
                         "global_step": state.best_global_step,
                         **state.best_val_metrics},
            "lead_time_rmse": {"last": state.val_lead_time or None,
                               "best": state.best_val_lead_time or None},
            # ENS statistics (config section `ens`): last validation, last training sample
            "ens": ({"val_last": state.ens_val or None,
                     "train_last": state.ens_train or None,
                     "samples_per_cube": cfg.run.val_num_samples}
                    if cfg.ens.enabled else None),
        },
        "compute": {
            "wall_time_s": wall_seconds,
            "train_time_s": train_seconds,
            "validation_time_s": metrics.inference_time,
            "mean_epoch_train_time_s": (train_seconds / len(state.epoch_seconds)
                                        if state.epoch_seconds else None),
            "mean_train_cubes_per_s": (float(np.mean(state.cubes_per_second))
                                       if state.cubes_per_second else None),
            "gpu_hours": wall_seconds * rt.accelerator.num_processes / 3600,
            "peak_gpu_memory_allocated_gb": gpu_memory[0] if gpu_memory else None,
            "peak_gpu_memory_reserved_gb": gpu_memory[1] if gpu_memory else None,
        },
        "checkpoints": {
            "dir": relative_to_project(outputs.checkpoints),
            "best": state.checkpoints.get("best"),
            "saved": sorted((info for name, info in state.checkpoints.items() if name != "best"),
                            key=lambda info: info["global_step"]),
        },
        "outputs": {
            "root": relative_to_project(outputs.root),
            "config": relative_to_project(outputs.config_json),
            "history": relative_to_project(outputs.history_csv),
            "lead_time": relative_to_project(state.lead_time_csv),
            "log": relative_to_project(outputs.log_file),
            "curves": relative_to_project(outputs.curves),
            "forecasts": relative_to_project(outputs.examples),
        },
    }


def write_training_info(cfg,
                        rt,
                        outputs,
                        state,
                        lr_scheduler):
    if rt.accelerator.is_main_process:
        write_json(outputs.info_json,
                   training_info(cfg,
                                 rt,
                                 outputs,
                                 state,
                                 lr_scheduler))


def update_figures(rt,
                   outputs,
                   state):
    if not rt.accelerator.is_main_process:
        return
    plot_diffusion_history(outputs.history_csv,
                           outputs.curves,
                           best_epoch=state.best_epoch)
    plot_lead_time_rmse(state.lead_time_csv,
                        os.path.join(outputs.curves, "rmse_by_lead_time.png"),
                        best_epoch=state.best_epoch)


def mark_run_ended(cfg,
                   rt,
                   outputs,
                   state,
                   lr_scheduler,
                   error=None):
    """
    Closes training_info.json: completed, or failed / interrupted with the
    error, so a crashed run still says how far it got and why it stopped.
    """
    if error is None:
        state.status = "completed"
    elif isinstance(error, KeyboardInterrupt):
        state.status = "interrupted"
    else:
        state.status = "failed"
        state.error = "".join(traceback.format_exception_only(type(error), error)).strip()
    state.ended_at = datetime.datetime.now().isoformat(timespec="seconds")
    write_training_info(cfg,
                        rt,
                        outputs,
                        state,
                        lr_scheduler)
    logger.info(f"run {state.status}: {outputs.info_json}")
    rt.tracker.summary({"best/epoch": state.best_epoch,
                        "best/global_step": state.best_global_step,
                        **{f"best/{k}": v for k, v in (state.best_val_metrics or {}).items()}})
    rt.tracker.finish(state.status,
                      state.global_step)


# ================================ TRAINING ================================

def _save_step_checkpoint(cfg,
                          rt,
                          outputs,
                          state,
                          name):
    """
    accelerate state in checkpoints/<name>, recorded for training_info.json.
    Before a checkpoint-<step>, the oldest ones beyond checkpoints_total_limit
    are removed. Main process only, as the save itself.
    """
    config_accelerate = cfg.accelerate
    ckptdir = outputs.checkpoints
    if name.startswith("checkpoint-") and config_accelerate.checkpoints_total_limit is not None:
        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        checkpoints = _step_checkpoints(ckptdir)
        if len(checkpoints) >= config_accelerate.checkpoints_total_limit:
            num_to_remove = len(checkpoints) - config_accelerate.checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
            for removing_checkpoint in removing_checkpoints:
                shutil.rmtree(os.path.join(ckptdir, removing_checkpoint))
                state.checkpoints.pop(removing_checkpoint, None)

    save_path = os.path.join(ckptdir, name)
    rt.accelerator.save_state(save_path)
    state.checkpoints[name] = {"folder": name,
                               "global_step": state.global_step,
                               "epochs_completed": state.epochs_completed}
    logger.info(f"Saved state to {save_path}")


def save_forecast_examples(cfg,
                           outputs,
                           state,
                           forecast,
                           cube_paths,
                           epoch):
    """
    results/forecasts/val_epoch_XXX.png: the first diffusion.n_forecast_examples
    cubes of the first validation batch (always the same ones, the validation
    loader does not shuffle), see save_forecast_grid.
    """
    n = min(cfg.diffusion.n_forecast_examples, forecast["pred"].shape[0])
    if n == 0:
        return
    names = [os.path.splitext(os.path.basename(str(p)))[0] for p in cube_paths[:n]] if cube_paths else None
    path = os.path.join(outputs.examples,
                        f"val_epoch_{epoch:03d}.png" if state.mode == "train" else "validate_only.png")
    save_forecast_grid(forecast["context"][:n].float().cpu().numpy(),
                       forecast["target"][:n].float().cpu().numpy(),
                       forecast["pred"][:n].float().cpu().numpy(),
                       path,
                       n_columns=cfg.diffusion.forecast_lead_columns,
                       names=names,
                       title=f"validation, epoch {epoch} (step {state.global_step})")


def train_loop(cfg,
               rt,
               outputs,
               state,
               optimizer,
               lr_scheduler,
               conditioner,
               denoiser,
               sampler,
               model,
               first_stage_model,
               train_dataloader,
               val_dataloader,
               loss_fn):
    """
    Il training vero e proprio. `state` (TrainState) arriva da configure_train()
    e viene fatto avanzare qui: global_step, resume_step, best_latter_rmse e le
    metriche in state.metrics cambiano mentre il loop gira, quindi si leggono e
    si scrivono sempre sull'oggetto, non su copie locali.
    After every epoch: history.csv, curves, training_info.json; at every
    validation also lead_time.csv and the forecast figure.
    """
    accelerator = rt.accelerator
    weight_dtype = rt.weight_dtype
    config_accelerate = cfg.accelerate
    config_diffusion = cfg.diffusion
    args = cfg.run
    config_data = cfg.data
    noise_image_num = config_diffusion.network_config.params.noise_image_num
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config_accelerate.gradient_accumulation_steps)
    # the columns of the validation metrics, also on the epochs without one
    val_keys = list(validation_rmse([0.0] * len(VAL_SUM_KEYS)))

    # ENS (config section `ens`): scoring of the forecasts, shared with evaluate_ens.py.
    # ens_columns = the columns it adds to history.csv, empty (they stay empty on the
    # epochs where nothing is scored, so the header never changes).
    ens_cfg = cfg.ens
    ens_eval = EnsEvaluator(cfg, accelerator, train_dataloader) if ens_cfg.enabled else None
    ens_columns = ens_eval.columns() if ens_eval is not None else {}
    if ens_eval is not None:
        logger.info(f"[ens] on: validation={ens_cfg.validation}, train={ens_cfg.train}"
                    + (f" ({ens_cfg.train_cubes} cubes every {ens_cfg.train_every_epochs} "
                       f"epoch(s), {'fixed' if ens_cfg.train_fixed_subset else 'random'} sample)"
                       if ens_cfg.train else "")
                    + f" | K={args.val_num_samples} predictions per cube, best kept"
                    f" | cloud mask {'on' if ens_cfg.cloud_mask else 'off'}")

    write_training_info(cfg,
                        rt,
                        outputs,
                        state,
                        lr_scheduler)
    # Cronometro e picco di memoria partono da qui: il caricamento del
    # checkpoint, che e' gia' avvenuto, non c'entra con il costo del training.
    state.metrics.start()

    # Training Loop
    logger.info('Start training loop')

    for epoch in range(state.first_epoch, config_accelerate.num_train_epochs):
        model.train()
        conditioner.train()
        train_loss = 0.0
        # epoch means for history.csv
        epoch_loss_sum = 0.0
        epoch_rmse_sum = 0.0
        epoch_steps = 0
        epoch_cubes = 0
        epoch_start = time.time()
        '''
        BUG: qui `train_dataloader` veniva riassegnato, e la riassegnazione
        sopravviveva all'epoca in cui serviva. Codice originale:

            if args.resume_from_checkpoint and epoch == first_epoch and resume_step > 0:
                train_dataloader = skip_first_batches(train_dataloader, resume_step)
                progress_bar.update(resume_step//config_accelerate.gradient_accumulation_steps)
                resume_step = 0

        `skip_first_batches` non salta i batch al volo: costruisce un dataloader
        nuovo con un SkipBatchSampler, il cui __len__ e' len(batch_sampler) -
        skip_batches. Scrivendo quel dataloader in `train_dataloader`, cioe'
        nella stessa variabile su cui itera ogni epoca, il troncamento diventava
        permanente. La guardia `resume_step > 0` impediva al ramo di rieseguirsi,
        ma non ripristinava il dataloader intero: tutte le epoche successive
        ripartivano gia' amputate.

        Effetto misurato sul run ripreso da checkpoint-77000, con
        len(train_dataloader) = 1766 (14133 cubi / batch 8, drop_last=True):
        resume_step = 77000 % 1766 = 1062, quindi da li' in avanti ogni epoca ha
        eseguito 1766 - 1062 = 704 step invece di 1766, per 57 epoche di fila.
        Le epoche avanzavano cosi' 2.5 volte piu' in fretta di quanto assuma il
        calcolo di num_train_epochs a inizio script, che divide max_train_steps
        per 1766: il bound di 100 epoche e' stato raggiunto a global_step 117128
        invece che a max_train_steps = 176600, e il training e' terminato da solo
        a due terzi del budget, senza errori e senza nulla a log.

        Lo stesso era gia' successo nel run precedente, ripreso da checkpoint-1000:
        resume_step = 1000 e quindi 766 step per epoca.

        Lo skip agisce sul batch sampler e shuffle e' True, quindi i batch
        scartati cambiavano a ogni epoca: nessun cubo e' stato ignorato in modo
        sistematico, ma ogni "epoca" copriva circa il 40% del training set.
        '''
        # Il dataloader ridotto serve solo per l'epoca in cui si riprende, quindi
        # resta in una variabile locale: `train_dataloader` deve continuare a
        # riferirsi al loader intero per tutte le epoche successive.
        epoch_dataloader = train_dataloader
        if args.validate_only:
            # Niente training: lista vuota, si salta dritti alla validazione
            # in fondo all'epoca.
            epoch_dataloader = []
        # One bar per epoch, over its optimization steps; only on the main
        # process of each machine. In the resumed epoch it starts from the
        # steps already done.
        progress_bar = tqdm(total=num_update_steps_per_epoch,
                            initial=0,
                            desc=f"Epoch {epoch + 1}/{config_accelerate.num_train_epochs}",
                            disable=not accelerator.is_local_main_process or args.validate_only,
                            dynamic_ncols=True)
        if not args.validate_only and epoch == state.first_epoch and state.resume_step > 0:
            epoch_dataloader = skip_first_batches(train_dataloader, state.resume_step)
            progress_bar.update(state.resume_step // config_accelerate.gradient_accumulation_steps)
            state.resume_step = 0

        # Start Train!
        for step, batch in enumerate(epoch_dataloader):
            for batch_key in batch.keys():
                if torch.is_tensor(batch[batch_key]):
                    batch[batch_key] = batch[batch_key].to(dtype=weight_dtype)

            x = batch[config_diffusion.input_key]  # 遥感图像
            with accelerator.accumulate(model):
                # video inference
                B, T, C, H, W = x.shape
                x_future = x[:, -noise_image_num:, ...]  # 获取

                x_past = x[:, :-noise_image_num, ...]
                x_past_mean = torch.mean(x_past, dim=1).contiguous()  # b,c,h,w

                x = x.view(B * T, C, H, W).contiguous()
                x = encode_first_stage(x,
                                       first_stage_model,
                                       config_diffusion)
                x_past_mean = encode_first_stage(x_past_mean,
                                                 first_stage_model,
                                                 config_diffusion)

                x = x.to(dtype=weight_dtype)
                x_past_mean = x_past_mean.to(dtype=weight_dtype)
                _, C, H, W = x.shape
                # 修改x形状为（b,t,c,h,w)
                x = x.view(B, T, C, H, W).contiguous()
                assert x.dtype == weight_dtype
                # calc loss
                loss, true_latent_rmse, true_latent_rmse_mean, prediction_sample, future_sample, noise_added_output_sample = \
                    loss_fn(model, denoiser, conditioner, x, batch, x_future, x_past_mean,
                            # the loss decodes a few samples for logging: it
                            # expects a one-argument decode(latent) function
                            functools.partial(decode_first_stage,
                                              first_stage_model=first_stage_model,
                                              config_diffusion=config_diffusion))  # b

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(config_data.params.batch_size)).mean()
                train_loss += avg_loss.item() / config_accelerate.gradient_accumulation_steps
                epoch_loss_sum += avg_loss.item()
                epoch_rmse_sum += true_latent_rmse.item()
                epoch_steps += 1
                epoch_cubes += B * accelerator.num_processes

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config_accelerate.max_grad_norm)
                    accelerator.clip_grad_norm_(conditioner.parameters(), config_accelerate.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes; Check gradient accumulation
            if accelerator.sync_gradients:
                progress_bar.update(1)
                state.global_step += 1
                rt.tracker.log({"step/train_loss": train_loss,
                                "step/latent_rmse": true_latent_rmse.item(),
                                "step/latent_rmse_context_mean": true_latent_rmse_mean.item(),
                                "step/lr": lr_scheduler.get_last_lr()[0]},
                               step=state.global_step)
                train_loss = 0.0
                if state.global_step % config_accelerate.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        _save_step_checkpoint(cfg,
                                              rt,
                                              outputs,
                                              state,
                                              f"checkpoint-{state.global_step}")
                    accelerator.wait_for_everyone()
                if state.global_step in config_accelerate.checkpointing_steps_list:
                    if accelerator.is_main_process:
                        _save_step_checkpoint(cfg,
                                              rt,
                                              outputs,
                                              state,
                                              f"save-checkpoint-{state.global_step}")
                    accelerator.wait_for_everyone()

            # running means of the epoch, as in phase 1. Only what is worth
            # watching goes into the bar: more fields would make the line wider
            # than the terminal, and tqdm would then print a new line at every
            # update instead of redrawing the bar
            progress_bar.set_postfix(loss=f"{epoch_loss_sum / epoch_steps:.4f}",
                                     rmse=f"{epoch_rmse_sum / epoch_steps:.4f}",
                                     lr=f"{lr_scheduler.get_last_lr()[0]:.1e}")
        progress_bar.close()

        train_seconds = time.time() - epoch_start
        if epoch_steps:
            state.train_metrics = {"loss": epoch_loss_sum / epoch_steps,
                                   "latent_rmse": epoch_rmse_sum / epoch_steps}
            state.epoch_seconds.append(train_seconds)
            state.cubes_per_second.append(epoch_cubes / train_seconds)
            state.epochs_completed = epoch + 1
            logger.info(f"[diffusion] epoca {epoch + 1}/{config_accelerate.num_train_epochs} | "
                        f"step {state.global_step} | TRAIN loss={state.train_metrics['loss']:.6f} "
                        f"latent_rmse={state.train_metrics['latent_rmse']:.6f} | "
                        f"{train_seconds:.0f}s train")

        is_last_epoch = (
                epoch == config_accelerate.num_train_epochs - 1
                or state.global_step >= config_accelerate.max_train_steps
        )

        val_rmse = None
        n_cubes = None
        val_seconds = None
        ens_row = dict(ens_columns)             # what the ENS adds to this epoch's history row
        if ens_eval is not None:
            ens_eval.scoring_seconds = 0.0
        if (epoch % config_accelerate.validate_epoch_interval == 0 or is_last_epoch
                or args.validate_only):
            validate_start_time = time.time()  # <--- START TIMER INFERENZA
            validate_metric_rgbn = EarthnetX_Metric(rgbn=True).to(accelerator.device, dtype=weight_dtype)
            validate_metric_ndvi = EarthnetX_Metric(rgbn=False).to(accelerator.device, dtype=weight_dtype)
            validate_metric_arvi = EarthnetX_Metric(rgbn=False).to(accelerator.device, dtype=weight_dtype)

            # Somme degli errori quadratici e conteggi, nell'ordine di
            # VAL_SUM_KEYS: si accumulano su tutti i batch e la radice si prende
            # alla fine, cosi' le RMSE sono quelle dell'intero validation set e
            # non la media di quelle dei singoli batch.
            val_totals = torch.zeros(len(VAL_SUM_KEYS), device=accelerator.device,
                                     dtype=torch.float64)
            # per lead time: model SE, baseline SE, count
            lead_totals = torch.zeros(3, noise_image_num, device=accelerator.device,
                                      dtype=torch.float64)
            n_val_batches = 0
            # ENS: one row per cube, filled batch by batch with the forecasts the RMSEs
            # already need (no second inference)
            ens_val_rows = [] if (ens_eval is not None and ens_cfg.validation) else None

            # Immagini di confronto, un PNG per cubo (ground truth sopra,
            # predizione sotto, con il nome del .npz di origine), solo
            # all'ultima epoca o con --validate_only. In multi-GPU ogni processo
            # salva i cubi del proprio shard, e i nomi sono distinti, quindi
            # non si pestano i piedi.
            save_images_dir = (os.path.join(outputs.examples, "final")
                               if (is_last_epoch or args.validate_only) else None)
            val_img_counter = {"n": 0}
            if save_images_dir is not None:
                logger.info(
                    f"[validation] ultima epoca: salvo le immagini di confronto "
                    f"in {save_images_dir}"
                    + (f" (max {args.save_val_images_max})"
                       if args.save_val_images_max else " (tutti i cubi)")
                )
            # one line per validation, over its batches, like the training bar
            val_bar = tqdm(total=min(len(val_dataloader), config_accelerate.validate_batch_num),
                           desc=f"Validation {epoch + 1}",
                           disable=not accelerator.is_local_main_process,
                           dynamic_ncols=True)
            for batch_num, batch in enumerate(val_dataloader):
                if batch_num >= config_accelerate.validate_batch_num:
                    break
                cloud = batch.pop("cloud_mask", None)    # for the ENS only, not for the model
                for batch_key in batch.keys():
                    if torch.is_tensor(batch[batch_key]):
                        batch[batch_key] = batch[batch_key].to(dtype=weight_dtype)
                validate_log, batch_sums, forecast = log_videos_local_and_wandb(
                    batch=batch,
                    model=model,
                    conditioner=conditioner,
                    first_stage_model=first_stage_model,
                    sampler=sampler,
                    denoiser=denoiser,
                    input_key=config_diffusion.input_key,
                    config_diffusion=config_diffusion,
                    noise_image_num=noise_image_num,
                    past_weigth=config_diffusion.loss_fn_config.params.past_weight,
                    val_num_samples=args.val_num_samples,
                    save_dir=save_images_dir,
                    save_counter=val_img_counter,
                    save_limit=args.save_val_images_max,
                    metrics_rgbn=validate_metric_rgbn,
                    metrics_ndvi=validate_metric_ndvi,
                    metrics_arvi=validate_metric_arvi,
                    return_all_samples=ens_val_rows is not None,
                )
                if ens_val_rows is not None:
                    ens_eval.add_batch(ens_val_rows, forecast["pred_samples"],
                                       forecast["target"], cloud, batch.get("cube_path"))
                val_totals += batch_sums
                lead_totals[0] += forecast["lead_se"]
                lead_totals[1] += forecast["lead_baseline_se"]
                lead_totals[2] += forecast["lead_count"]
                n_val_batches += 1
                # running value on this process's batches; the final one is
                # computed below over all processes
                running = validation_rmse(val_totals)
                val_bar.update(1)
                val_bar.set_postfix(latter_rmse=f"{running['latter_rmse']:.4f}",
                                    latent_rmse=f"{running['predict_latent_rmse']:.4f}")
                if batch_num == 0 and accelerator.is_main_process:
                    save_forecast_examples(cfg,
                                           outputs,
                                           state,
                                           forecast,
                                           batch.get("cube_path"),
                                           epoch + 1)
                accelerator.wait_for_everyone()
            val_bar.close()

            if save_images_dir is not None:
                logger.info(
                    f"[validation] salvate {val_img_counter['n']} immagini di "
                    f"confronto in {save_images_dir} "
                    f"(corrispondenze in {os.path.join(save_images_dir, 'index.tsv')})"
                )

            # --- RMSE sull'intero validation set ----------------------------
            # In multi-GPU ogni processo ha visto solo il suo shard: si sommano
            # gli accumulatori di tutti prima di dividere. gather concatena
            # lungo la dimensione 0, quindi si manda una riga per processo e si
            # somma. Cosi' il numero e' lo stesso ovunque e la decisione sul
            # modello migliore non diverge tra i rank.
            val_totals = accelerator.gather(val_totals.unsqueeze(0)).sum(dim=0)
            lead_totals = accelerator.gather(lead_totals.unsqueeze(0)).sum(dim=0)
            val_rmse = validation_rmse(val_totals)
            n_cubes = int(val_totals[VAL_SUM_KEYS.index("n_cubes")])
            lead_rmse = torch.sqrt(lead_totals[:2] / lead_totals[2].clamp(min=1)).tolist()
            if ens_val_rows is not None:
                ens_row.update(ens_eval.finish(ens_val_rows, "val"))
                state.ens_val = {"epoch": epoch + 1,
                                 **{k: ens_row[f"val_{k}"] for k in ENS_KEYS},
                                 "n_cubes": ens_row["val_ens_n_cubes"]}
                logger.info(f"[validation] ENS={_fmt_ens(ens_row['val_ENS'])} "
                            f"(MAD={_fmt_ens(ens_row['val_MAD'])} OLS={_fmt_ens(ens_row['val_OLS'])} "
                            f"EMD={_fmt_ens(ens_row['val_EMD'])} SSIM={_fmt_ens(ens_row['val_SSIM'])}) "
                            f"P-MAE={_fmt_ens(ens_row['val_P_MAE'])} N-MAE={_fmt_ens(ens_row['val_N_MAE'])} | "
                            f"{ens_row['val_ens_n_cubes']} cubes, best of K={args.val_num_samples}")
            val_seconds = time.time() - validate_start_time

            state.metrics.record_validation(val_rmse, n_cubes, epoch + 1,
                                            state.global_step,
                                            val_seconds)
            state.val_metrics = dict(val_rmse)
            state.val_epoch = epoch + 1
            state.val_lead_time = {"epoch": epoch + 1,
                                   "model": lead_rmse[0],
                                   "baseline_context_mean": lead_rmse[1]}

            logger.info(
                f"[validation] epoca {epoch + 1} | step {state.global_step} | "
                f"{n_val_batches} batch per processo, {n_cubes} cubi in totale"
            )
            for k, v in val_rmse.items():
                logger.info(f"[validation]   {k}: {v:.6f}")

            if accelerator.is_main_process:
                for lead, (model_rmse, baseline_rmse) in enumerate(zip(*lead_rmse), start=1):
                    append_history_row(state.lead_time_csv, {
                        "epoch": epoch + 1,
                        "global_step": state.global_step,
                        "lead": lead,
                        "model_rmse": model_rmse,
                        "baseline_rmse": baseline_rmse,
                    })

            # --- salvataggio del modello migliore ---------------------------
            # Il criterio e' latter_rmse: l'errore sugli ultimi 10 dei 20 frame
            # predetti, cioe' la parte piu' lontana dal contesto e quella che
            # separa davvero un modello che estrapola da uno che copia l'ultimo
            # frame visto.
            # In --validate_only non si salva niente: i pesi sono quelli del
            # checkpoint appena caricato.
            if (not args.validate_only) and val_rmse["latter_rmse"] < state.best_latter_rmse:
                previous = state.best_latter_rmse
                state.best_latter_rmse = val_rmse["latter_rmse"]
                state.best_epoch = epoch + 1
                state.best_global_step = state.global_step
                state.best_val_metrics = dict(val_rmse)
                state.best_val_lead_time = dict(state.val_lead_time)
                if accelerator.is_main_process:
                    accelerator.save_state(state.best_ckpt_path)
                    with open(os.path.join(state.best_ckpt_path, "best_metric.json"),
                              "w", encoding="utf-8") as fh:
                        json.dump({"epoch": epoch + 1, "global_step": state.global_step,
                                   "n_cubes": n_cubes, "k": args.val_num_samples,
                                   **val_rmse}, fh, indent=2)
                    state.checkpoints["best"] = {"folder": "best",
                                                 "epoch": epoch + 1,
                                                 "global_step": state.global_step,
                                                 "latter_rmse": state.best_latter_rmse}
                    logger.info(
                        f"[validation] nuovo minimo di latter_rmse: "
                        f"{state.best_latter_rmse:.6f} (prima {previous:.6f}), "
                        f"stato salvato in {state.best_ckpt_path}"
                    )
                accelerator.wait_for_everyone()
            elif not args.validate_only:
                logger.info(
                    f"[validation] latter_rmse {val_rmse['latter_rmse']:.6f} non "
                    f"migliora il minimo {state.best_latter_rmse:.6f}, niente da salvare"
                )

        # --- ENS on a sample of training cubes -------------------------------------
        # Full sampling of ens.train_cubes cubes (no flip, eval mode), as in validation:
        # it says how far the model is from its own training cubes. It is inference, so it
        # goes into state.metrics.inference_time and not into the training time.
        if (ens_eval is not None and ens_cfg.train and not args.validate_only
                and ((epoch + 1) % ens_cfg.train_every_epochs == 0 or is_last_epoch)):
            train_eval_start = time.time()
            ens_train_rows = []
            for batch in ens_eval.train_loader(epoch):
                cloud = batch.pop("cloud_mask", None)
                batch = ens_eval.to_device(batch)
                for batch_key in batch.keys():
                    if torch.is_tensor(batch[batch_key]):
                        batch[batch_key] = batch[batch_key].to(dtype=weight_dtype)
                _, _, forecast = log_videos_local_and_wandb(
                    batch=batch,
                    model=model,
                    conditioner=conditioner,
                    first_stage_model=first_stage_model,
                    sampler=sampler,
                    denoiser=denoiser,
                    input_key=config_diffusion.input_key,
                    config_diffusion=config_diffusion,
                    noise_image_num=noise_image_num,
                    past_weigth=config_diffusion.loss_fn_config.params.past_weight,
                    val_num_samples=args.val_num_samples,
                    metrics_rgbn=None,
                    metrics_ndvi=None,
                    metrics_arvi=None,
                    return_all_samples=True,
                )
                ens_eval.add_batch(ens_train_rows, forecast["pred_samples"],
                                   forecast["target"], cloud, batch.get("cube_path"))
            ens_row.update(ens_eval.finish(ens_train_rows, "train"))
            ens_row["ens_train_eval_time_s"] = time.time() - train_eval_start
            state.metrics.inference_time += ens_row["ens_train_eval_time_s"]
            state.ens_train = {"epoch": epoch + 1,
                               **{k: ens_row[f"train_{k}"] for k in ENS_KEYS},
                               "n_cubes": ens_row["train_ens_n_cubes"]}
            logger.info(f"[ens] epoch {epoch + 1} | TRAIN sample: ENS={_fmt_ens(ens_row['train_ENS'])} "
                        f"(MAD={_fmt_ens(ens_row['train_MAD'])} OLS={_fmt_ens(ens_row['train_OLS'])} "
                        f"EMD={_fmt_ens(ens_row['train_EMD'])} SSIM={_fmt_ens(ens_row['train_SSIM'])}) "
                        f"P-MAE={_fmt_ens(ens_row['train_P_MAE'])} N-MAE={_fmt_ens(ens_row['train_N_MAE'])} | "
                        f"{ens_row['train_ens_n_cubes']} cubes, best of K={args.val_num_samples}, "
                        f"{ens_row['ens_train_eval_time_s']:.0f}s")

        if ens_eval is not None:
            ens_row["ens_scoring_time_s"] = ens_eval.scoring_seconds or None

        if accelerator.is_main_process:
            # the K-sample columns only when K > 1: with K = 1 they repeat predict_latent_rmse
            multi_sample = args.val_num_samples > 1
            history_row = {
                "epoch": epoch + 1,
                "global_step": state.global_step,
                "lr": lr_scheduler.get_last_lr()[0],
                "train_loss": state.train_metrics.get("loss") if epoch_steps else None,
                "train_latent_rmse": state.train_metrics.get("latent_rmse") if epoch_steps else None,
                **{f"val_{k}": (val_rmse[k] if val_rmse is not None
                                and (multi_sample or "_of_k_" not in k) else None)
                   for k in val_keys},
                "val_n_cubes": n_cubes,
                "epoch_train_time_s": train_seconds if epoch_steps else None,
                "epoch_val_time_s": val_seconds,
                "train_cubes_per_s": epoch_cubes / train_seconds if epoch_steps else None,
                **ens_row,
            }
            append_history_row(state.history_csv,
                               history_row)
            rt.tracker.log({f"epoch/{k}": v for k, v in history_row.items()},
                           step=state.global_step)
        update_figures(rt,
                       outputs,
                       state)
        write_training_info(cfg,
                            rt,
                            outputs,
                            state,
                            lr_scheduler)
        rt.tracker.sync(state.global_step)

        # In --validate_only una sola passata di validazione e si esce.
        if args.validate_only:
            logger.info("[validate_only] validazione completata, esco senza addestrare.")
            break

        # Uscita per max_train_steps: in fondo all'epoca, non prima della
        # validazione, altrimenti l'ultima epoca finirebbe senza essere valutata
        # e senza poter aggiornare il modello migliore.
        if state.global_step >= config_accelerate.max_train_steps:
            break

    for line in state.metrics.summary_lines():
        logger.info(line)

    # the final state, to continue the training later from where it ended
    if (not args.validate_only and state.global_step > 0
            and f"checkpoint-{state.global_step}" not in _step_checkpoints(outputs.checkpoints)):
        if accelerator.is_main_process:
            _save_step_checkpoint(cfg,
                                  rt,
                                  outputs,
                                  state,
                                  f"checkpoint-{state.global_step}")
        accelerator.wait_for_everyone()
    accelerator.end_training()
    return state
