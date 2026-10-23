"""
Condition Predictor (Mamba): building, checkpoints, and its use in phase 2.

The checkpoint of a condition run (checkpoints/best.pt, last.pt) is
self-contained: next to the weights it stores
  - "arch":    model name, parameters and window sizes, to rebuild the same network;
  - "series":  how the weather was reduced and normalized (meso_agg, frames,
               cells, lo / hi of every variable);
  - "dataset": name and id of dataset.json, the cubes it was trained on.
Phase 2 rebuilds the model from "arch" and, before using it, checks that the
DiT reads the weather the way the model was trained on (load_condition_predictor).
"""

import importlib
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.config import TrainCfg, project_path
from src.dataset.dataset_info import dataset_identity, load_dataset_info
from src.dataset.meso_series import variable_names

logger = logging.getLogger(__name__)

# condition.model.name -> module with the Model class. Imported on demand:
# S_Mamba needs mamba_ssm (CUDA), PatchMixer does not.
MODEL_MODULES = {
    "S_Mamba": "src.model.mamba.S_Mamba",
    "PatchMixer": "src.model.mamba.PatchMixer",
}


# ================================== MODEL ==================================

def build_condition_model(name: str,
                          params: dict,
                          n_context: int,
                          n_predict: int,
                          n_channels: int) -> nn.Module:
    """
    The network `name` with the architecture `params`, reading n_context
    frames of n_channels variables and forecasting n_predict frames:
    (B, n_context, n_channels) -> (B, n_predict, n_channels).
    """
    if name not in MODEL_MODULES:
        raise ValueError(f"unknown condition model {name!r}, available: {sorted(MODEL_MODULES)}")
    try:
        module = importlib.import_module(MODEL_MODULES[name])
    except ImportError as error:
        if error.name != "mamba_ssm":
            raise
        raise SystemExit("S_Mamba needs mamba_ssm, which is not installed: on a machine with an NVIDIA "
                         "GPU run `uv pip install setuptools wheel packaging ninja` and then "
                         "`uv pip install causal-conv1d mamba-ssm --no-build-isolation` (as setup_env.sh "
                         "does), or use condition.model.name=PatchMixer") from error
    configs = SimpleNamespace(seq_len=n_context, pred_len=n_predict, enc_in=n_channels, **params)
    return module.Model(configs)


def default_device() -> torch.device:
    """cuda when available: the Mamba kernels of S_Mamba run only there."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_arch(cfg: TrainCfg) -> dict:
    """The architecture of cfg.condition, as stored in the checkpoints."""
    cond = cfg.condition
    return {"name": cond.model.name,
            "params": OmegaConf.to_container(cond.model.params, resolve=True),
            "n_context": cond.n_context,
            "n_predict": cond.n_predict,
            "n_channels": cond.n_channels}


def build_from_arch(arch: dict) -> nn.Module:
    return build_condition_model(arch["name"], arch["params"], arch["n_context"],
                                 arch["n_predict"], arch["n_channels"])


def series_contract(cfg: TrainCfg, lo: np.ndarray, hi: np.ndarray) -> dict:
    """How the weather reaches the model: stored in the checkpoints, checked by phase 2."""
    cond = cfg.condition
    return {"meso_agg": cond.meso_agg,
            "T": cond.T,
            "steps_per_frame": cond.steps_per_frame,
            "center_crop": cond.center_crop,
            "variables": variable_names(cfg),
            "lo": [float(v) for v in lo],
            "hi": [float(v) for v in hi]}


def load_condition_checkpoint(path: str, device=None) -> tuple[nn.Module, dict]:
    """(model in eval mode on `device`, default cuda if available; checkpoint dict) of a condition checkpoint."""
    device = default_device() if device is None else device
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("arch", "series", "model_state_dict"):
        if key not in ckpt:
            raise ValueError(f"{path} is not a condition checkpoint (no {key!r})")
    model = build_from_arch(ckpt["arch"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt


# ================================== RUNS ==================================

def completed_runs(outputs_dir: str):
    """(run folder, training_info.json) of the completed runs under outputs_dir, newest first."""
    root = project_path(outputs_dir)
    if not root.is_dir():
        return
    for run in sorted(os.listdir(root), reverse=True):
        info_json = root / run / "artifacts" / "training_info.json"
        if not info_json.is_file():
            continue
        info = json.loads(info_json.read_text())
        if info.get("run", {}).get("status") == "completed":
            yield root / run, info


def phase2_requirements(cfg: TrainCfg) -> dict:
    """
    What phase 2 needs from a condition predictor: the weather as the DiT reads
    it (data.params) and its frames. The same checks as load_condition_predictor.
    """
    data = cfg.data.params
    return {"meso_agg": data.meso_agg,
            "T": data.T,
            "steps_per_frame": data.meso_steps_per_frame,
            "center_crop": data.meso_center_crop,
            "n_context": data.T - data.noise_image_num,
            "n_predict": data.noise_image_num}


def _run_mismatches(info: dict, required: dict) -> list:
    """The requirements of phase 2 a run (its training_info.json) does not meet."""
    series = info.get("dataset", {}).get("series", {})
    arch = info.get("model", {}).get("arch", {})
    found = {**{k: series.get(k) for k in ("meso_agg", "T", "steps_per_frame", "center_crop")},
             "n_context": arch.get("n_context"), "n_predict": arch.get("n_predict")}
    return [f"{key}={found[key]!r} (needed {value!r})" for key, value in required.items()
            if found[key] != value]


def resolve_condition_path(cfg: TrainCfg) -> str:
    """
    condition.pretrained_path = 'latest' becomes the best.pt of the most recent
    COMPLETED condition run (under condition.outputs_dir) trained on the current
    dataset whose weather and frames fit phase 2 (phase2_requirements): a newer
    run made otherwise never hides an older one that fits. Any other value is a
    path and is left as it is. The config is updated in place, so the path
    actually used ends up in the logs.
    """
    cond = cfg.condition
    if cond.pretrained_path != "latest":
        return cond.pretrained_path
    current = dataset_identity(load_dataset_info(cfg.data.params.dataset_json))
    required = phase2_requirements(cfg)
    newest_mismatch = None
    for run_dir, info in completed_runs(cond.outputs_dir):
        best = run_dir / "checkpoints" / "best.pt"
        if not best.is_file() or (info.get("dataset", {}).get("identity") or {}).get("id") != current["id"]:
            continue
        mismatches = _run_mismatches(info, required)
        if not mismatches:
            cond.pretrained_path = str(best)
            logger.info(f"condition pretrained_path=latest -> {best}")
            return str(best)
        newest_mismatch = newest_mismatch or (run_dir.name, mismatches)
    hint = (f" The newest run on this dataset, {newest_mismatch[0]}, has "
            f"{', '.join(newest_mismatch[1])}: e.g. the DiT reads data.params.meso_agg, the model "
            f"was trained with condition.meso_agg." if newest_mismatch else "")
    raise FileNotFoundError(
        f"condition.pretrained_path=latest: no completed condition run in {cond.outputs_dir} "
        f"trained on dataset {current['name']} (id {current['id']}) that fits phase 2 "
        f"({', '.join(f'{k}={v!r}' for k, v in required.items())}).{hint} Train one with "
        f"src/scripts/run_condition_train.py, or set condition.pretrained_path to a checkpoint.")


# ================================= PHASE 2 =================================

class ConditionPredictor(nn.Module):
    """
    The frozen Condition Predictor as phase 2 uses it.

    Input: the weather of a cube as the DiT reads it (meso_condition_image),
    (B, T, n_channels) normalized in [-1, 1]. Output: the same tensor with the
    last n_predict frames replaced by the forecast made from the first
    n_context ones, clipped to [-1, 1] like the normalized observations.
    """

    def __init__(self, model: nn.Module, n_context: int, n_predict: int):
        super().__init__()
        self.model = model
        self.n_context = n_context
        self.n_predict = n_predict
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True):
        # always frozen: a call to .train() on the parent module must not
        # switch dropout / BatchNorm back on
        return super().train(False)

    @torch.no_grad()
    def forward(self, meso: torch.Tensor) -> torch.Tensor:
        context = meso[:, :self.n_context]
        forecast = self.model(context.float()).clamp(-1.0, 1.0)
        return torch.cat([context, forecast.to(meso.dtype)], dim=1)


def load_condition_predictor(cfg: TrainCfg, device=None) -> ConditionPredictor:
    """
    The Condition Predictor of condition.pretrained_path ('latest' or a path),
    frozen, on `device` (default cuda if available), after checking that it
    fits the data phase 2 reads:
      - same dataset (dataset.json id);
      - same frames: n_context + n_predict = data.params.T, n_predict = noise_image_num;
      - same weather: data.params.meso_agg, steps and cells equal to those of the
        model, and the same min / max as cond_stats.json (the normalization of the DiT).
    """
    device = default_device() if device is None else device
    path = resolve_condition_path(cfg)
    model, ckpt = load_condition_checkpoint(path, device)
    arch, series = ckpt["arch"], ckpt["series"]
    data = cfg.data.params
    errors = []

    current = dataset_identity(load_dataset_info(data.dataset_json))
    trained_on = ckpt.get("dataset") or {}
    if trained_on.get("id") != current["id"]:
        errors.append(f"trained on dataset {trained_on.get('name')} (id {trained_on.get('id')}), "
                      f"phase 2 uses {current['name']} (id {current['id']})")
    if arch["n_context"] + arch["n_predict"] != data.T or arch["n_predict"] != data.noise_image_num:
        errors.append(f"forecasts {arch['n_context']} -> {arch['n_predict']} frames, the DiT uses "
                      f"{data.T - data.noise_image_num} -> {data.noise_image_num}")
    for key, dit_key in (("meso_agg", "meso_agg"),
                         ("steps_per_frame", "meso_steps_per_frame"),
                         ("center_crop", "meso_center_crop")):
        if series[key] != data[dit_key]:
            errors.append(f"weather with {key}={series[key]!r}, the DiT reads it with "
                          f"data.params.{dit_key}={data[dit_key]!r}")
    cond_stats_path = Path(project_path(data.cond_stats_json))
    if cond_stats_path.is_file() and not errors:
        cond_stats = json.loads(cond_stats_path.read_text())
        # cond_stats.json is recomputed by the DiT dataloader when its parameters
        # change: it is comparable only when computed the same way as the series
        comparable = (cond_stats.get("meso_agg") == series["meso_agg"]
                      and cond_stats.get("T") == series["T"]
                      and cond_stats.get("meso_center_crop") == series["center_crop"]
                      and cond_stats.get("steps_per_frame") == series["steps_per_frame"])
        if not comparable:
            logger.info(f"{cond_stats_path} computed with other parameters: normalization not compared")
        elif not (np.allclose(cond_stats["meso"]["lo"], series["lo"])
                  and np.allclose(cond_stats["meso"]["hi"], series["hi"])):
            errors.append(f"normalization (min / max) different from {cond_stats_path}")
    if errors:
        raise RuntimeError(f"condition predictor {path} does not fit phase 2:\n  - "
                           + "\n  - ".join(errors))

    logger.info(f"condition predictor {arch['name']} from {path}: "
                f"{arch['n_context']} -> {arch['n_predict']} frames, meso_agg={series['meso_agg']}")
    return ConditionPredictor(model, arch["n_context"], arch["n_predict"]).to(device)
