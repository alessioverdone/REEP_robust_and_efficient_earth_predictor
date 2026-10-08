"""
Output folder of a training run.

Every run writes ONLY into its own folder, named after the start time:

    registry/outputs/<phase>/<YYYY-MM-DD_HH-MM-SS>/
        artifacts/
            config.json          resolved config, every value explicit
            config.yaml          same config, reloadable with --cfgdir
            training_info.json   run, dataset, model, compute, metrics, checkpoints
            history.csv          one row per epoch
            train.log            the run's log
        checkpoints/             best.pt, last.pt, epoch_XXX.pt
        results/
            curves/              metric curves (train / val / test)
            <examples>/          inference examples: reconstructions/ for the
                                 autoencoder, forecasts/ for the diffusion model

This module only knows about folders and files; what goes into
training_info.json is decided by the training code of each phase.
"""

import datetime
import json
import math
import os
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from accelerate import PartialState
from accelerate.utils import broadcast_object_list
from omegaconf import OmegaConf
from torch.utils.flop_counter import FlopCounterMode

from src.config import PROJECT_DIR, TrainCfg, project_path, save_config


@dataclass
class RunOutputs:
    timestamp: str
    root: str
    artifacts: str
    checkpoints: str
    results: str
    curves: str
    examples: str

    @property
    def config_json(self) -> str:
        return os.path.join(self.artifacts, "config.json")

    @property
    def config_yaml(self) -> str:
        return os.path.join(self.artifacts, "config.yaml")

    @property
    def info_json(self) -> str:
        return os.path.join(self.artifacts, "training_info.json")

    @property
    def history_csv(self) -> str:
        return os.path.join(self.artifacts, "history.csv")

    @property
    def log_file(self) -> str:
        return os.path.join(self.artifacts, "train.log")


def run_outputs_at(root: str,
                   examples_dirname: str) -> RunOutputs:
    """The RunOutputs of an existing run folder (no folder is created)."""
    root = Path(root)
    results = root / "results"
    return RunOutputs(timestamp=root.name,
                      root=str(root),
                      artifacts=str(root / "artifacts"),
                      checkpoints=str(root / "checkpoints"),
                      results=str(results),
                      curves=str(results / "curves"),
                      examples=str(results / examples_dirname))


def create_run_outputs(base_dir: str,
                       examples_dirname: str = "reconstructions") -> RunOutputs:
    """
    Creates <base_dir>/<start timestamp>/ with its subfolders. Two runs started
    in the same second get a numeric suffix instead of sharing the folder.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = project_path(base_dir)
    root = base / timestamp
    suffix = 1
    while root.exists():
        root = base / f"{timestamp}_{suffix}"
        suffix += 1

    outputs = run_outputs_at(str(root),
                             examples_dirname)
    for path in (outputs.artifacts,
                 outputs.checkpoints,
                 outputs.curves,
                 outputs.examples):
        os.makedirs(path, exist_ok=True)
    return outputs


def create_shared_run_outputs(base_dir: str,
                              examples_dirname: str) -> RunOutputs:
    """
    create_run_outputs for a run launched on several processes (accelerate):
    the main process creates the folder and the others receive its path, so
    they all write into the same run instead of one folder per process.
    """
    state = PartialState()
    root = [None]
    if state.is_main_process:
        root[0] = create_run_outputs(base_dir,
                                     examples_dirname).root
    if state.num_processes > 1:
        broadcast_object_list(root)
    return run_outputs_at(root[0],
                          examples_dirname)


def save_run_config(cfg: TrainCfg,
                    outputs: RunOutputs):
    """config.json with every value resolved, config.yaml reloadable as is."""
    write_json(outputs.config_json,
               OmegaConf.to_container(cfg, resolve=True))
    save_config(cfg,
                outputs.config_yaml)


def _json_safe(obj):
    """inf/nan are not valid JSON: they become null. Tuples become lists."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_json(path: str,
               data: dict):
    """
    Atomic write: the file is written next to the target and then renamed, so
    a run killed mid-write never leaves a truncated json behind.
    """
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_json_safe(data),
                  f,
                  indent=2,
                  default=str)
    os.replace(tmp, path)


def relative_to_project(path: str) -> str:
    """Path as seen from the project root, when it lives under it."""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_DIR))
    except ValueError:
        return str(path)


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args],
                            cwd=PROJECT_DIR,
                            capture_output=True,
                            text=True,
                            check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def environment_info() -> dict:
    """Where and with what the run is executed: host, software, GPU, git commit."""
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({"name": props.name,
                         "memory_gb": round(props.total_memory / 1024 ** 3, 2)})
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpus": gpus,
        "command": " ".join(sys.argv),
        "git_commit": _git("rev-parse", "HEAD") or None,
        # tracked files modified with respect to the commit
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }


def count_parameters(model: torch.nn.Module) -> dict:
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


@torch.no_grad()
def measure_flops(fn,
                  *inputs) -> int:
    """FLOPs of one call fn(*inputs), counted by torch on the operators it runs."""
    with FlopCounterMode(display=False) as counter:
        fn(*inputs)
    return int(counter.get_total_flops())


def file_size_mb(path: str) -> float:
    return round(os.path.getsize(path) / 1024 ** 2, 2)
