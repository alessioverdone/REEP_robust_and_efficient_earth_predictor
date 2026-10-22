"""
Live report of a training run on Weights & Biases (config section `wandb`).

The run folder under registry/outputs/ stays the reference; W&B mirrors it:

    metrics   every value written to history.csv (one row per epoch), plus the
              per-step training loss of phase 2, logged as they are produced
    files     artifacts/ (config, training_info.json, history.csv, lead_time.csv,
              train.log) and results/ (curves and example PNGs), uploaded with
              the "live" policy, so W&B follows them while the run rewrites them
    images    the example PNGs of results/<examples>/ (reconstructions or
              forecasts), also logged as media at the step they appear

checkpoints/ is never uploaded. With wandb.enabled=false, and on every process
but the main one, the Tracker does nothing, so the training code calls it
unconditionally.
"""

import logging
from pathlib import Path

import wandb
from omegaconf import OmegaConf

from src.config import TrainCfg
from src.training.run_outputs import RunOutputs

logger = logging.getLogger(__name__)


class Tracker:
    """A W&B run bound to a run folder; every method is a no-op without one."""

    def __init__(self,
                 run=None,
                 outputs: RunOutputs = None,
                 cfg: TrainCfg = None):
        self.run = run
        self.outputs = outputs
        self.cfg = cfg
        self._synced = set()

    @property
    def active(self) -> bool:
        return self.run is not None

    def log(self,
            metrics: dict,
            step: int):
        """Numeric values only: None (e.g. no validation this epoch) is left out."""
        if not self.active:
            return
        values = {k: v for k, v in metrics.items()
                  if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if values:
            self.run.log(values,
                         step=step)

    def summary(self,
                values: dict):
        """Final values shown in the W&B run table (e.g. test metrics, best epoch)."""
        if self.active:
            self.run.summary.update({k: v for k, v in values.items() if v is not None})

    def sync(self,
             step: int):
        """
        Uploads the files of artifacts/ and results/ not uploaded yet (the
        "live" policy keeps following the ones already uploaded) and logs the
        new example PNGs as images at `step`.
        """
        if not self.active:
            return
        root = Path(self.outputs.root)
        examples = Path(self.outputs.examples)
        new_images = []
        new_dirs = set()
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if (not path.is_file() or relative.parts[0] not in ("artifacts", "results")
                    or path.suffix == ".tmp" or str(relative) in self._synced):
                continue
            self._synced.add(str(relative))
            new_dirs.add(path.parent)
            # only the figures of the epochs (epoch_XXX/, epochs_XXX/): the per-band PNGs of the
            # final export (val/, iid/, ...) have a cube per folder and would flood the report
            if (path.suffix == ".png" and examples in path.parents
                    and path.relative_to(examples).parts[0].startswith("epoch")):
                new_images.append(path)
        if self.cfg.wandb.upload_files:
            # one save per folder with new files (wandb prints a line per call);
            # files already uploaded match again and are simply kept. "*.*"
            # matches the files but not the subfolders (each folder with new files is saved)
            for folder in sorted(new_dirs):
                self.run.save(str(folder / "*.*"),
                              base_path=str(root),
                              policy="live")
        if new_images and self.cfg.wandb.log_images:
            self.run.log({examples.name: [wandb.Image(str(p), caption=str(p.relative_to(examples))) for p in new_images]},
                         step=step)

    def finish(self,
               status: str,
               step: int):
        """Last sync, then closes the W&B run with the outcome of the training."""
        if not self.active:
            return
        self.sync(step)
        self.run.summary["status"] = status
        self.run.finish(exit_code=0 if status == "completed" else 1)
        self.run = None


def start_tracker(cfg: TrainCfg,
                  outputs: RunOutputs,
                  phase: str,
                  is_main_process: bool = True) -> Tracker:
    """
    Opens the W&B run of this training, named like the run folder and grouped
    by run.project_name, with the resolved config. Its local files go into
    <run folder>/wandb/. Call it once the config is fully resolved.
    """
    if not (cfg.wandb.enabled and is_main_process):
        return Tracker()
    run = wandb.init(project=cfg.wandb.project,
                     entity=cfg.wandb.entity,
                     mode=cfg.wandb.mode,
                     name=f"{phase}/{cfg.run.project_name}/{outputs.timestamp}",
                     group=cfg.run.project_name,
                     job_type=phase,
                     tags=[phase, *cfg.wandb.tags],
                     config=OmegaConf.to_container(cfg, resolve=True),
                     dir=outputs.root)
    logger.info(f"[wandb] {cfg.wandb.mode} run {run.name}: {run.url or outputs.root + '/wandb'}")
    tracker = Tracker(run,
                      outputs,
                      cfg)
    tracker.sync(step=0)
    return tracker
