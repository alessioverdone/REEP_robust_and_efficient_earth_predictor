"""
Phase "condition": training of the Condition Predictor (Mamba), which forecasts
the weather of the target frames of a cube from the weather of its context
frames. Phase 2 then conditions the DiT on that forecast.

The launch script is src/scripts/run_condition_train.py and has the same
structure as the other phases: configuration, data, model, training. No
constants here: every number comes from cfg.condition (schema in src/config.py,
values in registry/configurations/*.yaml).

Data: the series extracted by src/dataset/meso_series.py (one table frames x
variables per cube), normalized to [-1, 1] with the min / max of the train
series, as the DiT normalizes its weather. The loss is the MSE in that
normalized space, so the variables weigh the same whatever their units. The
metrics are reported both normalized and, per variable, in the units of the
cubes, next to those of two simple forecasts the model has to beat:
persistence (the last context frame repeated) and the context mean.

Everything the run produces goes into its own folder (src/training/run_outputs.py):
checkpoints in checkpoints/, config, history, log and training_info.json in
artifacts/, curves, forecast examples and statistics in results/.
"""

import datetime
import logging
import os
import time
import traceback
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset.dataset_info import dataset_identity, load_dataset_info
from src.dataset.meso_series import (MesoSeriesDataset, denormalize, ensure_series, load_split,
                                     load_stats, make_loaders, normalize, series_dir, variable_names)
from src.model.autoencoder_utils import append_history_row, save_checkpoint
from src.model.condition_utils import build_from_arch, model_arch, series_contract
from src.training.run_outputs import (count_parameters, environment_info, file_size_mb, measure_flops,
                                      relative_to_project, write_json)
from src.training.state import ConditionState
from src.utils.metrics import MAE, MSE, RMSE
from src.utils.plots import plot_condition_lead_rmse, plot_history, save_condition_examples

logger = logging.getLogger(__name__)

BASELINES = ("persistence", "context_mean")


# ================================== DATA ==================================

@dataclass
class ConditionData:
    loaders: dict           # split -> DataLoader of (context, target), normalized
    raw: dict               # split -> series (n_cubes, n_frames, n_channels), units of the cubes
    lo: np.ndarray          # normalization, min / max of every variable on train
    hi: np.ndarray
    info: dict              # series.json


def setup_dataloader(cfg) -> ConditionData:
    """
    The series of the train and val splits, their DataLoaders and the
    normalization. The series are extracted on the spot when missing, see
    ensure_series in src/dataset/meso_series.py.
    """
    info = ensure_series(cfg)
    lo, hi = load_stats(cfg)
    raw = {split: load_split(cfg, split)[0] for split in ("train", "val")}
    for split, series in raw.items():
        if len(series) == 0:
            raise SystemExit(f"[condition] no cube of the {split} split has a complete weather "
                             f"series (see series.json in {series_dir(cfg)})")
    loaders = make_loaders(cfg, {split: normalize(series, lo, hi) for split, series in raw.items()})
    logger.info(f"[condition] series in {series_dir(cfg)}: train {len(raw['train'])} cubes, "
                f"val {len(raw['val'])} cubes, {cfg.condition.T} frames x {cfg.condition.n_channels} variables")
    return ConditionData(loaders=loaders, raw=raw, lo=lo, hi=hi, info=info)


# ================================== MODEL ==================================

def instantiate_model(cfg, rt):
    """The network of cfg.condition.model, on the device of the runtime."""
    arch = model_arch(cfg)
    model = build_from_arch(arch).to(rt.device)
    logger.info(f"[condition] {arch['name']}: {count_parameters(model)['total'] / 1e3:.1f}k parameters, "
                f"{arch['n_context']} -> {arch['n_predict']} frames x {arch['n_channels']} variables")
    return model


def prepare_training(cfg, model):
    """AdamW, and a scheduler that reduces the lr when val_loss stops improving."""
    cond = cfg.condition
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cond.lr,
                                  weight_decay=cond.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="min",
                                                           factor=cond.lr_scheduler_factor,
                                                           patience=cond.lr_scheduler_patience)
    return optimizer, scheduler


def configure_train(cfg, outputs) -> ConditionState:
    """The state the loop starts from: always a new training, in its own run folder."""
    logger.info(f"[condition] outputs del run in {outputs.root}")
    return ConditionState(history_csv=outputs.history_csv,
                          started_at=datetime.datetime.now().isoformat(timespec="seconds"))


# =============================== EVALUATION ===============================

@torch.no_grad()
def predict_series(cfg, rt, model, normalized: np.ndarray) -> tuple:
    """(context, target, forecast) of every cube, normalized. The model is left in its mode."""
    cond = cfg.condition
    loader = DataLoader(MesoSeriesDataset(normalized, cond.n_context, cond.n_predict),
                        batch_size=cond.batch_size,
                        shuffle=False)
    was_training = model.training
    model.eval()
    contexts, targets, forecasts = [], [], []
    for context, target in loader:
        forecasts.append(model(context.to(rt.device)).float().cpu().numpy())
        contexts.append(context.numpy())
        targets.append(target.numpy())
    model.train(was_training)
    return np.concatenate(contexts), np.concatenate(targets), np.concatenate(forecasts)


def baseline_forecasts(context_units: np.ndarray, n_predict: int) -> dict:
    """
    The simple forecasts the model has to beat, made from the OBSERVED context
    in the units of the cubes (not from the normalized one, which is clipped to
    the train range): the last frame repeated, and the mean of the context.
    """
    return {"persistence": np.repeat(context_units[:, -1:], n_predict, axis=1),
            "context_mean": np.repeat(context_units.mean(axis=1, keepdims=True), n_predict, axis=1)}


def forecast_metrics(target: np.ndarray,
                     forecast: np.ndarray,
                     target_units: np.ndarray,
                     forecast_units: np.ndarray,
                     variables) -> dict:
    """
    The errors of a forecast (n_cubes, n_predict, n_channels), given both
    normalized (target, forecast) and in the units of the cubes:
        loss, rmse, mae          on the normalized weather, all the variables together
                                 (loss is the MSE the model is trained on)
        rmse_<var>, mae_<var>    each variable in the units of the cubes, against the
                                 observed series (not clipped by the normalization)
        rmse_by_lead             normalized RMSE of every forecast frame
    """
    metrics = {"loss": float(MSE(forecast, target)),
               "rmse": float(RMSE(forecast, target)),
               "mae": float(MAE(forecast, target))}
    for j, name in enumerate(variables):
        metrics[f"rmse_{name}"] = float(RMSE(forecast_units[..., j], target_units[..., j]))
        metrics[f"mae_{name}"] = float(MAE(forecast_units[..., j], target_units[..., j]))
    metrics["rmse_by_lead"] = np.sqrt(np.mean((forecast - target) ** 2, axis=(0, 2))).tolist()
    return metrics


def evaluate_series(cfg, rt, model, raw: np.ndarray, lo, hi, with_baselines: bool = True) -> dict:
    """
    Model (and baselines) on the series `raw` (n_cubes, n_frames, n_channels),
    in the units of the cubes: the first n_context frames are read, the next
    n_predict forecast. Returns {"n_cubes", "model": metrics, "baselines":
    {name: metrics}, "arrays": (context, target, forecast) in the units of the cubes}.
    """
    cond = cfg.condition
    variables = variable_names(cfg)
    context_units = raw[:, :cond.n_context]
    target_units = raw[:, cond.n_context:cond.n_context + cond.n_predict]
    _, target, forecast = predict_series(cfg, rt, model, normalize(raw, lo, hi))
    forecast_units = denormalize(forecast, lo, hi)
    result = {"n_cubes": len(raw),
              "model": forecast_metrics(target, forecast, target_units, forecast_units, variables),
              "baselines": {},
              "arrays": (context_units, target_units, forecast_units)}
    if with_baselines:
        for name, baseline_units in baseline_forecasts(context_units, cond.n_predict).items():
            result["baselines"][name] = forecast_metrics(target, normalize(baseline_units, lo, hi),
                                                         target_units, baseline_units, variables)
    return result


def example_indices(n_cubes: int, n: int) -> np.ndarray:
    """n cubes evenly spread over the split, always the same ones."""
    return np.linspace(0, n_cubes - 1, num=min(n, n_cubes)).astype(int) if n else np.array([], dtype=int)


def save_examples(cfg, result: dict, indices: np.ndarray, path: str, title: str):
    """Forecast examples (context, observed, forecast) of the cubes `indices` of an evaluation."""
    if not len(indices):
        return
    context, target, forecast = (a[indices] for a in result["arrays"])
    save_condition_examples(context, target, forecast, variable_names(cfg), path, title=title)


def write_split_statistics(path: str, split: str, checkpoint: str, result: dict):
    """results/statistics/metrics_<split>.json: the model and the baselines on one split."""
    write_json(path, {"split": split,
                      "checkpoint": checkpoint,
                      "n_cubes": result["n_cubes"],
                      "model": result["model"],
                      "baselines": result["baselines"]})


def lead_rmse_curves(result: dict) -> dict:
    return {"model": result["model"]["rmse_by_lead"],
            **{name: m["rmse_by_lead"] for name, m in result["baselines"].items()}}


def scalar_metrics(metrics: dict) -> dict:
    """The metrics without the per-lead list, for history.csv and the logs."""
    return {k: v for k, v in metrics.items() if k != "rmse_by_lead"}


# ============================ DESCRIZIONE DEL RUN ============================

def describe_run(cfg, rt, outputs, state, model, data: ConditionData):
    """
    The parts of training_info.json fixed for the whole run: environment,
    dataset and series, model, computational cost, and the baselines on the
    validation set. Stored in state.static and written right away.
    """
    cond = cfg.condition
    dataset = load_dataset_info(cfg.data.params.dataset_json)

    sample = torch.zeros(1, cond.n_context, cond.n_channels, device=rt.device)
    model.eval()
    try:
        flops = measure_flops(model, sample)
    except Exception as error:      # custom CUDA kernels the counter cannot follow
        logger.info(f"[condition] FLOPs not measured: {error}")
        flops = None
    model.train()

    baselines = evaluate_series(cfg, rt, model, data.raw["val"], data.lo, data.hi)["baselines"]
    state.baselines = baselines

    state.static = {
        "environment": environment_info(),
        "dataset": {
            "identity": dataset_identity(dataset),
            "dataset_json": cfg.data.params.dataset_json,
            "series_dir": relative_to_project(str(series_dir(cfg))),
            "series": series_contract(cfg, data.lo, data.hi),
            "splits": {split: entry for split, entry in data.info["splits"].items()},
            "batches_per_epoch": {split: len(loader) for split, loader in data.loaders.items()},
        },
        "model": {
            "class": type(model).__module__,
            "arch": model_arch(cfg),
            "parameters": count_parameters(model),
        },
        "flops_per_cube": None if flops is None else {
            "forward_mflops": flops / 1e6,
            "train_step_mflops_estimate": 3 * flops / 1e6,
        },
    }
    for name, metrics in baselines.items():
        logger.info(f"[condition] baseline {name} on val: loss={metrics['loss']:.6f} "
                    f"rmse={metrics['rmse']:.6f}")


def training_info(cfg, outputs, state, optimizer) -> dict:
    """The full content of training_info.json at this point of the run."""
    cond = cfg.condition
    metrics = state.metrics
    gpu_memory = metrics.gpu_memory_gb()
    train_seconds = float(sum(state.epoch_seconds))
    wall_seconds = metrics.wall_time if metrics.start_time else 0.0
    best = state.checkpoints.get("best.pt")
    return {
        "run": {
            "phase": "condition",
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
        "model": state.static.get("model", {}),
        "training": {
            "epochs_planned": cond.num_epochs,
            "epochs_completed": state.epochs_completed,
            "global_step": state.global_step,
            "batch_size": cond.batch_size,
            "optimizer": {"name": type(optimizer).__name__,
                          "lr": cond.lr,
                          "weight_decay": cond.weight_decay,
                          "grad_clip_norm": cond.grad_clip_norm},
            "scheduler": {"name": "ReduceLROnPlateau",
                          "monitor": "val_loss",
                          "factor": cond.lr_scheduler_factor,
                          "patience": cond.lr_scheduler_patience},
            "lr_final": optimizer.param_groups[0]["lr"],
            "lr_reductions": state.lr_reductions,
            "early_stopping": {"enabled": cond.early_stopping_patience > 0,
                               "monitor": "val_loss",
                               "patience": cond.early_stopping_patience,
                               "min_delta": cond.early_stopping_min_delta,
                               "epochs_without_improvement": state.epochs_without_improvement,
                               "triggered": state.stopped_early,
                               "stopped_at_epoch": state.epochs_completed if state.stopped_early else None},
            "loss": "MSE on the weather normalized to [-1, 1]",
        },
        "metrics": {
            "selection_criterion": "val_loss",
            "train_last": state.train_metrics,
            "val_last": state.val_metrics,
            "val_best": {"epoch": state.best_epoch, **state.best_val_metrics},
            "val_baselines": {name: scalar_metrics(m) for name, m in state.baselines.items()},
        },
        "compute": {
            **(state.static.get("flops_per_cube") or {}),
            "wall_time_s": wall_seconds,
            "train_time_s": train_seconds,
            "validation_time_s": metrics.inference_time,
            "mean_epoch_train_time_s": (train_seconds / len(state.epoch_seconds)
                                        if state.epoch_seconds else None),
            "mean_train_cubes_per_s": (float(np.mean(state.cubes_per_second))
                                       if state.cubes_per_second else None),
            "gpu_hours": wall_seconds / 3600,
            "peak_gpu_memory_allocated_gb": gpu_memory[0] if gpu_memory else None,
            "peak_gpu_memory_reserved_gb": gpu_memory[1] if gpu_memory else None,
        },
        "checkpoints": {
            "saved_to_disk": cond.save_checkpoints,
            "dir": relative_to_project(outputs.checkpoints),
            "best": None if best is None else {
                "file": "best.pt",
                "path": relative_to_project(os.path.join(outputs.checkpoints, "best.pt")),
                "epoch": best["epoch"],
                "criterion": "val_loss",
                "value": state.best_val_metrics.get("loss"),
            },
            "saved": [state.checkpoints[name] for name in sorted(state.checkpoints)],
        },
        "outputs": {
            "root": relative_to_project(outputs.root),
            "config": relative_to_project(outputs.config_json),
            "history": relative_to_project(outputs.history_csv),
            "log": relative_to_project(outputs.log_file),
            "curves": relative_to_project(outputs.curves),
            "forecasts": relative_to_project(outputs.examples),
            "statistics": relative_to_project(os.path.join(outputs.results, "statistics")),
        },
    }


def write_training_info(cfg, outputs, state, optimizer):
    write_json(outputs.info_json,
               training_info(cfg, outputs, state, optimizer))


def update_figures(outputs, state):
    plot_history(outputs.history_csv,
                 outputs.curves,
                 metrics=("loss",),
                 best_epoch=state.best_epoch)


def mark_run_ended(cfg, rt, outputs, state, optimizer, error=None):
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
    write_training_info(cfg, outputs, state, optimizer)
    logger.info(f"[condition] run {state.status}: {outputs.info_json}")
    rt.tracker.summary({"best/epoch": state.best_epoch,
                        **{f"best/val_{k}": v for k, v in state.best_val_metrics.items()}})
    rt.tracker.finish(state.status, state.global_step)


# ================================ TRAINING ================================

def _checkpoint_extra(cfg, state, data, train_metrics, val_metrics):
    """
    What is saved next to the weights: the contract with phase 2
    (architecture, series, dataset), see src/model/condition_utils.py.
    """
    return {
        "arch": model_arch(cfg),
        "series": series_contract(cfg, data.lo, data.hi),
        "dataset": state.static["dataset"]["identity"],
        "best_metric": state.best_metric,
        "best_epoch": state.best_epoch,
        "save_best_by": "loss",
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }


def _save_run_checkpoint(outputs, state, name, model, optimizer, scheduler, epoch, extra):
    """Saves checkpoints/<name> and records it for training_info.json."""
    path = os.path.join(outputs.checkpoints, name)
    save_checkpoint(path, model, optimizer, epoch, scheduler=scheduler, extra=extra)
    state.checkpoints[name] = {"file": name,
                               "epoch": epoch + 1,
                               "size_mb": file_size_mb(path),
                               "val_loss": extra["val_metrics"]["loss"]}
    return path


def update_early_stopping(cfg, state, val_loss):
    """
    Counts the epochs without an improvement of val_loss larger than min_delta
    and sets state.stopped_early once they reach the patience (0 = never).
    """
    cond = cfg.condition
    if val_loss < state.early_stop_reference - cond.early_stopping_min_delta:
        state.early_stop_reference = val_loss
        state.epochs_without_improvement = 0
    else:
        state.epochs_without_improvement += 1
    state.stopped_early = (cond.early_stopping_patience > 0
                           and state.epochs_without_improvement >= cond.early_stopping_patience)


def train_condition(cfg, rt, outputs, state, model, optimizer, scheduler, data: ConditionData):
    """
    The training loop. After every epoch: validation, history.csv, checkpoints,
    curves, training_info.json, and every examples_every epochs the forecast
    examples of the same validation cubes (results/forecasts/epoch_XXX/val.png).
    """
    cond = cfg.condition
    examples = example_indices(len(data.raw["val"]), cond.n_examples)
    write_training_info(cfg, outputs, state, optimizer)
    state.metrics.start()

    for epoch in range(state.start_epoch, cond.num_epochs):
        model.train()
        running, n_cubes = 0.0, 0
        epoch_start = time.time()

        for context, target in data.loaders["train"]:
            context = context.to(rt.device, non_blocking=True)
            target = target.to(rt.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(context), target)
            loss.backward()
            if cond.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cond.grad_clip_norm)
            optimizer.step()

            bs = context.size(0)
            n_cubes += bs
            running += loss.item() * bs
            state.global_step += 1

        epoch_seconds = time.time() - epoch_start
        state.epoch_seconds.append(epoch_seconds)
        state.cubes_per_second.append(n_cubes / epoch_seconds)
        train_metrics = {"loss": running / n_cubes}

        val_start = time.time()
        result = evaluate_series(cfg, rt, model, data.raw["val"], data.lo, data.hi, with_baselines=False)
        val_metrics = scalar_metrics(result["model"])
        val_seconds = time.time() - val_start
        state.metrics.record_validation(val_metrics, result["n_cubes"], epoch + 1,
                                        state.global_step, val_seconds)
        state.train_metrics = train_metrics
        state.val_metrics = val_metrics
        state.epochs_completed = epoch + 1

        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["loss"])
        lr_after = optimizer.param_groups[0]["lr"]
        if lr_after < lr_before:
            state.lr_reductions.append({"epoch": epoch + 1, "lr": lr_after})
            logger.info(f"[condition] lr ridotto: {lr_before:.2e} -> {lr_after:.2e}")

        logger.info(f"[condition] epoca {epoch + 1}/{cond.num_epochs} | "
                    f"TRAIN loss={train_metrics['loss']:.6f} | "
                    f"VAL loss={val_metrics['loss']:.6f} rmse={val_metrics['rmse']:.6f} "
                    f"mae={val_metrics['mae']:.6f} | {epoch_seconds:.1f}s train, {val_seconds:.1f}s val")

        history_row = {"epoch": epoch + 1,
                       "global_step": state.global_step,
                       "lr": lr_before,
                       "train_loss": train_metrics["loss"],
                       **{f"val_{k}": v for k, v in val_metrics.items()},
                       "epoch_train_time_s": epoch_seconds,
                       "epoch_val_time_s": val_seconds,
                       "train_cubes_per_s": n_cubes / epoch_seconds}
        append_history_row(state.history_csv, history_row)
        rt.tracker.log({f"epoch/{k}": v for k, v in history_row.items()}, step=state.global_step)

        is_best = val_metrics["loss"] < state.best_metric
        if is_best:
            state.best_metric = val_metrics["loss"]
            state.best_epoch = epoch + 1
            state.best_val_metrics = dict(val_metrics)
            state.best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"[condition] nuovo BEST: val_loss={val_metrics['loss']:.6f} (epoca {epoch + 1})")

        update_early_stopping(cfg, state, val_metrics["loss"])

        if cond.save_checkpoints:
            extra = _checkpoint_extra(cfg, state, data, train_metrics, val_metrics)
            _save_run_checkpoint(outputs, state, "last.pt", model, optimizer, scheduler, epoch, extra)
            if is_best:
                _save_run_checkpoint(outputs, state, "best.pt", model, optimizer, scheduler, epoch, extra)

        if cond.examples_every and (epoch + 1) % cond.examples_every == 0:
            # one folder per epoch: the W&B tracker shows as images the PNGs of epoch_XXX/
            epoch_dir = os.path.join(outputs.examples, f"epoch_{epoch + 1:03d}")
            os.makedirs(epoch_dir, exist_ok=True)
            save_examples(cfg, result, examples, os.path.join(epoch_dir, "val.png"),
                          title=f"validation, epoch {epoch + 1}")

        update_figures(outputs, state)
        write_training_info(cfg, outputs, state, optimizer)
        rt.tracker.sync(state.global_step)

        if state.stopped_early:
            logger.info(f"[condition] early stopping all'epoca {epoch + 1}: "
                        f"{state.epochs_without_improvement} epoche senza miglioramenti di val_loss "
                        f"oltre min_delta={cond.early_stopping_min_delta}")
            break

    for line in state.metrics.summary_lines():
        logger.info(line)
    logger.info(f"[condition] training terminato, storia in {state.history_csv}")
    return state


def final_validation(cfg, rt, outputs, state, model, data: ConditionData):
    """
    The best weights on the validation set, next to the baselines:
    results/statistics/metrics_val.json, results/curves/rmse_by_lead.png and
    results/forecasts/final.png. The test splits are evaluated separately, once,
    with src/scripts/evaluate_condition.py.
    """
    if state.best_state_dict is not None:
        model.load_state_dict(state.best_state_dict)
    result = evaluate_series(cfg, rt, model, data.raw["val"], data.lo, data.hi)
    statistics = os.path.join(outputs.results, "statistics")
    os.makedirs(statistics, exist_ok=True)
    write_split_statistics(os.path.join(statistics, "metrics_val.json"), "val", "best", result)
    plot_condition_lead_rmse(lead_rmse_curves(result),
                             os.path.join(outputs.curves, "rmse_by_lead.png"),
                             title=f"weather forecast error by lead time, validation (epoch {state.best_epoch})")
    save_examples(cfg, result, example_indices(result["n_cubes"], cfg.condition.n_examples),
                  os.path.join(outputs.examples, "final.png"),
                  title=f"validation, best epoch {state.best_epoch}")
    model_rmse = result["model"]["rmse"]
    for name, metrics in result["baselines"].items():
        verdict = "beats" if model_rmse < metrics["rmse"] else "does NOT beat"
        logger.info(f"[condition] val rmse: model {model_rmse:.6f} {verdict} {name} {metrics['rmse']:.6f}")
    rt.tracker.summary({f"final/val_{k}": v for k, v in scalar_metrics(result["model"]).items()})
    return result
