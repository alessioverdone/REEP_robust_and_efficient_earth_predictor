"""
Hyperparameter search with Optuna, built on top of the training scripts.

A sweep is a yaml in registry/configurations/ (schema: SweepCfg below). It
names the phase, the base config, the overrides fixed for every trial and the
search space, all as dotlist keys of src/config.py. Every trial is one run of
the phase script, launched as a subprocess with `--set` overrides: the training
code is the same one used for a single run and knows nothing about the sweep.

    registry/outputs/sweeps/<name>/
        sweep.yaml           the study definition (a resumed sweep must match it)
        study.db             the Optuna study (SQLite): rerun the sweep to resume it
        sweep.log            the log of the sweep
        trials.csv           one row per trial: state, value, sampled values, run folder
        best.json            the best completed trial
        trial_XXXX/
            console.log      stdout and stderr of the trial's process
            <timestamp>/     the run folder, as in registry/outputs/<phase>/

The value of a trial is read from the `metric` column of its history.csv: the
best value over the rows in train mode (the same epoch as checkpoints/best),
the last one in validate_only mode. While the trial runs, every new row is
reported to Optuna as the best value so far, at step = its epoch, and the
pruner may stop the trial: the sweep then sends SIGINT to the trial's process
group, the run closes as "interrupted" and Optuna records it as PRUNED.

Trial folders stay outside registry/outputs/<phase>/, so they are never picked
by `latest` (diffusion.first_stage_config.pretrained_path, run.resume_from_checkpoint).
A diffusion sweep resolves `pretrained_path: latest` once, at the creation of
the study, and every trial (also after a resume) uses that VAE.
"""

import csv
import functools
import json
import logging
import os
import signal
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional

import optuna
from accelerate import PartialState
from omegaconf import OmegaConf

from src.config import PROJECT_DIR, TrainCfg, load_config, project_path, resolve_config_path, use_project_dir
from src.dataset.dataset_info import load_dataset_info
from src.dataset.meso_series import ensure_series
from src.model.diffusion_utils import resolve_first_stage_path
from src.training.run_outputs import relative_to_project


# ================================================ SCHEMA ================================================

@dataclass
class ParamCfg:
    """One dimension of the search space."""
    type: str = "float"                                 # float | int | categorical
    low: Optional[float] = None                         # float, int
    high: Optional[float] = None
    step: Optional[float] = None                        # null = continuous (float) or 1 (int)
    log: bool = False                                   # sample on a log scale (not with step)
    # categorical: a list of values, or {name: value} for values Optuna cannot
    # store, such as lists (block_out_channels): Optuna then samples the name
    choices: Any = None

    TYPES: ClassVar[tuple[str, ...]] = ("float", "int", "categorical")


@dataclass
class PruningCfg:
    enabled: bool = True
    n_startup_trials: int = 5                           # trials completed before any pruning
    n_warmup_steps: int = 0                             # epochs of a trial before it can be pruned
    interval_steps: int = 1


@dataclass
class SweepCfg:
    name: str = ""                                      # the study, and its folder under outputs_dir
    phase: str = "diffusion"                            # autoencoder | condition | diffusion
    # train: every trial trains a model. validate_only (diffusion): every trial
    # validates the weights of `checkpoint`, for parameters that need no
    # training (sampler, guider)
    mode: str = "train"
    base_config: list[str] = field(default_factory=lambda: ["configuration_0.yaml"])
    checkpoint: Optional[str] = None                    # validate_only: a checkpoint folder (e.g. <run>/checkpoints/best)
    metric: str = "val_latter_rmse"                     # a column of history.csv
    direction: str = "minimize"                         # minimize | maximize
    # total of the study, earlier launches included (trials stopped by Ctrl-C do not count)
    n_trials: int = 20
    sampler: str = "tpe"                                # tpe | random
    seed: int = 0                                       # seed of the sampler, not of the trainings (run.seed)
    n_startup_trials: int = 10                          # tpe: random trials before the model kicks in
    multivariate: bool = True                           # tpe: model the parameters jointly
    outputs_dir: str = "registry/outputs/sweeps"
    # command before the script; null = python for the autoencoder and the
    # condition predictor, `python -m accelerate.commands.launch` (accelerate launch) for diffusion
    launcher: Optional[list[str]] = None
    poll_seconds: int = 30                              # how often history.csv is read
    stop_timeout_seconds: int = 600                     # after SIGINT, wait this long before SIGKILL
    pruning: PruningCfg = field(default_factory=PruningCfg)
    fixed: dict[str, Any] = field(default_factory=dict)       # dotlist key -> value, for every trial
    params: dict[str, ParamCfg] = field(default_factory=dict)  # dotlist key -> search space

    PHASES: ClassVar[tuple[str, ...]] = ("autoencoder", "condition", "diffusion")
    MODES: ClassVar[tuple[str, ...]] = ("train", "validate_only")
    DIRECTIONS: ClassVar[tuple[str, ...]] = ("minimize", "maximize")
    SAMPLERS: ClassVar[tuple[str, ...]] = ("tpe", "random")
    SCRIPTS: ClassVar[dict[str, str]] = {"autoencoder": "src/scripts/run_autoencoder_train.py",
                                         "condition": "src/scripts/run_condition_train.py",
                                         "diffusion": "src/scripts/run_diffusion_train.py"}
    # keys the sweep sets itself for every trial
    MANAGED_KEYS: ClassVar[tuple[str, ...]] = ("run.project_name", "run.cfgdir", "run.set",
                                               "run.validate_only", "run.resume_from_checkpoint",
                                               "autoencoder.outputs_dir", "condition.outputs_dir",
                                               "diffusion.outputs_dir", "wandb.tags")
    # fields that may change between two launches of the same study
    RESUMABLE_FIELDS: ClassVar[tuple[str, ...]] = ("n_trials", "launcher", "poll_seconds",
                                                   "stop_timeout_seconds")


class TrialFailed(Exception):
    """A trial that could not produce its metric: Optuna records it as FAIL and the sweep goes on."""


def trial_failed(trial: optuna.Trial,
                 reason: str) -> TrialFailed:
    """The exception to raise, with the first line of the reason stored in the trial (trials.csv)."""
    trial.set_user_attr("fail_reason",
                        reason.splitlines()[0])
    return TrialFailed(reason)


# ================================================ CONFIG ================================================

def validate_sweep(sweep: SweepCfg):
    """Allowed values and consistency of the sweep, collecting every error before raising."""
    errors = []
    if not sweep.name:
        errors.append("name: empty, it names the study and its folder")
    for key, allowed in (("phase", SweepCfg.PHASES),
                         ("mode", SweepCfg.MODES),
                         ("direction", SweepCfg.DIRECTIONS),
                         ("sampler", SweepCfg.SAMPLERS)):
        if sweep[key] not in allowed:
            errors.append(f"{key}: {sweep[key]!r}, allowed {allowed}")
    if sweep.mode == "validate_only":
        if sweep.phase != "diffusion":
            errors.append("mode validate_only exists only for phase diffusion")
        if not sweep.checkpoint or sweep.checkpoint == "latest":
            errors.append("mode validate_only needs `checkpoint`, the path of a checkpoint folder "
                          "('latest' would look into the trial folder, which is empty)")
    elif sweep.checkpoint is not None:
        errors.append("checkpoint: only for mode validate_only")
    if not sweep.metric.startswith("val_"):
        errors.append(f"metric: {sweep.metric!r}, the selection must use a val_ column of history.csv")
    if not sweep.params:
        errors.append("params: empty search space")

    for key in [*sweep.fixed, *sweep.params]:
        if any(key == managed or key.startswith(f"{managed}.") for managed in SweepCfg.MANAGED_KEYS):
            errors.append(f"{key}: set by the sweep for every trial, it cannot be fixed or searched")
    for key in sweep.params:
        if key in sweep.fixed:
            errors.append(f"{key}: both in fixed and in params")
        if key.startswith("preprocessing."):
            errors.append(f"{key}: the preprocessed dataset must already exist, fix "
                          f"preprocessing.* in `fixed` and run src/dataset/preprocessing.py first")
        if sweep.phase == "diffusion" and key.startswith("autoencoder."):
            errors.append(f"{key}: a diffusion sweep uses one VAE for every trial, its "
                          f"architecture cannot change")
        if sweep.phase == "condition" and (key.startswith("data.") or key == "condition.meso_agg"):
            errors.append(f"{key}: a condition sweep reads the weather series extracted once "
                          f"for every trial, how they are built cannot change")

    for key, param in sweep.params.items():
        if param.type not in ParamCfg.TYPES:
            errors.append(f"params.{key}.type: {param.type!r}, allowed {ParamCfg.TYPES}")
        elif param.type == "categorical":
            if not param.choices:
                errors.append(f"params.{key}: categorical without choices")
        elif param.low is None or param.high is None or param.low >= param.high:
            errors.append(f"params.{key}: needs low < high")
        elif param.log and (param.step is not None or param.low <= 0):
            errors.append(f"params.{key}: log scale needs low > 0 and no step")

    if errors:
        raise ValueError("Invalid sweep:\n  - " + "\n  - ".join(errors))


def load_sweep(path: Path) -> SweepCfg:
    """The sweep yaml on the SweepCfg schema (keys and types checked), validated."""
    sweep = OmegaConf.merge(OmegaConf.structured(SweepCfg),
                            OmegaConf.load(path))
    validate_sweep(sweep)
    return sweep


def check_study_definition(sweep: SweepCfg,
                           study_dir: Path):
    """
    Saves the sweep as <study_dir>/sweep.yaml at the first launch. A later
    launch of the same study must define it the same way (only the fields in
    RESUMABLE_FIELDS may change): otherwise the trials of one study would come
    from different search spaces or base configs.
    """
    path = study_dir / "sweep.yaml"
    current = OmegaConf.to_container(sweep,
                                     resolve=True)
    if path.is_file():
        saved = OmegaConf.to_container(OmegaConf.load(path),
                                       resolve=True)
        changed = sorted(key for key in set(current) | set(saved)
                         if key not in SweepCfg.RESUMABLE_FIELDS and current.get(key) != saved.get(key))
        if changed:
            raise SystemExit(f"sweep {sweep.name!r} already exists in {study_dir} with a different "
                             f"definition ({', '.join(changed)}). Give the new sweep another name, "
                             f"or restore {path}.")
    OmegaConf.save(sweep,
                   path)


def format_override(key: str,
                    value: Any) -> str:
    """A `--set` item: the value as JSON, which the dotlist parser reads as yaml."""
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value,
                                       resolve=True)
    return f"{key}={json.dumps(value)}"


def base_config_paths(sweep: SweepCfg) -> list[str]:
    return [str(resolve_config_path(path).resolve()) for path in sweep.base_config]


def check_inputs(sweep: SweepCfg,
                 cfg: TrainCfg) -> dict:
    """
    The checkpoint of validate_only and the preprocessed dataset the trials
    read must already exist. For the condition phase the weather series are
    extracted here, once, if missing: never inside a trial. Returns the dataset.json.
    """
    if sweep.mode == "validate_only" and not project_path(sweep.checkpoint).is_dir():
        raise SystemExit(f"checkpoint: {sweep.checkpoint} is not a folder (e.g. "
                         f"registry/outputs/diffusion/<run>/checkpoints/best)")
    dataset_json = (cfg.autoencoder.dataset_json if sweep.phase == "autoencoder"
                    else cfg.data.params.dataset_json)
    dataset = load_dataset_info(dataset_json)
    if sweep.phase == "condition":
        ensure_series(cfg)
    return dataset


def pin_first_stage(sweep: SweepCfg,
                    cfg: TrainCfg,
                    study: optuna.Study) -> Optional[str]:
    """
    The VAE of a diffusion sweep. `latest` is resolved at the creation of the
    study and stored in it, so a resumed sweep keeps the same VAE even if a
    newer autoencoder run finished in between. None for the autoencoder phase
    and for an explicit pretrained_path, which the trials already share.
    """
    if sweep.phase != "diffusion" or cfg.diffusion.first_stage_config.pretrained_path != "latest":
        return None
    path = study.user_attrs.get("first_stage_pretrained_path")
    if path is None:
        # resolve_first_stage_path logs through the accelerate logger, which
        # needs an accelerate state: a single-process one, nothing is launched
        PartialState()
        path = str(Path(resolve_first_stage_path(cfg)).resolve())
        study.set_user_attr("first_stage_pretrained_path",
                            path)
    logging.getLogger(__name__).info(f"[sweep] VAE of every trial: {path}")
    return path


# ================================================ TRIAL ================================================

def suggest_value(trial: optuna.Trial,
                  key: str,
                  param: ParamCfg) -> Any:
    """One sampled value, as it goes into the config."""
    if param.type == "float":
        return trial.suggest_float(key,
                                   param.low,
                                   param.high,
                                   step=param.step,
                                   log=param.log)
    if param.type == "int":
        return trial.suggest_int(key,
                                 int(param.low),
                                 int(param.high),
                                 step=int(param.step or 1),
                                 log=param.log)
    choices = OmegaConf.to_container(param.choices,
                                     resolve=True)
    if isinstance(choices, dict):
        name = trial.suggest_categorical(key,
                                         list(choices))
        return choices[name]
    return trial.suggest_categorical(key,
                                     choices)


def trial_command(sweep: SweepCfg,
                  overrides: list[str]) -> list[str]:
    launcher = (list(sweep.launcher) if sweep.launcher is not None
                else [sys.executable] if sweep.phase in ("autoencoder", "condition")
                else [sys.executable, "-m", "accelerate.commands.launch"])
    command = [*launcher,
               SweepCfg.SCRIPTS[sweep.phase],
               "--project_name", sweep.name,
               "--cfgdir", *base_config_paths(sweep)]
    if sweep.mode == "validate_only":
        command += ["--validate_only",
                    "--resume_from_checkpoint", str(project_path(sweep.checkpoint).resolve())]
    return command + ["--set", *overrides]


def trial_environment() -> dict:
    """The environment of the sweep, with the project folders the scripts import from on PYTHONPATH."""
    env = dict(os.environ)
    paths = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    for required in (str(PROJECT_DIR / "src" / "model"), str(PROJECT_DIR)):
        if required not in [str(Path(p).resolve()) for p in paths]:
            paths.insert(0,
                         required)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def find_run_dir(trial_dir: Path) -> Optional[Path]:
    """The run folder the trial's script created inside trial_dir (the only subfolder)."""
    runs = sorted(p for p in trial_dir.iterdir() if p.is_dir())
    return runs[-1] if runs else None


def read_metric(history_csv: Path,
                metric: str) -> list[tuple[int, float]]:
    """
    (epoch, value) of the rows of history.csv that have `metric`. A last line
    without newline is still being written and is left for the next read.
    """
    if not history_csv.is_file():
        return []
    lines = history_csv.read_text(encoding="utf-8").splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines = lines[:-1]
    reader = csv.DictReader(lines)
    rows = list(reader)
    if reader.fieldnames and metric not in reader.fieldnames:
        raise TrialFailed(f"metric {metric!r} is not a column of {history_csv}: "
                          f"{', '.join(c for c in reader.fieldnames if c.startswith('val_'))}")
    return [(int(row["epoch"]), float(row[metric])) for row in rows if row.get(metric)]


def best_value(values: list[float],
               direction: str) -> float:
    return min(values) if direction == "minimize" else max(values)


def stop_process(process: subprocess.Popen,
                 timeout: int):
    """
    SIGINT to the trial's process group (the launcher and every worker): the
    run closes as "interrupted", as with Ctrl-C. SIGKILL if it does not exit
    within `timeout` seconds.
    """
    if process.poll() is not None:
        return
    os.killpg(process.pid,
              signal.SIGINT)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logging.getLogger(__name__).warning(f"[sweep] pid {process.pid} still alive {timeout}s "
                                            f"after SIGINT, SIGKILL")
        os.killpg(process.pid,
                  signal.SIGKILL)
        process.wait()


def console_tail(path: Path,
                 n_lines: int = 30) -> str:
    lines = path.read_text(encoding="utf-8",
                           errors="replace").splitlines()
    return "\n".join(lines[-n_lines:])


def run_trial(sweep: SweepCfg,
              study_dir: Path,
              base_overrides: list[str],
              base_tags: list[str],
              trial: optuna.Trial) -> float:
    """The Optuna objective: one run of the phase script, its metric read from history.csv."""
    logger = logging.getLogger(__name__)
    values = {}
    for key, param in sweep.params.items():
        values[key] = suggest_value(trial,
                                    key,
                                    param)
    trial_dir = study_dir / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True,
                    exist_ok=True)
    managed = {f"{sweep.phase}.outputs_dir": str(trial_dir),
               "wandb.tags": [*base_tags, "sweep", sweep.name, f"trial_{trial.number:04d}"]}
    overrides = [*base_overrides,
                 *(format_override(key,
                                   value) for key, value in {**values, **managed}.items())]
    trial.set_user_attr("values",
                        values)
    trial.set_user_attr("trial_dir",
                        relative_to_project(trial_dir))

    # the same validation the run would do at start-up, without launching it
    try:
        load_config(*base_config_paths(sweep),
                    overrides=overrides)
    except (Exception, SystemExit) as error:
        raise trial_failed(trial,
                           f"invalid config: {error}") from error

    command = trial_command(sweep,
                            overrides)
    logger.info(f"[sweep] trial {trial.number}: {json.dumps(values)}")
    console = trial_dir / "console.log"
    reported = set()
    with open(console, "w", encoding="utf-8") as log_file:
        log_file.write(" ".join(command) + "\n\n")
        log_file.flush()
        process = subprocess.Popen(command,
                                   cwd=PROJECT_DIR,
                                   env=trial_environment(),
                                   stdout=log_file,
                                   stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            run_dir = None
            while True:
                try:
                    process.wait(timeout=sweep.poll_seconds)
                    finished = True
                except subprocess.TimeoutExpired:
                    finished = False
                run_dir = run_dir or find_run_dir(trial_dir)
                if run_dir is not None and trial.user_attrs.get("run_dir") is None:
                    trial.set_user_attr("run_dir",
                                        relative_to_project(run_dir))
                try:
                    rows = (read_metric(run_dir / "artifacts" / "history.csv",
                                        sweep.metric) if run_dir is not None else [])
                except TrialFailed as error:
                    raise trial_failed(trial,
                                       str(error)) from error
                for i, (epoch, _) in enumerate(rows):
                    if epoch not in reported:
                        trial.report(best_value([v for _, v in rows[:i + 1]],
                                                sweep.direction),
                                     step=epoch)
                        reported.add(epoch)
                if finished:
                    break
                if trial.should_prune():
                    logger.info(f"[sweep] trial {trial.number} pruned at epoch {max(reported)}")
                    raise optuna.TrialPruned()
        except KeyboardInterrupt:
            logger.info(f"[sweep] interrupted: stopping trial {trial.number}")
            trial.set_user_attr("fail_reason",
                                "interrupted by the user")
            raise
        finally:
            # pruned, failed metric, Ctrl-C or an error of the sweep: the
            # trial never outlives this function (no-op once it has exited)
            stop_process(process,
                         sweep.stop_timeout_seconds)

    if process.returncode != 0:
        raise trial_failed(trial,
                           f"exit code {process.returncode}, end of {console}:\n"
                           f"{console_tail(console)}")
    if not rows:
        raise trial_failed(trial,
                           f"no {sweep.metric} in the history.csv of {run_dir}")
    values_found = [v for _, v in rows]
    return values_found[-1] if sweep.mode == "validate_only" else best_value(values_found,
                                                                             sweep.direction)


# ================================================ RESULTS ================================================

def write_results(sweep: SweepCfg,
                  study_dir: Path,
                  study: optuna.Study,
                  trial: Optional[optuna.trial.FrozenTrial] = None):
    """
    trials.csv and best.json, rewritten after every trial (it is also an
    Optuna callback, hence the unused `trial`).
    """
    rows = []
    for frozen in study.trials:
        rows.append({"number": frozen.number,
                     "state": frozen.state.name,
                     sweep.metric: frozen.value,
                     "last_epoch": frozen.last_step,
                     "duration_s": (round(frozen.duration.total_seconds()) if frozen.duration else None),
                     **{key: json.dumps(value) for key, value in frozen.user_attrs.get("values", {}).items()},
                     "run_dir": frozen.user_attrs.get("run_dir"),
                     "fail_reason": frozen.user_attrs.get("fail_reason")})
    if rows:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with open(study_dir / "trials.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh,
                                    fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        best = study.best_trial
        summary = {"study": sweep.name,
                   "metric": sweep.metric,
                   "direction": sweep.direction,
                   "best_trial": best.number,
                   "value": best.value,
                   "values": best.user_attrs.get("values"),
                   "run_dir": best.user_attrs.get("run_dir"),
                   "trials": {state.name: sum(t.state == state for t in study.trials)
                              for state in optuna.trial.TrialState}}
        with open(study_dir / "best.json", "w", encoding="utf-8") as fh:
            json.dump(summary,
                      fh,
                      indent=2)


def counts_toward_budget(trial: optuna.trial.FrozenTrial) -> bool:
    """
    Completed, pruned and failed trials use up n_trials. Not the ones stopped
    by Ctrl-C, nor the ones the heartbeat marked FAIL after the sweep itself
    was killed (no fail_reason): a resumed sweep runs them again.
    """
    if trial.state in (optuna.trial.TrialState.COMPLETE,
                       optuna.trial.TrialState.PRUNED):
        return True
    reason = trial.user_attrs.get("fail_reason")
    return trial.state == optuna.trial.TrialState.FAIL and reason not in (None, "interrupted by the user")


def setup_logging(study_dir: Path):
    """Console and <study_dir>/sweep.log, Optuna's own messages included."""
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(),
                    logging.FileHandler(study_dir / "sweep.log",
                                        encoding="utf-8")):
        handler.setFormatter(formatter)
        root.addHandler(handler)
    optuna.logging.disable_default_handler()
    optuna.logging.enable_propagation()


def make_sampler(sweep: SweepCfg) -> optuna.samplers.BaseSampler:
    if sweep.sampler == "random":
        return optuna.samplers.RandomSampler(seed=sweep.seed)
    return optuna.samplers.TPESampler(seed=sweep.seed,
                                      n_startup_trials=sweep.n_startup_trials,
                                      multivariate=sweep.multivariate)


def make_pruner(sweep: SweepCfg) -> optuna.pruners.BasePruner:
    # one validation per trial in validate_only: nothing to compare along the way
    if not sweep.pruning.enabled or sweep.mode == "validate_only":
        return optuna.pruners.NopPruner()
    return optuna.pruners.MedianPruner(n_startup_trials=sweep.pruning.n_startup_trials,
                                       n_warmup_steps=sweep.pruning.n_warmup_steps,
                                       interval_steps=sweep.pruning.interval_steps)


def run_sweep(sweep_file: str):
    """Creates or resumes the study of `sweep_file` and runs its missing trials, one at a time."""
    use_project_dir()
    sweep = load_sweep(resolve_config_path(sweep_file))
    # base config and inputs first: a launch that fails here leaves no study behind
    base_overrides = [format_override(key,
                                      value) for key, value in sweep.fixed.items()]
    cfg = load_config(*base_config_paths(sweep),
                      overrides=base_overrides)
    dataset = check_inputs(sweep,
                           cfg)

    study_dir = project_path(sweep.outputs_dir) / sweep.name
    study_dir.mkdir(parents=True,
                    exist_ok=True)
    check_study_definition(sweep,
                           study_dir)
    setup_logging(study_dir)
    logger = logging.getLogger(__name__)
    logger.info(f"[sweep] dataset {dataset['name']} (id {dataset['id']}) in {cfg.paths.preprocessed_dir}")
    # multivariate TPE and the heartbeat are flagged experimental by Optuna
    warnings.filterwarnings("ignore",
                            category=optuna.exceptions.ExperimentalWarning)

    # the heartbeat marks as FAIL the trials left RUNNING by a sweep that was killed
    storage = optuna.storages.RDBStorage(f"sqlite:///{study_dir / 'study.db'}",
                                         heartbeat_interval=60,
                                         grace_period=180)
    study = optuna.create_study(storage=storage,
                                sampler=make_sampler(sweep),
                                pruner=make_pruner(sweep),
                                study_name=sweep.name,
                                direction=sweep.direction,
                                load_if_exists=True)
    first_stage_path = pin_first_stage(sweep,
                                       cfg,
                                       study)
    if first_stage_path is not None:
        base_overrides.append(format_override("diffusion.first_stage_config.pretrained_path",
                                              first_stage_path))

    done = sum(counts_toward_budget(t) for t in study.trials)
    remaining = sweep.n_trials - done
    logger.info(f"[sweep] {sweep.name}: {sweep.phase}/{sweep.mode}, {sweep.direction} {sweep.metric}, "
                f"{done} trials done, {max(remaining, 0)} to run, folder {study_dir}")
    callback = functools.partial(write_results,
                                 sweep,
                                 study_dir)
    try:
        if remaining > 0:
            study.optimize(functools.partial(run_trial,
                                             sweep,
                                             study_dir,
                                             base_overrides,
                                             list(cfg.wandb.tags)),
                           n_trials=remaining,
                           catch=(TrialFailed,),
                           callbacks=[callback])
    finally:
        write_results(sweep,
                      study_dir,
                      study)

    if (study_dir / "best.json").is_file():
        best = study.best_trial
        logger.info(f"[sweep] best trial {best.number}: {sweep.metric}={best.value:.6f} "
                    f"{json.dumps(best.user_attrs.get('values'))} -> {best.user_attrs.get('run_dir')}")
    else:
        logger.info("[sweep] no completed trial yet")
    logger.info(f"[sweep] results in {study_dir / 'trials.csv'}")
