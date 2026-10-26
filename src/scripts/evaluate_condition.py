"""
Evaluation of a trained Condition Predictor on the validation set and on the
EarthNet test splits, next to the baselines (persistence, context mean).

    python src/scripts/evaluate_condition.py --model latest --split iid ood
    python src/scripts/evaluate_condition.py --model registry/outputs/condition/<run> --split val

--model is a run folder, a piece of its name, or `latest` (the most recent
completed condition run). The weights are checkpoints/best.pt; the series are
the ones extracted by src/dataset/meso_series.py with the run's own
configuration (the test splits are extracted when data/raw/<split>_test_split
exists). For every split it writes into <run>/results/statistics/:
    metrics_<split>.json     model and baselines: errors normalized and per variable
                             in the units of the cubes, RMSE by lead time
    rmse_by_lead_<split>.png
    forecasts_<split>.png    forecast examples

Only the splits with the frames the model was trained on (iid, ood: 10 + 20)
are evaluated; extreme and seasonal have longer horizons and need a rollout.
The test splits are for the final number of the chosen model: the choices
(architecture, sizes) are made on the validation set.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    # launched as a file (or Run from the IDE): the project root goes on the path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import logging

import numpy as np
import torch

from src.config import load_config, load_saved_config, project_path, resolve_config_path, use_project_dir
from src.dataset.meso_series import check_series_fit, load_split
from src.model.condition_utils import completed_runs, load_condition_checkpoint
from src.training.train_condition import (evaluate_series, example_indices, lead_rmse_curves,
                                          save_examples, write_split_statistics)
from src.utils.plots import plot_condition_lead_rmse

logger = logging.getLogger(__name__)


def find_run(model: str, outputs_dir: str) -> Path:
    """The run folder of --model: a path, `latest`, or a piece of a run name under outputs_dir."""
    path = project_path(model)
    if path.is_dir():
        return path
    runs = [run_dir for run_dir, _ in completed_runs(outputs_dir)]
    if model == "latest":
        if not runs:
            raise SystemExit(f"no completed condition run in {outputs_dir}")
        return runs[0]
    matches = [r for r in runs if model in r.name]
    if len(matches) != 1:
        raise SystemExit(f"--model {model!r}: {len(matches)} completed runs in {outputs_dir} match, "
                         f"one is needed ({[r.name for r in matches]})")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="latest",
                        help="run folder, piece of its name, or latest (default)")
    parser.add_argument("--split", nargs="+", default=["iid", "ood"],
                        help="splits to evaluate: val, iid, ood (default: iid ood)")
    parser.add_argument("--cfgdir", default="configuration_0.yaml",
                        help="config used only to find the runs (condition.outputs_dir)")
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        level=logging.INFO)
    config_path = resolve_config_path(args.cfgdir).resolve()
    use_project_dir()
    run_dir = find_run(args.model, load_config(config_path).condition.outputs_dir)
    cfg = load_saved_config(run_dir / "artifacts" / "config.yaml")
    cond = cfg.condition

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_condition_checkpoint(str(run_dir / "checkpoints" / "best.pt"), device)
    # the series on disk must still be the ones of the training (dataset, reduction)
    check_series_fit(cfg, ckpt)
    lo = np.asarray(ckpt["series"]["lo"], dtype=np.float32)
    hi = np.asarray(ckpt["series"]["hi"], dtype=np.float32)
    rt = argparse.Namespace(device=device)
    logger.info(f"[evaluate] {run_dir.name}: {ckpt['arch']['name']}, epoch {ckpt.get('best_epoch')}")

    statistics = run_dir / "results" / "statistics"
    statistics.mkdir(parents=True, exist_ok=True)
    for split in args.split:
        raw, _ = load_split(cfg, split)
        if raw.shape[1] != cond.T:
            logger.warning(f"[evaluate] {split}: {raw.shape[1]} frames, the model forecasts "
                           f"{cond.n_context} -> {cond.n_predict}: skipped (needs a rollout)")
            continue
        result = evaluate_series(cfg, rt, model, raw, lo, hi)
        write_split_statistics(str(statistics / f"metrics_{split}.json"), split, "best", result)
        plot_condition_lead_rmse(lead_rmse_curves(result), str(statistics / f"rmse_by_lead_{split}.png"),
                                 title=f"weather forecast error by lead time, {split}")
        save_examples(cfg, result, example_indices(result["n_cubes"], cond.n_examples),
                      str(statistics / f"forecasts_{split}.png"), title=split)
        line = " | ".join(f"{name} {m['rmse']:.6f}" for name, m in result["baselines"].items())
        logger.info(f"[evaluate] {split}: {result['n_cubes']} cubes, rmse model "
                    f"{result['model']['rmse']:.6f} | {line}")
    logger.info(f"[evaluate] results in {statistics}")


if __name__ == "__main__":
    main()
