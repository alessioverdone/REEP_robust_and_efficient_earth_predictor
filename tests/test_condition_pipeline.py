"""
End-to-end tests of the condition phase on the synthetic EarthNet dataset:
training (both architectures), checkpoint, evaluation on the test splits,
hyperparameter sweep, and the use of the trained model in phase 2.

The scripts run as separate processes, as they do for real.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.config import CONFIGS_DIR, load_config
from src.dataset.dataset_info import load_dataset_info
from src.model.condition_utils import load_condition_checkpoint, load_condition_predictor
from tests.conftest import run_script

DEFAULT_YAML = CONFIGS_DIR / "configuration_0.yaml"
EPOCHS = 3


def history(run: Path) -> list:
    with open(run / "artifacts" / "history.csv", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def runs(earthnet) -> dict:
    """One short training per architecture: {model name: run folder}."""
    outputs = earthnet["root"] / "outputs" / "condition"
    folders = {}
    for name in ("PatchMixer", "S_Mamba"):
        before = set(outputs.glob("*")) if outputs.is_dir() else set()
        run_script("src/scripts/run_condition_train.py", "--project_name", f"test_{name}",
                   "--set", *earthnet["overrides"], f"condition.model.name={name}",
                   f"condition.num_epochs={EPOCHS}", "condition.batch_size=2", "condition.examples_every=1")
        (folders[name],) = set(outputs.glob("*")) - before
    return folders


# ================================ TRAINING ================================

@pytest.mark.parametrize("name", ["PatchMixer", "S_Mamba"])
def test_run_completes_with_all_its_outputs(runs, name):
    run = runs[name]
    info = json.loads((run / "artifacts" / "training_info.json").read_text())
    assert info["run"]["status"] == "completed" and info["run"]["phase"] == "condition"
    assert info["model"]["arch"]["name"] == name
    assert info["training"]["epochs_completed"] == EPOCHS
    assert set(info["metrics"]["val_baselines"]) == {"persistence", "context_mean"}
    for path in ("artifacts/config.yaml", "artifacts/train.log", "checkpoints/best.pt", "checkpoints/last.pt",
                 "results/curves/loss.png", "results/curves/rmse_by_lead.png", "results/forecasts/final.png",
                 "results/forecasts/epoch_001/val.png", "results/statistics/metrics_val.json"):
        assert (run / path).is_file(), path
    rows = history(run)
    assert [int(r["epoch"]) for r in rows] == list(range(1, EPOCHS + 1))
    assert {"train_loss", "val_loss", "val_rmse", "val_rmse_precipitation", "val_mae_max_temperature"} <= set(rows[0])


@pytest.mark.parametrize("name", ["PatchMixer", "S_Mamba"])
def test_training_reduces_the_loss(runs, name):
    rows = history(runs[name])
    assert float(rows[-1]["train_loss"]) < float(rows[0]["train_loss"])


def test_checkpoint_is_self_contained(runs, earthnet, device):
    """best.pt rebuilds the model alone and carries the contract with phase 2."""
    run = runs["S_Mamba"]
    model, ckpt = load_condition_checkpoint(str(run / "checkpoints" / "best.pt"), device)
    cfg = load_config(DEFAULT_YAML, overrides=earthnet["overrides"])
    assert ckpt["arch"]["name"] == "S_Mamba"
    assert ckpt["series"]["meso_agg"] == "mean" and len(ckpt["series"]["lo"]) == 5
    assert ckpt["dataset"]["id"] == load_dataset_info(cfg.data.params.dataset_json)["id"]
    with torch.no_grad():
        assert model(torch.zeros(2, 10, 5, device=device)).shape == (2, 20, 5)
    # the final validation uses the best weights: the val_loss of the best epoch
    final = json.loads((run / "results" / "statistics" / "metrics_val.json").read_text())
    assert final["model"]["loss"] == pytest.approx(ckpt["val_metrics"]["loss"], rel=1e-4)


# =============================== EVALUATION ===============================

def test_evaluation_on_the_test_splits(runs):
    run = runs["S_Mamba"]
    run_script("src/scripts/evaluate_condition.py", "--model", run, "--split", "iid", "ood")
    statistics = run / "results" / "statistics"
    for split in ("iid", "ood"):
        metrics = json.loads((statistics / f"metrics_{split}.json").read_text())
        assert metrics["split"] == split and metrics["n_cubes"] == 3
        assert set(metrics["baselines"]) == {"persistence", "context_mean"}
        assert len(metrics["model"]["rmse_by_lead"]) == 20
        assert (statistics / f"rmse_by_lead_{split}.png").is_file()


# ================================= SWEEP =================================

def test_sweep_over_the_condition_phase(earthnet, tmp_path):
    fixed = {key: yaml.safe_load(value) for key, value in (o.split("=", 1) for o in earthnet["overrides"])}
    fixed.update({"condition.model.name": "PatchMixer", "condition.num_epochs": 2, "condition.batch_size": 2})
    sweep = {"name": "condition_test", "phase": "condition", "metric": "val_loss", "n_trials": 2,
             "sampler": "random", "outputs_dir": str(tmp_path), "poll_seconds": 1,
             "pruning": {"enabled": False}, "fixed": fixed,
             "params": {"condition.lr": {"type": "float", "low": 1e-4, "high": 3e-3, "log": True}}}
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(sweep))
    run_script("src/scripts/run_sweep.py", "--sweep", path)
    with open(tmp_path / "condition_test" / "trials.csv", newline="") as f:
        trials = list(csv.DictReader(f))
    assert [t["state"] for t in trials] == ["COMPLETE", "COMPLETE"]
    assert json.loads((tmp_path / "condition_test" / "best.json").read_text())["metric"] == "val_loss"


# ================================= PHASE 2 =================================

def test_phase_2_latest_finds_no_predictor_for_another_weather(runs, earthnet):
    """The DiT of configuration_0.yaml reads meso_agg=last, every run was trained on means."""
    cfg = load_config(DEFAULT_YAML, overrides=earthnet["overrides"])
    with pytest.raises(FileNotFoundError, match="meso_agg='mean'"):
        load_condition_predictor(cfg)


def test_phase_2_refuses_an_explicit_predictor_for_another_weather(runs, earthnet):
    best = runs["S_Mamba"] / "checkpoints" / "best.pt"
    cfg = load_config(DEFAULT_YAML, overrides=[*earthnet["overrides"], f"condition.pretrained_path={best}"])
    with pytest.raises(RuntimeError, match="meso_agg"):
        load_condition_predictor(cfg)


def test_latest_skips_a_newer_run_that_does_not_fit(runs, earthnet, tmp_path):
    """A newer completed run made with other cells must not hide the older one that fits."""
    outputs = tmp_path / "condition"
    older = outputs / "2026-01-01_00-00-00"
    newer = outputs / "2026-01-02_00-00-00"
    source = runs["S_Mamba"]
    for run, center_crop in ((older, 2), (newer, 4)):
        (run / "artifacts").mkdir(parents=True)
        (run / "checkpoints").mkdir()
        (run / "checkpoints" / "best.pt").write_bytes((source / "checkpoints" / "best.pt").read_bytes())
        info = json.loads((source / "artifacts" / "training_info.json").read_text())
        info["dataset"]["series"]["center_crop"] = center_crop
        (run / "artifacts" / "training_info.json").write_text(json.dumps(info))
    cfg = load_config(DEFAULT_YAML, overrides=[*earthnet["overrides"], "data.params.meso_agg=mean",
                                               f"condition.outputs_dir={outputs}"])
    from src.model.condition_utils import resolve_condition_path
    assert resolve_condition_path(cfg) == str(older / "checkpoints" / "best.pt")


def test_evaluation_refuses_series_extracted_otherwise(runs, earthnet):
    """Series re-extracted with other parameters or another dataset are not the ones of the model."""
    from src.dataset.meso_series import check_series_fit
    cfg = load_config(DEFAULT_YAML, overrides=earthnet["overrides"])
    _, ckpt = load_condition_checkpoint(str(runs["S_Mamba"] / "checkpoints" / "best.pt"), "cpu")
    check_series_fit(cfg, ckpt)                                   # the series of the training: accepted
    for key, value in (("center_crop", 4), ("meso_agg", "last")):
        changed = {**ckpt, "series": {**ckpt["series"], key: value}}
        with pytest.raises(SystemExit, match=key):
            check_series_fit(cfg, changed)
    with pytest.raises(SystemExit, match="dataset id"):
        check_series_fit(cfg, {**ckpt, "dataset": {"name": "other", "id": "000000000000"}})


def test_phase_2_loads_the_latest_predictor(runs, earthnet, device):
    cfg = load_config(DEFAULT_YAML, overrides=[*earthnet["overrides"], "data.params.meso_agg=mean"])
    predictor = load_condition_predictor(cfg, device)
    assert cfg.condition.pretrained_path == str(runs["S_Mamba"] / "checkpoints" / "best.pt")

    meso = torch.rand(4, 30, 5, device=device) * 2 - 1
    out = predictor(meso)
    assert out.shape == meso.shape
    assert torch.equal(out[:, :10], meso[:, :10])          # the observed context stays as it is
    assert out[:, 10:].abs().max() <= 1.0                  # the forecast stays in the normalized range
    assert not any(p.requires_grad for p in predictor.parameters())
    predictor.train()
    assert not predictor.model.training                    # frozen also inside a module in train mode


def test_baselines_use_the_observed_context(earthnet):
    """Persistence repeats the observed last frame, also outside the train range (not the clipped one)."""
    from argparse import Namespace
    from src.training.train_condition import evaluate_series

    class Zero(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 20, x.shape[2])

    cfg = load_config(DEFAULT_YAML, overrides=earthnet["overrides"])
    lo, hi = np.zeros(5, dtype=np.float32), np.ones(5, dtype=np.float32)
    raw = np.full((2, 30, 5), 0.5, dtype=np.float32)
    raw[:, 9] = 3.0                                  # last context frame, above the train max (1)
    result = evaluate_series(cfg, Namespace(device=torch.device("cpu")), Zero(), raw, lo, hi)
    # persistence forecasts 3.0 against an observed 0.5: error 2.5 in the units of the cubes
    assert result["baselines"]["persistence"]["mae_precipitation"] == pytest.approx(2.5)
    assert result["baselines"]["context_mean"]["mae_precipitation"] == pytest.approx(0.25)
