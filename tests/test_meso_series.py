"""
Tests of src/dataset/meso_series.py on the synthetic EarthNet dataset of
tests/earthnet_fixtures.py, preprocessed by the REEP preprocessing (fixture
`earthnet` in tests/conftest.py).
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import CONFIGS_DIR, load_config
from src.dataset import meso_series
from src.dataset.dataloader_s2_dif import aggregate_meso, load_meso_dynamic
from tests.earthnet_fixtures import expected_series, weather


def make_cfg(earthnet, *extra):
    return load_config(CONFIGS_DIR / "configuration_0.yaml", overrides=[*earthnet["overrides"], *extra])


@pytest.fixture(scope="module")
def extracted(earthnet):
    cfg = make_cfg(earthnet)
    return cfg, meso_series.build_series(cfg, overwrite=True)


def test_every_split_is_extracted(extracted):
    _, info = extracted
    assert {split: entry["cubes"] for split, entry in info["splits"].items()} == {
        "train": 4, "val": 2, "iid": 3, "ood": 3}
    assert all(entry["left_out"] == 0 for entry in info["splits"].values())


def test_series_follow_the_definition(extracted):
    """Mean of the 2x2 central cells, then mean of the 5 days of every frame, per variable."""
    cfg, _ = extracted
    raw_dir = Path(cfg.data.params.data_root)
    for split in ("train", "val"):
        series, cubes = meso_series.load_split(cfg, split)
        assert series.shape == (len(cubes), 30, 5) and series.dtype == np.float32
        for values, name in zip(series, cubes):
            with np.load(raw_dir / name) as z:
                np.testing.assert_allclose(values, expected_series(z["mesodynamic"], 30), rtol=1e-5)


def test_test_splits_join_context_and_target(extracted):
    """
    iid has all the 150 days in the context file, ood 50 in the context and 100
    in the target: with the same weather the two splits give the same series.
    """
    cfg, _ = extracted
    iid, _ = meso_series.load_split(cfg, "iid")
    ood, _ = meso_series.load_split(cfg, "ood")
    np.testing.assert_allclose(iid, ood)
    for c, values in enumerate(iid):
        np.testing.assert_allclose(values, expected_series(weather(150, 100 + c), 30), rtol=1e-5)


def test_aggregate_meso_is_the_computation_of_load_meso_dynamic(earthnet):
    path = next((earthnet["root"] / "raw" / "train").rglob("*.npz"))
    with np.load(path) as z:
        md = z["mesodynamic"]
    for agg in ("mean", "last", "first", "sum"):
        np.testing.assert_array_equal(aggregate_meso(md, agg=agg), load_meso_dynamic(path, agg=agg))
        np.testing.assert_allclose(aggregate_meso(md, agg=agg), expected_series(md, 30, agg=agg), rtol=1e-5)


def test_stats_are_the_min_max_of_the_train_series(extracted):
    cfg, _ = extracted
    lo, hi = meso_series.load_stats(cfg)
    train, _ = meso_series.load_split(cfg, "train")
    np.testing.assert_allclose(lo, train.min(axis=(0, 1)))
    np.testing.assert_allclose(hi, train.max(axis=(0, 1)))


def test_stats_equal_cond_stats_with_the_aggregation_of_the_dit(earthnet):
    """With the aggregation of the DiT (data.params.meso_agg = last) the min / max are those of cond_stats.json."""
    cfg = make_cfg(earthnet, "condition.meso_agg=last")
    meso_series.build_series(cfg, overwrite=True)
    lo, hi = meso_series.load_stats(cfg)
    cond_stats = json.loads(Path(cfg.data.params.cond_stats_json).read_text())
    assert cond_stats["meso_agg"] == "last"
    np.testing.assert_allclose(lo, cond_stats["meso"]["lo"], rtol=1e-6)
    np.testing.assert_allclose(hi, cond_stats["meso"]["hi"], rtol=1e-6)


def test_normalization_round_trip(extracted):
    cfg, _ = extracted
    lo, hi = meso_series.load_stats(cfg)
    val, _ = meso_series.load_split(cfg, "val")
    norm = meso_series.normalize(val, lo, hi)
    assert norm.min() >= -1.0 and norm.max() <= 1.0
    inside = (val >= lo) & (val <= hi)          # outside the train range the normalization clips
    np.testing.assert_allclose(meso_series.denormalize(norm, lo, hi)[inside], val[inside], rtol=1e-5, atol=1e-4)


def test_dataset_pairs_context_and_target(extracted):
    cfg, _ = extracted
    train, _ = meso_series.load_split(cfg, "train")
    context, target = meso_series.MesoSeriesDataset(train, 10, 20)[0]
    assert context.shape == (10, 5) and target.shape == (20, 5)
    assert torch.equal(context, torch.from_numpy(train[0, :10]))
    assert torch.equal(target, torch.from_numpy(train[0, 10:]))


def test_cubes_with_missing_weather_are_left_out():
    complete = np.zeros((30, 5), dtype=np.float32)
    missing = complete.copy()
    missing[3, 0] = np.nan
    names, kept, left_out = meso_series._collect(["a", "b", "c"],
                                                 [(complete, None), (missing, None), (None, "unreadable")])
    assert names == ["a"] and len(kept) == 1 and set(left_out) == {"b", "c"}


def test_existing_series_are_not_overwritten(extracted):
    cfg, _ = extracted
    with pytest.raises(SystemExit, match="overwrite"):
        meso_series.build_series(cfg)


def test_series_built_otherwise_are_refused(extracted):
    cfg, _ = extracted
    path = meso_series.series_dir(cfg) / "series.json"
    original = path.read_text()
    changed = json.loads(original)
    changed["parameters"]["center_crop"] = 4
    path.write_text(json.dumps(changed))
    try:
        with pytest.raises(SystemExit, match="center_crop"):
            meso_series.ensure_series(cfg)
    finally:
        path.write_text(original)
    assert meso_series.ensure_series(cfg)["parameters"] == meso_series.series_parameters(cfg)


def test_a_frame_without_days_is_missing_whatever_the_aggregation():
    """With 'sum' (nansum) a frame with all its days missing would become 0: it must stay missing."""
    md = weather(150, seed=0)
    md[:, :, 1, 10:15] = np.nan                      # sea pressure: all the days of frame 2
    for agg in ("mean", "last", "first", "sum"):
        meso_series._init_worker({"n_channels": 5, "T": 30, "steps_per_frame": 5,
                                  "center_crop": 2, "meso_agg": agg})
        series = meso_series._reduce(md, 30, "test")
        assert np.isnan(series[2, 1]), agg
        assert np.isfinite(np.delete(series.reshape(-1), 2 * 5 + 1)).all(), agg
