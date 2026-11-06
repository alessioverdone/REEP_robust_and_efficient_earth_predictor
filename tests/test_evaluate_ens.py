"""
Evaluation of the DiT on the test splits (src/scripts/evaluate_ens.py): the pieces that do
not need a trained model.

- the autoregressive windows and their chain of contexts
- the dataset that reads context and target files and joins them in memory
- the per-cube seed of the scoring, and the file names
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.conftest import PROJECT_DIR

sys.path.insert(0, str(PROJECT_DIR / "src" / "scripts"))
sys.path.insert(0, str(PROJECT_DIR / "src" / "model"))

import evaluate_ens as ens                     # noqa: E402
from src.dataset.dataloader_s2_dif import S2DiffusionCubeDataset   # noqa: E402
from src.dataset.test_cubes import EarthNetTestDataset, list_test_pairs, read_test_cube   # noqa: E402
from src.utils import ens_metrics as em        # noqa: E402
from tests.earthnet_fixtures import HW, make_earthnet   # noqa: E402


# =========================== autoregressive windows ===========================

@pytest.mark.parametrize("n_context, n_target, starts", [
    (20, 40, [10, 30]),                       # extreme: 2 blocks
    (70, 140, [60, 80, 100, 120, 140, 160, 180]),   # seasonal: 7 blocks
])
def test_blocks_cover_the_target_window(n_context, n_target, starts):
    plan = ens.autoregressive_plan(n_context, n_target, 10, 20)
    assert [plan.start(k) for k in range(plan.n_windows)] == starts
    # every block ends where the next predicted frames begin, and the last one at the end
    assert plan.start(plan.n_windows - 1) + plan.window_length == n_context + n_target
    # the context of block k is the end of the frames covered so far
    for k in range(plan.n_windows):
        assert plan.start(k) + plan.window_context == n_context + k * plan.window_predict


def test_blocks_need_a_whole_number_of_windows():
    with pytest.raises(SystemExit, match="multipli"):
        ens.autoregressive_plan(20, 45, 10, 20)
    with pytest.raises(SystemExit, match="contesto"):
        ens.autoregressive_plan(5, 40, 10, 20)


def test_window_item_slices_images_and_weather_together():
    plan = ens.autoregressive_plan(20, 40, 10, 20)
    item = {"imgs": torch.arange(60.0).view(60, 1, 1, 1), "meso_condition_image": torch.arange(60.0).view(60, 1),
            "highres_condition_image": torch.zeros(1, 2, 2)}
    window = ens.window_item(item, plan, 1)
    assert window["imgs"].shape[0] == window["meso_condition_image"].shape[0] == 30
    assert window["imgs"][0, 0, 0, 0] == 30 and window["meso_condition_image"][0, 0] == 30
    assert window["highres_condition_image"] is item["highres_condition_image"]


def test_autoregressive_chain_feeds_the_last_predicted_frames(monkeypatch):
    """Block k reads the last 10 frames predicted by block k-1; the K chains are independent."""
    from src.model import diffusion_utils as du

    calls = []

    def fake_sample(batch, context, model, conditioner, first_stage, sampler, denoiser, cfg,
                    noise_image_num, past_weight, n_samples=1):
        calls.append((context.clone(), batch["meso_condition_image"].clone()))
        base = 100.0 * len(calls)
        b, _, c, h, w = context.shape
        frames = base + torch.arange(noise_image_num, dtype=torch.float32).view(1, -1, 1, 1, 1)
        return [frames.expand(b, noise_image_num, c, h, w).clone() for _ in range(n_samples)]

    monkeypatch.setattr(du, "sample_forecast", fake_sample)
    cfg = SimpleNamespace(diffusion=SimpleNamespace(
        input_key="imgs", loss_fn_config=SimpleNamespace(params=SimpleNamespace(past_weight=0.1))))
    bundle = SimpleNamespace(cfg=cfg, model=None, conditioner=None, first_stage_model=None,
                             sampler=None, denoiser=None)
    plan = ens.autoregressive_plan(20, 40, 10, 20)
    imgs = torch.arange(60.0).view(1, 60, 1, 1, 1).expand(2, 60, 1, 2, 2).clone()
    meso = torch.arange(60.0).view(1, 60, 1).expand(2, 60, 5).clone()
    batch = {"imgs": imgs, "meso_condition_image": meso}

    preds = ens.predict_autoregressive(batch, bundle, plan, n_samples=2)

    assert len(preds) == 2 and preds[0].shape == (2, 40, 1, 2, 2)
    # calls: block 0 of chain 0, block 0 of chain 1, block 1 of chain 0, block 1 of chain 1
    assert len(calls) == 4
    first_context, first_meso = calls[0]
    assert torch.equal(first_context[0, :, 0, 0, 0], torch.arange(10.0, 20.0))   # last observed frames
    assert torch.equal(first_meso[0, :, 0], torch.arange(10.0, 40.0))            # weather of the 30 frames
    second_context, second_meso = calls[2]
    assert torch.equal(second_context[0, :, 0, 0, 0], torch.arange(10.0, 20.0) + 100.0)   # chain 0, block 0: 110..119
    assert torch.equal(second_meso[0, :, 0], torch.arange(30.0, 60.0))
    # the chains do not mix: chain 1 continues from its own predictions
    assert torch.equal(calls[3][0][0, :, 0, 0, 0], torch.arange(10.0, 20.0) + 200.0)
    assert torch.equal(preds[0][0, :20, 0, 0, 0], torch.arange(20.0) + 100.0)
    assert torch.equal(preds[0][0, 20:, 0, 0, 0], torch.arange(20.0) + 300.0)


# ======================== cubes read from two files ==========================

def stats_for(n_bands=4):
    lo = np.zeros((1, n_bands, 1, 1), dtype=np.float32)
    hi = np.ones((1, n_bands, 1, 1), dtype=np.float32)
    return lo, hi


def test_dataset_joins_context_and_target_in_memory(tmp_path):
    make_earthnet(tmp_path)
    for split, ctx_days in (("iid", 150), ("ood", 50)):        # ood: weather split between the files
        test_dir = tmp_path / "raw" / f"{split}_test_split"
        pairs = list_test_pairs(test_dir)
        assert len(pairs) == 3 and all(not v.exists() for v, _, _ in pairs)   # virtual paths
        lo, hi = stats_for()
        common = dict(highres_lo=lo, highres_hi=hi, meso_lo=np.zeros(5), meso_hi=np.full(5, 2000.0),
                      static_lo=np.zeros(1), static_hi=np.full(1, 400.0), T=30, n_predict=20,
                      return_paths=True, dtype=torch.float32, resize=32)
        by = {str(v): (c, t) for v, c, t in pairs}
        dataset = EarthNetTestDataset([v for v, _, _ in pairs], by, cloud=(0.2, 100.0), **common)
        item = dataset[0]
        assert item["imgs"].shape == (30, 4, 32, 32) and item["meso_condition_image"].shape == (30, 5)
        assert item["cloud_mask"].shape == (20, 1, 32, 32)
        assert item["cube_path"] == str(pairs[0][0])

        # the same cube written as ONE file (what the old evaluation saved on disk)
        _, ctx, tgt = pairs[0]
        hd, hs, md, _ = read_test_cube(ctx, tgt, 10, 20, 4, 5)
        assert hd.shape[3] == 30 and md.shape[3] == 150
        joined = tmp_path / "joined.npz"
        np.savez(joined, highresdynamic=hd, highresstatic=hs, mesodynamic=md)
        reference = S2DiffusionCubeDataset([joined], **common)[0]
        for key in ("imgs", "highres_condition_image", "meso_condition_image"):
            assert torch.equal(item[key], reference[key]), (split, key)


def test_read_test_cube_refuses_a_cube_with_other_frames(tmp_path):
    make_earthnet(tmp_path)
    (_, ctx, tgt), *_ = list_test_pairs(tmp_path / "raw" / "iid_test_split")
    with pytest.raises(ValueError, match="expected 20 context frames"):
        read_test_cube(ctx, tgt, 20, 20, 4, 5)
    with pytest.raises(ValueError, match="days of mesodynamic"):
        read_test_cube(ctx, tgt, 10, 20, 4, 6)


# ============================ scoring and names ==============================

def test_cube_scores_do_not_depend_on_the_order_of_evaluation():
    rng = np.random.RandomState(0)
    lo, hi = np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)
    target = torch.tensor(rng.uniform(-1, 1, (2, 6, 4, 16, 16)), dtype=torch.float32)
    preds = [target + 0.1 * torch.tensor(rng.randn(*target.shape), dtype=torch.float32)]
    paths = ["a/cube1.npz", "b/cube2.npz"]
    both = em.score_batch(preds, target, None, paths, lo, hi, seed=7)
    second_alone = em.score_batch([p[1:] for p in preds], target[1:], None, paths[1:], lo, hi, seed=7)
    assert both[1]["OLS"] == second_alone[0]["OLS"]
    assert em.cube_seed("a/cube1.npz", 7) != em.cube_seed("b/cube2.npz", 7)


def test_autoregressive_files_are_named_after_the_mode(tmp_path):
    spec = SimpleNamespace(name="seasonal")
    plan = ens.autoregressive_plan(70, 140, 10, 20)
    assert ens.split_tag(spec, plan) == "seasonal_autoreg"
    assert ens.split_tag(spec, None) == "seasonal"
    args = SimpleNamespace(subset=50, subset_seed=0, max_cubes=0)
    payload = {"per_cube": []}
    # the stem of the result files: metrics_results_<tag>_K_<K><suffix>
    json_path, csv_path = ens.save_results(tmp_path, "seasonal_autoreg", 5, payload,
                                           ens.selection_suffix(args))
    assert json_path.name == "metrics_results_seasonal_autoreg_K_5_subset50.json"
    assert csv_path.name == "metrics_results_seasonal_autoreg_K_5_subset50.csv"
