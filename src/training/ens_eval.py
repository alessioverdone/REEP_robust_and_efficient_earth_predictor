"""
ENS (EarthNetScore) durante il training di fase 2, config section `ens`.

Il codice dello scoring e' quello di src/scripts/evaluate_ens.py (src/utils/ens_metrics.py):
qui c'e' solo la colla con il training loop.

    validation   i cubi del validation set, nelle epoche in cui si fa validazione:
                 si riusano le predizioni gia' generate per le RMSE, quindi
                 l'inferenza non si paga due volte (il dataset di validation e'
                 avvolto in MaskedCubeDataset, che aggiunge la maschera nuvole)
    train        un campione di cubi di train (config ens.train_cubes), a fine
                 epoca ogni ens.train_every_epochs: per questi cubi si fa un
                 sampling completo, senza flip e con il modello in eval(), come in
                 validazione. Servono predizioni vere, quindi non si possono
                 calcolare sui batch dello step di training, dove il modello vede
                 il futuro vero con un po' di rumore e fa un solo passo di denoising

Ogni cubo e' predetto K = run.val_num_samples volte e di ognuno conta la predizione
con l'ENS piu' alto (best-of-K, come evaluate_ens.py --num-samples K).
"""

import copy
import gc
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from accelerate.utils import gather_object
from torch.utils.data import DataLoader

from src.utils.ens_metrics import (MaskedCubeDataset, aggregate, score_batch, unique_rows)

# the statistics written in history.csv, as val_<name> and train_<name>
ENS_KEYS = ("ENS", "MAD", "OLS", "EMD", "SSIM", "SSIM_RAW", "P_MAE", "N_MAE")


logger = logging.getLogger(__name__)


def evaluate_test_splits(cfg, rt, outputs, state):
    """
    End of the diffusion training: src/scripts/evaluate_ens.py on the splits of
    config section `test_eval`, on the checkpoint it names. The results go where the
    script puts them when launched by hand, <run>/results/statistics/.

    Call it after train_loop and after the training objects are released
    (release_gpu_memory): the DiT is loaded again from the checkpoint on disk, as the
    script does, so the evaluation is the one a later manual run would give.
    Only the main process evaluates (the script is single-process); the other ranks
    return at once. A failure is logged and swallowed: the training is over and its
    checkpoints are saved, evaluate_ens.py can be rerun by hand.
    Returns the payloads of the evaluated splits (None when skipped).
    """
    test_eval = cfg.test_eval
    if not test_eval.enabled or cfg.run.validate_only or state.global_step <= 0:
        return None
    if not rt.accelerator.is_main_process:
        return None

    checkpoint = test_eval.checkpoint
    if checkpoint == "best" and not os.path.isfile(
            os.path.join(outputs.checkpoints, "best", "model.safetensors")):
        logger.warning("[test_eval] checkpoints/best not found (no validation improved the "
                       "minimum?): using the latest checkpoint")
        checkpoint = "latest"
    k = int(test_eval.num_samples or cfg.run.val_num_samples)
    argv = ["--model", outputs.root,
            "--split", *test_eval.splits,
            "--checkpoint", checkpoint,
            "--num-samples", str(k),
            "--seed", str(test_eval.seed),
            "--device", str(rt.accelerator.device)]
    if test_eval.subset:
        argv += ["--subset", str(test_eval.subset)]

    logger.info(f"[test_eval] evaluate_ens.py {' '.join(argv)}")
    start = time.time()
    try:
        # lazy: the script pulls in the whole inference stack (and edits sys.path)
        from src.scripts import evaluate_ens
        payloads = evaluate_ens.evaluate_run(evaluate_ens.parse_args(argv))
    except (Exception, SystemExit) as error:     # noqa: BLE001 - never lose a finished run
        logger.exception(f"[test_eval] failed, run kept as completed: {error}")
        return None

    for payload in payloads:
        split = payload["run"]["split"]
        rt.tracker.summary({f"test/{split}/{key}": value
                            for key, value in payload["aggregate"].items()
                            if isinstance(value, (int, float))})
    logger.info(f"[test_eval] {len(payloads)} splits evaluated in {time.time() - start:.0f}s, "
                f"results in {os.path.join(outputs.results, 'statistics')}")
    return payloads


def release_gpu_memory(accelerator):
    """Drops what the accelerator still references (the caller deletes its own names) and frees the cache."""
    accelerator.free_memory()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def wrap_with_cloud_mask(dataset, ens_cfg):
    """
    `dataset` (S2DiffusionCubeDataset con return_paths) piu' la chiave "cloud_mask" in
    ogni item: i pixel validi dei frame target, letti dai .npz originali. I frame
    target iniziano dopo il contesto (il dataset legge i primi T istanti di ogni cubo).
    """
    sources = {str(f): (Path(f), dataset.n_context) for f in dataset.files}
    return MaskedCubeDataset(dataset, sources, dataset.n_predict,
                             ens_cfg.cloud_prob_threshold, ens_cfg.cloud_prob_scale,
                             dataset.resize)


class EnsEvaluator:
    """Scoring dei cubi, scelta dei cubi di train e aggregazione tra i processi."""

    def __init__(self, cfg, accelerator, train_dataloader):
        self.cfg = cfg.ens
        self.k = int(cfg.run.val_num_samples)
        self.seed = int(cfg.run.seed)
        self.accelerator = accelerator
        self.batch_size = int(cfg.data.params.batch_size)
        self.num_workers = int(cfg.data.params.num_workers)
        # a prepared DataLoader still exposes its dataset
        self.train_dataset = train_dataloader.dataset
        self.lo = np.asarray(self.train_dataset.highres_lo).reshape(-1).astype(np.float32)
        self.hi = np.asarray(self.train_dataset.highres_hi).reshape(-1).astype(np.float32)
        self.scoring_seconds = 0.0          # CPU time of the scoring, since the last reset

    # -- columns of history.csv -----------------------------------------------
    def columns(self) -> dict:
        """Every column, empty: history.csv keeps the same header on every epoch."""
        cols = {}
        for split, used in (("val", self.cfg.validation), ("train", self.cfg.train)):
            if used:
                cols.update({f"{split}_{key}": None for key in ENS_KEYS})
                cols[f"{split}_ens_n_cubes"] = None
        cols["ens_scoring_time_s"] = None
        if self.cfg.train:
            cols["ens_train_eval_time_s"] = None
        return cols

    # -- scoring ----------------------------------------------------------------
    def add_batch(self, rows: list, pred_samples, target, cloud, cube_paths) -> None:
        """Scores the cubes of a batch (the K predictions of each) into `rows`."""
        start = time.time()
        rows.extend(score_batch(pred_samples, target, cloud, cube_paths, self.lo, self.hi,
                                self.cfg.ssim_data_range, self.cfg.scoring))
        self.scoring_seconds += time.time() - start

    def finish(self, rows: list, split: str) -> dict:
        """
        The rows of all the processes -> {val_ENS: ..., val_ens_n_cubes: ...}. A
        process only scored its shard of the cubes: the statistics are computed
        once, on all of them (mean of each sub-score over the cubes, then the
        harmonic mean).
        """
        rows = unique_rows(gather_object(rows))
        scored = [r for r in rows if r["ENS"] is not None]
        summary = aggregate(rows) if rows else {key: None for key in ENS_KEYS}
        result = {f"{split}_{key}": summary.get(key) for key in ENS_KEYS}
        result[f"{split}_ens_n_cubes"] = len(scored)
        return result

    # -- the training sample ------------------------------------------------------
    def train_loader(self, epoch: int):
        """
        The cubes to score on the training set, as a loader over THIS process's share
        of them: ens.train_cubes in all, the same every time with
        ens.train_fixed_subset or a new random sample otherwise. No flip, no
        shuffle; every process makes the same choice (it depends only on the seed)
        and takes every num_processes-th cube, so nothing is scored twice.

        Not passed through accelerator.prepare(): the accelerator keeps every
        prepared loader and saves its state in each checkpoint, so one new loader
        per epoch would pile up there.
        """
        files = list(self.train_dataset.files)
        rng = random.Random(self.seed if self.cfg.train_fixed_subset else self.seed + epoch)
        chosen = sorted(rng.sample(files, min(self.cfg.train_cubes, len(files))))
        chosen = chosen[self.accelerator.process_index::self.accelerator.num_processes]

        dataset = copy.copy(self.train_dataset)
        dataset.files = chosen
        dataset.do_hflip = False
        dataset.return_paths = True
        if self.cfg.cloud_mask:
            dataset = wrap_with_cloud_mask(dataset, self.cfg)
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=False)

    def to_device(self, batch: dict) -> dict:
        """The tensors of a batch of train_loader() on the accelerator's device."""
        return {k: (v.to(self.accelerator.device) if torch.is_tensor(v) else v)
                for k, v in batch.items()}
