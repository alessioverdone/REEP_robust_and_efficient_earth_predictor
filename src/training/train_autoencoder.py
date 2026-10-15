"""
Fase 1 dell'esperimento: addestramento dell'autoencoder che comprime le
immagini Sentinel-2 nel latente su cui poi lavora il diffusion model.

Lo script di lancio e' src/scripts/run_autoencoder_train.py e ha la stessa
struttura di quello della fase 2: configurazione, dati, modello, training. Qui
dentro non ci sono costanti: ogni numero viene da cfg.autoencoder (schema in
src/config.py, valori in registry/configurations/*.yaml).

Everything the run produces goes into its own folder (src/training/run_outputs.py):
checkpoints in checkpoints/, config, history, log and training_info.json in
artifacts/, curves and reconstruction examples in results/.
"""

import csv
import datetime
import logging
import os
import time
import traceback

import numpy as np
import torch
from tqdm import tqdm

from src.dataset.S2FlatDataset import make_dataloaders
from src.dataset.dataset_info import dataset_identity, load_dataset_info
from src.model.autoencoder_utils import (append_history_row, compute_batch_psnr,
                                         compute_ssim_torch, get_vae, load_checkpoint,
                                         reconstruction_loss_fn, resize_images,
                                         save_checkpoint)
from src.training.run_outputs import (count_parameters, environment_info, file_size_mb,
                                      measure_flops, relative_to_project, write_json)
from src.training.state import AutoencoderState, monitored_value
from src.utils.plots import plot_history, save_reconstruction_grid

logger = logging.getLogger(__name__)


# ================================== DATI ==================================

def setup_dataloader(cfg):
    """
    I DataLoader (train, val), la composizione dei due split e il numero
    di bande.

    I dati arrivano dal preprocessing gia' normalizzati in [-1, 1] e appiattiti
    a immagini singole, one file per split (the splits of dataset.json, the
    same the diffusion model uses): qui si apre il memmap e si divide in batch.
    Il numero di bande lo detta il file, non una costante da tenere allineata a
    mano.
    """
    auto = cfg.autoencoder
    loaders, composition = make_dataloaders(train_npy=auto.train_npy,
                                            val_npy=auto.val_npy,
                                            manifest=auto.manifest,
                                            batch_size=auto.batch_size,
                                            num_workers=auto.num_workers,
                                            prefetch_factor=auto.prefetch_factor,
                                            persistent_workers=auto.persistent_workers)

    # Downsampling happens on the GPU (see resize_images): only shrinking is
    # allowed, upsampling would invent detail the data does not have.
    native = loaders["train"].dataset.image_size
    if auto.image_size > min(native):
        raise SystemExit(
            f"autoencoder.image_size={auto.image_size} (from data.params.resize) "
            f"exceeds the native resolution {native} of {auto.train_npy}: "
            f"only downsampling is supported."
        )
    logger.info(f"[autoencoder] resolution: native {native[0]}x{native[1]} "
                f"-> training at {auto.image_size}x{auto.image_size}")
    return loaders, composition, loaders["train"].dataset.n_channels


# ================================= MODELLO =================================

def instantiate_autoencoder(cfg,
                            rt,
                            n_channels):
    """
    L'AutoencoderKL descritto da cfg.autoencoder, sul device del runtime.

    Le stesse chiavi le rilegge la fase 2 (diffusion.first_stage_config.params
    le interpola da qui) per ricostruire un modello identico e caricarci sopra
    questi pesi: se le due fasi divergono, il training del diffusion si ferma
    subito con un messaggio esplicito.
    """
    if n_channels != cfg.data.params.n_channels:
        raise SystemExit(
            f"il .npy del preprocessing ha {n_channels} bande, ma il config ne "
            f"dichiara data.params.n_channels={cfg.data.params.n_channels}. La "
            f"fase 2 costruisce il first stage su quel valore: vanno allineati."
        )

    vae = get_vae(
        in_channels=n_channels,
        out_channels=n_channels,
        latent_channels=cfg.autoencoder.latent_channels,
        block_out_channels=tuple(cfg.autoencoder.block_out_channels),
        layers_per_block=cfg.autoencoder.layers_per_block,
        norm_num_groups=cfg.autoencoder.norm_num_groups,
        device=rt.device,
    )

    total = sum(p.numel() for p in vae.parameters())
    logger.info(f"[autoencoder] AutoencoderKL: {total / 1e6:.1f}M parametri, "
                f"latent_channels={cfg.autoencoder.latent_channels}, "
                f"down_ratio={2 ** (len(cfg.autoencoder.block_out_channels) - 1)}")
    return vae


def prepare_training(cfg, vae):
    """Optimizer e scheduler. Lo scheduler segue la metrica di save_best_by."""
    auto = cfg.autoencoder
    optimizer = torch.optim.AdamW(vae.parameters(),
                                  lr=auto.lr,
                                  weight_decay=auto.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",                 # il valore monitorato e' gia' orientato al minimo
        factor=auto.lr_scheduler_factor,
        patience=auto.lr_scheduler_patience,
    )
    return optimizer, scheduler


def _inherit_history(resume_from,
                     history_csv,
                     start_epoch):
    """
    When resuming from a checkpoint of an earlier run, copy that run's
    history.csv rows up to the resumed epoch, so the curves of the new run
    start from the beginning of the training and not from the resume point.
    """
    previous = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(resume_from))),
                            "artifacts",
                            "history.csv")
    if not os.path.isfile(previous):
        logger.info(f"[autoencoder] no history.csv next to {resume_from}: "
                    f"the curves start from epoch {start_epoch + 1}")
        return
    with open(previous, newline="") as f:
        rows = [row for row in csv.DictReader(f) if int(row["epoch"]) <= start_epoch]
    for row in rows:
        append_history_row(history_csv,
                           row)
    logger.info(f"[autoencoder] {len(rows)} history rows inherited from {previous}")


def configure_train(cfg,
                    outputs,
                    vae,
                    optimizer,
                    scheduler):
    """
    Lo stato da cui il loop riparte: epoca iniziale e miglior valore storico,
    ripresi da cfg.autoencoder.resume_from se c'e'. The run itself always
    writes into its own new folder, also when resuming.
    """
    auto = cfg.autoencoder
    state = AutoencoderState(history_csv=outputs.history_csv,
                             started_at=datetime.datetime.now().isoformat(timespec="seconds"))

    if auto.resume_from is not None:
        state.start_epoch, ckpt = load_checkpoint(auto.resume_from,
                                                  vae,
                                                  optimizer=optimizer,
                                                  scheduler=scheduler)
        state.best_metric = ckpt.get("best_metric", state.best_metric)
        state.best_epoch = ckpt.get("best_epoch")
        state.early_stop_reference = ckpt.get("early_stop_reference", state.best_metric)
        state.epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
        state.epochs_completed = state.start_epoch
        state.resumed_from = relative_to_project(auto.resume_from)
        _inherit_history(auto.resume_from,
                         state.history_csv,
                         state.start_epoch)
        logger.info(f"[autoencoder] ripreso da {auto.resume_from}: "
                    f"start_epoch={state.start_epoch}, "
                    f"best_{auto.save_best_by}={state.best_metric}")

    logger.info(f"[autoencoder] outputs del run in {outputs.root}")
    return state


# ============================ DESCRIZIONE DEL RUN ============================

def describe_run(cfg,
                 rt,
                 outputs,
                 state,
                 vae,
                 loaders,
                 composition):
    """
    The parts of training_info.json fixed for the whole run: environment,
    dataset, model and its computational cost. Stored in state.static and
    written right away, so the json exists from the first second of the run.
    """
    auto = cfg.autoencoder
    native = loaders["train"].dataset.image_size
    n_channels = loaders["train"].dataset.n_channels
    down_ratio = 2 ** (len(auto.block_out_channels) - 1)
    latent_size = auto.image_size // down_ratio

    # FLOPs of one image at the training resolution, forward only. Encoder and
    # decoder separately: phase 2 runs the encoder on every frame it trains on.
    image = torch.zeros(1, n_channels, auto.image_size, auto.image_size, device=rt.device)
    latent = torch.zeros(1, auto.latent_channels, latent_size, latent_size, device=rt.device)
    vae.eval()
    encoder_flops = measure_flops(lambda x: vae.encode(x).latent_dist.mean,
                                  image)
    decoder_flops = measure_flops(lambda z: vae.decode(z).sample,
                                  latent)
    vae.train()

    # name and id of the preprocessed dataset: saved in the checkpoints, phase 2
    # refuses a VAE trained on a different one
    dataset = load_dataset_info(auto.dataset_json)
    state.static = {
        "environment": environment_info(),
        "dataset": {
            "identity": dataset_identity(dataset),
            "dataset_json": auto.dataset_json,
            "selection": dataset.get("selection"),
            "train_npy": auto.train_npy,
            "val_npy": auto.val_npy,
            "manifest": auto.manifest,
            "stats_json": f"{auto.preprocessed_dir}/stats.json",
            "normalization": "per-band quantile scaling to [-1, 1] (see stats_json)",
            "T": auto.T,
            "n_channels": n_channels,
            "native_resolution": list(native),
            "training_resolution": [auto.image_size, auto.image_size],
            "resize_method": "area interpolation on GPU" if auto.image_size != min(native) else None,
            "splits": composition,
            "batches_per_epoch": {name: len(loader) for name, loader in loaders.items()
                                  if loader is not None},
        },
        "model": {
            "class": type(vae).__name__,
            "architecture": {
                "in_channels": n_channels,
                "latent_channels": auto.latent_channels,
                "block_out_channels": list(auto.block_out_channels),
                "layers_per_block": auto.layers_per_block,
                "norm_num_groups": auto.norm_num_groups,
                "down_ratio": down_ratio,
            },
            "parameters": count_parameters(vae),
            "input_shape": [n_channels, auto.image_size, auto.image_size],
            "latent_shape": [auto.latent_channels, latent_size, latent_size],
            "compression_ratio": (n_channels * auto.image_size ** 2)
                                 / (auto.latent_channels * latent_size ** 2),
        },
        "flops_per_image": {
            "encoder_gflops": encoder_flops / 1e9,
            "decoder_gflops": decoder_flops / 1e9,
            "forward_gflops": (encoder_flops + decoder_flops) / 1e9,
            # backward costs about twice the forward
            "train_step_gflops_estimate": 3 * (encoder_flops + decoder_flops) / 1e9,
        },
    }
    logger.info(f"[autoencoder] dataset {dataset['name']} (id {dataset['id']})")
    logger.info(f"[autoencoder] {state.static['model']['parameters']['total'] / 1e6:.2f}M "
                f"parametri, {(encoder_flops + decoder_flops) / 1e9:.2f} GFLOPs/immagine "
                f"(forward, {auto.image_size}x{auto.image_size})")


def training_info(cfg,
                  outputs,
                  state,
                  optimizer):
    """The full content of training_info.json at this point of the run."""
    auto = cfg.autoencoder
    metrics = state.metrics
    gpu_memory = metrics.gpu_memory_gb()
    train_seconds = float(sum(state.epoch_seconds))
    wall_seconds = metrics.wall_time if metrics.start_time else 0.0

    best = state.checkpoints.get("best.pt")
    return {
        "run": {
            "phase": "autoencoder",
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
            "epochs_planned": auto.num_epochs,
            "epochs_completed": state.epochs_completed,
            "start_epoch": state.start_epoch,
            "resumed_from": state.resumed_from,
            "global_step": state.global_step,
            "batch_size": auto.batch_size,
            "optimizer": {"name": type(optimizer).__name__,
                          "lr": auto.lr,
                          "weight_decay": auto.weight_decay,
                          "grad_clip_norm": auto.grad_clip_norm},
            "scheduler": {"name": "ReduceLROnPlateau",
                          "monitor": f"val_{auto.save_best_by}",
                          "factor": auto.lr_scheduler_factor,
                          "patience": auto.lr_scheduler_patience},
            "lr_final": optimizer.param_groups[0]["lr"],
            "lr_reductions": state.lr_reductions,
            "early_stopping": {"enabled": auto.early_stopping_patience > 0,
                               "monitor": f"val_{auto.save_best_by}",
                               "patience": auto.early_stopping_patience,
                               "min_delta": auto.early_stopping_min_delta,
                               "epochs_without_improvement": state.epochs_without_improvement,
                               "triggered": state.stopped_early,
                               "stopped_at_epoch": (state.epochs_completed
                                                    if state.stopped_early else None)},
            "loss": "MSE",
        },
        "metrics": {
            "selection_criterion": f"val_{auto.save_best_by}",
            "train_last": state.train_metrics,
            "val_last": state.val_metrics,
            "val_best": {"epoch": state.best_epoch,
                         **state.best_val_metrics},
        },
        # diffusion.scale_factor / shift_factor for phase 2, also saved in best.pt
        "scale_factor": state.scale_factor or None,
        "compute": {
            **state.static.get("flops_per_image", {}),
            "wall_time_s": wall_seconds,
            "train_time_s": train_seconds,
            "validation_time_s": metrics.inference_time,
            "mean_epoch_train_time_s": (train_seconds / len(state.epoch_seconds)
                                        if state.epoch_seconds else None),
            "mean_train_images_per_s": (float(np.mean(state.images_per_second))
                                        if state.images_per_second else None),
            "gpu_hours": wall_seconds / 3600,     # single-GPU training
            "peak_gpu_memory_allocated_gb": gpu_memory[0] if gpu_memory else None,
            "peak_gpu_memory_reserved_gb": gpu_memory[1] if gpu_memory else None,
        },
        "checkpoints": {
            "saved_to_disk": auto.save_checkpoints,
            "dir": relative_to_project(outputs.checkpoints),
            "best": None if best is None else {
                "file": "best.pt",
                "path": relative_to_project(os.path.join(outputs.checkpoints, "best.pt")),
                "epoch": best["epoch"],
                "criterion": f"val_{auto.save_best_by}",
                "value": state.best_val_metrics.get(auto.save_best_by),
            },
            "saved": [state.checkpoints[name] for name in sorted(state.checkpoints)],
        },
        "outputs": {
            "root": relative_to_project(outputs.root),
            "config": relative_to_project(outputs.config_json),
            "history": relative_to_project(outputs.history_csv),
            "log": relative_to_project(outputs.log_file),
            "curves": relative_to_project(outputs.curves),
            "reconstructions": relative_to_project(outputs.examples),
        },
    }


def write_training_info(cfg,
                        outputs,
                        state,
                        optimizer):
    write_json(outputs.info_json,
               training_info(cfg,
                             outputs,
                             state,
                             optimizer))


def update_figures(outputs,
                   state):
    plot_history(outputs.history_csv,
                 outputs.curves,
                 metrics=("loss", "mse", "psnr", "ssim"),
                 best_epoch=state.best_epoch)


def mark_run_ended(cfg,
                   rt,
                   outputs,
                   state,
                   optimizer,
                   error=None):
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
    write_training_info(cfg,
                        outputs,
                        state,
                        optimizer)
    logger.info(f"[autoencoder] run {state.status}: {outputs.info_json}")
    rt.tracker.summary({"best/epoch": state.best_epoch,
                        **{f"best/val_{k}": v for k, v in state.best_val_metrics.items()},
                        "latent/scale_factor": state.scale_factor.get("value"),
                        "latent/shift_factor": state.scale_factor.get("shift_factor")})
    rt.tracker.finish(state.status,
                      state.global_step)


# ================================ TRAINING ================================

@torch.no_grad()
def evaluate_vae(cfg, rt, vae, dataloader, desc="Validation"):
    """Una passata su un set di valutazione: loss, mse, psnr, ssim."""
    data_range = cfg.autoencoder.data_range
    vae.eval()

    running = {"loss": 0.0, "mse": 0.0, "psnr": 0.0, "ssim": 0.0}
    n_samples = 0
    n_batches = 0

    for batch in tqdm(dataloader, desc=desc, leave=False):
        imgs = resize_images(batch["tif"].to(rt.device, non_blocking=True),
                             cfg.autoencoder.image_size)

        # media della posterior, non un campione: i latenti di riferimento
        # devono essere deterministici (stessa scelta della fase 2)
        recon = vae.decode(vae.encode(imgs).latent_dist.mean).sample
        loss = reconstruction_loss_fn(recon, imgs)

        bs = imgs.size(0)
        n_samples += bs
        n_batches += 1
        running["loss"] += loss.item() * bs
        running["mse"] += loss.item() * bs
        running["psnr"] += compute_batch_psnr(recon, imgs, data_range=data_range)
        running["ssim"] += compute_ssim_torch(recon, imgs, data_range=data_range).item()

    return {
        "loss": running["loss"] / n_samples,
        "mse": running["mse"] / n_samples,
        "psnr": running["psnr"] / n_batches,
        "ssim": running["ssim"] / n_batches,
    }, n_samples


def example_batch(dataset,
                  n):
    """n images evenly spread over the dataset, always the same ones."""
    if n == 0:
        return None
    indices = np.linspace(0, len(dataset) - 1, num=min(n, len(dataset))).astype(int)
    return torch.stack([dataset[int(i)]["tif"] for i in indices])


@torch.no_grad()
def save_examples(cfg,
                  rt,
                  vae,
                  imgs,
                  path,
                  title):
    """Reconstruction grid of `imgs` (input | recon | error), see save_reconstruction_grid."""
    if imgs is None:
        return
    was_training = vae.training
    vae.eval()
    imgs = resize_images(imgs.to(rt.device),
                         cfg.autoencoder.image_size)
    recon = vae.decode(vae.encode(imgs).latent_dist.mean).sample
    save_reconstruction_grid(imgs.float().cpu().numpy(),
                             recon.float().clamp(-1, 1).cpu().numpy(),
                             path,
                             title=title)
    vae.train(was_training)


def _checkpoint_extra(cfg, state, train_metrics, val_metrics):
    """
    Cosa si salva accanto ai pesi. L'architettura ci va per intero: e' il
    contratto con la fase 2, che la rilegge per controllare di stare
    ricostruendo lo stesso modello prima di caricare lo state_dict.
    """
    auto = cfg.autoencoder
    return {
        "arch": {
            "in_channels": cfg.data.params.n_channels,
            "latent_channels": auto.latent_channels,
            "block_out_channels": list(auto.block_out_channels),
            "layers_per_block": auto.layers_per_block,
            "norm_num_groups": auto.norm_num_groups,
            # resolution the weights were trained at: the latent statistics
            # (and so diffusion.scale_factor) depend on it
            "image_size": auto.image_size,
        },
        # the preprocessed dataset (name + id of dataset.json) the weights were
        # trained on: phase 2 refuses them on a different one
        "dataset": state.static["dataset"]["identity"],
        "best_metric": state.best_metric,
        "best_epoch": state.best_epoch,
        "early_stop_reference": state.early_stop_reference,
        "epochs_without_improvement": state.epochs_without_improvement,
        "save_best_by": auto.save_best_by,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "latent_channels": auto.latent_channels,
    }


def _save_run_checkpoint(cfg,
                         outputs,
                         state,
                         name,
                         vae,
                         optimizer,
                         scheduler,
                         epoch,
                         extra):
    """Saves checkpoints/<name> and records it for training_info.json."""
    path = os.path.join(outputs.checkpoints, name)
    save_checkpoint(path, vae, optimizer, epoch, scheduler=scheduler, extra=extra)
    state.checkpoints[name] = {
        "file": name,
        "epoch": epoch + 1,
        "size_mb": file_size_mb(path),
        f"val_{cfg.autoencoder.save_best_by}": extra["val_metrics"][cfg.autoencoder.save_best_by],
    }
    return path


def update_early_stopping(cfg,
                          state,
                          monitor_value):
    """
    Counts the epochs without an improvement larger than min_delta on the
    monitored value (already oriented to the minimum) and sets
    state.stopped_early once they reach the patience. Disabled with patience 0.
    """
    auto = cfg.autoencoder
    if monitor_value < state.early_stop_reference - auto.early_stopping_min_delta:
        state.early_stop_reference = monitor_value
        state.epochs_without_improvement = 0
    else:
        state.epochs_without_improvement += 1
    state.stopped_early = (auto.early_stopping_patience > 0
                           and state.epochs_without_improvement >= auto.early_stopping_patience)


def train_vae(cfg,
              rt,
              outputs,
              state,
              vae,
              optimizer,
              scheduler,
              loaders):
    """
    Il loop di training. Fa avanzare `state` (epoca, step, miglior metrica e
    metriche in state.metrics) come train_loop() fa con TrainState nella fase 2.
    After every epoch: history.csv, checkpoints, curves, training_info.json,
    and every examples_every epochs the reconstruction examples.
    """
    auto = cfg.autoencoder
    data_range = auto.data_range
    val_examples = example_batch(loaders["val"].dataset,
                                 auto.n_recon_examples)
    write_training_info(cfg,
                        outputs,
                        state,
                        optimizer)
    state.metrics.start()

    for epoch in range(state.start_epoch, auto.num_epochs):
        vae.train()

        running = {"loss": 0.0, "mse": 0.0, "psnr": 0.0, "ssim": 0.0}
        n_samples = 0
        n_batches = 0
        epoch_start = time.time()

        pbar = tqdm(loaders["train"], desc=f"Epoch {epoch + 1}/{auto.num_epochs}")
        for batch in pbar:
            imgs = resize_images(batch["tif"].to(rt.device, non_blocking=True),
                                 auto.image_size)
            optimizer.zero_grad(set_to_none=True)

            posterior = vae.encode(imgs).latent_dist
            z = posterior.rsample() if hasattr(posterior, "rsample") else posterior.sample()
            recon = vae.decode(z).sample

            loss = reconstruction_loss_fn(recon, imgs)
            loss.backward()

            if auto.grad_clip_norm is not None and auto.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(vae.parameters(), auto.grad_clip_norm)
            optimizer.step()

            with torch.no_grad():
                ssim_val = compute_ssim_torch(recon, imgs, data_range=data_range)
                psnr_val = compute_batch_psnr(recon, imgs, data_range=data_range)

            bs = imgs.size(0)
            n_samples += bs
            n_batches += 1
            state.global_step += 1
            running["loss"] += loss.item() * bs
            running["mse"] += loss.item() * bs
            running["psnr"] += psnr_val
            running["ssim"] += ssim_val.item()

            pbar.set_postfix({
                "loss": f"{running['loss'] / n_samples:.5f}",
                "mse": f"{running['mse'] / n_samples:.5f}",
                "ssim": f"{running['ssim'] / n_batches:.5f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        epoch_seconds = time.time() - epoch_start
        state.epoch_seconds.append(epoch_seconds)
        state.images_per_second.append(n_samples / epoch_seconds)

        train_metrics = {
            "loss": running["loss"] / n_samples,
            "mse": running["mse"] / n_samples,
            "psnr": running["psnr"] / n_batches,
            "ssim": running["ssim"] / n_batches,
        }

        validate_start_time = time.time()
        val_metrics, n_val_images = evaluate_vae(cfg, rt, vae, loaders["val"])
        val_seconds = time.time() - validate_start_time
        state.metrics.record_validation(val_metrics, n_val_images, epoch + 1,
                                        state.global_step,
                                        val_seconds)
        state.train_metrics = train_metrics
        state.val_metrics = val_metrics
        state.epochs_completed = epoch + 1

        # Un solo criterio per scheduler e checkpoint migliore, quello del
        # config: per psnr e ssim il valore e' cambiato di segno, cosi' qui si
        # guarda sempre il minimo.
        monitor_value = monitored_value(val_metrics, auto.save_best_by)
        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step(monitor_value)
        lr_after = optimizer.param_groups[0]["lr"]
        if lr_after < lr_before:
            state.lr_reductions.append({"epoch": epoch + 1, "lr": lr_after})
            logger.info(f"[autoencoder] lr ridotto: {lr_before:.2e} -> {lr_after:.2e}")

        logger.info(
            f"[autoencoder] epoca {epoch + 1}/{auto.num_epochs} | "
            f"TRAIN loss={train_metrics['loss']:.6f} mse={train_metrics['mse']:.6f} "
            f"psnr={train_metrics['psnr']:.4f} ssim={train_metrics['ssim']:.4f} | "
            f"VAL loss={val_metrics['loss']:.6f} mse={val_metrics['mse']:.6f} "
            f"psnr={val_metrics['psnr']:.4f} ssim={val_metrics['ssim']:.4f} | "
            f"{epoch_seconds:.0f}s train, {val_seconds:.0f}s val"
        )

        history_row = {
            "epoch": epoch + 1,
            "global_step": state.global_step,
            "lr": lr_before,
            "train_loss": train_metrics["loss"],
            "train_mse": train_metrics["mse"],
            "train_psnr": train_metrics["psnr"],
            "train_ssim": train_metrics["ssim"],
            "val_loss": val_metrics["loss"],
            "val_mse": val_metrics["mse"],
            "val_psnr": val_metrics["psnr"],
            "val_ssim": val_metrics["ssim"],
            "epoch_train_time_s": epoch_seconds,
            "epoch_val_time_s": val_seconds,
            "train_images_per_s": n_samples / epoch_seconds,
        }
        append_history_row(state.history_csv,
                           history_row)
        rt.tracker.log({f"epoch/{k}": v for k, v in history_row.items()},
                       step=state.global_step)

        is_best = monitor_value < state.best_metric
        if is_best:
            state.best_metric = monitor_value
            state.best_epoch = epoch + 1
            state.best_val_metrics = dict(val_metrics)
            state.best_state_dict = {k: v.detach().cpu().clone()
                                     for k, v in vae.state_dict().items()}
            logger.info(f"[autoencoder] nuovo BEST su {auto.save_best_by}: "
                        f"{val_metrics[auto.save_best_by]:.6f} (epoca {epoch + 1})")

        update_early_stopping(cfg,
                              state,
                              monitor_value)

        # only last.pt and best.pt, and none at all with save_checkpoints=false
        if auto.save_checkpoints:
            extra = _checkpoint_extra(cfg, state, train_metrics, val_metrics)
            _save_run_checkpoint(cfg, outputs, state, "last.pt",
                                 vae, optimizer, scheduler, epoch, extra)
            if is_best:
                _save_run_checkpoint(cfg, outputs, state, "best.pt",
                                     vae, optimizer, scheduler, epoch, extra)

        if auto.examples_every and (epoch + 1) % auto.examples_every == 0:
            save_examples(cfg,
                          rt,
                          vae,
                          val_examples,
                          os.path.join(outputs.examples, f"val_epoch_{epoch + 1:03d}.png"),
                          title=f"validation, epoch {epoch + 1}")

        update_figures(outputs,
                       state)
        write_training_info(cfg,
                            outputs,
                            state,
                            optimizer)
        rt.tracker.sync(state.global_step)

        if state.stopped_early:
            logger.info(f"[autoencoder] early stopping all'epoca {epoch + 1}: "
                        f"{state.epochs_without_improvement} epoche senza miglioramenti "
                        f"di val_{auto.save_best_by} oltre "
                        f"min_delta={auto.early_stopping_min_delta}")
            break

    for line in state.metrics.summary_lines():
        logger.info(line)
    logger.info(f"[autoencoder] training terminato, storia in {state.history_csv}")
    return state


# ============================== SCALE FACTOR ==============================

@torch.no_grad()
def compute_latent_stats(cfg,
                         device,
                         vae,
                         loader):
    """
    Mean and std of the latents the diffusion model will see (latent_dist.mean,
    the same encode as phase 2), at the training resolution, over
    autoencoder.scale_factor_batches batches of `loader`: global and per
    latent channel. scale_factor = 1 / std, shift_factor = mean.
    """
    auto = cfg.autoencoder
    vae.eval()
    # per-channel sums: the global mean/std follow from them, and the
    # per-channel values tell whether a single shift/scale is enough
    ch_sum, ch_sum_sq, values_per_channel, n_images = None, None, 0, 0
    for i, batch in enumerate(loader):
        if i >= auto.scale_factor_batches:
            break
        imgs = resize_images(batch["tif"].to(device, non_blocking=True),
                             auto.image_size)
        z = vae.encode(imgs).latent_dist.mean.double()
        batch_sum = z.sum(dim=(0, 2, 3))
        batch_sum_sq = (z * z).sum(dim=(0, 2, 3))
        ch_sum = batch_sum if ch_sum is None else ch_sum + batch_sum
        ch_sum_sq = batch_sum_sq if ch_sum_sq is None else ch_sum_sq + batch_sum_sq
        values_per_channel += z.numel() // z.size(1)
        n_images += imgs.size(0)

    n_values = values_per_channel * ch_sum.numel()
    mean = ch_sum.sum().item() / n_values
    std = max(ch_sum_sq.sum().item() / n_values - mean ** 2, 0.0) ** 0.5
    ch_mean = ch_sum / values_per_channel
    ch_std = (ch_sum_sq / values_per_channel - ch_mean ** 2).clamp(min=0.0).sqrt()
    return {"value": 1.0 / std,
            "shift_factor": mean,
            "latent_mean": mean,
            "latent_std": std,
            "per_channel": {"mean": ch_mean.tolist(),
                            "std": ch_std.tolist()},
            "n_images": n_images}


def log_latent_stats(stats):
    logger.info(f"[autoencoder] latenti: mean={stats['latent_mean']:+.5f} "
                f"std={stats['latent_std']:.5f} su {stats['n_images']} immagini "
                f"-> diffusion.scale_factor={stats['value']:.6f}, "
                f"diffusion.shift_factor={stats['shift_factor']:+.6f}")
    logger.info(f"[autoencoder] latenti per canale: mean="
                f"[{', '.join(f'{m:+.3f}' for m in stats['per_channel']['mean'])}] std="
                f"[{', '.join(f'{v:.3f}' for v in stats['per_channel']['std'])}]")


def measure_scale_factor(cfg,
                         rt,
                         outputs,
                         state,
                         vae,
                         loaders):
    """
    diffusion.scale_factor / shift_factor and the per-channel statistics of
    the best weights (compute_latent_stats on the training set). Saved in state
    (training_info.json) and in best.pt, where phase 2 reads them when the
    corresponding diffusion keys are null.
    """
    if state.best_state_dict is None:
        logger.info("[autoencoder] nessun peso migliore in questo run: scale_factor non misurato")
        return state

    vae.load_state_dict(state.best_state_dict)
    state.scale_factor = {**compute_latent_stats(cfg,
                                                 rt.device,
                                                 vae,
                                                 loaders["train"]),
                          "weights": f"best (epoch {state.best_epoch})"}
    log_latent_stats(state.scale_factor)
    mean, std = state.scale_factor["latent_mean"], state.scale_factor["latent_std"]
    if abs(mean) > 0.5 * std:
        # only a latent_normalization with shift removes the offset; with
        # 'scale' the diffusion model sees it
        logger.warning(f"[autoencoder] latents not centered (|mean| = {abs(mean):.3f} > "
                       f"std / 2 = {std / 2:.3f}): phase 2 needs a diffusion.latent_normalization "
                       f"with shift (the default) to remove it")

    best_path = os.path.join(outputs.checkpoints, "best.pt")
    if cfg.autoencoder.save_checkpoints and os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        ckpt["scale_factor"] = 1.0 / std
        ckpt["shift_factor"] = mean
        ckpt["latent_stats"] = state.scale_factor
        torch.save(ckpt, best_path)
        state.checkpoints["best.pt"]["scale_factor"] = 1.0 / std
        state.checkpoints["best.pt"]["shift_factor"] = mean
        logger.info(f"[autoencoder] scale_factor e shift_factor salvati in {best_path}")
    return state
