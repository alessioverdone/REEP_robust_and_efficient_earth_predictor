import os
import csv
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

def save_checkpoint(path, vae, optimizer, epoch, extra=None, scheduler=None):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": vae.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()

    if extra is not None:
        ckpt.update(extra)

    torch.save(ckpt, path)


def load_checkpoint(path, vae, optimizer=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    vae.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    start_epoch = ckpt.get("epoch", 0) + 1
    return start_epoch, ckpt


def get_vae(in_channels=4,
            out_channels=4,
            latent_channels=4,
            block_out_channels=(64, 128, 256),
            layers_per_block=2,
            norm_num_groups=32,
            device="cpu"):
    down_block_types = tuple(["DownEncoderBlock2D"] * len(block_out_channels))
    up_block_types = tuple(["UpDecoderBlock2D"] * len(block_out_channels))

    vae = AutoencoderKL(in_channels=in_channels,
                        out_channels=out_channels,
                        latent_channels=latent_channels,
                        down_block_types=down_block_types,
                        up_block_types=up_block_types,
                        block_out_channels=block_out_channels,
                        layers_per_block=layers_per_block,
                        norm_num_groups=norm_num_groups).to(device)

    return vae


def resize_images(imgs,
                  size):
    """
    Downsample a batch [B, C, H, W] to [B, C, size, size].

    Area interpolation averages the source pixels, so values in [-1, 1] stay
    in [-1, 1] and no aliasing is introduced. No-op if already at that size.
    """
    if tuple(imgs.shape[-2:]) == (size, size):
        return imgs
    return F.interpolate(imgs,
                         size=(size, size),
                         mode="area")


@torch.no_grad()
def encode_first_stage(first_stage_model, input_tensor, scale_factor=1.0):
    """
    Encode per usare il VAE come first-stage model.

    input_tensor: [B, C, H, W]
    output: latent scalato
    """
    z = first_stage_model.encode(input_tensor)

    if isinstance(z, dict):
        z = z["latent_dist"].sample()
    elif hasattr(z, "latent_dist"):
        z = z.latent_dist.sample()
    elif hasattr(z, "sample"):
        z = z.sample()

    z = scale_factor * z
    return z


@torch.no_grad()
def decode_first_stage(first_stage_model, input_tensor, scale_factor=1.0):
    """
    Decode per usare il VAE come first-stage model.

    input_tensor: latent scalato
    output: immagine ricostruita
    """
    z = input_tensor / scale_factor

    out = first_stage_model.decode(z)

    if isinstance(out, dict):
        out = out["sample"]
    elif hasattr(out, "sample"):
        out = out.sample

    return out


def gaussian_kernel(window_size=11, sigma=1.5, channels=1, device="cpu"):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    kernel_2d = torch.outer(g, g).unsqueeze(0).unsqueeze(0)
    kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)

    return kernel_2d


def compute_ssim_torch(
    recon,
    target,
    data_range=2.0,
    window_size=11,
    sigma=1.5,
    eps=1e-12,
):
    """
    SSIM medio scalare torch.
    Usato solo come metrica, non nella loss.
    """
    _, c, _, _ = recon.shape

    kernel = gaussian_kernel(
        window_size=window_size,
        sigma=sigma,
        channels=c,
        device=recon.device,
    )

    padding = window_size // 2

    mu_x = F.conv2d(recon, kernel, padding=padding, groups=c)
    mu_y = F.conv2d(target, kernel, padding=padding, groups=c)

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(recon * recon, kernel, padding=padding, groups=c) - mu_x2
    sigma_y2 = F.conv2d(target * target, kernel, padding=padding, groups=c) - mu_y2
    sigma_xy = F.conv2d(recon * target, kernel, padding=padding, groups=c) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + eps
    )

    return ssim_map.mean()


def reconstruction_loss_fn(recon, target):
    """
    Loss di ricostruzione: SOLO MSE.
    """
    loss = F.mse_loss(recon, target)
    return loss


@torch.no_grad()
def compute_batch_psnr(recon, target, data_range=2.0, eps=1e-12):
    mse = F.mse_loss(recon, target, reduction="mean").item()
    psnr = 10.0 * math.log10((data_range ** 2) / (mse + eps))
    return psnr


def append_history_row(csv_path, row_dict):
    file_exists = os.path.exists(csv_path)

    if file_exists:
        # a row with columns the file does not have yet (e.g. the ENS columns, in a
        # run resumed from a history written without them): the file is rewritten
        # with the wider header, the old rows keep their cells and get empty ones
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            header = list(reader.fieldnames or [])
            old_rows = list(reader) if set(row_dict) - set(header) else []
        new_columns = [k for k in row_dict if k not in header]
        if new_columns:
            header += new_columns
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                writer.writerows(old_rows)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row_dict)