"""
Adds the per-channel latent statistics to an autoencoder run that does not
have them (runs before 2026-10-08), so phase 2 can use
diffusion.latent_normalization=shift_scale_per_channel with it:

    python src/scripts/measure_latent_stats.py --run_dir registry/outputs/autoencoder/<run>

Same measurement as the end of phase 1 (compute_latent_stats, best weights,
training split of the run's own dataset and resolution). Only per_channel is
added: the global scale_factor / shift_factor already recorded by the run are
kept as they are, so a phase-2 run that used them still resolves the same
values. Writes into the run's checkpoints/best.pt (latent_stats) and
artifacts/training_info.json (scale_factor).
"""

import argparse
import datetime
import json
import logging
import os
import tempfile

import torch
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from src.config import TrainCfg, config_from_cli
from src.training.run_outputs import write_json
from src.training.train_autoencoder import (compute_latent_stats, instantiate_autoencoder,
                                            log_latent_stats, setup_dataloader)
from src.training.runtime import AutoencoderRuntime

logger = logging.getLogger(__name__)


def known_keys_only(saved,
                    schema,
                    prefix=""):
    """
    The keys of a saved config that the current schema still has. A run saved
    before a key was removed (e.g. run.workdir) would otherwise not reload.
    Returns (filtered config, dropped keys).
    """
    kept, dropped = {}, []
    for key, value in saved.items():
        if key not in schema:
            dropped.append(prefix + key)
        elif isinstance(value, dict) and OmegaConf.is_dict(schema[key]):
            kept[key], inner = known_keys_only(value,
                                               schema[key],
                                               prefix + key + ".")
            dropped += inner
        else:
            kept[key] = value
    return kept, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir",
                        required=True,
                        help="autoencoder run folder, registry/outputs/autoencoder/<run>")
    run_dir = os.path.abspath(parser.parse_args().run_dir)
    best_path = os.path.join(run_dir, "checkpoints", "best.pt")
    info_path = os.path.join(run_dir, "artifacts", "training_info.json")
    for path in (best_path, info_path):
        if not os.path.isfile(path):
            raise SystemExit(f"{path} not found: not a completed autoencoder run")

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        level=logging.INFO)
    # the run's own config: same dataset, resolution and number of batches
    saved = OmegaConf.to_container(OmegaConf.load(os.path.join(run_dir, "artifacts", "config.yaml")))
    saved, dropped = known_keys_only(saved,
                                     OmegaConf.structured(TrainCfg))
    if dropped:
        logger.info(f"keys of the run's config.yaml no longer in the schema, ignored: {dropped}")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        OmegaConf.save(OmegaConf.create(saved), fh.name)
    try:
        cfg = config_from_cli(["--project_name", "measure_latent_stats",
                               "--cfgdir", fh.name])
    finally:
        os.remove(fh.name)
    set_seed(cfg.run.seed)
    rt = AutoencoderRuntime(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                            datenow=os.path.basename(run_dir))

    loaders, _, n_channels = setup_dataloader(cfg)
    vae = instantiate_autoencoder(cfg,
                                  rt,
                                  n_channels)
    ckpt = torch.load(best_path,
                      map_location="cpu",
                      weights_only=False)
    vae.load_state_dict(ckpt["model_state_dict"])
    measured = compute_latent_stats(cfg,
                                    rt.device,
                                    vae,
                                    loaders["train"])
    log_latent_stats(measured)

    with open(info_path, encoding="utf-8") as fh:
        info = json.load(fh)
    recorded = info.get("scale_factor") or {}
    if recorded:
        # the global values stay those of the run (shift_factor = its latent
        # mean, as phase 2 already resolves it for such a run)
        stats = {"shift_factor": recorded["latent_mean"],
                 **recorded}
    else:
        stats = {**measured,
                 "weights": f"best.pt (epoch {ckpt.get('epoch')})"}
    stats["per_channel"] = measured["per_channel"]
    stats["per_channel_measured_by"] = (f"src/scripts/measure_latent_stats.py on "
                                        f"{datetime.datetime.now().isoformat(timespec='seconds')}, "
                                        f"{measured['n_images']} images")
    if recorded:
        logger.info(f"kept the global values recorded by the run: scale_factor="
                    f"{stats['value']:.6f}, shift_factor={stats['shift_factor']:+.6f} "
                    f"(this measurement: {measured['value']:.6f}, {measured['shift_factor']:+.6f})")

    ckpt["latent_stats"] = stats
    ckpt.setdefault("scale_factor", stats["value"])
    ckpt.setdefault("shift_factor", stats["shift_factor"])
    torch.save(ckpt, best_path)
    info["scale_factor"] = stats
    write_json(info_path,
               info)
    logger.info(f"per-channel statistics written into {best_path} and {info_path}")


if __name__ == "__main__":
    main()
