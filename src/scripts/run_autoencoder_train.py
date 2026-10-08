"""
Fase 1: addestramento dell'autoencoder (il first stage congelato della fase 2).

    python src/scripts/run_autoencoder_train.py                      (o Run dall'IDE)
    python src/scripts/run_autoencoder_train.py --project_name vae_v1 \
        --cfgdir configuration_0.yaml --set autoencoder.num_epochs=50

Every run writes into registry/outputs/autoencoder/<start timestamp>/, see
src/training/run_outputs.py.
"""

from src.config import config_from_cli
from src.training.run_outputs import create_run_outputs, save_run_config
from src.training.runtime import setup_autoencoder_runtime
from src.training.tracking import start_tracker
from src.training.train_autoencoder import (configure_train, describe_run,
                                            instantiate_autoencoder, mark_run_ended,
                                            measure_scale_factor, prepare_training,
                                            setup_dataloader, test_vae, train_vae)

# Lancio dall'IDE (Run/Debug, che non passa argomenti): valgono questi. Da
# terminale --cfgdir e --project_name li scavalcano, e config_from_cli scrive a
# video quando sta usando il ripiego al posto della riga di comando.
DEFAULT_CONFIG_FILE = "configuration_0.yaml"
DEFAULT_PROJECT_NAME = "autoencoder_debug"


def main(config_file=DEFAULT_CONFIG_FILE,
         project_name=DEFAULT_PROJECT_NAME):
    # Configuration
    cfg = config_from_cli(config_file=config_file,       # schema di src/config.py <- yaml di --cfgdir <- flag <- --set, validato
                          project_name=project_name)
    outputs = create_run_outputs(cfg.autoencoder.outputs_dir)  # registry/outputs/autoencoder/<timestamp>/{artifacts,checkpoints,results}
    rt = setup_autoencoder_runtime(cfg, outputs)         # device, seed, tf32, log in artifacts/train.log
    save_run_config(cfg, outputs)                        # artifacts/config.json + config.yaml

    # Setup Dataloader
    loaders, composition, n_channels = setup_dataloader(cfg)

    # Get model
    vae = instantiate_autoencoder(cfg, rt, n_channels)

    # Prepare training
    optimizer, scheduler = prepare_training(cfg, vae)

    # Configure training
    # state = da che epoca ripartire, miglior valore storico della metrica di
    # autoencoder.save_best_by, metriche e info che finiscono in training_info.json.
    state = configure_train(cfg, outputs, vae, optimizer, scheduler)
    describe_run(cfg, rt, outputs, state, vae, loaders, composition)
    rt.tracker = start_tracker(cfg,                      # live report on W&B (wandb.enabled)
                               outputs,
                               "autoencoder")

    # Training Loop + test of the best checkpoint + scale factor
    try:
        train_vae(cfg,
                  rt,
                  outputs,
                  state,
                  vae,
                  optimizer,
                  scheduler,
                  loaders)
        test_vae(cfg,
                 rt,
                 outputs,
                 state,
                 vae,
                 loaders)
        # diffusion.scale_factor of the best weights, saved in best.pt for phase 2
        measure_scale_factor(cfg,
                             rt,
                             outputs,
                             state,
                             vae,
                             loaders)
    except BaseException as error:
        mark_run_ended(cfg,
                       rt,
                       outputs,
                       state,
                       optimizer,
                       error=error)
        raise
    mark_run_ended(cfg,
                   rt,
                   outputs,
                   state,
                   optimizer)


if __name__ == "__main__":
    main()
