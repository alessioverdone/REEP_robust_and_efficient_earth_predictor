"""
Fase 2: addestramento del diffusion model

    python src/scripts/run_diffusion_train.py                      (o Run dall'IDE)
    accelerate launch src/scripts/run_diffusion_train.py --project_name diffusion_v1 \
        --cfgdir configuration_0.yaml

Every run writes into registry/outputs/diffusion/<start timestamp>/, see
src/training/run_outputs.py.
"""
import sys
from pathlib import Path

from accelerate.logging import get_logger

from src.config import config_from_cli
from src.model.diffusion_utils import (configure_train, describe_run, instantiate_models,
                                       mark_run_ended, prepare_training, setup_dataloader,
                                       train_loop)
from src.training.ens_eval import check_ens_dependencies, evaluate_test_splits, release_gpu_memory
from src.training.run_outputs import create_shared_run_outputs
from src.training.runtime import setup_runtime
from src.training.tracking import start_tracker

logger = get_logger(__name__, log_level="INFO")

# Lancio dall'IDE (Run/Debug, che non passa argomenti): valgono questi. Da
# terminale --cfgdir e --project_name li scavalcano, e config_from_cli scrive a
# video quando sta usando il ripiego al posto della riga di comando.
DEFAULT_CONFIG_FILE = "configuration_0.yaml"
DEFAULT_PROJECT_NAME = "diffusion_debug"


def add_sgm_to_path():
    # The diffusion config instantiates its modules from strings like
    # "sgm.modules...", importable only with src/model on the path: the
    # equivalent of `export PYTHONPATH=src/model`, without the shell.
    model_dir = str(Path(__file__).resolve().parents[1] / "model")
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)


def main(config_file=DEFAULT_CONFIG_FILE,
         project_name=DEFAULT_PROJECT_NAME):
    add_sgm_to_path()

    # Configuration
    cfg = config_from_cli(config_file=config_file,       # schema di src/config.py <- yaml di --cfgdir <- flag <- --set, validato
                          project_name=project_name)
    outputs = create_shared_run_outputs(cfg.diffusion.outputs_dir,  # registry/outputs/diffusion/<timestamp>/{artifacts,checkpoints,results}
                                        "forecasts")
    rt = setup_runtime(cfg,                              # accelerator, dtype, seed, tf32, log in artifacts/train.log
                       outputs)

    # The ENS (in the loop, and in the test evaluation at the end) needs earthnet and
    # scikit-image. With ens.enabled a missing one stops the run now; with only test_eval
    # it is a warning (the training does not need them, the evaluation would just fail).
    if cfg.ens.enabled:
        check_ens_dependencies()
    elif cfg.test_eval.enabled:
        try:
            check_ens_dependencies()
        except ImportError as error:
            logger.warning(f"[test_eval] it will fail at the end of the training: {error}")

    # Get model
    model, conditioner, first_stage_model, first_stage_info = instantiate_models(cfg,
                                                                                 rt)

    # Setup Dataloader
    train_dataloader, val_dataloader, train_len, overrode_max_train_steps = setup_dataloader(cfg,
                                                                                             rt,
                                                                                             first_stage_info,
                                                                                             model,
                                                                                             conditioner,
                                                                                             first_stage_model)

    # Prepare training
    # model, conditioner and dataloaders are replaced by their accelerator.prepare() versions
    (num_update_steps_per_epoch, denoiser,
     optimizer, lr_scheduler, loss_fn, sampler,
     model, conditioner, train_dataloader, val_dataloader) = prepare_training(cfg,
                                                                              rt,
                                                                              outputs,
                                                                              model,
                                                                              conditioner,
                                                                              train_dataloader,
                                                                              val_dataloader,
                                                                              overrode_max_train_steps)

    # Configure training
    # state = a che punto e' il training (epoca e step da cui ripartire, minimo
    # storico di latter_rmse) e cosa finisce in training_info.json, history.csv
    # e nelle figure. Saves artifacts/config.json + config.yaml, with the VAE
    # path, scale_factor and resumed checkpoint already resolved.
    state = configure_train(cfg,
                            rt,
                            outputs,
                            model,
                            conditioner,
                            num_update_steps_per_epoch,
                            denoiser)
    describe_run(cfg,
                 rt,
                 outputs,
                 state,
                 model,
                 conditioner,
                 first_stage_model,
                 first_stage_info,
                 train_dataloader,
                 val_dataloader)
    # live report on W&B (wandb.enabled), opened with the config fully resolved
    rt.tracker = start_tracker(cfg,
                               outputs,
                               "diffusion",
                               rt.accelerator.is_main_process)

    # Training Loop, then the test-set evaluation. Both are inside the try, so that an
    # interruption (e.g. Ctrl+C during the evaluation, which can take hours) closes
    # training_info.json as "interrupted" instead of leaving it on "running".
    try:
        train_loop(cfg,
                   rt,
                   outputs,
                   state,
                   optimizer,
                   lr_scheduler,
                   conditioner,
                   denoiser,
                   sampler,
                   model,
                   first_stage_model,
                   train_dataloader,
                   val_dataloader,
                   loss_fn)

        # Test-set evaluation (config section test_eval): evaluate_ens.py on iid / ood /
        # extreme / seasonal, reloading the checkpoint from disk. The training objects are
        # dropped first (the scheduler still refers to the optimizer, so the parameters
        # of the DiT stay on the GPU next to the copy the evaluation loads: a few hundred
        # MB for a DiT-B). A failure of the evaluation itself is logged and does not fail
        # the run (see evaluate_test_splits).
        if cfg.test_eval.enabled:
            del (model, conditioner, first_stage_model, optimizer, denoiser, sampler,
                 loss_fn, train_dataloader, val_dataloader)
            release_gpu_memory(rt.accelerator)
            evaluate_test_splits(cfg, rt, outputs, state)
    except BaseException as error:
        mark_run_ended(cfg,
                       rt,
                       outputs,
                       state,
                       lr_scheduler,
                       error=error)
        raise
    mark_run_ended(cfg,
                   rt,
                   outputs,
                   state,
                   lr_scheduler)


if __name__ == "__main__":
    main()
