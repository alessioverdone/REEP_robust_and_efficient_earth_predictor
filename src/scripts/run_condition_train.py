"""
Condition Predictor (Mamba): training of the model that forecasts the weather of
the target frames from the weather of the context frames.

    python src/scripts/run_condition_train.py                      (o Run dall'IDE)
    python src/scripts/run_condition_train.py --project_name mamba_v1 \
        --cfgdir configuration_0.yaml --set condition.model.params.d_model=128

It needs the preprocessed dataset (src/dataset/preprocessing.py); the weather
series are extracted on the first launch (src/dataset/meso_series.py). Every run
writes into registry/outputs/condition/<start timestamp>/, see
src/training/run_outputs.py. The test splits are evaluated separately, once,
with src/scripts/evaluate_condition.py.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    # launched as a file (or Run from the IDE): the project root goes on the path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import config_from_cli
from src.training.run_outputs import create_run_outputs, save_run_config
from src.training.runtime import setup_condition_runtime
from src.training.tracking import start_tracker
from src.training.train_condition import (configure_train, describe_run, final_validation,
                                          instantiate_model, mark_run_ended, prepare_training,
                                          setup_dataloader, train_condition)

# Lancio dall'IDE (Run/Debug, che non passa argomenti): valgono questi. Da
# terminale --cfgdir e --project_name li scavalcano, e config_from_cli scrive a
# video quando sta usando il ripiego al posto della riga di comando.
DEFAULT_CONFIG_FILE = "configuration_0.yaml"
DEFAULT_PROJECT_NAME = "condition_debug"


def main(config_file=DEFAULT_CONFIG_FILE,
         project_name=DEFAULT_PROJECT_NAME):
    # Configuration
    cfg = config_from_cli(config_file=config_file,          # schema di src/config.py <- yaml di --cfgdir <- flag <- --set, validato
                          project_name=project_name)
    outputs = create_run_outputs(cfg.condition.outputs_dir,  # registry/outputs/condition/<timestamp>/{artifacts,checkpoints,results}
                                 "forecasts")
    rt = setup_condition_runtime(cfg, outputs)              # device, seed, tf32, log in artifacts/train.log
    save_run_config(cfg, outputs)                           # artifacts/config.json + config.yaml

    # Setup Dataloader
    data = setup_dataloader(cfg)

    # Get model
    model = instantiate_model(cfg, rt)

    # Prepare training
    optimizer, scheduler = prepare_training(cfg, model)

    # Configure training
    state = configure_train(cfg, outputs)
    describe_run(cfg, rt, outputs, state, model, data)
    rt.tracker = start_tracker(cfg,                         # live report on W&B (wandb.enabled)
                               outputs,
                               "condition")

    # Training Loop + final validation of the best weights
    try:
        train_condition(cfg,
                        rt,
                        outputs,
                        state,
                        model,
                        optimizer,
                        scheduler,
                        data)
        final_validation(cfg,
                         rt,
                         outputs,
                         state,
                         model,
                         data)
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
