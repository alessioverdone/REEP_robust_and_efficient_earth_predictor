"""
Hyperparameter search with Optuna over the training scripts of either phase.

    python src/scripts/run_sweep.py                                (or Run from the IDE)
    python src/scripts/run_sweep.py --sweep sweep_diffusion_0.yaml

The sweep yaml (a name in registry/configurations/ or a path) defines the
study: see SweepCfg in src/training/sweep.py. Rerunning the same yaml resumes
the study and runs only the missing trials. Results go into
registry/outputs/sweeps/<name>/ (trials.csv, best.json, one folder per trial).
"""

import argparse

from src.training.sweep import run_sweep


def main(sweep_file: str = "sweep_diffusion_0.yaml"):
    parser = argparse.ArgumentParser(description="Hyperparameter search with Optuna.")
    parser.add_argument("--sweep",
                        type=str,
                        default=sweep_file,
                        help="sweep yaml: a name in registry/configurations/ or a path "
                             f"(default: {sweep_file})")
    args = parser.parse_args()
    run_sweep(args.sweep)


if __name__ == "__main__":
    main()
