"""
Shared set-up of the test suite.

    python -m pytest                    (from the project root)

- The project root goes on sys.path, so the tests import `src...` as the
  scripts do.
- mamba_ssm runs only on NVIDIA GPUs (its selective scan is a CUDA kernel).
  Without a GPU, or without the package, tests/stubs goes on the path and
  `mamba_ssm.Mamba` is a CPU stand-in (see tests/stubs/mamba_ssm). The tests of
  everything AROUND the Mamba blocks then run on any machine; the tests marked
  `cuda` use the real kernels and are skipped without them.
- `earthnet` builds, once per session, a small synthetic EarthNet dataset and
  runs the REEP preprocessing on it: the tests of the condition phase read the
  dataset.json and cond_stats.json the real pipeline writes.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
STUBS_DIR = Path(__file__).resolve().parent / "stubs"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def _real_mamba_available() -> bool:
    """True when the real Mamba kernels can run: a CUDA device and mamba_ssm installed."""
    if not torch.cuda.is_available():
        return False
    try:
        import mamba_ssm  # noqa: F401
    except ImportError:
        return False
    return True


REAL_MAMBA = _real_mamba_available()
if not REAL_MAMBA:
    sys.path.insert(0, str(STUBS_DIR))


@pytest.fixture
def device() -> torch.device:
    """cuda with the real Mamba kernels, cpu with the stand-in."""
    return torch.device("cuda" if REAL_MAMBA else "cpu")


def script_env() -> dict:
    """Environment of the scripts launched by the tests: the project (and the stand-in) on the path."""
    paths = [str(PROJECT_DIR), str(PROJECT_DIR / "src" / "model")]
    if not REAL_MAMBA:
        paths.append(str(STUBS_DIR))
    # tiny runs: one thread is enough, and it avoids the OpenMP clashes of some conda envs
    return {**os.environ, "PYTHONPATH": os.pathsep.join(paths), "OMP_NUM_THREADS": "1"}


def run_script(*args, timeout: int = 600) -> subprocess.CompletedProcess:
    """Runs a script of the project with python; fails the test with its output if it fails."""
    result = subprocess.run([sys.executable, *map(str, args)], cwd=PROJECT_DIR, env=script_env(),
                            capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, f"{args[0]} failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
    return result


@pytest.fixture(scope="session")
def earthnet(tmp_path_factory) -> dict:
    """
    A synthetic EarthNet dataset, preprocessed by src/dataset/preprocessing.py:
    {"root", "overrides"}, where overrides are the --set values that point the
    configuration at it (raw cubes, preprocessed folder, run outputs).
    """
    from tests.earthnet_fixtures import make_earthnet

    root = tmp_path_factory.mktemp("earthnet")
    raw_train = make_earthnet(root)
    overrides = [f"paths.raw_dir={raw_train}",
                 f"paths.preprocessed_dir={root / 'preprocessed'}",
                 f"paths.outputs_dir={root / 'outputs'}",
                 "preprocessing.limit=0",
                 "preprocessing.val_frac=0.34",
                 "preprocessing.workers=1",
                 "condition.extract_workers=1"]
    run_script("src/dataset/preprocessing.py", "--stats-only", "--force", "--set", *overrides)
    return {"root": root, "overrides": overrides}


def pytest_report_header(config):
    return ("mamba_ssm: real CUDA kernels" if REAL_MAMBA
            else "mamba_ssm: CPU stand-in from tests/stubs (tests marked `cuda` are skipped)")


def pytest_collection_modifyitems(config, items):
    if REAL_MAMBA:
        return
    skip = pytest.mark.skip(reason="needs an NVIDIA GPU with mamba_ssm installed")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip)
