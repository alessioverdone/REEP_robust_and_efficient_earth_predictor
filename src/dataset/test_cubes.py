"""
The cubes of the EarthNet test splits (iid, ood, extreme, seasonal), read straight from
the original files.

A test cube is split in two files, data/raw/<split>_test_split/context/<tile>/context_<cube>.npz
and .../target/<tile>/target_<cube>.npz. The dataset of the diffusion model reads one
sequence per cube (context frames first, then the target frames), so the two files are
joined in memory, in the worker that reads the cube, right before the model sees it:
nothing is written to disk. (The evaluation used to save a joined copy of every cube,
weather included, which took hundreds of GB.)

The cube is identified by a virtual path, <split_dir>/merged/<tile>/<cube>.npz, that never
exists on disk: it only names the cube in the batches and in the results, and sorts the
cubes like the files did, so --subset and the exported cubes pick the same ones as before.

    pairs   = list_test_pairs(test_dir)                   [(virtual path, context, target)]
    dataset = EarthNetTestDataset(files=[...], pairs={str(virtual): (context, target)}, ...)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.dataset.dataloader_s2_dif import (HIGHRES_DYNAMIC_KEY, HIGHRES_STATIC_KEY,
                                           MESO_DYNAMIC_KEY, S2DiffusionCubeDataset)
from src.utils.ens_metrics import cloud_mask_tensor

logger = logging.getLogger(__name__)

# folder name of the virtual path of a cube (see the module docstring)
VIRTUAL_DIRNAME = "merged"

# virtual path of a cube -> (context file, target file)
CubePairs = Mapping[str, Tuple[Path, Path]]


def list_test_pairs(test_dir: Path) -> List[Tuple[Path, Path, Path]]:
    """
    (virtual path, context file, target file) of every cube of a test split. The two
    files live in context/<tile>/ and target/<tile>/ with the same name except for the
    prefix (context_<cube>.npz, target_<cube>.npz).
    """
    ctx_root, tgt_root = test_dir / "context", test_dir / "target"
    for root in (ctx_root, tgt_root):
        if not root.is_dir():
            raise SystemExit(f"{root} not found: download the test split in {test_dir} "
                             f"(or pass --test-root).")

    def key(path: Path, root: Path) -> Tuple[str, str]:
        rel = path.relative_to(root)
        return rel.parent.as_posix(), rel.name.split("_", 1)[-1]

    targets = {key(p, tgt_root): p for p in tgt_root.rglob("*.npz")}
    pairs, missing = [], 0
    for ctx in sorted(ctx_root.rglob("*.npz")):
        tile, cube = key(ctx, ctx_root)
        tgt = targets.get((tile, cube))
        if tgt is None:
            missing += 1
            continue
        pairs.append((test_dir / VIRTUAL_DIRNAME / tile / cube, ctx, tgt))
    if missing:
        logger.warning(f"{missing} context files without the matching target in {tgt_root}")
    if not pairs:
        raise SystemExit(f"no context/target pair in {test_dir}")
    return pairs


def read_test_cube(ctx_path: Path, tgt_path: Path, n_context: int, n_target: int,
                   n_bands: int, steps_per_frame: int
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    (highresdynamic, highresstatic, mesodynamic, target highresdynamic) of a test cube,
    in the format the training dataset reads from one file:

    - highresdynamic: the frames of the context and the target joined on the time
      axis, keeping the first n_bands bands (blue, green, red, nir; the others are masks);
    - highresstatic, mesostatic: from the context;
    - mesodynamic: the days of the context and those of the target joined, unless the
      context already covers all of them;
    - the last array is the highresdynamic of the target with ALL its channels: the cloud
      channels of the frames to predict, for the cloud mask.

    If something does not fit (frames, bands, days of weather) it raises ValueError with
    the shapes, instead of producing a shifted cube.
    """
    with np.load(ctx_path) as ctx, np.load(tgt_path) as tgt:
        hd_c, hd_t = ctx[HIGHRES_DYNAMIC_KEY], tgt[HIGHRES_DYNAMIC_KEY]
        if hd_c.shape[3] != n_context or hd_t.shape[3] != n_target:
            raise ValueError(
                f"{ctx_path.name}: expected {n_context} context frames and {n_target} target "
                f"frames, found {hd_c.shape[3]} and {hd_t.shape[3]} "
                f"(shape {hd_c.shape} / {hd_t.shape})")
        if min(hd_c.shape[2], hd_t.shape[2]) < n_bands:
            raise ValueError(f"{ctx_path.name}: {n_bands} bands needed, found "
                             f"{hd_c.shape[2]} in the context and {hd_t.shape[2]} in the target")

        hd = np.concatenate([hd_c[:, :, :n_bands], hd_t[:, :, :n_bands]], axis=3)

        static_src = ctx if HIGHRES_STATIC_KEY in ctx.files else tgt
        if HIGHRES_STATIC_KEY not in static_src.files:
            raise ValueError(f"{ctx_path.name}: {HIGHRES_STATIC_KEY} missing in context and target")
        hs = static_src[HIGHRES_STATIC_KEY]

        need = (n_context + n_target) * steps_per_frame
        md_c = ctx[MESO_DYNAMIC_KEY]
        md_t = tgt[MESO_DYNAMIC_KEY] if MESO_DYNAMIC_KEY in tgt.files else None
        if md_c.shape[3] >= need:
            md = md_c
        elif md_t is not None and md_c.shape[3] + md_t.shape[3] == need:
            md = np.concatenate([md_c, md_t], axis=3)
        else:
            raise ValueError(
                f"{ctx_path.name}: {need} days of {MESO_DYNAMIC_KEY} needed "
                f"({n_context + n_target} frames x {steps_per_frame}), found "
                f"{md_c.shape[3]} in the context and "
                f"{'none' if md_t is None else md_t.shape[3]} in the target")
    return hd, hs, md, hd_t


class EarthNetTestDataset(S2DiffusionCubeDataset):
    """
    S2DiffusionCubeDataset over test cubes made of two files. `files` are the virtual
    paths (the "cube_path" of the items), `pairs` maps each one to its (context, target)
    files; the frame windows are the dataset's own (n_context = T - n_predict, n_predict).

    With `cloud` = (threshold, prob_scale) every item also carries "cloud_mask", the valid
    pixels of the target frames, built from the target array already in memory: the cube
    is read once.
    """

    def __init__(self, files: Sequence[Path], pairs: CubePairs,
                 cloud: Optional[Tuple[float, float]] = None, **kwargs):
        super().__init__(files=files, **kwargs)
        missing = [str(f) for f in self.files if str(f) not in pairs]
        if missing:
            raise KeyError(f"no context/target pair for {len(missing)} cubes, e.g. {missing[0]}")
        self.pairs: Dict[str, Tuple[Path, Path]] = {str(f): tuple(pairs[str(f)])
                                                    for f in self.files}
        self.cloud = cloud

    def __getitem__(self, idx: int):
        path = self.files[idx]
        ctx_path, tgt_path = self.pairs[str(path)]
        hd, hs, md, hd_target = read_test_cube(Path(ctx_path), Path(tgt_path), self.n_context,
                                               self.n_predict, self.n_channels,
                                               self.steps_per_frame)
        item = self._build_item(path, hd, hs, md)
        if self.cloud is not None:
            mask = cloud_mask_tensor(hd_target, self.cloud[0], self.cloud[1], self.resize)
            if item.pop("hflip", False):
                import torch
                mask = torch.flip(mask, dims=[-1])
            item["cloud_mask"] = mask
        return item
