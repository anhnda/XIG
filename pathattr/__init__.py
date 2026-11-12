"""
pathattr — path-attribution toolkit (PyTorch). LIG has been dropped.

    oracle      model access + forward/backward counting
    path_eval   the single, aligned evaluation of a discrete path
    paths       straight, per-pixel schedules (dither = "dnoise"), SPI
    estimators  left / mid / random / Gauss–Legendre, cv_complete (any tensor shape)
    complete    completeness corrections; 'support' keeps Dummy exactly
    masks       partitions (SLIC / grid, multi-scale), Shapley-kernel masks, game values
    tpa         uniform (SPI) / Gibbs / entropic aggregation (core) / TPA-R, KernelSHAP
    tokens      token-level paths, [MASK] game, Shapley reference, ERASER scores
    methods     IG, IDGI, Guided IG, dither/dnoise, SPI, smooth IG, TPA, KernelSHAP
    metrics     ins/del, localisation, checkerboard, infidelity, approximate ROAD
    data        ImageNet-Seg (gtsegs_ijcv.mat) and folder loaders
"""

from .oracle import CountingModule, Oracle
from .path_eval import PathEval, eval_path, legacy_df

__all__ = ["CountingModule", "Oracle", "PathEval", "eval_path", "legacy_df"]
