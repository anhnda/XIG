"""
pathattr — path-attribution toolkit for the LIG revision (PyTorch).

    oracle      model access + forward/backward counting
    path_eval   the single, aligned evaluation of a discrete path (bug B1 fix)
    paths       straight, per-pixel schedules (dither = "dnoise"), SPI
    estimators  left / mid / random / Gauss–Legendre, cv_complete
    methods     IG, IDGI, Guided IG (faithful port), dither/dnoise, SPI
    metrics     insertion/deletion (any baseline), localisation, checkerboard
    data        ImageNet-Seg (gtsegs_ijcv.mat) and folder loaders
"""

from .oracle import CountingModule, Oracle
from .path_eval import PathEval, eval_path, legacy_df

__all__ = ["CountingModule", "Oracle", "PathEval", "eval_path", "legacy_df"]
