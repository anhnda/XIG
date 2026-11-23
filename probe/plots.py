"""Thiết lập matplotlib dùng chung (Agg, kiểu trục thống nhất)."""

from __future__ import annotations


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e4e3dc", "grid.linewidth": 0.6,
                         "axes.edgecolor": "#b5b4ab", "font.size": 9, "lines.linewidth": 2,
                         "lines.markersize": 5, "legend.frameon": False})
    return plt
