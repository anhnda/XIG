"""Tiện ích báo cáo: đọc CSV, định dạng số, bảng markdown."""

from __future__ import annotations

import math
import os

import numpy as np


def _read(out, name):
    import pandas as pd
    p = os.path.join(out, f"{name}.csv")
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return None


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return "✓" if v else "✗"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf"
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e4 or a < 1e-3:
        return f"{v:.2e}"
    return f"{v:.3f}" if a < 10 else f"{v:.1f}"


def md(df, cols=None, index=False):
    if df is None or not len(df):
        return "_(không có dữ liệu)_\n"
    d = df.reset_index() if index else df
    cols = cols or list(d.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in d[cols].iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"
