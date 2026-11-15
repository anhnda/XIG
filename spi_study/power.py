"""
Power analysis (Giai đoạn 0, protocol 6.5).

Đọc kết quả từng ảnh đã có trên 200 ảnh (JSONL của runner cũ `experiments.run`, ví dụ
runs/axis_resnet50_black.jsonl), lấy trung bình qua seed theo ảnh, tính
d_j = IoU(SPI 8×24) − IoU(IG N=24), SD_d, rồi n = ((z_{0.9875} + z_{0.8})·SD_d / δ)².
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict

import numpy as np

from .stats import n_required, z


def per_image(path: str, method: str, metric: str) -> dict:
    acc = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("error") or r.get("method") != method:
                continue
            v = r.get(metric)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            acc[r["image"]].append(float(v))
    return {im: float(np.mean(v)) for im, v in acc.items()}


def run_power(cfg: dict, old_jsonl: str, out_dir: str) -> dict:
    pc, gc = cfg["power"], cfg["gate1"]
    if not old_jsonl or not os.path.exists(old_jsonl):
        raise SystemExit(f"cần JSONL cũ (local.old_jsonl hoặc --old-jsonl), nhận {old_jsonl!r}")
    spi = per_image(old_jsonl, pc["spi_method"], pc["metric"])
    ig = per_image(old_jsonl, pc["ig_method"], pc["metric"])
    common = sorted(set(spi) & set(ig))
    if len(common) < 10:
        raise SystemExit(f"chỉ {len(common)} ảnh chung giữa '{pc['spi_method']}' và '{pc['ig_method']}' trong {old_jsonl}")
    d = np.array([spi[i] - ig[i] for i in common])
    sd = float(d.std(ddof=1))
    delta, pw = float(gc["delta"]), float(gc["power"])
    alpha2 = 1.0 - float(gc["conf"])
    n = n_required(sd, delta, alpha2, pw)
    grid = {s: n_required(s, delta, alpha2, pw) for s in (0.08, 0.10, 0.12, 0.15)}
    lines = ["# Power analysis (Giai đoạn 0)\n",
             f"Nguồn: `{old_jsonl}`; SPI = `{pc['spi_method']}`, IG = `{pc['ig_method']}`, metric = {pc['metric']}.\n",
             f"- Số ảnh chung: {len(common)}",
             f"- Trung bình d = {d.mean():+.4f}, SD_d = {sd:.4f}",
             f"- δ = {delta}, mức hai phía α = {alpha2:.4f} (z = {z(1 - alpha2 / 2):.4f}), power = {pw}",
             f"- **Số ảnh cần: n = {n}**\n",
             "| SD_d | " + " | ".join(f"{s}" for s in grid) + " |",
             "|---|" + "---|" * len(grid),
             "| n | " + " | ".join(str(v) for v in grid.values()) + " |\n",
             "Ghi n vào `data.imagenet.n_images` trong config.yaml TRƯỚC khi khoá.",
             "Lưu ý: d ở đây đo trên estimator 8×24 so với IG N=24 (cùng ngân sách cũ), không phải tham chiếu population;",
             "SD_d của tham chiếu population có thể khác. Gate 1 tính lại power từ SD_d thật."]
    text = "\n".join(lines) + "\n"
    os.makedirs(out_dir, exist_ok=True)
    rec = {"n_common": len(common), "mean_d": float(d.mean()), "sd_d": sd, "n_required": n, "grid": grid,
           "source": old_jsonl, "spi_method": pc["spi_method"], "ig_method": pc["ig_method"]}
    with open(os.path.join(out_dir, "power.md"), "w") as fh:
        fh.write(text)
    with open(os.path.join(out_dir, "power.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    print(text)
    return rec
