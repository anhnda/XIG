"""
PATH_FAMILIES Phase 1 — kiểm tích hợp phần KHÔNG cần GPU của runner trên dữ liệu tổng hợp: pairs.csv / emp.csv / cost.csv giả
cho dev và test ⇒ select → power_table → lock_payload → write_report. Không mô hình, không ảnh.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from probe import config as C  # noqa: E402
from probe import pathfam as PFM  # noqa: E402


def _fake_stage(cfg, stage, configs, n_img, dsets, rng):
    out = os.path.join(PFM.root(cfg), stage)
    os.makedirs(out, exist_ok=True)
    pairs, emp, cost = [], [], []
    for i in range(n_img):
        im = f"seg{90000 + i:05d}"
        scale = rng.lognormal(0, 0.5)
        for c in configs:
            nm = PFM.cname(c)
            h = PFM.h_of(c)
            base = {"stage": stage, "image": im, "config": nm, **c, "rule": "rand", "M": 1}
            for dn in dsets:
                for s_ in ("sel", "eval"):
                    L = scale * (10 + 8 * (1 - h) ** 2 + rng.normal(0, 0.3))          # L(μ) giảm theo h
                    Ve = scale * (5 + 80 * h ** 2) * (1.3 if c["family"] == "cmix" else 1.0)
                    Vo = Ve * 1.05
                    pairs.append({**base, "dset": dn, "set": s_, "group": "all", "L_mu": L, "Ve": Ve, "Vo": Vo})
                    if s_ == "eval" and dn == dsets[0]:
                        e = {**base, "dset": dn, "m": 16}
                        for K in (4, 16):
                            for cp in ("indep", "rev"):
                                jf = L + ((Ve + Vo) if cp == "indep" else 2 * Ve) / K
                                e[f"Jf_{cp}_{K}"] = jf
                                e[f"Jemp_{cp}_{K}"] = jf + rng.normal(0, 0.05 * jf)
                                e[f"Jemp_{cp}_{K}_ng"] = 4
                        emp.append(e)
            cost.append({**base, "grads": 32 * c["N"], "seconds": 1.0, "paths": 32})
    for nm, rows in (("pairs", pairs), ("emp", emp), ("cost", cost)):
        pd.DataFrame(rows).to_csv(os.path.join(out, f"{nm}.csv"), index=False)


def test_runner_offline_pipeline(tmp_path):
    cfg = C.load(os.path.join(ROOT, "config.yaml"))
    cfg["local"] = {**cfg.get("local", {}), "out": str(tmp_path / "runs")}
    cfg["pathfam"]["protocol"]["bootstrap"] = 300
    rng = np.random.default_rng(0)
    _fake_stage(cfg, "dev", PFM.dev_configs(cfg), 24, ["del_sq"], rng)
    PFM.select(cfg)
    sel = PFM.load_sel(cfg)
    for B in ("256", "1024"):
        assert set(sel["chosen"][B]) == {"ln", "cmix", "smooth", "spi"}
        for fam, s in sel["chosen"][B].items():
            assert s["config"].startswith(fam + "_")
            assert abs(sum(s["p_select"].values()) - 1) < 1e-9
    pt = PFM.power_table(cfg, sel)
    assert len(pt["rows"]) == 4 and 48 <= pt["n_test"] <= 128
    pl = PFM.lock_payload(cfg)
    assert pl["n_test"] == pt["n_test"] and "spi_a1_N32" in pl["test_configs"]
    tc = PFM.test_configs(cfg, sel)
    _fake_stage(cfg, "test", tc, 40, ["del_sq", "ins_sq", "sym_sq"], rng)
    rp = PFM.write_report(cfg)
    txt = open(rp).read()
    assert "NI" in txt and "J_indep" in txt and "Cùng exploration h" in txt and "Kiểm toán chi phí" in txt
    assert "KHÔNG KÉM HƠN" in txt or "CHƯA KẾT LUẬN" in txt or "KÉM HƠN" in txt
    json.dumps(pl, default=str)
