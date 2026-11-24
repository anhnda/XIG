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


def _fake_ext_meta(cfg, stage, configs, n_img, rng):
    out = os.path.join(PFM.root(cfg), stage)
    rows = []
    for i in range(n_img):
        im = f"seg{90000 + i:05d}"
        for c in configs:
            nm = PFM.cname(c)
            ests = ["mean_all"] + [f"{cp}_B{B}_K{B // c['N']}" for B in (256, 1024) for cp in ("rev", "indep") if (B // c["N"]) % 2 == 0]
            for e in ests:
                rows.append({"stage": stage, "image": im, "config": nm, **c, "estimator": e,
                             **{m: float(rng.uniform(0, 1)) for m in ("insdel_blur", "road", "road_morf", "road_lerf", "iou", "pointing", "auroc_mask", "ap_mask")}})
    pd.DataFrame(rows).to_csv(os.path.join(out, "ext.csv"), index=False)
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump({"time": "2999-01-01T00:00:00", "images": [f"seg{90000 + i:05d}" if stage == "test" else f"seg{80000 + i:05d}"
                                                               for i in range(n_img)], "sc": {"configs": configs}}, fh, default=str)


def test_verify_diag_freeze_offline(tmp_path):
    import pytest
    from probe import pathfam_verify as PV
    cfg = C.load(os.path.join(ROOT, "config.yaml"))
    cfg["local"] = {**cfg.get("local", {}), "out": str(tmp_path / "runs")}
    cfg["pathfam"]["protocol"]["bootstrap"] = 300
    rng = np.random.default_rng(1)
    _fake_stage(cfg, "dev", PFM.dev_configs(cfg), 24, ["del_sq"], rng)
    _fake_ext_meta(cfg, "dev", [], 24, rng)
    PFM.select(cfg)
    PFM.lock(cfg)
    n = json.load(open(PFM.lock_path(cfg)))["payload"]["n_test"]
    tc = PFM.test_configs(cfg, PFM.load_sel(cfg))
    _fake_stage(cfg, "test", tc, n, ["del_sq"], rng)
    _fake_ext_meta(cfg, "test", tc, n, rng)
    PFM.write_report(cfg)
    txt = open(PV.verify(cfg)).read()
    assert "| digest khoá = digest payload tính lại | ✓" in txt
    assert "| số ảnh test = n trong khoá | ✓" in txt
    t = pd.read_csv(os.path.join(PFM.root(cfg), "test", "pairs.csv"))
    assert len(t)
    # sai khác với runner ≈ 0
    import re
    diffs = [float(x) for x in re.findall(r"\| (\d\.\d+e-\d+|0) \|", txt)]
    assert all(d < 1e-9 for d in diffs)
    PV.diag(cfg)
    assert os.path.exists(os.path.join(PFM.root(cfg), "PATHFAM_J_VS_EXTERNAL.md"))
    PV.freeze(cfg)
    with pytest.raises(SystemExit):
        PFM.lock(cfg)


def test_independent_stats_match_runner():
    from probe import pathfam_verify as PV
    rng = np.random.default_rng(3)
    ji = rng.lognormal(2.5, 0.4, 64)
    jf = ji * (1 + rng.normal(0.02, 0.05, 64))
    a = PV.three_tests(ji, jf, 0.10, 2000, 61)
    r = PFM.tests_for(ji, jf, 0.10, 2000, 61, 0.05)
    assert abs(a["g"] - r["g"]) < 1e-12
    for k in ("ni", "sup", "tost"):
        assert abs(a[f"p_{k}_bt"] - r[f"p_{k}"]) < 1e-12
    from probe.infid_posthoc import holm
    ps = [0.01, 0.04, 0.03, 0.2]
    assert np.allclose(PV.my_holm(ps), holm(ps))
    assert abs(PV.j_from_raw(np.array([1.0]), np.array([2.0]), np.array([3.0]), 4, "rev")[0] - 2.0) < 1e-12
