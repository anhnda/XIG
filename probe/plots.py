"""Đồ thị cho báo cáo pilot (matplotlib, PNG). Màu theo họ đường (cố định), quy tắc theo kiểu nét/marker."""

from __future__ import annotations

import os

import numpy as np

PATH_COLOR = {"ig": "#2a78d6", "spi10": "#eb6834", "guided": "#1baf7a", "blur": "#eda100"}
RULE_STYLE = {"canon_left": (":", "v"), "canon_mid": ("-.", "s"), "canon_rand": ("-", "o"),
              "idgi": ("--", "D"), "idgi_raw": ((0, (1, 1)), "x"), "idgi_rand": ((0, (5, 1, 1, 1)), "^")}
MUTED = "#6b6a63"


def _color(p):
    return PATH_COLOR.get(p, MUTED)


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e4e3dc", "grid.linewidth": 0.6,
                         "axes.edgecolor": "#b5b4ab", "font.size": 9, "lines.linewidth": 2,
                         "lines.markersize": 5, "legend.frameon": False})
    return plt


def _read(out, name):
    from .report import _read as r
    return r(out, name)


def conv(out, plt):
    df = _read(out, "conv")
    if df is None:
        return None
    paths = list(dict.fromkeys(df["path"]))
    fig, ax = plt.subplots(2, len(paths), figsize=(3.4 * len(paths), 6), squeeze=False)
    Nmax = df["N"].max()
    for j, p in enumerate(paths):
        d = df[df.path == p]
        for r, g in d.groupby("rule"):
            ls, mk = RULE_STYLE.get(r, ("-", "o"))
            m = g.assign(a=g.resid_rel.abs()).groupby("N")["a"].median()
            ax[0, j].plot(m.index, m.values, ls=ls, marker=mk, color=_color(p) if r.startswith("canon") else MUTED,
                          label=r)
            h = g[g.N < Nmax].groupby("N")["dist_ref"].median()
            ax[1, j].plot(h.index, h.values, ls=ls, marker=mk, color=_color(p) if r.startswith("canon") else MUTED,
                          label=r)
        for i in range(2):
            ax[i, j].set_xscale("log", base=2)
            ax[i, j].set_yscale("symlog", linthresh=1e-4)
            ax[i, j].set_xlabel("N")
        ax[0, j].set_title(p)
    ax[0, 0].set_ylabel("median |ΣA − D| / |D|")
    ax[1, 0].set_ylabel(f"median ‖A_N − A_ref‖/‖A_ref‖ (ref: N={Nmax})")
    ax[0, -1].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    path = os.path.join(out, "plot_convergence.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def heatmaps(out, plt):
    import torch
    p = os.path.join(out, "heatmaps.pt")
    if not os.path.exists(p):
        return []
    H = torch.load(p, weights_only=False)
    files = []
    from .data import MEAN, STD
    for im, h in H.items():
        keys = list(h["maps"])
        paths = list(dict.fromkeys(k.split("|")[0] for k in keys))
        rls = list(dict.fromkeys(k.split("|")[1] for k in keys))
        fig, ax = plt.subplots(len(paths), len(rls) + 1, figsize=(2.0 * (len(rls) + 1), 2.0 * len(paths)),
                               squeeze=False)
        img = (h["x"][0].float() * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1)).clamp(0, 1)
        for i, pth in enumerate(paths):
            ax[i, 0].imshow(img.permute(1, 2, 0).numpy())
            ax[i, 0].contour(h["mask"].numpy().astype(float), levels=[0.5], colors="white", linewidths=0.8)
            ax[i, 0].set_ylabel(pth)
            for j, r in enumerate(rls):
                m = h["maps"].get(f"{pth}|{r}")
                a = ax[i, j + 1]
                if m is not None:
                    m = m.float().numpy()
                    a.imshow(m, cmap="Blues", vmin=0, vmax=np.quantile(m, 0.995) + 1e-30)
                if i == 0:
                    a.set_title(r, fontsize=8)
        for a in ax.flat:
            a.set_xticks([]); a.set_yticks([]); a.grid(False)
        fig.suptitle(f"{im}: Σ_c |A| (thang màu riêng từng ô, cắt ở phân vị 99.5%)", fontsize=9)
        fig.tight_layout()
        f = os.path.join(out, f"plot_heatmap_{im}.png")
        fig.savefig(f)
        plt.close(fig)
        files.append(f)
    return files


def idgi_conc(out, plt, primary):
    dg, mp, st = _read(out, "idgi_diag"), _read(out, "maps"), _read(out, "idgi_steps")
    if dg is None or mp is None:
        return None
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    agg = dg.groupby(["image", "path"]).agg(P_M=("P_M", "mean"), area=("area", "first"),
                                            top1=("top1_w", "mean")).reset_index()
    piv = mp[mp.rule.isin([primary, "idgi"])].pivot_table(index=["image", "path"], columns="rule", values="iou")
    piv = piv.reset_index()
    piv["dIoU"] = piv["idgi"] - piv[primary]
    m = agg.merge(piv, on=["image", "path"])
    for p, g in m.groupby("path"):
        ax[0].scatter(g.P_M - g.area, g.dIoU, s=28, color=_color(p), label=p, edgecolor="white", linewidth=1)
    ax[0].axhline(0, color=MUTED, lw=0.8); ax[0].axvline(0, color=MUTED, lw=0.8)
    ax[0].set_xlabel("P_M − diện tích mask"); ax[0].set_ylabel(f"IoU(idgi) − IoU({primary})")
    ax[0].legend(fontsize=7)
    paths = list(dict.fromkeys(agg.path))
    data = [agg[agg.path == p]["top1"].values for p in paths]
    bp = ax[1].boxplot(data, patch_artist=True, widths=0.5)
    for patch, p in zip(bp["boxes"], paths):
        patch.set_facecolor(_color(p)); patch.set_alpha(0.6)
    ax[1].set_xticks(range(1, len(paths) + 1), paths)
    ax[1].set_ylabel("khối lượng top-1% của g²/‖g‖² (trọng số |ΔF_k|)")
    if st is not None and len(st):
        im0 = st.image.iloc[0]
        for p, g in st[st.image == im0].groupby("path"):
            ax[2].plot(g.t, g.q, color=_color(p), label=p, lw=1.5)
        ax[2].axhline(float(dg[dg.image == im0].area.iloc[0]), color=MUTED, lw=0.8, ls="--", label="diện tích mask")
        ax[2].set_xlabel("t = k/N"); ax[2].set_ylabel("q_k(mask)"); ax[2].set_title(im0, fontsize=8)
        ax[2].legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(out, "plot_idgi_concentration.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def signal(out, plt, primary):
    df = _read(out, "signal")
    if df is None or not len(df):
        return None
    d = df[df.rule.isin([primary, "idgi"]) & (df.sign > 0)]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for k, loc in enumerate(["inside", "outside"]):
        a = ax[k]
        g0 = d[d["loc"].isin([loc, "any"])]
        for (p, r), g in g0[g0["shape"] == "square"].groupby(["path", "rule"]):
            ls, mk = RULE_STYLE.get(r, ("-", "o"))
            m = g.groupby("n_elements")["retention"].median()
            a.plot(m.index, m.values, ls=ls, marker=mk, color=_color(p), label=f"{p} {r}")
        sc = g0[g0["shape"] == "scatter"]
        for (p, r), g in sc.groupby(["path", "rule"]):
            a.scatter(g.n_elements.median() * 1.15, g.retention.median(), marker="*", s=80, color=_color(p),
                      edgecolor="white", linewidth=1)
        toy = g0[(g0.rule == "idgi") & (g0.path == "ig") & (g0["shape"] == "square")]
        if len(toy) and "idgi_pred_toy" in toy:
            m = toy.groupby("n_elements")["idgi_pred_toy"].median()
            a.plot(m.index, m.values, color=MUTED, lw=1, ls=(0, (2, 2)), label="toy dự đoán (ig, idgi)")
        a.axhline(1, color=MUTED, lw=0.8); a.axhline(0, color=MUTED, lw=0.8)
        a.set_xscale("log"); a.set_xlabel("số phần tử của R (★ = phân tán)"); a.set_title(f"R {loc}")
    ax[0].set_ylabel("retention = Σ_R ΔA / Σ A*  (median)")
    ax[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    path = os.path.join(out, "plot_signal_retention.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def trap(out, plt, primary, cfg):
    from .components import predicted_attenuation
    df = _read(out, "trap")
    if df is None or not len(df):
        return None
    eb = cfg["exp_b"]
    d = df[df.rule.isin([primary, "idgi"]) & (df.u == "high")]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    a = ax[0]
    g0 = d[np.isclose(d.eta, float(eb["eta_ref"]))]
    for (p, r), g in g0.groupby(["path", "rule"]):
        ls, mk = RULE_STYLE.get(r, ("-", "o"))
        m = g.groupby("rho")["rel_ig"].median()
        a.plot(m.index, m.values, ls=ls, marker=mk, color=_color(p), label=f"{p} {r}")
    spi = cfg["paths"].get("spi10")
    if spi:
        rr = np.geomspace(min(eb["rho"]) / 1.5, max(eb["rho"]) * 1.5, 50)
        a.plot(rr, [predicted_attenuation(1 / (spi["alpha"] + 1), r) for r in rr], color=MUTED, lw=1,
               ls=(0, (2, 2)), label="SPI dự đoán a(ρ) (xấp xỉ)")
    a.set_xscale("log"); a.set_yscale("symlog", linthresh=1e-3)
    a.set_xlabel("ρ"); a.set_ylabel(f"‖ΔA‖ / ‖ΔA_IG,{primary}‖ (median)"); a.set_title(f"η = {eb['eta_ref']}")
    a.legend(fontsize=6, ncol=2)
    a = ax[1]
    g1 = d[np.isclose(d.rho, float(eb["rho_ref"]))]
    for (p, r), g in g1.groupby(["path", "rule"]):
        ls, mk = RULE_STYLE.get(r, ("-", "o"))
        m = g.groupby("eta")["rel_A"].median()
        a.plot(m.index, m.values, ls=ls, marker=mk, color=_color(p), label=f"{p} {r}")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("η (mục tiêu)"); a.set_ylabel("‖ΔA‖ / ‖A(F)‖ (median)"); a.set_title(f"ρ = {eb['rho_ref']}")
    fig.tight_layout()
    path = os.path.join(out, "plot_trap_sensitivity.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def spi_k(out, plt, primary):
    df = _read(out, "spi_k")
    if df is None or not len(df):
        return None
    df = df.copy()
    df["grp"] = df.apply(lambda r: f"K={r.K}" + (" deb" if r.debiased else ""), axis=1)
    order = sorted(df.grp.unique(), key=lambda s: (int(s.split("=")[1].split()[0]), s))
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, metric in zip(ax, ["iou", "energy_mask", "l2"]):
        pos = 0
        ticks, labels = [], []
        for r, col in ((primary, PATH_COLOR["spi10"]), ("idgi", MUTED)):
            for gname in order:
                v = df[(df.rule == r) & (df.grp == gname)][metric].dropna().values
                if not len(v):
                    continue
                b = a.boxplot([v], positions=[pos], widths=0.6, patch_artist=True)
                b["boxes"][0].set_facecolor(col); b["boxes"][0].set_alpha(0.55)
                ticks.append(pos); labels.append(f"{'canon' if r == primary else 'idgi'}\n{gname}")
                pos += 1
            pos += 0.6
        a.set_xticks(ticks, labels, fontsize=6)
        a.set_title(metric)
        if metric == "l2":
            a.set_yscale("log")
    fig.tight_layout()
    path = os.path.join(out, "plot_spi_k.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def make_all(out: str, cfg: dict) -> list:
    plt = _setup()
    primary = "canon_rand" if "rand" in cfg["quadrature"] else "canon_left"
    files = []
    for f in (lambda: conv(out, plt), lambda: idgi_conc(out, plt, primary), lambda: signal(out, plt, primary),
              lambda: trap(out, plt, primary, cfg), lambda: spi_k(out, plt, primary)):
        try:
            r = f()
            if r:
                files.append(r)
        except Exception as e:                       # một đồ thị lỗi không chặn báo cáo
            print(f"[plots] lỗi: {type(e).__name__}: {e}")
    try:
        files += heatmaps(out, plt)
    except Exception as e:
        print(f"[plots] heatmap lỗi: {type(e).__name__}: {e}")
    for f in files:
        print(f"[plots] {f}")
    return files