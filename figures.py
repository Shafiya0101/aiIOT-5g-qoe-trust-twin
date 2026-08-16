"""
AI4T Project 2 -- Cross-Domain Trust in 4G/5G YouTube QoE
Component 5 of 5: PUBLICATION FIGURES

Produces the six figures the brief requires:
  Fig 1  dataset / twin-state description
  Fig 2  baseline comparison (in-domain vs out-of-domain F1)
  Fig 3  leakage: random row split vs session-level split
  Fig 4  reliability diagram before and after calibration  [calibration result]
  Fig 5  feature-distribution shift (PSI)
  Fig 6  trust indicator decision map                      [limitation analysis]

All figures are saved at 300 dpi, greyscale-safe, with font sizes chosen for
a two-column IEEE page.
"""

from __future__ import annotations

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import (STATIC_CONTEXTS, add_persistence, feature_columns,
                    load_replayed)
from trust import (_calibrator, build_base_models, three_way_session_split,
                   trust_band, trust_score)

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(_HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
})
C = {"a": "#1f4e79", "b": "#c0504d", "c": "#7f7f7f", "d": "#4f81bd"}


def _save(fig, name):
    p = os.path.join(FIGDIR, name)
    fig.savefig(p)
    plt.close(fig)
    print(f"  saved {name}")


# --------------------------------------------------------------------------
def fig1_dataset(df):
    """Twin-state description: event rarity across the eight environments."""
    piv = df.pivot_table(index="tech", columns="context", values="y", aggfunc="mean") * 100
    cnt = df.pivot_table(index="tech", columns="context", values="eid", aggfunc="nunique")

    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    im = ax.imshow(piv.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ax.set_yticks(range(len(piv.index)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i,j]:.2f}%\n({cnt.values[i,j]} sess.)",
                    ha="center", va="center", fontsize=7,
                    color="white" if piv.values[i, j] > 4 else "black")
    ax.set_title("Stall rate within the 5 s prediction horizon, by environment")
    fig.colorbar(im, ax=ax, label="positive rate (%)")
    _save(fig, "fig1_dataset_description.png")


def fig2_baselines(gap):
    """In-domain vs out-of-domain F1 for every model and transfer."""
    exps = gap["experiment"].unique()
    fig, axes = plt.subplots(1, len(exps), figsize=(7.2, 2.6), sharey=True)
    for ax, e in zip(np.atleast_1d(axes), exps):
        g = gap[gap["experiment"] == e].sort_values("model")
        x = np.arange(len(g)); w = 0.38
        ax.bar(x - w/2, g["f1_in"], w, label="in-domain", color=C["a"])
        ax.bar(x + w/2, g["f1_out"], w, label="out-of-domain", color=C["b"])
        ax.set_xticks(x, [m[:4] for m in g["model"]], rotation=0)
        ax.set_title(e, fontsize=9)
        ax.set_ylim(0, 0.65)
    np.atleast_1d(axes)[0].set_ylabel("F1-score")
    np.atleast_1d(axes)[0].legend(loc="upper right")
    fig.suptitle("Model performance collapses across technology, not mobility", y=1.04)
    _save(fig, "fig2_baseline_comparison.png")


def fig3_leakage(leak):
    """The methodological control: split rule changes the headline number."""
    m = ["f1", "precision", "recall", "pr_auc"]
    lab = ["F1", "Precision", "Recall", "PR-AUC"]
    x = np.arange(len(m)); w = 0.38
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.bar(x - w/2, leak.iloc[0][m].values, w, label="random row split", color=C["b"])
    ax.bar(x + w/2, leak.iloc[1][m].values, w, label="session-level split", color=C["a"])
    for i, k in enumerate(m):
        a, b = leak.iloc[0][k], leak.iloc[1][k]
        if b > 0:
            ax.text(i, max(a, b) + 0.02, f"+{100*(a-b)/b:.0f}%",
                    ha="center", fontsize=7, color=C["b"])
    ax.set_xticks(x, lab); ax.set_ylabel("score"); ax.set_ylim(0, 0.95)
    ax.legend(); ax.set_title("Row-level splitting inflates every metric")
    _save(fig, "fig3_leakage.png")


def fig4_reliability(df, feats):
    """Reliability diagrams: uncalibrated vs Platt, in- and out-of-domain."""
    src = df[df["tech"] == "4G"].reset_index(drop=True)
    tgt = df[df["tech"] == "5G"].reset_index(drop=True)
    tr, ca, te = three_way_session_split(src)
    X = src[feats].values.astype("float32"); y = src["y"].values
    Xt, yt = tgt[feats].values.astype("float32"), tgt["y"].values

    rf = build_base_models()["RandomForest"]
    rf.fit(X[tr], y[tr])
    cal = _calibrator(rf, "sigmoid"); cal.fit(X[ca], y[ca])

    def curve(model, Xe, ye, bins=10):
        p = model.predict_proba(Xe)[:, 1]
        edges = np.linspace(0, 1, bins + 1)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p > lo) & (p <= hi)
            if m.sum() >= 20:
                xs.append(p[m].mean()); ys.append(ye[m].mean())
        return xs, ys

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
    for ax, (Xe, ye, ttl) in zip(axes, [(X[te], y[te], "In-domain (4G)"),
                                        (Xt, yt, "Out-of-domain (5G)")]):
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
        xs, ys = curve(rf, Xe, ye);  ax.plot(xs, ys, "o-", color=C["b"], ms=3, label="uncalibrated")
        xs, ys = curve(cal, Xe, ye); ax.plot(xs, ys, "s-", color=C["a"], ms=3, label="Platt")
        ax.set_title(ttl); ax.set_xlabel("predicted probability")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0].set_ylabel("observed frequency"); axes[0].legend(loc="upper left")
    fig.suptitle("Calibration works in-domain but not under domain shift", y=1.03)
    _save(fig, "fig4_reliability.png")


def fig5_shift(shift):
    """Which variables actually move between 4G and 5G."""
    top = shift.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.barh(top["feature"], top["psi"], color=C["a"])
    for thr, lab, col in [(0.10, "moderate", C["c"]), (0.25, "major", C["b"])]:
        ax.axvline(thr, ls="--", lw=0.9, color=col)
        ax.text(thr, -0.8, lab, fontsize=7, color=col, ha="center")
    ax.set_xlabel("Population Stability Index (4G vs 5G)")
    ax.set_title("Radio-quality dynamics shift most between technologies")
    _save(fig, "fig5_distribution_shift.png")


def fig6_trust(mean_psi, val_f1):
    """Limitation analysis: identical confidence, opposite decision."""
    confs = np.linspace(0.5, 1.0, 120)
    psis = np.linspace(0.0, 0.6, 120)
    Z = np.array([[trust_score(c, p, val_f1) for c in confs] for p in psis])

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="RdYlGn",
                   extent=[confs[0], confs[-1], psis[0], psis[-1]], vmin=0, vmax=0.6)
    cs = ax.contour(confs, psis, Z, levels=[0.20, 0.50], colors="k", linewidths=0.9)
    ax.clabel(cs, fmt={0.20: "REJECT/VERIFY", 0.50: "VERIFY/TRUST"}, fontsize=6)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(mean_psi, color="k", lw=0.8, ls="--")
    ax.text(0.52, 0.02, "4G deployment (PSI=0)", fontsize=6.5)
    ax.text(0.52, mean_psi + 0.02, f"5G deployment (PSI={mean_psi:.2f})", fontsize=6.5)
    ax.plot([0.95, 0.95], [0.0, mean_psi], "ko-", ms=4, lw=1.2)
    ax.set_xlabel("model confidence"); ax.set_ylabel("distribution shift (mean PSI)")
    ax.set_title("Same 95% confidence, opposite operational decision")
    fig.colorbar(im, ax=ax, label="trust score")
    _save(fig, "fig6_trust_map.png")


# --------------------------------------------------------------------------
def main():
    print("loading results...")
    df = add_persistence(load_replayed())
    df["mobility"] = np.where(df["context"].isin(STATIC_CONTEXTS), "static", "mobile")
    feats = feature_columns(df)

    gap = pd.read_csv(os.path.join(_HERE, "results_gap.csv"))
    leak = pd.read_csv(os.path.join(_HERE, "results_leakage.csv"))
    shift = pd.read_csv(os.path.join(_HERE, "results_shift_4G_to_5G.csv"))
    cal = pd.read_csv(os.path.join(_HERE, "results_calibration.csv"))

    hl = cal[(cal["experiment"] == "4G -> 5G") & (cal["model"] == "RandomForest")
             & (cal["calibration"] == "isotonic")]
    val_f1 = float(hl[hl["condition"] == "in-domain"]["f1"].iloc[0])
    mean_psi = float(hl["mean_psi"].iloc[0])

    print("generating figures...")
    fig1_dataset(df)
    fig2_baselines(gap)
    fig3_leakage(leak)
    fig4_reliability(df, feats)
    fig5_shift(shift)
    fig6_trust(mean_psi, val_f1)
    print(f"\nall figures written to {FIGDIR}")


if __name__ == "__main__":
    main()
