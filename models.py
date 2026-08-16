"""
AI4T Project 2 -- Cross-Domain Trust in 4G/5G YouTube QoE
Component 2 of 5: BASELINES, MODELS, AND DOMAIN-AWARE EVALUATION

Answers the parts of the brief that require:
  - a comparison of Logistic Regression, Random Forest and Gradient Boosting
  - in-domain versus out-of-domain F1
  - the generalization gap
  - calibration degradation across domains

CRITICAL METHODOLOGICAL POINT
-----------------------------
Splits are made by SESSION, never by row. Second 40 and second 41 of the same
streaming session share almost all of their feature values, so a random row
split places near-duplicates on both sides of the train/test boundary and
produces a badly inflated score. We demonstrate this explicitly in
`leakage_demonstration()` because it is a result worth reporting, not just a
mistake worth avoiding.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
RANDOM_STATE = 42
HORIZON_S = 5

# Columns that describe the instance rather than the network state.
META_COLS = ["eid", "context", "tech", "ts", "y"]

# The four measurement contexts split into a mobility axis.
STATIC_CONTEXTS = {"Indoor", "Outdoor"}
MOBILE_CONTEXTS = {"Pedestrian", "Mobility"}


# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------

def load_replayed(path: str | None = None) -> pd.DataFrame:
    path = path or os.path.join(_HERE, "replayed.pkl")
    df = pd.read_pickle(path)
    df = df.sort_values(["eid", "ts"]).reset_index(drop=True)
    df["mobility"] = np.where(df["context"].isin(STATIC_CONTEXTS), "static", "mobile")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Everything that is not metadata. All of these are computed from the
    trailing window only, so none of them can see the future."""
    return [c for c in df.columns if c not in META_COLS + ["mobility"]]


def add_persistence(df: pd.DataFrame, horizon_s: int = HORIZON_S) -> pd.DataFrame:
    """The naive baseline required by the brief.

    Persistence says: 'the near future looks like the recent past'. Concretely,
    predict a stall in (t, t+H] if a stall actually began in (t-H, t].

    That past fact is exactly y evaluated at time t-H, so we obtain it with a
    time-aligned self-join. This uses only information an operator already had
    at time t -- it is a baseline, not a leak.
    """
    left = df[["eid", "ts", "y"]].copy()
    right = df[["eid", "ts", "y"]].copy()
    right["ts"] = right["ts"] + pd.Timedelta(seconds=horizon_s)
    right = right.rename(columns={"y": "persistence"})
    merged = left.merge(right[["eid", "ts", "persistence"]], on=["eid", "ts"], how="left")
    df = df.copy()
    df["persistence"] = merged["persistence"].fillna(0).astype(int).values
    return df


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """ECE: average gap between predicted confidence and observed frequency.

    A model can have excellent F1 and a terrible ECE. That divergence is the
    central claim of this project, so ECE is reported everywhere alongside F1.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (y_prob > lo) & (y_prob <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(y_prob)) * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)


def evaluate(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    pos = int(y_true.sum())
    return {
        "n": len(y_true),
        "n_pos": pos,
        "base_rate": float(y_true.mean()),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_prob) if pos else np.nan,
        "roc_auc": roc_auc_score(y_true, y_prob) if 0 < pos < len(y_true) else np.nan,
        "brier": brier_score_loss(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
    }


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def build_models() -> dict:
    """The three models named in the brief.

    class_weight='balanced' matters here: the positive class is 2.6% of rows,
    so an unweighted model can reach 97% accuracy by never predicting a stall.
    """
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "LogisticRegression": Pipeline([
            ("imp", imp()),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=RANDOM_STATE)),
        ]),
        "RandomForest": Pipeline([
            ("imp", imp()),
            ("clf", RandomForestClassifier(n_estimators=100, min_samples_leaf=20,
                                           class_weight="balanced_subsample",
                                           n_jobs=2, random_state=RANDOM_STATE)),
        ]),
        # HistGradientBoosting is the histogram-based implementation. It is
        # 10-50x faster than GradientBoostingClassifier at this data size and
        # handles NaN natively. Naive GradientBoosting exhausted our machine.
        "GradientBoosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, max_depth=6,
            class_weight="balanced", random_state=RANDOM_STATE),
    }


def fit_predict(model, Xtr, ytr, Xte):
    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1]


# --------------------------------------------------------------------------
# Experiment 1 -- why the split rule matters
# --------------------------------------------------------------------------

def leakage_demonstration(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Same model, same data, two split rules. Reported as a figure."""
    X, y, g = df[feats].values, df["y"].values, df["eid"].values
    rows = []

    # (a) naive random row split -- what a leaky pipeline would do
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.permutation(len(df))
    cut = int(0.7 * len(df))
    tr, te = idx[:cut], idx[cut:]
    p = fit_predict(build_models()["RandomForest"], X[tr], y[tr], X[te])
    rows.append({"split": "random row split (leaky)", **evaluate(y[te], p)})

    # (b) grouped split -- sessions never appear on both sides
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=RANDOM_STATE)
    tr, te = next(gss.split(X, y, groups=g))
    p = fit_predict(build_models()["RandomForest"], X[tr], y[tr], X[te])
    rows.append({"split": "session-level split (correct)", **evaluate(y[te], p)})

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Experiment 2 -- in-domain versus out-of-domain
# --------------------------------------------------------------------------

def domain_experiment(df: pd.DataFrame, feats: list[str], column: str,
                      source: str, target: str) -> pd.DataFrame:
    """Train on `source`, evaluate in-domain and on `target`.

    In-domain testing still uses a held-out set of SESSIONS from the source
    domain, so the in-domain and out-of-domain numbers are comparable: neither
    has ever seen its test sessions during training.
    """
    src = df[df[column] == source]
    tgt = df[df[column] == target]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=RANDOM_STATE)
    tr_i, te_i = next(gss.split(src[feats].values, src["y"].values,
                                groups=src["eid"].values))
    Xtr, ytr = src[feats].values[tr_i].astype("float32"), src["y"].values[tr_i]
    Xin, yin = src[feats].values[te_i].astype("float32"), src["y"].values[te_i]
    Xout, yout = tgt[feats].values.astype("float32"), tgt["y"].values

    rows = []

    # persistence baseline, scored on the same test sets
    rows.append({"model": "Persistence", "condition": f"in-domain ({source})",
                 **evaluate(yin, src["persistence"].values[te_i])})
    rows.append({"model": "Persistence", "condition": f"out-of-domain ({target})",
                 **evaluate(yout, tgt["persistence"].values)})

    for name, model in build_models().items():
        model.fit(Xtr, ytr)
        p_in = model.predict_proba(Xin)[:, 1]
        p_out = model.predict_proba(Xout)[:, 1]
        rows.append({"model": name, "condition": f"in-domain ({source})",
                     **evaluate(yin, p_in)})
        rows.append({"model": name, "condition": f"out-of-domain ({target})",
                     **evaluate(yout, p_out)})

    out = pd.DataFrame(rows)
    out["experiment"] = f"{source} -> {target}"
    return out


def generalization_gap(res: pd.DataFrame) -> pd.DataFrame:
    """F1 gap and calibration degradation, per model, per experiment."""
    rows = []
    for (exp, model), g in res.groupby(["experiment", "model"]):
        ind = g[g["condition"].str.startswith("in-domain")].iloc[0]
        ood = g[g["condition"].str.startswith("out-of-domain")].iloc[0]
        rows.append({
            "experiment": exp, "model": model,
            "f1_in": ind["f1"], "f1_out": ood["f1"],
            "f1_gap": ind["f1"] - ood["f1"],
            "ece_in": ind["ece"], "ece_out": ood["ece"],
            "ece_degradation": ood["ece"] - ind["ece"],
            "brier_in": ind["brier"], "brier_out": ood["brier"],
            "base_rate_in": ind["base_rate"], "base_rate_out": ood["base_rate"],
        })
    return pd.DataFrame(rows).sort_values(["experiment", "model"])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    df = load_replayed()
    df = add_persistence(df)
    feats = feature_columns(df)
    print(f"{len(df):,} instants | {len(feats)} features | "
          f"{df['eid'].nunique()} sessions | positive rate {df['y'].mean():.4f}\n")

    print("=" * 72)
    print("EXPERIMENT 1  Does the split rule change the conclusion?")
    print("=" * 72)
    leak = leakage_demonstration(df, feats)
    print(leak[["split", "f1", "precision", "recall", "pr_auc", "ece"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n" + "=" * 72)
    print("EXPERIMENT 2  In-domain versus out-of-domain")
    print("=" * 72)
    experiments = [
        ("tech", "4G", "5G"),
        ("tech", "5G", "4G"),
        ("mobility", "static", "mobile"),
        ("mobility", "mobile", "static"),
    ]
    parts = []
    for col, s_, t_ in experiments:
        r = domain_experiment(df, feats, col, s_, t_)
        parts.append(r)
        pd.concat(parts, ignore_index=True).to_csv(
            os.path.join(_HERE, "results_domain.csv"), index=False)
        print(f"  [done] {s_} -> {t_}", flush=True)
    all_res = pd.concat(parts, ignore_index=True)

    for exp, g in all_res.groupby("experiment", sort=False):
        print(f"\n--- {exp} ---")
        print(g[["model", "condition", "base_rate", "f1", "precision",
                 "recall", "pr_auc", "brier", "ece"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n" + "=" * 72)
    print("GENERALIZATION GAP AND CALIBRATION DEGRADATION")
    print("=" * 72)
    gap = generalization_gap(all_res)
    print(gap[["experiment", "model", "f1_in", "f1_out", "f1_gap",
               "ece_in", "ece_out", "ece_degradation"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    all_res.to_csv(os.path.join(_HERE, "results_domain.csv"), index=False)
    gap.to_csv(os.path.join(_HERE, "results_gap.csv"), index=False)
    leak.to_csv(os.path.join(_HERE, "results_leakage.csv"), index=False)
    print("\nsaved: results_domain.csv, results_gap.csv, results_leakage.csv")


if __name__ == "__main__":
    main()
