"""
AI4T Project 2 -- Cross-Domain Trust in 4G/5G YouTube QoE
Component 3 of 5: CALIBRATION, DISTRIBUTION SHIFT, TRUST INDICATOR

The brief requires:
  - the best model to be calibrated (Platt scaling / isotonic regression)
  - one simple measure of feature-distribution shift
  - a trust indicator combining model confidence, observed distribution shift
    and validation performance

Design note on the calibration split
------------------------------------
Calibrating on data the model was trained on produces a falsely good
calibration curve. We therefore split the SOURCE domain three ways by session:
    train (fit the model) | calib (fit the calibrator) | test (report)
No session appears in more than one of them.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

# scikit-learn >= 1.6 removed cv="prefit" in favour of FrozenEstimator.
# This shim keeps the code working on both old and new versions, which
# matters because Colab and local installs are often different releases.
try:
    from sklearn.frozen import FrozenEstimator

    def _calibrator(fitted_model, method):
        return CalibratedClassifierCV(FrozenEstimator(fitted_model), method=method)
except ImportError:                                   # scikit-learn < 1.6
    def _calibrator(fitted_model, method):
        return CalibratedClassifierCV(fitted_model, method=method, cv="prefit")
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from models import (STATIC_CONTEXTS, add_persistence, evaluate,
                    expected_calibration_error, feature_columns, load_replayed)

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
RANDOM_STATE = 42


# --------------------------------------------------------------------------
# Distribution shift
# --------------------------------------------------------------------------

def population_stability_index(a: np.ndarray, b: np.ndarray, bins: int = 10) -> float:
    """PSI between a reference sample `a` and a new sample `b`.

    Chosen because it is the standard, easily explained shift measure in
    operational ML. Conventional reading:
        PSI < 0.10  negligible shift
        0.10-0.25   moderate shift
        PSI > 0.25  major shift -- model output should be treated with caution
    """
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 10 or len(b) < 10:
        return np.nan
    edges = np.quantile(a, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    if len(edges) < 3:
        return np.nan
    pa = np.histogram(a, bins=edges)[0].astype(float)
    pb = np.histogram(b, bins=edges)[0].astype(float)
    pa = np.clip(pa / pa.sum(), 1e-6, None)
    pb = np.clip(pb / pb.sum(), 1e-6, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def shift_report(src: pd.DataFrame, tgt: pd.DataFrame,
                 feats: list[str]) -> pd.DataFrame:
    """Per-feature PSI, source vs target domain."""
    rows = [{"feature": f,
             "psi": population_stability_index(src[f].values.astype(float),
                                               tgt[f].values.astype(float))}
            for f in feats]
    out = pd.DataFrame(rows).dropna().sort_values("psi", ascending=False)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# Trust indicator
# --------------------------------------------------------------------------

def trust_score(confidence: float, mean_psi: float, val_f1: float) -> float:
    """Combine the three ingredients the brief names, into [0, 1].

    trust = confidence_component * shift_penalty * validation_component

    - confidence_component : how far the calibrated probability is from the
      0.5 decision boundary, rescaled to [0,1]. A prediction sitting on the
      boundary carries little information.
    - shift_penalty        : exp(-PSI / 0.25). Equals 1 when the deployment
      distribution matches training, and decays as it diverges. 0.25 is the
      conventional "major shift" threshold, so a major shift costs ~63%.
    - validation_component : the model's measured F1 in the environment it was
      validated in. A model that never worked cannot be trusted anywhere.

    This is a decision aid, not a probability. The report must say so.
    """
    conf = abs(confidence - 0.5) * 2.0
    shift_penalty = float(np.exp(-max(mean_psi, 0.0) / 0.25))
    return float(np.clip(conf * shift_penalty * max(val_f1, 0.0), 0.0, 1.0))


def trust_band(score: float) -> str:
    if score >= 0.50:
        return "TRUST"
    if score >= 0.20:
        return "VERIFY"
    return "REJECT"


# --------------------------------------------------------------------------
# Calibration experiment
# --------------------------------------------------------------------------

def three_way_session_split(df: pd.DataFrame, seed: int = RANDOM_STATE):
    """Split a domain into train / calib / test by SESSION."""
    g = df["eid"].values
    idx_all = np.arange(len(df))
    s1 = GroupShuffleSplit(n_splits=1, test_size=0.40, random_state=seed)
    tr, rest = next(s1.split(idx_all, groups=g))
    rest_g = g[rest]
    s2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    ca_rel, te_rel = next(s2.split(rest, groups=rest_g))
    return tr, rest[ca_rel], rest[te_rel]


def build_base_models() -> dict:
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "RandomForest": Pipeline([
            ("imp", imp()),
            ("clf", RandomForestClassifier(n_estimators=100, min_samples_leaf=20,
                                           class_weight="balanced_subsample",
                                           n_jobs=2, random_state=RANDOM_STATE)),
        ]),
        "GradientBoosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, max_depth=6,
            class_weight="balanced", random_state=RANDOM_STATE),
    }


def calibration_experiment(df: pd.DataFrame, feats: list[str], column: str,
                           source: str, target: str) -> tuple:
    """Train on source, calibrate on source, evaluate in- and out-of-domain."""
    src = df[df[column] == source].reset_index(drop=True)
    tgt = df[df[column] == target].reset_index(drop=True)

    tr, ca, te = three_way_session_split(src)
    X = src[feats].values.astype("float32")
    y = src["y"].values
    Xt, yt = tgt[feats].values.astype("float32"), tgt["y"].values

    # distribution shift between the training environment and the target
    shift = shift_report(src.iloc[tr], tgt, feats)
    mean_psi = float(shift["psi"].mean())
    top_psi = shift.head(5)

    rows = []
    for name, model in build_base_models().items():
        model.fit(X[tr], y[tr])

        variants = {
            "uncalibrated": model,
            "platt": _calibrator(model, "sigmoid"),
            "isotonic": _calibrator(model, "isotonic"),
        }
        for vname, v in variants.items():
            if vname != "uncalibrated":
                v.fit(X[ca], y[ca])
            p_in = v.predict_proba(X[te])[:, 1]
            p_out = v.predict_proba(Xt)[:, 1]
            rows.append({"model": name, "calibration": vname,
                         "condition": "in-domain", **evaluate(y[te], p_in)})
            rows.append({"model": name, "calibration": vname,
                         "condition": "out-of-domain", **evaluate(yt, p_out)})

    res = pd.DataFrame(rows)
    res["experiment"] = f"{source} -> {target}"
    res["mean_psi"] = mean_psi
    return res, shift, top_psi, mean_psi


def main():
    df = add_persistence(load_replayed())
    df["mobility"] = np.where(df["context"].isin(STATIC_CONTEXTS), "static", "mobile")
    feats = feature_columns(df)

    all_res, shift_tables = [], {}
    for col, s, t in [("tech", "4G", "5G"), ("mobility", "mobile", "static")]:
        print(f"\n{'='*70}\nCALIBRATION: {s} -> {t}\n{'='*70}", flush=True)
        res, shift, top, mpsi = calibration_experiment(df, feats, col, s, t)
        all_res.append(res)
        shift_tables[f"{s}->{t}"] = shift
        print(f"mean PSI ({s} vs {t}) = {mpsi:.3f}")
        print("top-5 shifted features:")
        print(top.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print()
        print(res[["model", "calibration", "condition", "f1", "pr_auc",
                   "brier", "ece"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    res = pd.concat(all_res, ignore_index=True)
    res.to_csv(os.path.join(_HERE, "results_calibration.csv"), index=False)
    for k, v in shift_tables.items():
        v.to_csv(os.path.join(_HERE,
                 f"results_shift_{k.replace('->','_to_')}.csv"), index=False)

    # ---- trust indicator applied to the headline transfer ----------------
    print(f"\n{'='*70}\nTRUST INDICATOR (4G-trained model)\n{'='*70}")
    hl = res[(res["experiment"] == "4G -> 5G") &
             (res["model"] == "RandomForest") &
             (res["calibration"] == "isotonic")]
    val_f1 = float(hl[hl["condition"] == "in-domain"]["f1"].iloc[0])
    mpsi = float(hl["mean_psi"].iloc[0])

    demo = []
    for conf in [0.95, 0.80, 0.65, 0.55]:
        for env, psi in [("in-domain 4G", 0.0), (f"out-of-domain 5G", mpsi)]:
            sc = trust_score(conf, psi, val_f1)
            demo.append({"confidence": conf, "environment": env,
                         "mean_psi": round(psi, 3), "val_f1": round(val_f1, 3),
                         "trust": round(sc, 3), "decision": trust_band(sc)})
    demo = pd.DataFrame(demo)
    print(demo.to_string(index=False))
    demo.to_csv(os.path.join(_HERE, "results_trust.csv"), index=False)
    print("\nsaved: results_calibration.csv, results_shift_*.csv, results_trust.csv")


if __name__ == "__main__":
    main()
