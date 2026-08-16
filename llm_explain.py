"""
AI4T Project 2 -- Cross-Domain Trust in 4G/5G YouTube QoE
Component 4 of 5: GROUNDED LLM EXPLANATION

The brief is explicit about the LLM's role:
  - it does NOT train the model and does NOT control the system
  - it receives STRUCTURED EVIDENCE produced by the model
  - it produces a short explanation stating the predicted event, likely causes,
    confidence/limitations, and what an operator should verify
  - students must verify the explanation invents no values absent from the
    evidence

This module therefore does three things:
  1. builds the evidence dictionary for a given instant of the replay
  2. sends it to a real LLM (API) or writes prompts for manual submission
  3. checks every number in the returned text against the evidence
"""

from __future__ import annotations

import json
import os
import re
import warnings

import numpy as np
import pandas as pd

from models import (STATIC_CONTEXTS, add_persistence, feature_columns,
                    load_replayed)
from trust import (_calibrator, build_base_models, three_way_session_split,
                   shift_report, trust_band, trust_score)

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
N_CASES = 20
RANDOM_STATE = 42


# --------------------------------------------------------------------------
# 1. Evidence construction
# --------------------------------------------------------------------------

def build_evidence(row: pd.Series, prob: float, mean_psi: float,
                   val_f1: float, in_domain: bool) -> dict:
    """Everything the LLM is allowed to know. Nothing else may appear
    in its explanation. Values are rounded so that string-matching the
    numbers back out of the generated text is reliable."""
    r = lambda v, n=1: (None if pd.isna(v) else round(float(v), n))
    ts = trust_score(prob, 0.0 if in_domain else mean_psi, val_f1)
    return {
        "session_id": row["eid"],
        "environment": {
            "technology": row["tech"],
            "context": row["context"],
            "matches_training_environment": bool(in_domain),
        },
        "prediction": {
            "event": "video stall within next 5 seconds",
            "probability": r(prob, 3),
            "threshold": 0.5,
            "predicted_positive": bool(prob >= 0.5),
        },
        "radio_evidence": {
            "rsrp_dbm_last": r(row.get("rsrp_last")),
            "rsrp_dbm_mean_10s": r(row.get("rsrp_mean")),
            "rsrp_trend_10s": r(row.get("rsrp_delta")),
            "rsrq_db_last": r(row.get("rsrq_last")),
            "snr_db_last": r(row.get("snr_last")),
            "cqi_last": r(row.get("cqi_last")),
        },
        "throughput_evidence": {
            "downlink_kbps_last": r(row.get("dl_bitrate_last")),
            "downlink_kbps_mean_10s": r(row.get("dl_bitrate_mean")),
            "fraction_of_zero_throughput_seconds": r(row.get("dl_zero_frac"), 2),
        },
        "session_evidence": {
            "elapsed_seconds": r(row.get("elapsed_s"), 0),
            "handovers_so_far": r(row.get("n_handovers"), 0),
            "stalls_so_far": r(row.get("n_stalls_so_far"), 0),
        },
        "reliability": {
            "model": "RandomForest + Platt scaling",
            "validation_f1_in_training_environment": r(val_f1, 3),
            "distribution_shift_psi": r(0.0 if in_domain else mean_psi, 3),
            "trust_score": r(ts, 3),
            "trust_band": trust_band(ts),
        },
    }


PROMPT_TEMPLATE = """You are assisting a mobile-network operator.

Below is structured evidence produced by a stall-prediction model. Write a
short explanation (maximum 120 words) containing exactly these four parts:
1. the predicted event,
2. the likely causes,
3. the confidence and its limitations,
4. what the operator should verify.

STRICT RULES:
- Use ONLY numbers that appear in the evidence below. Do not invent, estimate,
  round differently, or infer any value that is not present.
- Do not recommend changing the network. You explain; you do not act.
- If the trust_band is REJECT, say clearly that the prediction should not be
  used for an automatic decision.

EVIDENCE:
{evidence}
"""


def build_prompt(evidence: dict) -> str:
    return PROMPT_TEMPLATE.format(evidence=json.dumps(evidence, indent=2))


# --------------------------------------------------------------------------
# 2. Calling a real LLM (optional)
# --------------------------------------------------------------------------

def call_llm(prompt: str) -> str | None:
    """Try Anthropic, then OpenAI, using whichever API key is in the
    environment. Returns None if no key is configured, in which case the
    prompts are written out for manual submission instead."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            c = anthropic.Anthropic()
            m = c.messages.create(model="claude-sonnet-4-6", max_tokens=400,
                                  messages=[{"role": "user", "content": prompt}])
            return m.content[0].text.strip()
        except Exception as e:
            print("  anthropic call failed:", e)
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            c = OpenAI()
            m = c.chat.completions.create(model="gpt-4o-mini", max_tokens=400,
                                          messages=[{"role": "user", "content": prompt}])
            return m.choices[0].message.content.strip()
        except Exception as e:
            print("  openai call failed:", e)
    return None


# --------------------------------------------------------------------------
# 3. Automated hallucination check
# --------------------------------------------------------------------------

def _numbers_in(obj) -> set:
    """Every numeric value present anywhere in the evidence."""
    out = set()
    if isinstance(obj, dict):
        for v in obj.values():
            out |= _numbers_in(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= _numbers_in(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.add(round(float(obj), 3))
    return out


def check_grounding(explanation: str, evidence: dict) -> dict:
    """Extract every number from the generated text and test whether it is
    supported by the evidence. This produces the 'free of unsupported
    statements' judgement objectively instead of by eye.

    Small tolerances are allowed for legitimate restatement: an LLM may write
    '5 seconds' (the horizon) or convert 1500 kbps to 1.5 Mbps.
    """
    allowed = _numbers_in(evidence) | {5.0, 0.5, 120.0, 10.0, 100.0}
    allowed |= {round(v / 1000.0, 3) for v in allowed}   # kbps -> Mbps
    allowed |= {round(v * 100.0, 3) for v in allowed}    # fraction -> percent

    found = [float(m) for m in re.findall(r"-?\d+\.?\d*", explanation)]
    unsupported = []
    for f in found:
        if not any(abs(f - a) <= max(0.05, abs(a) * 0.02) for a in allowed):
            unsupported.append(f)

    return {
        "n_numbers_in_text": len(found),
        "n_unsupported": len(unsupported),
        "unsupported_values": unsupported,
        "auto_grounded": len(unsupported) == 0,
    }


def check_completeness(explanation: str) -> dict:
    """Does the explanation cover the four required parts?"""
    t = explanation.lower()
    return {
        "mentions_event": any(k in t for k in ["stall", "rebuffer", "buffering"]),
        "mentions_cause": any(k in t for k in ["because", "due to", "caused",
                                               "throughput", "rsrp", "signal",
                                               "snr", "cqi", "handover"]),
        "mentions_confidence": any(k in t for k in ["confidence", "probability",
                                                    "trust", "uncertain",
                                                    "reliab", "limitation"]),
        "mentions_verify": any(k in t for k in ["verify", "check", "inspect",
                                                "monitor", "confirm", "review"]),
    }


# --------------------------------------------------------------------------
# 4. Case generation
# --------------------------------------------------------------------------

def main():
    df = add_persistence(load_replayed())
    df["mobility"] = np.where(df["context"].isin(STATIC_CONTEXTS), "static", "mobile")
    feats = feature_columns(df)

    src = df[df["tech"] == "4G"].reset_index(drop=True)
    tgt = df[df["tech"] == "5G"].reset_index(drop=True)

    print("fitting the model that will produce the evidence...")
    tr, ca, te = three_way_session_split(src)
    X = src[feats].values.astype("float32"); y = src["y"].values
    rf = build_base_models()["RandomForest"]; rf.fit(X[tr], y[tr])
    cal = _calibrator(rf, "sigmoid"); cal.fit(X[ca], y[ca])

    from sklearn.metrics import f1_score
    p_te = cal.predict_proba(X[te])[:, 1]
    val_f1 = f1_score(y[te], (p_te >= 0.5).astype(int), zero_division=0)
    mean_psi = float(shift_report(src.iloc[tr], tgt, feats)["psi"].mean())
    print(f"  validation F1 = {val_f1:.3f} | mean PSI (4G vs 5G) = {mean_psi:.3f}")

    # Select 20 cases spanning the situations an operator actually meets:
    # in-domain positives and negatives, and out-of-domain positives and
    # negatives. A sample of only easy cases would not test the LLM.
    rng = np.random.RandomState(RANDOM_STATE)
    te_df = src.iloc[te].reset_index(drop=True)
    te_p = p_te
    p_tgt = cal.predict_proba(tgt[feats].values.astype("float32"))[:, 1]

    picks = []
    for name, frame, probs, indom in [("in-domain", te_df, te_p, True),
                                      ("out-of-domain", tgt, p_tgt, False)]:
        pos = np.where(frame["y"].values == 1)[0]
        neg = np.where(frame["y"].values == 0)[0]
        hi = np.argsort(-probs)[:200]
        for pool, k in [(pos, 3), (neg, 2), (hi, 5)]:
            if len(pool) == 0:
                continue
            sel = rng.choice(pool, size=min(k, len(pool)), replace=False)
            for i in sel:
                picks.append((name, frame.iloc[int(i)], float(probs[int(i)]), indom))

    picks = picks[:N_CASES]
    print(f"selected {len(picks)} cases")

    rows, prompts = [], []
    for i, (grp, row, prob, indom) in enumerate(picks, start=1):
        ev = build_evidence(row, prob, mean_psi, val_f1, indom)
        pr = build_prompt(ev)
        prompts.append(f"{'='*70}\nCASE {i:02d}  ({grp})\n{'='*70}\n{pr}\n")

        expl = call_llm(pr)
        rec = {
            "case_id": i,
            "group": grp,
            "session_id": row["eid"],
            "actual_outcome": int(row["y"]),
            "predicted_probability": round(prob, 3),
            "trust_band": ev["reliability"]["trust_band"],
            "evidence_json": json.dumps(ev),
            "explanation": expl if expl else "",
        }
        if expl:
            rec.update(check_grounding(expl, ev))
            rec.update(check_completeness(expl))
        # columns for the HUMAN judgement the brief requires
        rec.update({"human_factually_correct": "",
                    "human_complete": "",
                    "human_free_of_unsupported": "",
                    "human_notes": ""})
        rows.append(rec)

    cases = pd.DataFrame(rows)
    cases.to_csv(os.path.join(_HERE, "llm_cases.csv"), index=False)
    with open(os.path.join(_HERE, "llm_prompts.txt"), "w") as fh:
        fh.write("\n".join(prompts))

    got_llm = cases["explanation"].str.len().gt(0).sum()
    print(f"\nwrote llm_cases.csv ({len(cases)} cases) and llm_prompts.txt")
    if got_llm:
        print(f"{got_llm} explanations generated automatically")
        print(f"auto-grounded (no unsupported numbers): "
              f"{int(cases['auto_grounded'].sum())}/{got_llm}")
    else:
        print("\nNo API key found, so no explanations were generated.")
        print("Open llm_prompts.txt, paste each of the 20 prompts into any")
        print("chat LLM, and paste the replies into the 'explanation' column")
        print("of llm_cases.csv. Then re-run the grounding check on them.")
    print("\nFill in the three human_* columns by hand -- that is the")
    print("manual evaluation of 20 cases the brief requires.")


if __name__ == "__main__":
    main()
