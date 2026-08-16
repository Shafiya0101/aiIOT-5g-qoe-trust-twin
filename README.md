# AI4T Mini-Project - Group 2

## Cross-Domain Trust in 4G/5G YouTube QoE Prediction

A trace-driven digital twin that replays real YouTube streaming sessions,
reconstructs the network state second by second, predicts whether a stall will
occur within the next 5 seconds, and estimates how far that prediction can be
trusted when the deployment environment differs from the training environment.

**Research question.** Can a QoE model trained in one radio environment be
trusted in a different one, and is model confidence a sufficient trust signal?

**Answer.** No. Classification performance and calibration are distinct
properties, and neither alone indicates whether a prediction is safe to use in
an unfamiliar environment.

---

## Dataset

Mustafa et al., *YouTube goes 5G* - https://github.com/razaulmustafa852/youtubegoes5g

1 Hz radio telemetry (RSRP, RSRQ, SNR, CQI, downlink bitrate, handover events)
joined to YouTube IFrame player events, across four measurement contexts
(Indoor, Outdoor, Pedestrian, Mobility) and two technologies (4G, 5G NSA).

After cleaning and joining: **262 sessions, 97,370 labelled instants,
2.63% positive rate.**

---

## How to reproduce

Open `AI4T_Project2_COMPLETE.ipynb` in Google Colab and run all cells
(about 12 minutes). It clones the dataset, writes the five source files,
runs every experiment and regenerates every figure and table in this repo.

Locally instead:

```bash
git clone https://github.com/razaulmustafa852/youtubegoes5g.git yt5g
pip install -r requirements.txt
python twin_replay.py    # replay, state, labels      -> replayed.pkl
python models.py         # baselines + 3 classifiers  -> results_domain/gap/leakage.csv
python trust.py          # calibration + trust        -> results_calibration/shift/trust.csv
python figures.py        # six figures                -> figures/
python llm_explain.py    # 20 evidence records        -> llm_cases.csv, llm_prompts.txt
```

---

## Repository contents

| File | Deliverable component |
|---|---|
| `twin_replay.py` | 1/5 - chronological replay, twin state, label definition |
| `models.py` | 2/5 - persistence baseline, LogReg, Random Forest, Gradient Boosting |
| `trust.py` | 3/5 - Platt/isotonic calibration, PSI shift, trust indicator |
| `llm_explain.py` | 4/5 - grounded LLM explanation + automated hallucination check |
| `figures.py` | 5/5 - the six publication figures |
| `results_*.csv` | all experimental results |
| `llm_cases.csv` | 20 LLM cases with evidence, explanation and human judgement |
| `figures/` | Figures 1-6, 300 dpi |

---

## Method summary

**Target.** A rebuffering event beginning in `(t, t+5s]`. Buffering before the
first `playing` event is startup delay, not a stall, and is excluded.

**State.** Features are computed from a 10 s trailing window only. The twin
issues no prediction during the first 10 s of a session, and the final 5 s are
unlabelled because their future is unobservable. No feature reads forward.

**Splitting.** By session, never by row. Adjacent seconds of the same session
are near-duplicates, so row-level splitting leaks information across the
train/test boundary.

**Domains.** Technology (4G vs 5G) and mobility (static vs mobile).

---

## Key results

| Finding | Evidence |
|---|---|
| Row-level splitting inflates results | PR-AUC 0.622 random vs 0.455 session-level (+37%) |
| Performance collapses across technology | Random Forest F1 0.561 in-domain (4G) vs 0.000 on 5G |
| Mobility transfers, technology does not | Random Forest 0.486 -> 0.487 across static/mobile |
| Calibration trades against classification | Platt cuts ECE ~12x but reduces F1 by about a third |
| Distribution shift is major | Mean PSI (4G vs 5G) = 0.433, threshold for "major" is 0.25 |
| Confidence alone is misleading | Identical 95% confidence: VERIFY in-domain, REJECT out-of-domain |
| A naive baseline can beat ML out-of-domain | Persistence F1 0.413 vs Random Forest 0.000 on 5G->4G |

---

## Limitations

- 63 sessions have telemetry but no player events and are excluded.
- Approximately 1-2% of telemetry rows are column-misaligned and are removed.
- The 5G Indoor subgroup is small (3,325 instants); no strong conclusion is
  drawn from it alone.
- The dataset's 5G is Non-Standalone (NSA), evidenced by inter-RAT 4G/5G
  handover events. No Standalone (SA) data is available, so the SA/NSA
  comparison in the original brief is replaced by 4G/5G and static/mobile.
- The dataset is observational. It shows that stalls can be anticipated from
  prior radio and throughput evidence; it does not establish that acting on a
  warning would prevent a stall.
