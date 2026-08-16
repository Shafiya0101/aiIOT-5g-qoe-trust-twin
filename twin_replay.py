"""
AI4T Project 2 -- Cross-Domain Trust in 5G/4G YouTube QoE
Component 1 of 5: CHRONOLOGICAL REPLAY + TWIN STATE + LABEL DEFINITION

This module is the "digital twin" core required by the brief:
  (1) it replays observations chronologically, one session at a time,
      never treating rows as independent shuffled samples;
  (2) it maintains a STATE holding the current network/service variables;
  (3) it emits a label for a FUTURE event (stall within the next H seconds).

Dataset: razaulmustafa852/youtubegoes5g
  Channel Logs/<Context>/<Eid>.csv  -- 1 Hz radio + throughput telemetry
  YouTuve QoE Events/events.csv     -- YouTube IFrame player state changes

Author: <group name>
"""

from __future__ import annotations

import csv
import glob
import os
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterator

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

#  Where the cloned dataset lives. By default we look for a folder named
#  "yt5g" sitting next to this script. Override with the AI4T_DATA env var,
#  e.g.  export AI4T_DATA=/Users/you/Downloads/youtubegoes5g
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("AI4T_DATA", os.path.join(_HERE, "yt5g"))

#  Prediction horizon. The telemetry is sampled at 1 Hz, so a 5 s horizon is
#  5 samples ahead. The brief for Project 2 explicitly names "a stall during
#  the next five seconds" as an admissible target, so H = 5.
HORIZON_S = 5

#  Length of the history window used to build rolling features. Kept short so
#  that a prediction can be issued only 10 s into a session.
WINDOW_S = 10

#  Radio / service columns we lift out of the raw channel logs.
#  NOTE on naming: in this dataset 'Level' is RSRP (dBm) and 'Qual' is RSRQ (dB).
#  We rename them so the report and the LLM evidence use standard 3GPP terms.
RAW_TO_STD = {
    "Level": "rsrp",
    "Qual": "rsrq",
    "SNR": "snr",
    "CQI": "cqi",
    "DL_bitrate": "dl_bitrate",
    "UL_bitrate": "ul_bitrate",
    "SecondCell_RSRP": "sc_rsrp",
    "SecondCell_RSRQ": "sc_rsrq",
    "SecondCell_SNR": "sc_snr",
}

NUMERIC_COLS = list(RAW_TO_STD.values())

#  Physically plausible ranges. Rows outside these are treated as corrupt.
#  ~1-2% of rows in this dataset are column-misaligned (numeric junk lands in
#  NetworkTech), and this is how we catch them without hand-editing the data.
VALID_RANGE = {
    "rsrp": (-140.0, -40.0),      # dBm
    "rsrq": (-30.0, 0.0),         # dB
    "snr": (-20.0, 40.0),         # dB
    "cqi": (0.0, 30.0),           # index
    "dl_bitrate": (0.0, 2_000_000.0),   # kbps as logged
    "ul_bitrate": (0.0, 2_000_000.0),
    "sc_rsrp": (-140.0, -40.0),
    "sc_rsrq": (-30.0, 0.0),
    "sc_snr": (-20.0, 40.0),
}


# --------------------------------------------------------------------------
# Low-level parsing helpers
# --------------------------------------------------------------------------

def _norm_eid(s: str) -> str:
    """Session ids are inconsistently cased between the two files
    ('4P7s2' in the log filename vs '4p7s2' in events.csv). Normalising is
    what takes the join from 3 sessions to 264."""
    return (s or "").strip().lower()


def _to_float(x) -> float:
    """Robust numeric cast. The logs use '-' for 'not measured'."""
    if x is None:
        return np.nan
    s = str(x).strip().strip('"')
    if s in ("", "-", "nan", "None", "NA"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_log_ts(x) -> datetime | None:
    """Channel-log timestamps look like '"\n2022.09.26_17.42.11"'."""
    if x is None:
        return None
    s = re.sub(r"[\r\n\"]", "", str(x)).strip()
    try:
        return datetime.strptime(s, "%Y.%m.%d_%H.%M.%S")
    except ValueError:
        return None


def _parse_event_ts(date_s: str, time_s: str) -> datetime | None:
    """events.csv splits the stamp into Date (dd/mm/YYYY) and Time (HH:MM:SS)."""
    try:
        return datetime.strptime(
            f"{str(date_s).strip()} {str(time_s).strip()}", "%d/%m/%Y %H:%M:%S"
        )
    except (ValueError, TypeError):
        return None


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

@dataclass
class Session:
    """One YouTube streaming session: telemetry + player events + domain tags."""
    eid: str
    context: str                 # Indoor | Outdoor | Pedestrian | Mobility
    tech: str                    # 4G | 5G  (dominant technology in the session)
    telemetry: pd.DataFrame      # 1 Hz, chronologically sorted, indexed by second
    stall_starts: list           # datetimes at which a rebuffering event begins
    playback_start: datetime | None  # first 'playing' event; startup buffering
                                     # before this is NOT counted as a stall
    n_dropped_rows: int = 0      # corrupt rows removed, reported in the paper


def load_events(root: str = DATA_ROOT) -> dict:
    """Read the player-event log and group it by (normalised) session id."""
    path = os.path.join(root, "YouTuve QoE Events", "events.csv")
    by_eid: dict[str, list] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            ts = _parse_event_ts(row.get("Date"), row.get("Time"))
            if ts is None:
                continue
            by_eid.setdefault(_norm_eid(row.get("Eid")), []).append(
                {
                    "ts": ts,
                    "category": (row.get("Category") or "").strip(),
                    "quality": (row.get("Quality") or "").strip().rstrip(","),
                    "time_stall": _to_float(row.get("TimeStall")),
                }
            )
    for eid in by_eid:
        by_eid[eid].sort(key=lambda r: r["ts"])
    return by_eid


def _clean_telemetry(rows: list) -> tuple[pd.DataFrame, int]:
    """Turn raw log rows into a clean, chronologically sorted 1 Hz frame.

    Returns the frame and the number of rows discarded, so the report can
    state the data-quality cost honestly rather than hiding it.
    """
    recs, dropped = [], 0
    for r in rows:
        ts = _parse_log_ts(r.get("Timestamp"))
        tech = (r.get("NetworkTech") or "").strip()
        # Column-misalignment guard: NetworkTech must be a technology string.
        if ts is None or tech not in ("2G", "3G", "4G", "5G"):
            dropped += 1
            continue
        rec = {"ts": ts, "tech": tech,
               "event": (r.get("EVENT") or "").strip(),
               "state": (r.get("State") or "").strip()}
        for raw, std in RAW_TO_STD.items():
            rec[std] = _to_float(r.get(raw))
        recs.append(rec)

    if not recs:
        return pd.DataFrame(), dropped

    df = pd.DataFrame(recs).sort_values("ts").reset_index(drop=True)

    # Range-check each physical quantity; out-of-range becomes NaN rather than
    # a silently wrong feature value.
    for col, (lo, hi) in VALID_RANGE.items():
        if col in df.columns:
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

    # Collapse duplicate timestamps (the logger occasionally double-writes).
    df = df.drop_duplicates(subset="ts", keep="first").reset_index(drop=True)
    return df, dropped


def load_sessions(root: str = DATA_ROOT, verbose: bool = True) -> list[Session]:
    """Load every session for which BOTH telemetry and player events exist."""
    events = load_events(root)
    sessions, skipped = [], Counter()

    for path in sorted(glob.glob(os.path.join(root, "Channel Logs", "*", "*.csv"))):
        eid = _norm_eid(os.path.splitext(os.path.basename(path))[0])
        context = os.path.basename(os.path.dirname(path))

        if eid not in events:
            skipped["no_player_events"] += 1
            continue

        with open(path, encoding="utf-8", errors="replace") as fh:
            raw_rows = list(csv.DictReader(fh))
        df, dropped = _clean_telemetry(raw_rows)

        if len(df) < WINDOW_S + HORIZON_S + 5:
            skipped["too_short"] += 1
            continue

        ev = events[eid]
        playing = [e["ts"] for e in ev if e["category"] == "playing"]
        playback_start = min(playing) if playing else None

        # A rebuffering event that happens *after* playback has begun is a
        # stall. Buffering before the first 'playing' event is startup delay,
        # a different QoE impairment, and is excluded from the target.
        stalls = [
            e["ts"] for e in ev
            if e["category"] == "buffering"
            and playback_start is not None
            and e["ts"] > playback_start
        ]

        tech_counts = Counter(df["tech"])
        tech = tech_counts.most_common(1)[0][0]
        if tech not in ("4G", "5G"):
            skipped["odd_tech"] += 1
            continue

        sessions.append(
            Session(eid=eid, context=context, tech=tech, telemetry=df,
                    stall_starts=sorted(stalls), playback_start=playback_start,
                    n_dropped_rows=dropped)
        )

    if verbose:
        print(f"loaded {len(sessions)} sessions   skipped: {dict(skipped)}")
    return sessions


# --------------------------------------------------------------------------
# The twin state
# --------------------------------------------------------------------------

@dataclass
class TwinState:
    """The state of the service twin at one instant of replay.

    This is deliberately a real object rather than a row of a feature matrix:
    it is updated incrementally as the replay advances, it only ever contains
    information observable at or before time t, and it is what gets serialised
    into the evidence dictionary handed to the LLM.
    """
    eid: str
    context: str
    tech: str
    t: datetime | None = None
    elapsed_s: float = 0.0

    # instantaneous radio + service variables
    current: dict = field(default_factory=dict)
    # rolling history, most recent last
    history: deque = field(default_factory=lambda: deque(maxlen=WINDOW_S))
    # cumulative session counters
    n_handovers: int = 0
    n_stalls_so_far: int = 0

    def update(self, row: pd.Series) -> None:
        """Advance the twin by one observation."""
        self.t = row["ts"]
        self.current = {c: row.get(c, np.nan) for c in NUMERIC_COLS}
        self.current["tech_is_5g"] = 1.0 if row["tech"] == "5G" else 0.0
        self.history.append(dict(self.current))
        if "HANDOVER" in str(row.get("event", "")):
            self.n_handovers += 1

    def features(self) -> dict | None:
        """Feature vector built ONLY from the window ending at t.

        Every quantity here is computable by an operator at time t. Nothing
        reads forward. This is the single most important property of the
        pipeline -- a leaked future value would inflate every metric in the
        report.
        """
        if len(self.history) < WINDOW_S:
            return None   # not enough history yet; twin stays in warm-up

        hist = pd.DataFrame(list(self.history))
        f: dict = {}

        for c in ["rsrp", "rsrq", "snr", "cqi", "dl_bitrate"]:
            s = hist[c]
            f[f"{c}_last"] = s.iloc[-1]
            f[f"{c}_mean"] = s.mean()
            f[f"{c}_std"] = s.std()
            f[f"{c}_min"] = s.min()
            f[f"{c}_max"] = s.max()
            # short-term trend: is the channel improving or collapsing?
            f[f"{c}_delta"] = s.iloc[-1] - s.iloc[0]
            f[f"{c}_slope"] = _slope(s.values)

        # secondary (NR/LTE dual-connectivity) cell, when reported
        for c in ["sc_rsrp", "sc_rsrq", "sc_snr"]:
            f[f"{c}_mean"] = hist[c].mean()

        # throughput collapse indicators -- the strongest physical precursor
        # of a rebuffering event
        dl = hist["dl_bitrate"]
        f["dl_zero_frac"] = float((dl.fillna(0) <= 0).mean())
        f["dl_ratio_last_to_mean"] = (
            dl.iloc[-1] / dl.mean() if dl.mean() and dl.mean() > 0 else np.nan
        )

        f["tech_is_5g"] = self.current.get("tech_is_5g", np.nan)
        f["n_handovers"] = float(self.n_handovers)
        f["n_stalls_so_far"] = float(self.n_stalls_so_far)
        f["elapsed_s"] = float(self.elapsed_s)
        return f


def _slope(v: np.ndarray) -> float:
    """Least-squares slope of a short series; NaN-safe."""
    v = np.asarray(v, dtype=float)
    m = ~np.isnan(v)
    if m.sum() < 2:
        return np.nan
    x = np.arange(len(v))[m]
    return float(np.polyfit(x, v[m], 1)[0])


# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------

def replay_session(sess: Session, horizon_s: int = HORIZON_S) -> pd.DataFrame:
    """Replay ONE session second by second and emit (state, label) pairs.

    The label is the required future event:
        y = 1  iff a rebuffering event starts in the interval (t, t + H]

    Rows before the twin has warmed up, and rows in the final H seconds
    (whose future is not observable), are not emitted.
    """
    df = sess.telemetry
    state = TwinState(eid=sess.eid, context=sess.context, tech=sess.tech)
    stalls = np.array([s.timestamp() for s in sess.stall_starts])
    t_end = df["ts"].iloc[-1]
    t0 = df["ts"].iloc[0]

    out = []
    for _, row in df.iterrows():          # <-- chronological, never shuffled
        state.elapsed_s = (row["ts"] - t0).total_seconds()
        state.update(row)

        # count stalls already seen, so the state knows session history
        state.n_stalls_so_far = int((stalls <= row["ts"].timestamp()).sum())

        feats = state.features()
        if feats is None:
            continue
        # the last H seconds have no observable future -> cannot be labelled
        if (t_end - row["ts"]).total_seconds() < horizon_s:
            continue

        lo = row["ts"].timestamp()
        hi = (row["ts"] + timedelta(seconds=horizon_s)).timestamp()
        y = int(((stalls > lo) & (stalls <= hi)).any())

        feats.update({
            "eid": sess.eid,
            "context": sess.context,
            "tech": sess.tech,
            "ts": row["ts"],
            "y": y,
        })
        out.append(feats)

    return pd.DataFrame(out)


def build_dataset(sessions: list[Session], horizon_s: int = HORIZON_S,
                  verbose: bool = True) -> pd.DataFrame:
    """Replay every session and concatenate. Session order is preserved."""
    frames = [replay_session(s, horizon_s) for s in sessions]
    frames = [f for f in frames if len(f)]
    data = pd.concat(frames, ignore_index=True)
    if verbose:
        print(f"replayed {len(frames)} sessions -> {len(data):,} labelled instants")
        print(f"positive rate: {data['y'].mean():.4f}  "
              f"({int(data['y'].sum()):,} stall-imminent instants)")
    return data


if __name__ == "__main__":
    if not os.path.isdir(DATA_ROOT):
        raise SystemExit(
            f"\nCannot find the dataset at: {DATA_ROOT}\n"
            "Clone it first:\n"
            "    git clone https://github.com/razaulmustafa852/youtubegoes5g.git yt5g\n"
            "or point AI4T_DATA at wherever you put it.\n"
        )
    sessions = load_sessions()
    data = build_dataset(sessions)
    out = os.path.join(_HERE, "replayed.pkl")
    data.to_pickle(out)
    print(f"\nsaved -> {out}")
    print("\nstall rate by domain:")
    print(data.groupby(["tech", "context"])["y"].agg(["size", "mean"]))
