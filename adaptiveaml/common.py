"""Shared helpers: configuration, paths, seeding, artifact I/O and the feature registry.

Everything that more than one pipeline stage needs lives here, so each stage file
(s01 ... s07) reads top to bottom as a single story.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- config
def load_config(path: str | Path = "configs/base.yaml") -> dict:
    with open(ROOT / path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = str(ROOT)
    return cfg


def p(cfg: dict, *parts: str) -> Path:
    """Path relative to the project root, created on demand."""
    out = ROOT.joinpath(*parts)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def art(cfg: dict, name: str) -> Path:
    return p(cfg, cfg["artifacts_dir"], name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------- artifact I/O
class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, cls=_Enc, indent=1)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- split
def split_of(cfg: dict) -> dict[str, str]:
    """Map each quarter label to its role: train / valid / test."""
    return {q: role for role in ("train", "valid", "test") for q in cfg["split"][role]}


def load_features(cfg: dict) -> pd.DataFrame:
    return pd.read_parquet(p(cfg, cfg["data"]["processed_dir"], "alert_features.parquet"))


# --------------------------------------------------------------------------- feature registry
# name -> (group, plain-language label, rationale). This is the disclosed feature set:
# the base paper criticises AML studies for hiding their features, so ours are documented.
# Prefix h_ = history window (days 30-365 before the alert); r_ = recent window (last 30 days).
REGISTRY: dict[str, tuple[str, str, str]] = {
    # Volume and tenure
    "h_n": ("Volume", "Transactions in history", "Scale of the client's normal activity"),
    "h_active_days": ("Volume", "Active days in history", "Regular user vs occasional user"),
    "h_span_days": ("Volume", "Length of history (days)", "Short tenure is a risk marker"),
    "h_txn_per_active_day": ("Volume", "Transactions per active day", "Intensity on the days the account is used"),
    # Dormancy
    "h_gap_max": ("Dormancy", "Longest quiet period (days)", "Dormant accounts that wake up"),
    "h_days_since_last": ("Dormancy", "Quiet days before the recent window", "Was the account asleep just before the alert?"),
    "h_inactive_week_share": ("Dormancy", "Share of weeks with no activity", "How intermittent the account is"),
    # Direction
    "h_credit_share": ("Direction", "Share of credits", "Money mostly coming in or going out"),
    "h_both_dir_day_share": ("Direction", "Days with both credit and debit", "Pass-through: money in and straight out"),
    "h_credit_debit_size_gap": ("Direction", "Credit vs debit typical size gap", "Large credits split into small debits or the reverse"),
    # Channel mix
    "h_share_cash": ("Channel", "Cash share", "Cash is the hardest channel to trace"),
    "h_share_card": ("Channel", "Card share", "Everyday retail behaviour"),
    "h_share_wire": ("Channel", "Wire share", "Account-to-account movement"),
    "h_share_international": ("Channel", "International share", "Cross-border movement"),
    "h_channel_entropy": ("Channel", "Channel diversity", "One channel only vs a mix"),
    "h_cash_credit_share": ("Channel", "Cash share among credits", "Cash deposits specifically"),
    # Size distribution (Size is log-standardised: these are relative, never money totals)
    "h_size_mean": ("Size", "Typical size (mean)", "Level of transaction sizes"),
    "h_size_median": ("Size", "Typical size (median)", "Robust level of sizes"),
    "h_size_std": ("Size", "Size variability", "Erratic vs uniform amounts"),
    "h_size_p95": ("Size", "Large-end size (95th pct)", "How big the big transactions are"),
    "h_size_max": ("Size", "Largest size", "Single extreme transaction"),
    "h_large_share": ("Size", "Share of large transactions", "Above the training 90th percentile"),
    # Timing
    "h_hour_entropy": ("Timing", "Time-of-day spread", "Scripted activity is unusually regular"),
    "h_gap_cv": ("Timing", "Irregularity of gaps", "Bursty vs evenly spaced activity"),
    # Recent window
    "r_n": ("Recent", "Transactions in last 30 days", "Activity right before the alert"),
    "r_n7": ("Recent", "Transactions in last 7 days", "Very recent activity"),
    "r_credit_share": ("Recent", "Recent share of credits", "Direction right before the alert"),
    "r_share_cash": ("Recent", "Recent cash share", "Channel right before the alert"),
    "r_share_international": ("Recent", "Recent international share", "Cross-border right before the alert"),
    "r_size_mean": ("Recent", "Recent typical size", "Sizes right before the alert"),
    "r_large_share": ("Recent", "Recent share of large transactions", "Large amounts right before the alert"),
    # Change: recent vs history
    "rate_ratio": ("Change", "Activity rate vs history", "Recent daily rate / historical daily rate"),
    "size_shift": ("Change", "Shift in typical size", "Recent mean size minus historical"),
    "cash_shift": ("Change", "Shift in cash share", "Recent cash share minus historical"),
    "intl_shift": ("Change", "Shift in international share", "Recent international share minus historical"),
    "no_history": ("Change", "No history before last 30 days", "Brand-new activity"),
    # Burst signal (produced by the expected-activity model in stage 4)
    "burst_resid": ("Burst", "Activity above expected", "Actual minus expected recent activity (log scale)"),
    "burst_z": ("Burst", "Activity above expected (confidence-adjusted)", "Residual divided by the model's own uncertainty"),
}

# Channel-direction profile: how big and how frequent each kind of transaction is, over the full
# year before the alert. 2 directions x 4 channels x 6 statistics = 48 features. This is the
# family of features used by the dataset's published benchmark, given readable names here.
PROFILE_STATS = {"n": "number", "mean": "typical size (mean)", "median": "typical size (median)",
                 "std": "size variability", "min": "smallest size", "max": "largest size"}
for _d in ("credit", "debit"):
    for _t in ("cash", "card", "wire", "international"):
        for _s, _lab in PROFILE_STATS.items():
            REGISTRY[f"p_{_d}_{_t}_{_s}"] = ("Profile", f"{_t.title()} {_d}s: {_lab}",
                                             f"Size and frequency profile of {_t} {_d}s over the year")
PROFILE_COLS = [c for c in REGISTRY if c.startswith("p_")]
BEHAVIOUR_COLS = [c for c in REGISTRY if not c.startswith("p_")]

COUNT_COLS = [c for c in PROFILE_COLS if c.endswith("_n")] + ["h_n", "h_active_days", "h_span_days", "h_txn_per_active_day", "h_gap_max",
              "h_days_since_last", "r_n", "r_n7", "rate_ratio", "h_gap_cv"]
BURST_COLS = ["burst_resid", "burst_z"]
# Inputs of the expected-activity regression: history window ONLY (no leakage into the target).
HISTORY_COLS = [c for c in REGISTRY if c.startswith("h_")]
# Inputs of the classifiers: everything.
CLF_COLS = list(REGISTRY)


def label(col: str) -> str:
    return REGISTRY[col][1] if col in REGISTRY else col


# --------------------------------------------------------------------------- time-aware validation
def forward_folds(quarters: pd.Series, train_quarters: list[str]):
    """Forward-chaining folds inside the training period: fit on Q1 -> validate on Q2,
    fit on Q1-Q2 -> validate on Q3, and so on. The model never validates on the past.
    Yields positional (fit_idx, val_idx) arrays, usable as a scikit-learn `cv`."""
    q = quarters.values
    for i in range(1, len(train_quarters)):
        fit = np.where(np.isin(q, train_quarters[:i]))[0]
        val = np.where(q == train_quarters[i])[0]
        if len(fit) and len(val):
            yield fit, val
