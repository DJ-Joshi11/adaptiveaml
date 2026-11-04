"""Generate a SMALL schema-faithful stand-in for SynthAML so the pipeline can be smoke-tested
without the 16M-row download.

THIS IS NOT SynthAML. Numbers produced from it are meaningless and must never be presented.
The pipeline marks every artifact built from it as data_source = "fixture" and the dashboard
shows a warning banner.

Usage:  python tests/make_fixture.py [n_alerts]
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
n_alerts = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
rng = np.random.default_rng(7)

quarters = [(y, q) for y in (2020, 2021) for q in (1, 2, 3, 4)]
qi = rng.integers(0, 8, n_alerts)
q_start = np.array([np.datetime64(f"{y}-{3*q-2:02d}-01") for y, q in quarters])[qi]
alert_true = q_start + rng.integers(0, 85, n_alerts).astype("timedelta64[D]")
alert_noisy = q_start + rng.integers(0, 85, n_alerts).astype("timedelta64[D]")  # date noise within quarter

latent = rng.normal(size=n_alerts)
burst = np.exp(rng.normal(0, 0.6, n_alerts))
cash_pref = rng.beta(2, 6, n_alerts)
score = 0.9 * np.log(burst) + 1.5 * (cash_pref - 0.25) + 0.5 * latent + 0.15 * (qi >= 4)
y = score > np.quantile(score, 0.83)

n_hist = np.maximum(5, rng.lognormal(5.0, 0.9, n_alerts)).astype(int)
n_recent = rng.poisson(n_hist / 335 * 30 * burst)
tot = n_hist + n_recent
aid = np.repeat(np.arange(n_alerts), tot)
is_recent = np.concatenate([np.r_[np.zeros(h, bool), np.ones(r, bool)] for h, r in zip(n_hist, n_recent)])
days_before = np.where(is_recent, rng.uniform(0, 30, aid.size), rng.uniform(30, 365, aid.size))
ts = alert_true[aid].astype("datetime64[s]") - (days_before * 86400).astype("timedelta64[s]")
u = rng.uniform(size=aid.size)
cp = cash_pref[aid] * np.where(is_recent & y[aid], 1.4, 1.0)
typ = np.where(u < cp, "Cash", np.where(u < cp + 0.45, "Card", np.where(u < cp + 0.6, "Wire", "International")))
entry = np.where(rng.uniform(size=aid.size) < 0.45 + 0.01 * y[aid], "Credit", "Debit")
size = rng.normal(0.1 * latent[aid] + 0.3 * (typ == "Wire"), 1.0) * rng.uniform(0.98, 1.02, aid.size)

out = ROOT / "data" / "sample"
out.mkdir(parents=True, exist_ok=True)
pl.DataFrame({"AlertID": np.arange(n_alerts), "Date": alert_noisy.astype(str),
              "Outcome": np.where(y, "Report", "Dismiss")}).write_csv(out / "synthetic_alerts.csv")
pl.DataFrame({"AlertID": aid, "Timestamp": ts.astype(str), "Entry": entry, "Type": typ,
              "Size": size.round(5)}).write_csv(out / "synthetic_transactions.csv")
print(f"fixture written: {n_alerts} alerts, {aid.size:,} transactions -> {out}")
