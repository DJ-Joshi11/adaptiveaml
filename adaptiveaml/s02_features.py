"""Stage 2 - Behavioural feature engineering.

Turns ~16M transactions into ONE row per alert. Two time windows are used everywhere:

    history window : days 30..365 before the alert   (prefix h_)
    recent window  : days  0..30  before the alert   (prefix r_)

The hard boundary at day 30 matters: the expected-activity regression in stage 4 predicts the
recent transaction count (r_n) from history features only. If a history feature looked inside
the last 30 days, the regression would be predicting something it had already seen.

Size is log-transformed and standardised by the dataset authors, so it is a RELATIVE amount.
We therefore use distribution statistics (mean, spread, percentiles) and never money totals.

A second family, the channel-direction profile (prefix p_), describes the size and frequency of
each kind of transaction (credit/debit x cash/card/wire/international) over the whole year.

Each alert's features are computed from that alert's own transactions only, so no information
flows between alerts. The single cross-alert quantity (the "large transaction" cut-off) is
learned on training quarters only.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .common import BURST_COLS, PROFILE_COLS, PROFILE_STATS, REGISTRY, p

TYPES = ["cash", "card", "wire", "international"]


def _entropy(shares: list[pl.Expr]) -> pl.Expr:
    """Shannon entropy of a discrete distribution given as share columns (0*log0 := 0)."""
    return -sum(pl.when(s > 0).then(s * s.log()).otherwise(0.0) for s in shares)


def run(cfg: dict) -> pl.DataFrame:
    d, w = cfg["data"], cfg["windows"]
    R, H = w["recent_days"], w["history_days"]
    alerts = pl.read_parquet(p(cfg, d["interim_dir"], "alerts.parquet"))
    txn = pl.read_parquet(p(cfg, d["interim_dir"], "transactions.parquet")).filter(pl.col("days_before") <= H)

    # "Large" = above the 90th percentile of Size among TRAINING alerts' transactions (no look-ahead).
    train_ids = alerts.filter(pl.col("role") == "train")["alert_id"]
    large_cut = float(txn.filter(pl.col("alert_id").is_in(train_ids.to_list()))["size"]
                      .quantile(w["large_size_quantile"]))
    txn = txn.with_columns((pl.col("size") > large_cut).alias("is_large"),
                           (pl.col("days_before") < R).alias("recent"))
    hist, rec = txn.filter(~pl.col("recent")), txn.filter(pl.col("recent"))

    # ---- history window: volume, direction, channel, size
    f_h = hist.group_by("alert_id").agg(
        pl.len().alias("h_n"),
        pl.col("days_before").max().alias("h_span_days"),
        (pl.col("days_before").min() - R).alias("h_days_since_last"),
        pl.col("is_credit").mean().alias("h_credit_share"),
        *[(pl.col("type") == t).mean().alias(f"h_share_{t}") for t in TYPES],
        ((pl.col("type") == "cash") & pl.col("is_credit")).sum().alias("_cash_cr"),
        pl.col("is_credit").sum().alias("_n_cr"),
        pl.col("size").mean().alias("h_size_mean"),
        pl.col("size").median().alias("h_size_median"),
        pl.col("size").std().alias("h_size_std"),
        pl.col("size").quantile(0.95).alias("h_size_p95"),
        pl.col("size").max().alias("h_size_max"),
        pl.col("is_large").mean().alias("h_large_share"),
        pl.col("size").filter(pl.col("is_credit")).median().alias("_cr_med"),
        pl.col("size").filter(~pl.col("is_credit")).median().alias("_dr_med"),
    ).with_columns(
        _entropy([pl.col(f"h_share_{t}") for t in TYPES]).alias("h_channel_entropy"),
        (pl.col("_cash_cr") / pl.col("_n_cr")).alias("h_cash_credit_share"),
        (pl.col("_cr_med") - pl.col("_dr_med")).alias("h_credit_debit_size_gap"),
    )

    # ---- history window: day-level behaviour (active days, pass-through days, quiet weeks)
    days = hist.with_columns(pl.col("days_before").floor().cast(pl.Int32).alias("day")).group_by("alert_id", "day").agg(
        pl.col("is_credit").any().alias("cr"), (~pl.col("is_credit")).any().alias("dr"))
    f_d = days.group_by("alert_id").agg(
        pl.len().alias("h_active_days"),
        (pl.col("cr") & pl.col("dr")).mean().alias("h_both_dir_day_share"),
        (pl.col("day") // 7).n_unique().alias("_weeks_active"),
    ).with_columns((1 - pl.col("_weeks_active") / ((H - R) / 7)).clip(0, 1).alias("h_inactive_week_share"))

    # ---- history window: gaps between consecutive transactions (dormancy and burstiness)
    gaps = (hist.sort("alert_id", "days_before")
            .with_columns(pl.col("days_before").diff().over("alert_id").alias("gap")).drop_nulls("gap"))
    f_g = gaps.group_by("alert_id").agg(
        pl.col("gap").max().alias("h_gap_max"),
        (pl.col("gap").std() / (pl.col("gap").mean() + 1e-9)).alias("h_gap_cv"))

    # ---- history window: spread of activity over the 24 hours of the day
    hours = hist.group_by("alert_id", "hour").len().with_columns(
        (pl.col("len") / pl.col("len").sum().over("alert_id")).alias("s"))
    f_t = hours.group_by("alert_id").agg((-(pl.col("s") * pl.col("s").log()).sum()).alias("h_hour_entropy"))

    # ---- recent window
    f_r = rec.group_by("alert_id").agg(
        pl.len().alias("r_n"),
        (pl.col("days_before") < 7).sum().alias("r_n7"),
        pl.col("is_credit").mean().alias("r_credit_share"),
        (pl.col("type") == "cash").mean().alias("r_share_cash"),
        (pl.col("type") == "international").mean().alias("r_share_international"),
        pl.col("size").mean().alias("r_size_mean"),
        pl.col("is_large").mean().alias("r_large_share"),
    )

    # ---- channel-direction cells: one group per (direction, channel), over the full year
    g = txn.with_columns((pl.when(pl.col("is_credit")).then(pl.lit("credit")).otherwise(pl.lit("debit"))
                          + "_" + pl.col("type")).alias("cell")).group_by("alert_id", "cell").agg(
        pl.col("size").min().alias("min"), pl.col("size").mean().alias("mean"),
        pl.col("size").median().alias("median"), pl.col("size").max().alias("max"),
        pl.col("size").std().alias("std"), pl.len().alias("n"), pl.col("size").sum().alias("sum"))
    # our profile features: 6 readable statistics per cell (no "sum": Size is not currency)
    prof = g.pivot(on="cell", index="alert_id", values=list(PROFILE_STATS))
    prof = prof.rename({c: "p_" + "_".join(c.split("_")[1:]) + "_" + c.split("_")[0] for c in prof.columns if c != "alert_id"})
    for c in PROFILE_COLS:                              # a cell that never occurs in the data at all
        if c not in prof.columns:
            prof = prof.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
    prof = prof.with_columns([pl.col(c).fill_null(0) for c in PROFILE_COLS if c.endswith("_n")])

    # ---- assemble one row per alert
    feats = alerts.select("alert_id", "quarter", "role", "y")
    for f in (f_h, f_d, f_g, f_t, f_r, prof):
        feats = feats.join(f, on="alert_id", how="left")
    feats = feats.with_columns([pl.col(c).fill_null(0) for c in PROFILE_COLS if c.endswith("_n")]).with_columns(
        pl.col("h_n").is_null().cast(pl.Int8).alias("no_history"),
        pl.col("h_n").fill_null(0), pl.col("h_active_days").fill_null(0),
        pl.col("r_n").fill_null(0), pl.col("r_n7").fill_null(0),
    ).with_columns(
        (pl.col("h_n") / pl.col("h_active_days").clip(1)).alias("h_txn_per_active_day"),
        # recent daily rate relative to historical daily rate (+1 smoothing keeps it finite)
        (((pl.col("r_n") + 1) / R) / ((pl.col("h_n") + 1) / (H - R))).alias("rate_ratio"),
        (pl.col("r_size_mean") - pl.col("h_size_mean")).alias("size_shift"),
        (pl.col("r_share_cash") - pl.col("h_share_cash")).alias("cash_shift"),
        (pl.col("r_share_international") - pl.col("h_share_international")).alias("intl_shift"),
    )
    cols = [c for c in REGISTRY if c not in BURST_COLS]
    feats = feats.select("alert_id", "quarter", "role", "y", *cols)
    feats.write_parquet(p(cfg, d["processed_dir"], "alert_features.parquet"))

    # ---- published-benchmark features: the SynthAML paper's own recipe, re-implemented so that
    #      "existing approach vs AdaptiveAML" can be compared on identical alerts and splits.
    #      7 statistics x 2 entry directions x 4 transaction types = 56 features per alert.
    wide = g.rename({"n": "count"}).pivot(on="cell", index="alert_id",
                                          values=["min", "mean", "median", "max", "std", "count", "sum"])
    bench = alerts.select("alert_id", "quarter", "role", "y").join(wide, on="alert_id", how="left").fill_null(0)
    bench.write_parquet(p(cfg, d["processed_dir"], "benchmark_features.parquet"))

    # ---- per-alert daily timeline for the dashboard's alert detail view (held-out alerts only)
    keep = alerts.filter(pl.col("role").is_in(["valid", "test"]))["alert_id"].to_list()
    tl = (txn.filter(pl.col("alert_id").is_in(keep))
          .with_columns(pl.col("days_before").floor().cast(pl.Int16).alias("day"))
          .group_by("alert_id", "day").agg(pl.len().alias("n"), pl.col("is_credit").sum().alias("n_credit"),
                                           (pl.col("type") == "cash").sum().alias("n_cash")))
    tl.write_parquet(p(cfg, d["processed_dir"], "alert_daily.parquet"))
    print(f"[2] built {len(cols)} features for {feats.height:,} alerts | large-size cut-off (train) = {large_cut:.3f}")
    return feats
