"""Stage 1 - Data ingestion and validation.

Reads the two SynthAML CSV files, normalises them into typed Parquet and writes a data-quality
report. Nothing is modelled here: the goal is to know exactly what the data is before using it.

Key ideas
- The prediction unit is the ALERT (one row in synthetic_alerts.csv). Each alert links to many
  transactions through AlertID (a 1:N relationship).
- Alert dates are only accurate to the quarter, so the only time information we take from the
  alert file is the quarter label used for the train / validation / test split.
- Every transaction gets `days_before`: how many days before the alert it happened. All feature
  windows in stage 2 are defined on this single column.
"""
from __future__ import annotations

import re

import polars as pl

from .common import art, p, save_json, split_of

ALIASES = {
    "alertid": "alert_id", "alert": "alert_id", "id": "alert_id",
    "date": "date", "alertdate": "date",
    "outcome": "outcome", "alertoutcome": "outcome",
    "timestamp": "timestamp", "time": "timestamp", "transactiontimestamp": "timestamp",
    "entry": "entry", "entrytype": "entry",
    "type": "type", "transactiontype": "type",
    "size": "size", "transactionsize": "size",
}


def _normalise_columns(df: pl.DataFrame) -> pl.DataFrame:
    ren = {}
    for c in df.columns:
        key = re.sub(r"[^a-z]", "", c.lower())
        if key in ALIASES:
            ren[c] = ALIASES[key]
    return df.rename(ren)


def _to_seconds(col: pl.Series, mode: str) -> tuple[pl.Series, str]:
    """Return the timestamp as float seconds plus the detected mode."""
    if col.dtype == pl.String:
        parsed = col.str.to_datetime(strict=False)
        if parsed.null_count() <= 0.01 * len(col):
            return parsed.dt.epoch("s").cast(pl.Float64), "datetime"
        col = col.cast(pl.Float64, strict=False)
    elif col.dtype in (pl.Datetime, pl.Date):
        return col.cast(pl.Datetime).dt.epoch("s").cast(pl.Float64), "datetime"
    col = col.cast(pl.Float64)
    if mode == "auto":
        # numeric timestamps: non-negative = seconds BEFORE the alert, non-positive = offsets forward in time
        mode = "seconds_before_alert" if col.min() >= 0 else "seconds_forward"
    return col, mode


def run(cfg: dict) -> dict:
    d = cfg["data"]
    alerts = _normalise_columns(pl.read_csv(p(cfg, d["alerts_csv"]), try_parse_dates=True))
    txn = _normalise_columns(pl.read_csv(p(cfg, d["transactions_csv"])))
    n_alert_rows, n_txn_rows = alerts.height, txn.height

    # ---- alerts: label and quarter
    alerts = alerts.with_columns(
        pl.col("date").cast(pl.String).str.to_date(strict=False).alias("date"),
        pl.col("outcome").cast(pl.String).str.to_lowercase().str.starts_with("r").cast(pl.Int8).alias("y"),
    ).with_columns(
        (pl.col("date").dt.year().cast(pl.String) + "Q" + pl.col("date").dt.quarter().cast(pl.String)).alias("quarter")
    )
    roles = split_of(cfg)
    alerts = alerts.with_columns(pl.col("quarter").replace_strict(roles, default="unused").alias("role"))

    # ---- transactions: types, direction, time relative to the alert
    secs, ts_mode = _to_seconds(txn["timestamp"], d.get("timestamp_mode", "auto"))
    txn = txn.with_columns(
        secs.alias("t"),
        pl.col("entry").cast(pl.String).str.to_lowercase().str.starts_with("c").alias("is_credit"),
        pl.col("type").cast(pl.String).str.to_lowercase().str.slice(0, 3)
        .replace({"cas": "cash", "car": "card", "wir": "wire", "int": "international"}).alias("type"),
        pl.col("size").cast(pl.Float32),
    )
    if ts_mode == "seconds_before_alert":
        txn = txn.with_columns((pl.col("t") / 86400).alias("days_before"))
    elif ts_mode == "seconds_forward":
        txn = txn.with_columns((-pl.col("t") / 86400).alias("days_before"))
    else:
        # Absolute datetimes. The alert date carries random noise inside its quarter, so it cannot
        # anchor the windows. We anchor on the alert's last observed transaction instead.
        txn = txn.with_columns(((pl.col("t").max().over("alert_id") - pl.col("t")) / 86400).alias("days_before"))
    txn = txn.with_columns(
        pl.col("days_before").cast(pl.Float32),
        ((pl.col("t") % 86400) // 3600).cast(pl.Int8).alias("hour"),
    ).select("alert_id", "days_before", "hour", "is_credit", "type", "size")

    # ---- data-quality report
    per_alert = txn.group_by("alert_id").len()
    known = set(alerts["alert_id"].to_list())
    by_q = (alerts.group_by("quarter").agg(pl.len().alias("alerts"), pl.col("y").sum().alias("reported"))
            .sort("quarter").with_columns((pl.col("reported") / pl.col("alerts")).alias("report_rate")))
    report = {
        "data_source": cfg.get("data_source", "synthaml"),
        "n_alerts": n_alert_rows,
        "n_transactions": n_txn_rows,
        "report_rate": float(alerts["y"].mean()),
        "timestamp_mode": ts_mode,
        "nulls_alerts": {c: int(alerts[c].null_count()) for c in ("alert_id", "date", "y")},
        "nulls_transactions": {c: int(txn[c].null_count()) for c in txn.columns},
        "duplicate_alert_ids": int(n_alert_rows - alerts["alert_id"].n_unique()),
        "alerts_without_transactions": int(len(known - set(per_alert["alert_id"].to_list()))),
        "transactions_without_alert": int(txn.filter(~pl.col("alert_id").is_in(list(known))).height),
        "entry_values": txn.group_by("is_credit").len().sort("is_credit").to_dicts(),
        "type_values": txn.group_by("type").len().sort("len", descending=True).to_dicts(),
        "txn_per_alert": {k: float(per_alert["len"].quantile(q)) for k, q in
                          (("p05", .05), ("median", .5), ("mean", None), ("p95", .95), ("max", 1.0)) if q is not None}
        | {"mean": float(per_alert["len"].mean())},
        "history_days_max": float(txn["days_before"].max()),
        "size_mean": float(txn["size"].mean()), "size_std": float(txn["size"].std()),
        "by_quarter": by_q.to_dicts(),
        "split": {r: int((alerts["role"] == r).sum()) for r in ("train", "valid", "test", "unused")},
    }
    alerts.write_parquet(p(cfg, d["interim_dir"], "alerts.parquet"))
    txn.write_parquet(p(cfg, d["interim_dir"], "transactions.parquet"))
    save_json(report, art(cfg, "data_quality.json"))
    print(f"[1] ingested {n_alert_rows:,} alerts / {n_txn_rows:,} transactions | "
          f"report rate {report['report_rate']:.3f} | timestamp mode: {ts_mode}")
    return report
