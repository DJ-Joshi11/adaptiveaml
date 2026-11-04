"""Stage 9 - Sequence analysis: Hidden Markov Models over each alert's transaction history.

The 84 features of stages 2-7 summarise a year of transactions and throw the ORDER away. Here the
history is treated as a sequence, and an HMM looks for hidden behavioural regimes (for example
"rapid small card debits" or "occasional large wire credits") and for how the client moves
between them.

Encoding: each transaction becomes one of 72 symbols
    direction (credit, debit) x channel (cash, card, wire, international)
    x size band (small / medium / large: tertiles of training transactions)
    x time since the previous transaction (< 1 hour, < 1 day, longer)
Transactions inside the first and last hour of each alert's one-year window are dropped: the
synthetic data piles clipped timestamps there, so their order carries no information.

Models (all fitted on 2020 training alerts only, see hmm.py for the algorithms)
    behaviour HMM       unsupervised, K states chosen by held-out log-likelihood and BIC;
                        Viterbi paths and state occupancy describe each alert
    report HMM / dismiss HMM
                        one HMM per outcome. The log-likelihood ratio between them is a
                        generative suspicion score; the dismiss-HMM likelihood alone is an
                        anomaly score ("how unlike normal behaviour is this sequence?").
                        Training alerts are CROSS-FITTED (scored by models that did not see them).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from .common import art, p, save_json
from .hmm import DiscreteHMM

TYPES = ["cash", "card", "wire", "international"]
SIZES = ["small", "medium", "large"]
GAPS = ["within the hour", "same day", "after a day or more"]
EDGE = 1 / 24                                                   # one hour, in days


def symbol_name(m: int) -> str:
    g, s, rest = m % 3, (m // 3) % 3, m // 9
    t, d = rest % 4, rest // 4
    return f"{TYPES[t]} {'credit' if d == 0 else 'debit'}, {SIZES[s]}, {GAPS[g]}"


def name_parts(marg: dict, lift: dict) -> tuple[str, str, str, str]:
    """Words for a regime: in each dimension, the most over-represented value among those that make
    up at least 15% of the regime (so a rare category with a big lift cannot name a regime)."""
    def pick(kd):
        share, lf = np.asarray(marg[kd]), np.asarray(lift[kd])
        ok = share >= 0.15
        return int(np.argmax(np.where(ok, lf, -1))) if ok.any() else int(np.argmax(share))
    return (TYPES[pick("channel")], ["credits", "debits"][pick("direction")],
            ["rapid", "same-day", "spread-out"][pick("gap")], SIZES[pick("size")])


def encode(cfg: dict, alerts: pl.DataFrame) -> tuple[pl.DataFrame, list[float]]:
    txn = pl.read_parquet(p(cfg, cfg["data"]["interim_dir"], "transactions.parquet"),
                          columns=["alert_id", "days_before", "is_credit", "type", "size"])
    H = cfg["windows"]["history_days"]
    txn = txn.filter((pl.col("days_before") > EDGE) & (pl.col("days_before") < H - EDGE))
    train_ids = alerts.filter(pl.col("role") == "train")["alert_id"]
    q = txn.filter(pl.col("alert_id").is_in(train_ids.to_list()))["size"].quantile
    cuts = [float(q(1 / 3)), float(q(2 / 3))]
    txn = (txn.sort(["alert_id", "days_before"], descending=[False, True])          # oldest first
           .with_columns((pl.col("days_before").shift(1).over("alert_id") - pl.col("days_before")).alias("gap")))
    t_idx = pl.col("type").replace_strict({t: i for i, t in enumerate(TYPES)}, default=0)
    s_idx = pl.when(pl.col("size") <= cuts[0]).then(0).when(pl.col("size") <= cuts[1]).then(1).otherwise(2)
    g_idx = pl.when(pl.col("gap").is_null()).then(2).when(pl.col("gap") < EDGE).then(0).when(pl.col("gap") < 1).then(1).otherwise(2)
    d_idx = pl.when(pl.col("is_credit")).then(0).otherwise(1)
    txn = txn.with_columns((((d_idx * 4 + t_idx) * 3 + s_idx) * 3 + g_idx).cast(pl.Int64).alias("sym"))
    seqs = txn.group_by("alert_id", maintain_order=True).agg(pl.col("sym"), pl.col("days_before"))
    return seqs, cuts


def fit_best(seqs, K, M, seed, n_iter=30, restarts=2):
    best = None
    for r in range(restarts):
        h = DiscreteHMM(K, M, seed=seed + r).fit(seqs, n_iter=n_iter)
        if best is None or h.history[-1] > best.history[-1]:
            best = h
    return best


def run(cfg: dict) -> dict:
    seed, M = cfg["seed"], 72
    rng = np.random.default_rng(seed)
    t0 = time.time()
    alerts = pl.read_parquet(p(cfg, cfg["data"]["interim_dir"], "alerts.parquet")).filter(
        pl.col("role").is_in(["train", "valid", "test"]))
    seqs_df, cuts = encode(cfg, alerts)
    meta = alerts.select("alert_id", "role", "quarter", "y").join(seqs_df, on="alert_id", how="inner").to_pandas()
    meta = meta[meta.sym.map(len) >= 2].reset_index(drop=True)
    S = [np.asarray(s, dtype=np.int64) for s in meta.sym]
    D = [np.asarray(d, dtype=np.float32) for d in meta.days_before]
    role, y, quarter = meta.role.values, meta.y.values, meta.quarter.values
    tr_idx, va_idx = np.where(role == "train")[0], np.where(role == "valid")[0]
    print(f"    encoded {sum(map(len, S)):,} transactions into {len(S):,} sequences ({time.time() - t0:.0f}s)")

    def sample(idx, n):
        return rng.choice(idx, min(n, len(idx)), replace=False)

    # ---- 1. how many hidden states? held-out log-likelihood per transaction, and BIC
    fit_s, val_s = sample(tr_idx, 1200), sample(va_idx, 600)
    n_fit = sum(len(S[i]) for i in fit_s)
    k_sel = []
    for K in (2, 3, 4, 5, 6):
        h = DiscreteHMM(K, M, seed=seed).fit([S[i] for i in fit_s], n_iter=20)
        ll_tr = h.history[-1] * n_fit
        val = h.loglik([S[i] for i in val_s]).sum() / sum(len(S[i]) for i in val_s)
        k_sel.append({"K": K, "val_ll_per_txn": float(val), "bic": float(-2 * ll_tr + h.n_params() * np.log(n_fit)),
                      "train_ll_per_txn": float(h.history[-1])})
        print(f"    {K} states: held-out log-lik per transaction {val:.4f} ({time.time() - t0:.0f}s)")
    K = min(k_sel, key=lambda r: r["bic"])["K"]

    # ---- 2. behaviour HMM: Baum-Welch on training alerts
    beh = fit_best([S[i] for i in sample(tr_idx, 2500)], K, M, seed, n_iter=40)
    print(f"    behaviour HMM fitted with {K} states ({time.time() - t0:.0f}s)")
    gam = beh.posteriors(S)
    paths = beh.viterbi(S)
    occ = np.vstack([g.mean(0) for g in gam])
    switches = np.array([(np.diff(v) != 0).sum() / max(len(v) - 1, 1) * 100 for v in paths])
    rec = [d < cfg["windows"]["recent_days"] for d in D]
    recent_occ = np.vstack([g[r].mean(0) if r.any() else g.mean(0) for g, r in zip(gam, rec)])
    hist_occ = np.vstack([g[~r].mean(0) if (~r).any() else g.mean(0) for g, r in zip(gam, rec)])
    shift = np.abs(recent_occ - hist_occ).sum(1) / 2                 # 0 = same mix of regimes, 1 = complete change

    # ---- 3. outcome-specific HMMs, cross-fitted on the training year
    def class_models(idx):
        rep, dis = idx[y[idx] == 1], idx[y[idx] == 0]
        hr = fit_best([S[i] for i in sample(rep, 900)], K, M, seed, n_iter=25, restarts=1)
        hd = fit_best([S[i] for i in sample(dis, 900)], K, M, seed, n_iter=25, restarts=1)
        return hr, hd

    n_txn = np.array([len(s) for s in S], dtype=float)
    ll_rep, ll_dis = np.zeros(len(S)), np.zeros(len(S))
    halves = [np.isin(quarter, cfg["split"]["train"][:2]), np.isin(quarter, cfg["split"]["train"][2:])]
    for fit_half, score_half in ((halves[0], halves[1]), (halves[1], halves[0])):
        hr, hd = class_models(np.where(fit_half & (role == "train"))[0])
        sc = np.where(score_half & (role == "train"))[0]
        ll_rep[sc], ll_dis[sc] = hr.loglik([S[i] for i in sc]), hd.loglik([S[i] for i in sc])
    hr, hd = class_models(tr_idx)
    later = np.where(role != "train")[0]
    ll_rep[later], ll_dis[later] = hr.loglik([S[i] for i in later]), hd.loglik([S[i] for i in later])
    llr = (ll_rep - ll_dis) / n_txn
    normal = ll_dis / n_txn
    print(f"    outcome HMMs fitted and cross-fitted ({time.time() - t0:.0f}s)")

    # ---- per-alert features
    feat = pd.DataFrame({"alert_id": meta.alert_id, "role": role, "quarter": quarter, "y": y,
                         "hmm_llr": llr, "hmm_normal_ll": normal, "hmm_switch_rate": switches, "hmm_regime_shift": shift,
                         "hmm_n": n_txn})
    for k in range(K):
        feat[f"hmm_occ_{k}"] = occ[:, k]
    feat.to_parquet(art(cfg, "sequence_features.parquet"))
    te = np.where(role == "test")[0]
    pd.DataFrame({"alert_id": meta.alert_id.values[te],
                  "states": [paths[i].astype(np.int8) for i in te],
                  "days": [D[i] for i in te],
                  "symbols": [S[i].astype(np.int8) for i in te]}).to_parquet(art(cfg, "viterbi_paths.parquet"))

    # ---- describe the states in words
    freq = np.bincount(np.concatenate([S[i] for i in tr_idx]), minlength=M) / sum(len(S[i]) for i in tr_idx)
    states = []
    for k in range(K):
        b = beh.B[k]
        lift = b / np.maximum(freq, 1e-9)
        top = [int(m) for m in np.argsort(-b)[:5]]
        dist = [int(m) for m in np.argsort(-(lift * (b > 0.01)))[:3]]
        marg = {"direction": [float(b.reshape(2, 4, 3, 3)[d].sum()) for d in range(2)],
                "channel": [float(b.reshape(2, 4, 3, 3)[:, t].sum()) for t in range(4)],
                "size": [float(b.reshape(2, 4, 3, 3)[:, :, s].sum()) for s in range(3)],
                "gap": [float(b.reshape(2, 4, 3, 3)[:, :, :, g].sum()) for g in range(3)]}
        # name each state by what it over-represents relative to all training transactions
        fr = freq.reshape(2, 4, 3, 3)
        base = {"direction": fr.sum((1, 2, 3)), "channel": fr.sum((0, 2, 3)), "size": fr.sum((0, 1, 3)), "gap": fr.sum((0, 1, 2))}
        lift_of = {kd: np.array(v) / np.maximum(base[kd], 1e-9) for kd, v in marg.items()}
        ch, dr, gp, sz = name_parts(marg, lift_of)
        states.append({"state": k, "name": f"{gp.title()} {sz} {ch} {dr}", "marginals": marg, "lifts": {kd: v.tolist() for kd, v in lift_of.items()},
                       "top": [{"symbol": symbol_name(m), "p": float(b[m])} for m in top],
                       "distinctive": [{"symbol": symbol_name(m), "lift": float(lift[m])} for m in dist],
                       "occupancy_report": float(occ[(role == "train") & (y == 1), k].mean()),
                       "occupancy_dismiss": float(occ[(role == "train") & (y == 0), k].mean()),
                       "expected_duration": float(1 / max(1 - beh.A[k, k], 1e-9))})

    # ---- how well do sequence scores rank alerts on their own? (test quarters)
    yt = y[te]
    def score_row(name, s):
        return {"score": name, "pr_auc": float(average_precision_score(yt, s)), "roc_auc": float(roc_auc_score(yt, s))}
    standalone = [score_row("Outcome log-likelihood ratio", llr[te]),
                  score_row("Unusualness vs normal behaviour", -normal[te]),
                  score_row("Regime shift in last 30 days", shift[te]),
                  score_row("Regime switching rate", switches[te])]
    bins = np.linspace(np.percentile(llr[te], 1), np.percentile(llr[te], 99), 41)
    res = {"K": K, "n_symbols": M, "size_cuts": cuts, "k_selection": k_sel,
           "pi": beh.pi, "A": beh.A, "history": beh.history, "states": states,
           "standalone": standalone, "base_rate_test": float(yt.mean()),
           "llr_hist": {"bins": bins, "report": np.histogram(llr[te][yt == 1], bins, density=True)[0],
                        "dismiss": np.histogram(llr[te][yt == 0], bins, density=True)[0]},
           "class_history": {"report": hr.history, "dismiss": hd.history},
           "n_sequences": len(S), "n_transactions": int(n_txn.sum()), "seconds": time.time() - t0}
    save_json(res, art(cfg, "sequences.json"))
    best = max(standalone, key=lambda r: r["pr_auc"])
    print(f"[9] HMM with {K} states | strongest stand-alone sequence score: {best['score']} PR-AUC {best['pr_auc']:.3f} "
          f"(no skill {yt.mean():.3f}) | {time.time() - t0:.0f}s")
    return res
