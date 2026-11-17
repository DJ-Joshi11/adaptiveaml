"""Stage 13 - Export everything the web dashboard needs as small static JSON files.

The dashboard (web/) is a static site: it never trains or scores anything, it only reads the
files written here into web/public/data/. Re-run this stage after any other stage and rebuild.

    summary.json     headline numbers, phase journey, ticker, insight text
    queue.json       every test alert: rank, risk, band, committee agreement, activity sparkline
    alerts/NN.json   per-alert detail, in 40 buckets (alert_id % 40) so the browser fetches little
    arena.json       model comparison across PBL-1 / PBL-2 / PBL-3, ablations, benchmark, baseline
    ensembles.json   ensemble diagnostics (forest, boosting, AdaBoost trace, stacking, BMA, GAM)
    clusters.json    behavioural profiles + a sample of the profile map
    sequences.json   HMM regimes, transitions, regime risk, example clients
    data.json        dataset structure, quality checks and exploratory statistics
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd

from .common import CLF_COLS, REGISTRY, ROOT, art, label, load_features, load_json, p

N_BUCKETS = 40


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if not np.isfinite(f) else round(f, 5)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_clean(obj), fh, separators=(",", ":"))


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def run(cfg: dict) -> dict:
    out_dir = ROOT / "web" / "public" / "data"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "alerts").mkdir(parents=True)

    fin, ens = load_json(art(cfg, "final.json")), load_json(art(cfg, "ensembles.json"))
    ev, cls = load_json(art(cfg, "evaluation.json")), load_json(art(cfg, "classification.json"))
    enr, ex = load_json(art(cfg, "enrichment.json")), load_json(art(cfg, "explain.json"))
    clu, sq = load_json(art(cfg, "clustering.json")), load_json(art(cfg, "sequences.json"))
    eda, dq, reg = load_json(art(cfg, "eda.json")), load_json(art(cfg, "data_quality.json")), load_json(art(cfg, "regression.json"))

    Q = pd.read_parquet(art(cfg, "final_scores.parquet")).sort_values("rank").reset_index(drop=True)
    F = load_features(cfg).set_index("alert_id")
    tr = F[F.role == "train"]
    SH = pd.read_parquet(art(cfg, "shap_test.parquet")).set_index("alert_id")
    LR = pd.read_parquet(art(cfg, "contributions.parquet")).set_index("alert_id")
    C = pd.read_parquet(art(cfg, "clusters.parquet")).set_index("alert_id")
    S = pd.read_parquet(art(cfg, "sequence_features.parquet")).set_index("alert_id")
    V = pd.read_parquet(art(cfg, "viterbi_paths.parquet")).set_index("alert_id")
    EA = pd.read_parquet(art(cfg, "expected_activity.parquet")).set_index("alert_id")
    P1 = pd.read_parquet(art(cfg, "predictions.parquet")).set_index("alert_id")
    daily = pd.read_parquet(p(cfg, cfg["data"]["processed_dir"], "alert_daily.parquet"))
    daily = daily[daily.alert_id.isin(Q.alert_id)]
    daily["week"] = (daily.day // 7).clip(upper=51)
    weekly = daily.groupby(["alert_id", "week"]).n.sum().unstack(fill_value=0).reindex(columns=range(52), fill_value=0)
    weekly = weekly.reindex(Q.alert_id, fill_value=0)

    K = sq["K"]
    regime_names = [f"R{k + 1} · {s['name']}" for k, s in enumerate(sq["states"])]
    km_prof = {r["cluster"]: r for r in clu["profiles"]["kmeans"]["train"]}
    gmm_prof = {r["cluster"]: r for r in clu["profiles"]["gmm"]["train"]}
    mu, sd = tr[CLF_COLS].mean(), tr[CLF_COLS].std().replace(0, 1)
    llr_test = S.loc[S.index.intersection(Q.alert_id), "hmm_llr"]

    # ------------------------------------------------------------------ queue + per-alert detail
    queue, buckets = [], {b: {} for b in range(N_BUCKETS)}
    votes = Q[[f"vote|{m}" for m in fin["committee"]]].values
    for i, r in enumerate(Q.itertuples(index=False)):
        a = int(r.alert_id)
        sh = SH.loc[a, CLF_COLS].astype(float)
        top_sh = sh.abs().sort_values(ascending=False).index[:8]
        lr = LR.loc[a, CLF_COLS].astype(float) if a in LR.index else None
        top_lr = lr.abs().sort_values(ascending=False).index[:6] if lr is not None else []
        x = F.loc[a]
        lead = top_sh[0]
        reason = f"{label(lead)} {'raises' if sh[lead] > 0 else 'lowers'} risk"
        queue.append({"id": a, "q": r.quarter, "risk": r.risk, "rank": r.rank, "band": r.band, "agree": r.agreement,
                      "y": int(r.y), "reason": reason, "spark": weekly.loc[a].values[::-1].reshape(26, 2).sum(1).tolist()})
        detail = {
            "id": a, "quarter": r.quarter, "y": int(r.y), "risk": r.risk, "rank": r.rank, "n": len(Q), "band": r.band,
            "agreement": r.agreement, "committee_sd": r.committee_sd,
            "votes": [{"model": m, "p": float(votes[i, j])} for j, m in enumerate(fin["committee"])],
            "bayes": {"p": float(P1.loc[a, "blr_prob"]), "lo": float(1 / (1 + np.exp(-(P1.loc[a, "blr_logit"] - 1.96 * P1.loc[a, "blr_sd"])))),
                      "hi": float(1 / (1 + np.exp(-(P1.loc[a, "blr_logit"] + 1.96 * P1.loc[a, "blr_sd"]))))},
            "shap": {"base": float(SH.loc[a, "bias"]), "rest": float(sh.drop(top_sh).sum()),
                     "top": [{"f": f, "label": label(f), "group": REGISTRY[f][0], "v": float(sh[f]), "x": float(x[f]) if pd.notna(x[f]) else None,
                              "z": float((x[f] - mu[f]) / sd[f]) if pd.notna(x[f]) else None} for f in top_sh]},
            "lr": [{"f": f, "label": label(f), "v": float(lr[f])} for f in top_lr],
            "weekly": weekly.loc[a].values.tolist(),     # index 0 = the week just before the alert
            "activity": {"actual": float(np.expm1(EA.loc[a, "actual_log"])), "expected": float(np.expm1(EA.loc[a, "expected_log"])),
                         "lo": float(np.expm1(EA.loc[a, "expected_log"] - 1.96 * EA.loc[a, "expected_sd"])),
                         "hi": float(np.expm1(EA.loc[a, "expected_log"] + 1.96 * EA.loc[a, "expected_sd"])),
                         "burst_z": float(x["burst_z"]) if pd.notna(x["burst_z"]) else None},
            "cluster": {"kmeans": int(C.loc[a, "kmeans"]), "km_rate": km_prof[int(C.loc[a, "kmeans"])]["report_rate"],
                        "km_name": km_prof[int(C.loc[a, "kmeans"])]["name"], "gmm": int(C.loc[a, "gmm"]),
                        "gmm_conf": float(C.loc[a, "gmm_conf"]), "gmm_rate": gmm_prof.get(int(C.loc[a, "gmm"]), {}).get("report_rate"),
                        "gmm_top": sorted([{"k": k, "p": float(C.loc[a, f"gmm_p{k}"])} for k in range(10)], key=lambda d: -d["p"])[:3],
                        "pc1": float(C.loc[a, "pc1"]), "pc2": float(C.loc[a, "pc2"])},
        }
        if a in S.index:
            s = S.loc[a]
            detail["hmm"] = {"llr": float(s.hmm_llr), "llr_pct": float((llr_test < s.hmm_llr).mean()),
                             "switch_rate": float(s.hmm_switch_rate), "shift": float(s.hmm_regime_shift),
                             "occ": [float(s[f"hmm_occ_{k}"]) for k in range(K)]}
        if a in V.index:
            st_, dy = np.asarray(V.loc[a, "states"]), np.asarray(V.loc[a, "days"])
            cut = np.flatnonzero(np.diff(st_)) + 1
            starts, ends = np.r_[0, cut], np.r_[cut, len(st_)]
            detail["regimes"] = [[round(float(dy[s0]), 1), round(float(dy[e0 - 1]), 1), int(st_[s0]), int(e0 - s0)]
                                 for s0, e0 in zip(starts, ends)]
        buckets[a % N_BUCKETS][a] = detail
    for b, d in buckets.items():
        dump(d, out_dir / "alerts" / f"{b}.json")
    dump(queue, out_dir / "queue.json")

    # ------------------------------------------------------------------ summary
    champ = next(r for r in fin["table"] if r["model"] == fin["champion"])
    pbl1 = next(r for r in fin["table"] if r["model"] == fin["pbl1_champion"])
    yt, risk = Q.y.values, Q.risk.values
    dec = pd.qcut(pd.Series(risk).rank(method="first"), 10, labels=False)
    deciles = [{"decile": 10 - int(d), "rate": float(yt[dec == d].mean()), "n": int((dec == d).sum()),
                "ci": wilson(int(yt[dec == d].sum()), int((dec == d).sum()))} for d in range(9, -1, -1)]
    bands = Q.band.value_counts().to_dict()
    band_rates = {b: float(Q[Q.band == b].y.mean()) for b in bands}
    agree_rates = [{"agreement": g, "n": int(len(d)), "rate": float(d.y.mean()),
                    "review_rate": float((d.band == "Review now").mean())} for g, d in Q.groupby("agreement")]
    # heatmap: report rate by risk decile x dominant regime (does the sequence view agree with the score?)
    Sd = S.reindex(Q.alert_id)
    dom = Sd[[f"hmm_occ_{k}" for k in range(K)]].values.argmax(1)
    heat = []
    for k in range(K):
        row = []
        for dd in range(10):
            m = (dec.values == 9 - dd) & (dom == k) & Sd.hmm_occ_0.notna().values
            row.append({"n": int(m.sum()), "rate": float(yt[m].mean()) if m.sum() >= 15 else None})
        heat.append({"regime": regime_names[k], "cells": row})
    reviewed = int((Q.band == "Review now").sum())
    caught = int(Q[Q.band == "Review now"].y.sum())
    summary = {
        "project": "AdaptiveAML", "n_test": len(Q), "n_reported": int(yt.sum()), "base_rate": float(yt.mean()),
        "budget": fin["budget"], "champion": fin["champion"], "champion_phase": fin["champion_phase"],
        "pbl1_champion": fin["pbl1_champion"], "verdict": fin["verdict"],
        "champion_row": champ, "pbl1_row": pbl1, "phases": fin["phases"],
        "reviewed": reviewed, "caught": caught, "deciles": deciles, "bands": bands, "band_rates": band_rates,
        "agreement": agree_rates, "heat": heat, "top": queue[:8],
        "data": {"alerts": dq["n_alerts"], "transactions": dq["n_transactions"], "report_rate": dq["report_rate"],
                 "split": dq["split"]},
        "n_candidates": len(fin["table"]), "challenger": fin["challenger"], "promoted": fin["promoted"],
        "challenger_row": next(r for r in fin["table"] if r["model"] == fin["challenger"]),
        "best_test_row": fin["table"][0], "valid_gap": fin["valid_gap_vs_pbl1"],
        "explainer_rank_corr": fin["explainer_rank_corr"], "champion_reason": fin["champion_reason"],
    }
    dump(summary, out_dir / "summary.json")

    # ------------------------------------------------------------------ arena (all model comparison)
    arena = {"final": fin, "pbl1": {k: ev[k] for k in ("table", "curves", "references", "by_quarter", "ablation_burst",
                                                       "ablation_features", "ablation_imbalance", "errors", "champion",
                                                       "champion_reason", "valid_gap_best_vs_transparent", "base_rate_test")
                                    if k in ev},
             "benchmark": ev["benchmark"], "tuning": {m: {"best": v["best_params"], "cv": v["cv_pr_auc"], "grid": v["grid"]}
                                                      for m, v in cls["models"].items()},
             "enrichment": enr, "explain": {k: ex[k] for k in ("coefs", "permutation", "permutation_model", "tree_importance",
                                                               "tree_rules", "rank_agreement_with_champion")},
             "regression": reg, "labels": {f: l for f, (_, l, _) in REGISTRY.items()}}
    dump(arena, out_dir / "arena.json")
    dump({**ens, "feature_names": [label(f) for f in CLF_COLS]}, out_dir / "ensembles.json")

    # ------------------------------------------------------------------ clusters
    rng = np.random.default_rng(cfg["seed"])
    Ct = C[C.role == "test"]
    samp = Ct.iloc[rng.choice(len(Ct), min(1800, len(Ct)), replace=False)]
    clu_out = {**clu, "scatter": [{"x": float(r.pc1), "y": float(r.pc2), "k": int(r.kmeans), "g": int(r.gmm), "r": int(r.y)}
                                  for r in samp.itertuples()]}
    dump(clu_out, out_dir / "clusters.json")

    # ------------------------------------------------------------------ sequences
    occ = [f"hmm_occ_{k}" for k in range(K)]
    SF = S.reset_index()
    SF["dominant"] = SF[occ].values.argmax(1)
    risk_rows = []
    for k in range(K):
        for period, roles in (("2020", ["train"]), ("2021", ["valid", "test"])):
            d = SF[SF.role.isin(roles) & (SF.dominant == k)]
            risk_rows.append({"k": k, "period": period, "rate": float(d.y.mean()) if len(d) else None,
                              "ci": wilson(int(d.y.sum()), len(d)), "n": len(d)})
    sw = V.states.map(lambda s: int((np.diff(s) != 0).sum()))
    ex_ids = [int(i) for i in sw[(sw >= 3) & (sw <= 12)].index[:400] if i in set(Q.alert_id)][:60]
    examples = [{"id": i, "switches": int(sw[i]), "rank": int(Q.loc[Q.alert_id == i, "rank"].iloc[0])} for i in ex_ids]
    dump({**sq, "regime_names": regime_names, "regime_risk": risk_rows,
          "overall_2021": float(SF[SF.role != "train"].y.mean()), "examples": examples}, out_dir / "sequences.json")

    # ------------------------------------------------------------------ data
    dump({"quality": dq, "eda": eda, "registry": [{"feature": f, "group": g, "label": l, "why": w} for f, (g, l, w) in REGISTRY.items()]},
         out_dir / "data.json")
    size = sum(f.stat().st_size for f in out_dir.rglob("*.json"))
    print(f"[13] exported dashboard data: {len(queue)} alerts, {size / 1e6:.1f} MB in web/public/data")
    return {"bytes": size}
