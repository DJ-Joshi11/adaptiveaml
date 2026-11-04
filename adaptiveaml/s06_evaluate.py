"""Stage 6 - Comparative performance analysis on quarters the models never saw.

Headline metric: PR-AUC (average precision). With ~17% positives a model that always answers
"dismiss" scores ~83% accuracy and catches nothing, so accuracy is never reported. A no-skill
ranking scores PR-AUC = base rate, which is the reference line on every chart.

What "fair" means here
    - one frozen set of predictions from stage 5, evaluated once
    - champion model chosen on VALIDATION quarters, then reported on TEST quarters
    - review budget = exactly the top 20% of the queue by score (a rank, so no labels are used);
      calibration and the safety threshold are fitted on validation only
    - 95% intervals from a stratified bootstrap over test alerts
    - model-vs-model differences from the SAME bootstrap resamples (paired), plus McNemar's test
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)

from .common import CLF_COLS, ROOT, art, load_features, load_json, save_json


def stratified_bootstrap(y: np.ndarray, n: int, rng) -> np.ndarray:
    """Index matrix (n, len(y)): resample reported and dismissed alerts separately so every
    resample keeps the same class balance."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    return np.hstack([rng.choice(pos, (n, len(pos))), rng.choice(neg, (n, len(neg)))])


def ci(v) -> list[float]:
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def boot_metric(metric, y, s, idx) -> np.ndarray:
    return np.array([metric(y[i], s[i]) for i in idx])


def thin(x, y, k=250):
    """Down-sample a curve to ~k points for plotting."""
    step = max(1, len(x) // k)
    return list(map(float, x[::step])), list(map(float, y[::step]))


def gain_curve(y, s):
    """Share of all reported alerts caught (y-axis) after reviewing the top x% of the queue."""
    order = np.argsort(-s)
    caught = np.cumsum(y[order]) / y.sum()
    frac = np.arange(1, len(y) + 1) / len(y)
    return frac, caught


def top_k(s: np.ndarray, budget: float) -> np.ndarray:
    """Flag exactly the top `budget` share of alerts by score (ties broken by queue order).
    Rank-based, so every model reviews the same number of alerts and uses no outcome labels."""
    k = int(np.ceil(budget * len(s)))
    flag = np.zeros(len(s), dtype=bool)
    flag[np.argsort(-s, kind="stable")[:k]] = True
    return flag


def reliability(y, prob, bins=10):
    edges = np.unique(np.quantile(prob, np.linspace(0, 1, bins + 1)))
    which = np.clip(np.digitize(prob, edges[1:-1]), 0, len(edges) - 2)
    return [{"pred": float(prob[which == b].mean()), "obs": float(y[which == b].mean()), "n": int((which == b).sum())}
            for b in range(len(edges) - 1) if (which == b).any()]


def mcnemar(y, a_flag, b_flag) -> float:
    """Exact McNemar test on the alerts where exactly one of two models is right."""
    a_ok, b_ok = a_flag == y, b_flag == y
    n01, n10 = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    return 1.0 if n01 + n10 == 0 else float(stats.binomtest(n01, n01 + n10, 0.5).pvalue)


def run(cfg: dict) -> dict:
    ev, rng = cfg["evaluation"], np.random.default_rng(cfg["seed"])
    P = pd.read_parquet(art(cfg, "predictions.parquet"))
    va, te = P[P.role == "valid"].reset_index(drop=True), P[P.role == "test"].reset_index(drop=True)
    yv, yt = va.y.values, te.y.values
    models = [c.split("|")[1] for c in P.columns if c.startswith("cal|")]
    base = float(yt.mean())
    idx = stratified_bootstrap(yt, ev["n_bootstrap"], rng)
    out: dict = {"base_rate_test": base, "base_rate_valid": float(yv.mean()), "n_test": len(te),
                 "n_test_reported": int(yt.sum()), "budget": ev["review_budget"],
                 "data_source": load_json(art(cfg, "data_quality.json"))["data_source"]}

    # ---- champion: best PR-AUC on VALIDATION (the test quarters play no part in the choice)
    valid_ap = {m: float(average_precision_score(yv, va[f"raw|{m}"])) for m in models}
    best_valid = max(valid_ap, key=valid_ap.get)
    champion, reason = best_valid, "highest PR-AUC on validation quarters"
    # Prefer the transparent model when it is statistically tied with the best one: its per-alert
    # reasons are exact, so the scores and the explanations then come from the same model.
    plain = "Logistic Regression"
    if best_valid != plain and plain in valid_ap:
        vidx = stratified_bootstrap(yv, ev["n_bootstrap"], rng)
        gap = (boot_metric(average_precision_score, yv, va[f"raw|{best_valid}"].values, vidx)
               - boot_metric(average_precision_score, yv, va[f"raw|{plain}"].values, vidx))
        out["valid_gap_best_vs_transparent"] = {"best": best_valid, "gap": float(gap.mean()), "ci": ci(gap)}
        if np.percentile(gap, 2.5) <= 0:
            champion = plain
            reason = (f"statistically tied with {best_valid} on validation quarters; "
                      "preferred because its per-alert reasons are exact")
    out["champion"], out["champion_reason"], out["valid_pr_auc"] = champion, reason, valid_ap
    out["best_on_validation"] = best_valid

    # ---- main table
    boot_ap = {m: boot_metric(average_precision_score, yt, te[f"raw|{m}"].values, idx) for m in models}
    rows, curves, flags = [], {}, {}
    for m in models:
        raw, cal = te[f"raw|{m}"].values, te[f"cal|{m}"].values
        flag = top_k(cal, ev["review_budget"])                                  # exactly the top 20% of the test queue
        cut = float(cal[flag].min())
        flags[m] = flag.astype(int)
        # safety point: lowest-risk share of the queue that can wait while keeping >= 95% of reports (on validation)
        vs = np.sort(va.loc[va.y == 1, f"cal|{m}"].values)
        safe_cut = float(vs[int(np.floor((1 - ev["safety_recall"]) * len(vs)))])
        diff = boot_ap[m] - boot_ap[champion]
        rows.append({
            "model": m, "pr_auc": float(average_precision_score(yt, raw)), "pr_auc_ci": ci(boot_ap[m]),
            "lift": float(average_precision_score(yt, raw) / base),
            "roc_auc": float(roc_auc_score(yt, raw)),
            "roc_auc_ci": ci(boot_metric(roc_auc_score, yt, raw, idx[:300])),
            "brier": float(brier_score_loss(yt, cal)),
            "recall_at_budget": float(yt[flag].sum() / yt.sum()), "precision_at_budget": float(yt[flag].mean()),
            "flagged_share": float(flag.mean()), "threshold": cut,
            "safe_deprioritised": float((cal < safe_cut).mean()), "safe_recall": float(yt[cal >= safe_cut].sum() / yt.sum()),
            "diff_vs_champion": float(diff.mean()), "diff_ci": ci(diff),
            "tied_with_champion": bool(m != champion and np.percentile(diff, 2.5) <= 0 <= np.percentile(diff, 97.5)),
            "valid_pr_auc": valid_ap[m],
        })
        pr, rc, _ = precision_recall_curve(yt, raw)
        fpr, tpr, _ = roc_curve(yt, raw)
        gx, gy = gain_curve(yt, raw)
        curves[m] = {"pr": thin(rc[::-1], pr[::-1]), "roc": thin(fpr, tpr), "gain": thin(gx, gy),
                     "reliability": reliability(yt, cal),
                     "op_point": [rows[-1]["recall_at_budget"], rows[-1]["precision_at_budget"]]}
        raw_is_prob = raw.min() >= 0 and raw.max() <= 1
        curves[m]["reliability_raw"] = reliability(yt, raw) if raw_is_prob else None
    for r in rows:
        r["mcnemar_p_vs_champion"] = None if r["model"] == champion else mcnemar(yt, flags[r["model"]], flags[champion])
    out["table"] = sorted(rows, key=lambda r: -r["pr_auc"])
    out["curves"] = curves

    # ---- reference rows: every model must be read against these
    refs = [{"model": "No skill (random ranking)", "pr_auc": base, "roc_auc": 0.5, "recall_at_budget": ev["review_budget"]}]
    for c in [c for c in P.columns if c.startswith("ref|")]:
        s = te[c].values
        top = top_k(s, ev["review_budget"])
        refs.append({"model": c.split("|")[1], "pr_auc": float(average_precision_score(yt, s)),
                     "pr_auc_ci": ci(boot_metric(average_precision_score, yt, s, idx)),
                     "roc_auc": float(roc_auc_score(yt, s)), "recall_at_budget": float(yt[top].sum() / yt.sum())})
    out["references"] = refs

    # ---- stability: PR-AUC per quarter (does performance decay as we move away from training?)
    out["by_quarter"] = [{"quarter": q, "role": g.role.iloc[0], "base_rate": float(g.y.mean()), "n": len(g),
                          **{m: float(average_precision_score(g.y, g[f"raw|{m}"])) for m in models}}
                         for q, g in P.groupby("quarter")]

    # ---- ablation 1: does the burst signal help? (paired bootstrap of the difference)
    out["ablation_burst"] = []
    for m in models:
        d = boot_ap[m] - boot_metric(average_precision_score, yt, te[f"noburst|{m}"].values, idx)
        out["ablation_burst"].append({
            "model": m, "with": float(average_precision_score(yt, te[f"raw|{m}"])),
            "without": float(average_precision_score(yt, te[f"noburst|{m}"])),
            "gain": float(d.mean()), "gain_ci": ci(d), "significant": bool(np.percentile(d, 2.5) > 0)})
    # ---- ablation 3: which feature family carries the signal?
    out["ablation_features"] = []
    for m in models:
        b_only = boot_metric(average_precision_score, yt, te[f"only_behaviour|{m}"].values, idx)
        p_only = boot_metric(average_precision_score, yt, te[f"only_profile|{m}"].values, idx)
        d = boot_ap[m] - p_only
        out["ablation_features"].append({
            "model": m, "all": float(average_precision_score(yt, te[f"raw|{m}"])),
            "behaviour_only": float(average_precision_score(yt, te[f"only_behaviour|{m}"])), "behaviour_ci": ci(b_only),
            "profile_only": float(average_precision_score(yt, te[f"only_profile|{m}"])), "profile_ci": ci(p_only),
            "behaviour_adds": float(d.mean()), "behaviour_adds_ci": ci(d), "significant": bool(np.percentile(d, 2.5) > 0)})
    # ---- ablation 2: imbalance strategy (compared on ranking quality and at the same review budget)
    out["ablation_imbalance"] = []
    for c in [c for c in P.columns if c.startswith("imb_")]:
        arm, m = c.split("|")[0][4:], c.split("|")[1]
        s = te[c].values
        top = top_k(s, ev["review_budget"])
        out["ablation_imbalance"].append({"model": m, "arm": arm, "pr_auc": float(average_precision_score(yt, s)),
                                          "recall_at_budget": float(yt[top].sum() / yt.sum())})

    # ---- existing approach vs AdaptiveAML, on the published split (train 2020, test all of 2021)
    ya = P.y.values
    idx_all = stratified_bootstrap(ya, 300, rng)
    def bench_row(label, family, col):
        s = P[col].values
        return {"model": label, "family": family, "roc_auc": float(roc_auc_score(ya, s)),
                "roc_auc_ci": ci(boot_metric(roc_auc_score, ya, s, idx_all)),
                "pr_auc": float(average_precision_score(ya, s)),
                "pr_auc_ci": ci(boot_metric(average_precision_score, ya, s, idx_all))}
    bench = [bench_row(c.split("|")[1], "Existing approach (56 summary features)", c) for c in P.columns if c.startswith("bench|")]
    bench += [bench_row(m, "AdaptiveAML (profile + behavioural features)", f"raw|{m}") for m in models]
    with open(ROOT / "configs" / "published_benchmark.yaml", encoding="utf-8") as fh:
        pub = yaml.safe_load(fh)
    out["benchmark"] = {"rows": bench, "base_rate": float(ya.mean()), "n": len(P),
                        "published": {k: v for k, v in pub["values"].items() if v is not None}, "published_source": pub["source"]}

    # ---- error analysis: what do the reported alerts we ranked low look like?
    F = load_features(cfg).set_index("alert_id")
    trn = F[F.role == "train"][CLF_COLS]
    Z = ((F.loc[te.alert_id, CLF_COLS] - trn.mean()) / trn.std().replace(0, 1)).reset_index(drop=True)
    hit, miss = (yt == 1) & (flags[champion] == 1), (yt == 1) & (flags[champion] == 0)
    gap = (Z[miss].mean() - Z[hit].mean()).dropna()
    out["errors"] = {"caught": int(hit.sum()), "missed": int(miss.sum()),
                     "profile": [{"feature": f, "missed_z": float(Z[miss][f].mean()), "caught_z": float(Z[hit][f].mean())}
                                 for f in gap.abs().sort_values(ascending=False).index[:8]]}

    # ---- scored alert queue for the dashboard (held-out quarters only)
    Q = P[["alert_id", "quarter", "role", "y"]].copy()
    Q["risk"] = P[f"cal|{champion}"]
    for m in models:
        Q[f"risk|{m}"] = P[f"cal|{m}"]
    Q["blr_sd"] = P["blr_sd"]
    # priority bands, per evaluation period: top 20% by rank = review now, rest split at the median score
    Q["band"] = "Low priority"
    for role in ("valid", "test"):
        r = (Q.role == role).values
        s = Q.loc[r, "risk"].values
        Q.loc[r, "band"] = np.where(top_k(s, ev["review_budget"]), "Review now",
                                    np.where(s >= np.median(s), "Standard", "Low priority"))
    Q["low_confidence"] = Q.blr_sd >= Q.blr_sd.quantile(0.9)
    Q.to_parquet(art(cfg, "scored_alerts.parquet"))

    save_json(out, art(cfg, "evaluation.json"))
    top = out["table"][0]
    print(f"[6] test base rate {base:.3f} | in service: {champion} ({reason}) | "
          f"best test PR-AUC: {top['model']} {top['pr_auc']:.3f} [{top['pr_auc_ci'][0]:.3f}, {top['pr_auc_ci'][1]:.3f}]")
    return out
