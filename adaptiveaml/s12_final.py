"""Stage 12 - Final comparative analysis: PBL-1 vs PBL-2 vs PBL-3 on the same untouched test quarters.

Rules (identical to stage 6, so every number is comparable across the three reviews)
    - one frozen set of scores per model, all trained on 2020 only
    - the final champion is chosen on VALIDATION quarters (2021 Q1-Q2); test plays no part
    - PR-AUC is the headline; 95% intervals from 1000 stratified bootstrap resamples of the test set
    - differences between models use the SAME resamples (paired) and McNemar's test on the
      review decision (top 20% of the queue)
    - if an ensemble does not beat the simpler model, the table says so
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score as AP
from sklearn.metrics import brier_score_loss, precision_recall_curve, roc_auc_score

from .common import ROOT, art, load_json, save_json
from .s05_classification import platt
from .s06_evaluate import boot_metric, ci, gain_curve, mcnemar, reliability, stratified_bootstrap, thin, top_k

PBL1 = ["Logistic Regression", "Naive Bayes", "LDA", "K-Nearest Neighbours", "Decision Tree", "SVM (RBF)"]
PBL2 = ["SVM + profiles + sequences", "Pattern model (profiles + sequences)", "HMM suspicion score"]
PBL3 = ["Random Forest", "Bagging (CART)", "XGBoost", "LightGBM", "XGBoost + profiles + sequences",
        "AdaBoost (from scratch)", "GAM (spline)", "Voting committee", "Stack: PBL-1 models",
        "Stack: PBL-1 + PBL-2", "Stack: all phases", "Bayesian model averaging"]
KIND = {"Random Forest": "Bagging", "Bagging (CART)": "Bagging", "XGBoost": "Boosting", "LightGBM": "Boosting",
        "XGBoost + profiles + sequences": "Boosting", "AdaBoost (from scratch)": "Boosting", "GAM (spline)": "Additive model",
        "Voting committee": "Committee", "Stack: PBL-1 models": "Stacking", "Stack: PBL-1 + PBL-2": "Stacking",
        "Stack: all phases": "Stacking", "Bayesian model averaging": "Model averaging",
        "SVM + profiles + sequences": "Enriched classifier", "Pattern model (profiles + sequences)": "Pattern model",
        "HMM suspicion score": "Generative score"}


def run(cfg: dict) -> dict:
    ev, rng = cfg["evaluation"], np.random.default_rng(cfg["seed"])
    P1 = pd.read_parquet(art(cfg, "predictions.parquet"))
    E = pd.read_parquet(art(cfg, "predictions_ensembles.parquet"))
    D = E.merge(P1[["alert_id"] + [f"raw|{m}" for m in PBL1] + ["blr_sd"]], on="alert_id", how="left")
    for m in PBL1:
        D[f"s|{m}"] = D[f"raw|{m}"]
    for m in PBL2 + PBL3:
        D[f"s|{m}"] = D[f"ens|{m}"]
    names = PBL1 + PBL2 + PBL3
    phase = {**{m: "PBL-1" for m in PBL1}, **{m: "PBL-2" for m in PBL2}, **{m: "PBL-3" for m in PBL3}}
    va, te = D[D.role == "valid"].reset_index(drop=True), D[D.role == "test"].reset_index(drop=True)
    yv, yt = va.y.values, te.y.values
    base = float(yt.mean())
    idx = stratified_bootstrap(yt, ev["n_bootstrap"], rng)
    pbl1_champ = load_json(art(cfg, "evaluation.json"))["champion"]

    valid_ap = {m: float(AP(yv, va[f"s|{m}"])) for m in names}
    # Champion-challenger rule (the same spirit as stage 6's "prefer the simpler model when tied"):
    # the PBL-1 champion stays in service unless the best new model (PBL-2 or PBL-3, chosen on
    # validation) beats it on VALIDATION with a paired bootstrap interval entirely above zero.
    challenger = max(PBL2 + PBL3, key=valid_ap.get)
    vidx = stratified_bootstrap(yv, ev["n_bootstrap"], rng)
    vgap = (boot_metric(AP, yv, va[f"s|{challenger}"].values, vidx) - boot_metric(AP, yv, va[f"s|{pbl1_champ}"].values, vidx))
    promoted = bool(np.percentile(vgap, 2.5) > 0)
    champion = challenger if promoted else pbl1_champ
    boot = {m: boot_metric(AP, yt, te[f"s|{m}"].values, idx) for m in names}
    cal = {m: platt(va[f"s|{m}"].values, yv)(te[f"s|{m}"].values) for m in names}
    flags = {m: top_k(te[f"s|{m}"].values, ev["review_budget"]) for m in names}

    rows = []
    for m in names:
        s = te[f"s|{m}"].values
        d = boot[m] - boot[pbl1_champ]
        dc = boot[m] - boot[champion]
        f = flags[m]
        vs = np.sort(platt(va[f"s|{m}"].values, yv)(va.loc[va.y == 1, f"s|{m}"].values))
        safe_cut = float(vs[int(np.floor((1 - ev["safety_recall"]) * len(vs)))])
        rows.append({
            "model": m, "phase": phase[m], "kind": KIND.get(m, "Single model"),
            "valid_pr_auc": valid_ap[m], "pr_auc": float(AP(yt, s)), "pr_auc_ci": ci(boot[m]),
            "lift": float(AP(yt, s) / base), "roc_auc": float(roc_auc_score(yt, s)),
            "brier": float(brier_score_loss(yt, cal[m])),
            "recall_at_budget": float(yt[f].sum() / yt.sum()), "precision_at_budget": float(yt[f].mean()),
            "safe_deprioritised": float((cal[m] < safe_cut).mean()), "safe_recall": float(yt[cal[m] >= safe_cut].sum() / yt.sum()),
            "diff_vs_pbl1": float(d.mean()), "diff_vs_pbl1_ci": ci(d),
            "verdict_vs_pbl1": ("reference" if m == pbl1_champ else "better" if np.percentile(d, 2.5) > 0
                                else "worse" if np.percentile(d, 97.5) < 0 else "tied"),
            "diff_vs_champion": float(dc.mean()), "diff_vs_champion_ci": ci(dc),
            "mcnemar_p_vs_pbl1": None if m == pbl1_champ else mcnemar(yt, flags[m].astype(int), flags[pbl1_champ].astype(int)),
        })
    rows.sort(key=lambda r: -r["pr_auc"])

    # ---- the best of each phase, chosen on validation (the honest way to compare phases)
    phases = []
    for ph, group in (("PBL-1", PBL1), ("PBL-2", PBL2), ("PBL-3", PBL3)):
        b = max(group, key=lambda m: valid_ap[m])
        r = next(x for x in rows if x["model"] == b)
        phases.append({"phase": ph, "model": b, **{k: r[k] for k in ("valid_pr_auc", "pr_auc", "pr_auc_ci", "roc_auc",
                                                                      "recall_at_budget", "precision_at_budget",
                                                                      "diff_vs_pbl1", "diff_vs_pbl1_ci", "verdict_vs_pbl1")}})

    # ---- ablation ladder and leave-one-out on the stack
    ladder = []
    prev = None
    for m in ["SVM (RBF)", "Stack: PBL-1 models", "Stack: PBL-1 + PBL-2", "Stack: all phases"]:
        step = {"model": m, "pr_auc": float(AP(yt, te[f"s|{m}"])), "pr_auc_ci": ci(boot[m]), "valid_pr_auc": valid_ap[m]}
        if prev:
            d = boot[m] - boot[prev]
            step.update(gain=float(d.mean()), gain_ci=ci(d))
        ladder.append(step)
        prev = m
    ens_info = load_json(art(cfg, "ensembles.json"))
    loo = []
    full = boot["Stack: all phases"]
    for r in ens_info["stacking"]["leave_one_out"]:
        s = te[f"stackloo|{r['removed']}"].values
        d = full - boot_metric(AP, yt, s, idx)
        loo.append({"removed": r["removed"], "phase": r["phase"], "pr_auc_without": float(AP(yt, s)),
                    "loss": float(d.mean()), "loss_ci": ci(d)})
    feat_abl = []
    for a, b in (("XGBoost", "XGBoost + profiles + sequences"), ("SVM (RBF)", "SVM + profiles + sequences")):
        d = boot[b] - boot[a]
        feat_abl.append({"model": a, "with_pbl2_features": float(AP(yt, te[f"s|{b}"])), "without": float(AP(yt, te[f"s|{a}"])),
                         "gain": float(d.mean()), "gain_ci": ci(d)})

    # ---- stability across the two test quarters
    by_q = [{"quarter": q, "n": len(g), "base_rate": float(g.y.mean()),
             **{p["model"]: float(AP(g.y, g[f"s|{p['model']}"])) for p in phases}}
            for q, g in te.groupby("quarter")]

    # ---- curves for the phase winners and the champion
    show = list(dict.fromkeys([p["model"] for p in phases] + [champion, "Random Forest", "XGBoost", "Stack: all phases", "Voting committee"]))
    curves = {}
    for m in show:
        s = te[f"s|{m}"].values
        pr, rc, _ = precision_recall_curve(yt, s)
        gx, gy = gain_curve(yt, s)
        curves[m] = {"pr": thin(rc[::-1], pr[::-1], 150), "gain": thin(gx, gy, 120), "reliability": reliability(yt, cal[m])}

    # ---- existing approach (published split: train 2020, test all of 2021) vs the final champion
    ya = D.y.values
    idx_all = stratified_bootstrap(ya, 300, rng)
    def all21(m):
        s = D[f"s|{m}"].values
        return {"model": m, "roc_auc": float(roc_auc_score(ya, s)), "roc_auc_ci": ci(boot_metric(roc_auc_score, ya, s, idx_all)),
                "pr_auc": float(AP(ya, s)), "pr_auc_ci": ci(boot_metric(AP, ya, s, idx_all))}
    bench_ev = load_json(art(cfg, "evaluation.json"))["benchmark"]
    with open(ROOT / "configs" / "published_benchmark.yaml", encoding="utf-8") as fh:
        pub = yaml.safe_load(fh)
    benchmark = {"existing": [r for r in bench_ev["rows"] if r["family"].startswith("Existing")],
                 "ours": [all21(m) for m in dict.fromkeys([pbl1_champ, champion, "XGBoost", "Random Forest"])],
                 "published": {k: v for k, v in pub["values"].items() if v is not None}, "n": len(D), "base_rate": float(ya.mean())}

    # ---- the final scored queue (test quarters): champion risk + how much the committee agrees
    committee = ["Logistic Regression", "LDA", "SVM (RBF)", "Random Forest", "XGBoost", "Pattern model (profiles + sequences)"]
    Q = te[["alert_id", "quarter", "y"]].copy()
    Q["risk"] = cal[champion]
    for m in committee:
        Q[f"vote|{m}"] = cal[m]
    # disagreement = spread of each model's PERCENTILE rank for the alert (probabilities are not
    # comparable across models at the top of the queue, ranks are)
    sd = np.std([pd.Series(te[f"s|{m}"].values).rank(pct=True).values for m in committee], axis=0)
    Q["committee_sd"] = sd
    lo, hi = np.quantile(sd, [0.5, 0.9])
    Q["agreement"] = np.where(sd >= hi, "Models disagree", np.where(sd >= lo, "Mixed", "Models agree"))
    Q["blr_sd"] = te.blr_sd.values
    f = top_k(Q.risk.values, ev["review_budget"])
    Q["band"] = np.where(f, "Review now", np.where(Q.risk >= np.median(Q.risk), "Standard", "Low priority"))
    Q["rank"] = (-Q.risk).rank(method="first").astype(int)
    Q.to_parquet(art(cfg, "final_scores.parquet"))

    champ_row = next(r for r in rows if r["model"] == champion)
    out = {"champion": champion, "champion_phase": phase[champion], "pbl1_champion": pbl1_champ,
           "challenger": challenger, "promoted": promoted,
           "champion_reason": (f"{challenger} (best new model on validation) beat the PBL-1 champion on validation "
                               "with a paired interval above zero, so it was promoted" if promoted else
                               f"the best new model on validation ({challenger}) did not beat the PBL-1 champion on "
                               "validation reliably (paired interval includes zero), so the incumbent stays in service"),
           "valid_gap_vs_pbl1": {"gap": float(vgap.mean()), "ci": ci(vgap)},
           "explainer_rank_corr": float(pd.Series(te[f"s|{champion}"].values).corr(pd.Series(te["s|XGBoost"].values), method="spearman")),
           "base_rate_test": base, "n_test": len(te), "n_test_reported": int(yt.sum()), "budget": ev["review_budget"],
           "table": rows, "phases": phases, "ladder": ladder, "stack_leave_one_out": loo, "feature_ablation": feat_abl,
           "by_quarter": by_q, "curves": curves, "benchmark": benchmark,
           "committee": committee, "agreement_cuts": [float(lo), float(hi)],
           "best_test": rows[0]["model"],
           "verdict": ("ensembles beat the PBL-1 champion" if any(r["verdict_vs_pbl1"] == "better" for r in rows if r["phase"] == "PBL-3")
                       else "no model is significantly better than the PBL-1 champion")}
    save_json(out, art(cfg, "final.json"))
    print(f"    challenger {challenger}: validation gap vs {pbl1_champ} {vgap.mean():+.3f} [{np.percentile(vgap, 2.5):+.3f}, {np.percentile(vgap, 97.5):+.3f}] -> {'promoted' if promoted else 'not promoted'}")
    print(f"[12] final champion {champion} ({phase[champion]}) test PR-AUC {champ_row['pr_auc']:.3f} "
          f"[{champ_row['pr_auc_ci'][0]:.3f}, {champ_row['pr_auc_ci'][1]:.3f}] | vs PBL-1 champion "
          f"{champ_row['diff_vs_pbl1']:+.3f} [{champ_row['diff_vs_pbl1_ci'][0]:+.3f}, {champ_row['diff_vs_pbl1_ci'][1]:+.3f}] -> {out['verdict']}")
    for p in phases:
        print(f"    {p['phase']} best on validation: {p['model']:<34} test {p['pr_auc']:.3f}  ({p['verdict_vs_pbl1']})")
    return out
