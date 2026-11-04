"""Stage 10 - Do behavioural profiles and sequence models add anything to the risk models?

Ablation on exactly the PBL-1 protocol: same six classifiers with their tuned settings, same
2020 training year, same 2021 Q3-Q4 test quarters, same stratified bootstrap.

    Baseline          the 86 features of stages 2-7
    + Profiles        cluster membership (K-Means), soft memberships and density (Gaussian mixture)
    + Sequences       HMM log-likelihood ratio, unusualness, regime occupancy, switching, regime shift
    + Both

Every added feature was itself fitted on 2020 alerts only (stages 8 and 9), so nothing from the
evaluation quarters leaks into training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .common import CLF_COLS, art, load_features, load_json, save_json
from .s05_classification import MODELS, build, raw_score
from .s06_evaluate import boot_metric, ci, stratified_bootstrap, top_k


def run(cfg: dict) -> dict:
    seed, ev = cfg["seed"], cfg["evaluation"]
    F = load_features(cfg)
    C = pd.read_parquet(art(cfg, "clusters.parquet"))
    S = pd.read_parquet(art(cfg, "sequence_features.parquet"))
    cl_cols = [c for c in C.columns if c.startswith(("km_", "gmm_p"))] + ["gmm_density", "gmm_entropy"]
    sq_cols = [c for c in S.columns if c.startswith("hmm_") and c != "hmm_n"]
    df = (F.merge(C[["alert_id"] + cl_cols], on="alert_id", how="left")
           .merge(S[["alert_id"] + sq_cols], on="alert_id", how="left"))
    tr, te = df[df.role == "train"], df[df.role == "test"].reset_index(drop=True)
    yt = te.y.values
    idx = stratified_bootstrap(yt, ev["n_bootstrap"], np.random.default_rng(seed))
    best = load_json(art(cfg, "classification.json"))["models"]
    sets = {"Baseline": CLF_COLS, "+ Profiles": CLF_COLS + cl_cols,
            "+ Sequences": CLF_COLS + sq_cols, "+ Both": CLF_COLS + cl_cols + sq_cols}

    rows, scores = [], {}
    for name in MODELS:
        base_boot = None
        for variant, cols in sets.items():
            m = build(name, cols, best[name]["best_params"], seed).fit(tr[cols], tr.y)
            s = raw_score(m, te[cols])
            scores[f"{variant}|{name}"] = s
            b = boot_metric(average_precision_score, yt, s, idx)
            if variant == "Baseline":
                base_boot = b
            d = b - base_boot
            rows.append({"model": name, "variant": variant, "pr_auc": float(average_precision_score(yt, s)),
                         "pr_auc_ci": ci(b), "recall_at_budget": float(yt[top_k(s, ev["review_budget"])].sum() / yt.sum()),
                         "gain": float(d.mean()), "gain_ci": ci(d),
                         "verdict": "baseline" if variant == "Baseline" else
                         ("adds signal" if np.percentile(d, 2.5) > 0 else "hurts" if np.percentile(d, 97.5) < 0 else "no measurable effect")})
        print(f"    {name:<22} " + "  ".join(f"{r['variant']} {r['pr_auc']:.3f}" for r in rows[-4:]))

    # stand-alone unsupervised / generative scores, no classifier involved
    def alone(label, s):
        s = np.nan_to_num(np.asarray(s, dtype=float), nan=np.nanmedian(s))
        return {"score": label, "pr_auc": float(average_precision_score(yt, s)),
                "pr_auc_ci": ci(boot_metric(average_precision_score, yt, s, idx[:300])),
                "recall_at_budget": float(yt[top_k(s, ev["review_budget"])].sum() / yt.sum())}
    Ct = C.set_index("alert_id").loc[te.alert_id]
    St = S.set_index("alert_id").reindex(te.alert_id)
    standalone = [alone("Report rate of the alert's profile (K-Means)", Ct.cluster_rate.values),
                  alone("Unusualness under the Gaussian mixture", Ct.gmm_density.values),
                  alone("Sequence log-likelihood ratio (HMM)", St.hmm_llr.values),
                  alone("Unusualness vs normal sequences (HMM)", -St.hmm_normal_ll.values),
                  alone("Regime shift in last 30 days (HMM)", St.hmm_regime_shift.values)]
    pd.DataFrame({"alert_id": te.alert_id, **{k: v for k, v in scores.items()}}).to_parquet(art(cfg, "predictions_enriched.parquet"))
    out = {"rows": rows, "standalone": standalone, "base_rate_test": float(yt.mean()), "n_test": len(te),
           "added": {"Profiles": cl_cols, "Sequences": sq_cols}}
    save_json(out, art(cfg, "enrichment.json"))
    top = max(rows, key=lambda r: r["pr_auc"])
    print(f"[10] best combination on test: {top['model']} {top['variant']} PR-AUC {top['pr_auc']:.3f} "
          f"(gain over its baseline {top['gain']:+.3f}, {top['verdict']})")
    return out
