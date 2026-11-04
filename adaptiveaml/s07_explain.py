"""Stage 7 - Explainability: why did an alert get its score?

Global (what the model learned overall)
    - logistic-regression coefficients on standardised features, with bootstrap intervals
    - permutation importance of the champion model on the validation quarters
    - the top of the decision tree as readable rules

Local (why THIS alert)
    For logistic regression the log-odds are a plain sum:
        logit(x) = intercept + sum_j  coef_j * standardised_value_j
    so each feature's contribution to one alert's score is exactly coef_j * value_j. This is a
    true additive explanation with no approximation; SHAP is only needed later for the ensembles.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.tree import export_text

from .common import CLF_COLS, REGISTRY, art, load_features, load_json, p, save_json


def run(cfg: dict) -> dict:
    df = load_features(cfg)
    tr, va = df[df.role == "train"], df[df.role == "valid"]
    held = df[df.role.isin(["valid", "test"])]
    mdir = lambda n: p(cfg, cfg["artifacts_dir"], "models", f"{n}.joblib")
    lr = joblib.load(mdir("Logistic Regression"))
    names = list(lr[0][0].get_feature_names_out())           # column order after preprocessing
    coef, intercept = lr[-1].coef_[0], float(lr[-1].intercept_[0])

    # ---- local: exact per-alert contributions (log-odds units)
    Zs = lr[:-1].transform(held[CLF_COLS])
    contrib = pd.DataFrame(Zs * coef, columns=names)
    contrib.insert(0, "alert_id", held.alert_id.values)
    contrib["intercept"] = intercept
    contrib.to_parquet(art(cfg, "contributions.parquet"))

    # ---- global: coefficient intervals by refitting on bootstrap resamples of the training alerts
    rng = np.random.default_rng(cfg["seed"])
    boots = []
    for _ in range(cfg["evaluation"]["n_coef_bootstrap"]):
        b = tr.iloc[rng.integers(0, len(tr), len(tr))]
        boots.append(clone(lr).fit(b[CLF_COLS], b.y)[-1].coef_[0])
    boots = np.array(boots)
    coefs = sorted(({"feature": f, "label": REGISTRY[f][1], "group": REGISTRY[f][0], "coef": float(c),
                     "lo": float(np.percentile(boots[:, j], 2.5)), "hi": float(np.percentile(boots[:, j], 97.5))}
                    for j, (f, c) in enumerate(zip(names, coef))), key=lambda r: -abs(r["coef"]))
    for r in coefs:
        r["stable"] = bool(r["lo"] > 0 or r["hi"] < 0)       # interval excludes zero

    # ---- global: permutation importance of the champion (drop in PR-AUC when a feature is shuffled)
    champion = load_json(art(cfg, "evaluation.json"))["champion"]
    champ = joblib.load(mdir(champion))
    sample = va.sample(min(len(va), 1000), random_state=cfg["seed"])
    pi = permutation_importance(champ, sample[CLF_COLS], sample.y, scoring="average_precision",
                                n_repeats=3, random_state=cfg["seed"], n_jobs=-1)
    perm = sorted(({"feature": f, "label": REGISTRY[f][1], "importance": float(m), "sd": float(s)}
                   for f, m, s in zip(CLF_COLS, pi.importances_mean, pi.importances_std)), key=lambda r: -r["importance"])

    # ---- global: decision tree as rules
    tree = joblib.load(mdir("Decision Tree"))
    tnames = list(tree[0][0].get_feature_names_out())
    timp = sorted(({"feature": f, "label": REGISTRY[f][1], "importance": float(v)}
                   for f, v in zip(tnames, tree[-1].feature_importances_)), key=lambda r: -r["importance"])[:12]
    rules = export_text(tree[-1], feature_names=[REGISTRY[f][1] for f in tnames], max_depth=3)

    # ---- does the transparent model agree with the champion's ranking?
    P = pd.read_parquet(art(cfg, "predictions.parquet"))
    agree = float(stats.spearmanr(P["raw|Logistic Regression"], P[f"raw|{champion}"])[0])

    out = {"coefs": coefs, "permutation": perm, "permutation_model": champion, "tree_importance": timp,
           "tree_rules": rules, "intercept": intercept, "rank_agreement_with_champion": agree}
    save_json(out, art(cfg, "explain.json"))
    print(f"[7] explanations written | top driver: {coefs[0]['label']} ({coefs[0]['coef']:+.2f}) | "
          f"linear-vs-champion rank agreement {agree:.2f}")
    return out
