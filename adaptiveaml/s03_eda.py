"""Stage 3 - Exploratory data analysis and statistical tests.

Every analysis here uses TRAINING quarters only. Looking at later quarters while deciding what
to build would leak the future through the analyst, even if no model ever saw the data.

Each block answers one question and its result drives a modelling decision:

  class balance          -> why accuracy is useless and PR-AUC is the headline metric
  Mann-Whitney U         -> which features separate reported from dismissed alerts
  chi-square, Cramer's V -> does the dominant transaction channel relate to the outcome
  Bayes' rule            -> how far one piece of evidence moves the prior P(report)
  Gaussian MLE           -> are features Gaussian within a class (Naive Bayes / LDA assumption)
  Spearman correlation   -> which features are collinear (motivates Ridge regularisation)
  Fisher discriminant    -> how linearly separable the two classes are
  two-sample KS          -> have feature distributions drifted between train and validation
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .common import BEHAVIOUR_COLS, BURST_COLS, REGISTRY, art, load_features, save_json


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (better than the normal interval for small n)."""
    if n == 0:
        return (0.0, 0.0)
    ph, den = k / n, 1 + z * z / n
    centre, half = ph + z * z / (2 * n), z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return ((centre - half) / den, (centre + half) / den)


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg correction: with ~35 tests some would look significant by chance."""
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.minimum.accumulate((pvals[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out


def run(cfg: dict) -> dict:
    df = load_features(cfg)
    feats = [c for c in REGISTRY if c not in BURST_COLS and c != "no_history"]
    tr, va = df[df.role == "train"], df[df.role == "valid"]
    rep, dis = tr[tr.y == 1], tr[tr.y == 0]
    prior = float(tr.y.mean())
    out: dict = {"prior": prior, "n_train": len(tr)}

    # ---- class balance per training quarter, with Wilson intervals
    out["balance"] = [{"quarter": q, "alerts": len(g), "reported": int(g.y.sum()), "rate": float(g.y.mean()),
                       "lo": wilson(int(g.y.sum()), len(g))[0], "hi": wilson(int(g.y.sum()), len(g))[1]}
                      for q, g in tr.groupby("quarter")]
    # the accuracy trap: a model that always answers "dismiss"
    out["accuracy_trap"] = {"accuracy": 1 - prior, "recall": 0.0, "pr_auc": prior, "roc_auc": 0.5}

    # ---- univariate signal: Mann-Whitney U with rank-biserial effect size
    rows = []
    for c in feats:
        a, b = rep[c].dropna(), dis[c].dropna()
        u, pv = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append({"feature": c, "label": REGISTRY[c][1], "group": REGISTRY[c][0],
                     "median_report": float(a.median()), "median_dismiss": float(b.median()),
                     "effect": float(2 * u / (len(a) * len(b)) - 1), "p": float(pv)})
    mw = pd.DataFrame(rows)
    mw["p_adj"] = bh_adjust(mw.p.values)
    mw = mw.reindex(mw.effect.abs().sort_values(ascending=False).index)
    out["mannwhitney"] = mw.to_dict("records")
    top = mw.feature.head(8).tolist()

    # ---- class-conditional distributions of the strongest features (shared bins)
    out["distributions"] = {}
    for c in top:
        lo, hi = np.nanpercentile(tr[c], [1, 99])
        bins = np.linspace(lo, hi, 31)
        out["distributions"][c] = {
            "bins": bins,
            "report": np.histogram(rep[c].dropna(), bins=bins, density=True)[0],
            "dismiss": np.histogram(dis[c].dropna(), bins=bins, density=True)[0]}

    # ---- dominant channel vs outcome: chi-square test of independence and Cramer's V
    share_cols = [c for c in feats if c.startswith("h_share_")]
    dom = tr[share_cols].fillna(0).idxmax(axis=1).str.replace("h_share_", "")
    ct = pd.crosstab(dom, tr.y)
    chi2, pv, dof, _ = stats.chi2_contingency(ct)
    out["channel"] = {"chi2": float(chi2), "p": float(pv), "dof": int(dof),
                      "cramers_v": float(np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))),
                      "table": [{"channel": k, "alerts": int(r.sum()), "rate": float(r.get(1, 0) / r.sum())}
                                for k, r in ct.iterrows()]}

    # ---- Bayes' rule: P(report | feature in its top decile) vs the prior
    #      posterior = likelihood x prior / evidence, estimated by counting
    bay = []
    for c in feats:
        hi_mask = tr[c] >= tr[c].quantile(0.9)
        lik = float(hi_mask[tr.y == 1].mean())          # P(top decile | report)
        evid = float(hi_mask.mean())                    # P(top decile)
        bay.append({"feature": c, "label": REGISTRY[c][1], "likelihood": lik, "evidence": evid,
                    "posterior": lik * prior / evid if evid > 0 else prior})
    out["bayes"] = sorted(bay, key=lambda r: -r["posterior"])

    # ---- Gaussian maximum-likelihood fits per class (mu = sample mean, sigma = sample std)
    out["gaussian"] = {}
    for c in top[:4]:
        entry = {}
        for name, g in (("report", rep), ("dismiss", dis)):
            x = g[c].dropna().values
            mu, sd = float(x.mean()), float(x.std())
            entry[name] = {"mu": mu, "sigma": sd, "skew": float(stats.skew(x)),
                           "kurtosis": float(stats.kurtosis(x)),
                           "loglik_per_obs": float(stats.norm.logpdf(x, mu, sd).mean())}
        out["gaussian"][c] = entry

    # ---- correlation structure (Spearman: robust to the heavy tails)
    #      shown for the behavioural features; the 48 profile features would make the map unreadable
    cf = [c for c in feats if c in BEHAVIOUR_COLS]
    corr = tr[cf].corr(method="spearman").fillna(0)
    out["corr"] = {"features": cf, "matrix": corr.values.round(3)}
    pairs = [(cf[i], cf[j], float(corr.values[i, j])) for i in range(len(cf)) for j in range(i)]
    out["corr_top"] = [{"a": a, "b": b, "rho": r} for a, b, r in sorted(pairs, key=lambda t: -abs(t[2]))[:10]]

    # ---- Fisher's linear discriminant: the 1-D projection that best separates the classes
    X = tr[feats].fillna(tr[feats].median())
    X = (X - X.mean()) / X.std().replace(0, 1)
    z = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, tr.y).decision_function(X)
    z1, z0 = z[tr.y.values == 1], z[tr.y.values == 0]
    bins = np.linspace(np.percentile(z, 0.5), np.percentile(z, 99.5), 41)
    out["fisher"] = {"bins": bins, "report": np.histogram(z1, bins=bins, density=True)[0],
                     "dismiss": np.histogram(z0, bins=bins, density=True)[0],
                     # Fisher criterion: between-class separation over within-class spread
                     "criterion": float((z1.mean() - z0.mean()) ** 2 / (z1.var() + z0.var()))}

    # ---- drift: two-sample Kolmogorov-Smirnov, training vs validation quarters
    ks = [{"feature": c, "label": REGISTRY[c][1], **dict(zip(("stat", "p"), map(float, stats.ks_2samp(
        tr[c].dropna(), va[c].dropna()))))} for c in feats]
    ksd = pd.DataFrame(ks)
    ksd["p_adj"] = bh_adjust(ksd.p.values)
    out["drift"] = ksd.sort_values("stat", ascending=False).to_dict("records")

    save_json(out, art(cfg, "eda.json"))
    print(f"[3] EDA on {len(tr):,} training alerts | prior {prior:.3f} | strongest feature: {top[0]} "
          f"(effect {mw.effect.iloc[0]:+.2f}) | Fisher criterion {out['fisher']['criterion']:.2f}")
    return out
