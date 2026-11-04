"""Stage 4 - Expected-activity regression and the burst signal.

Question: given how a client behaved in days 30-365 before the alert, how many transactions
SHOULD they have made in the last 30 days? The gap between actual and expected is the burst
signal handed to the classifiers.

    target  y = log(1 + transactions in the last 30 days)      (log tames the heavy tail)
    inputs  X = history-window features only (prefix h_)        (no overlap with the target)

Models
    Linear      ordinary least squares = maximum likelihood under Gaussian noise
    Ridge       least squares + L2 penalty = MAP estimate with a Gaussian prior on the weights
    Lasso       least squares + L1 penalty, drives weak coefficients exactly to zero
    Polynomial  degree-2 feature expansion (with Ridge) to capture curvature and interactions
    Huber       robust loss: quadratic for small errors, linear for large ones (outlier clients)
    Bayesian    Bayesian linear regression: a full predictive distribution (mean AND std)

Two burst features come out of this stage
    burst_resid = actual - expected                      (from the polynomial model)
    burst_z     = (actual - expected) / predictive std   (from the Bayesian model)

Leakage control: for training alerts both are CROSS-FITTED - the model that scores a training
quarter never saw that quarter. Otherwise training residuals would be artificially small and
the classifier would meet a differently distributed feature at test time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler

from .common import COUNT_COLS, HISTORY_COLS, art, forward_folds, load_features, p, save_json


def prep(cols: list[str]) -> Pipeline:
    """Shared preprocessing: log counts -> median-impute -> standardise. Fitted on training data only."""
    logs = [c for c in cols if c in COUNT_COLS]
    ct = ColumnTransformer([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), logs)],
                           remainder="passthrough", verbose_feature_names_out=False)
    return make_pipeline(ct, SimpleImputer(strategy="median"), StandardScaler())


def make(name: str, alpha: float | None = None) -> Pipeline:
    if name == "Linear":
        return make_pipeline(prep(HISTORY_COLS), LinearRegression())
    if name == "Ridge":
        return make_pipeline(prep(HISTORY_COLS), Ridge(alpha=alpha))
    if name == "Lasso":
        return make_pipeline(prep(HISTORY_COLS), Lasso(alpha=alpha, max_iter=20000))
    if name == "Polynomial":
        return make_pipeline(prep(HISTORY_COLS), PolynomialFeatures(2, include_bias=False),
                             StandardScaler(), Ridge(alpha=alpha))
    if name == "Huber":
        return make_pipeline(prep(HISTORY_COLS), HuberRegressor(alpha=alpha, max_iter=2000))
    if name == "Bayesian":
        return make_pipeline(prep(HISTORY_COLS), BayesianRidge())
    raise ValueError(name)


GRID = {"Linear": [None], "Ridge": [0.1, 1, 10, 100, 1000], "Lasso": [0.001, 0.003, 0.01, 0.03, 0.1],
        "Polynomial": [1, 10, 100, 1000, 10000], "Huber": [0.0001, 0.01, 1], "Bayesian": [None]}


def scores(y, yhat) -> dict:
    return {"mae": float(mean_absolute_error(y, yhat)), "rmse": float(np.sqrt(np.mean((y - yhat) ** 2))),
            "r2": float(r2_score(y, yhat))}


def bayes_predict(pipe: Pipeline, X) -> tuple[np.ndarray, np.ndarray]:
    return pipe[-1].predict(pipe[:-1].transform(X), return_std=True)


def run(cfg: dict) -> dict:
    df = load_features(cfg)
    tq = cfg["split"]["train"]
    tr, va, te = (df[df.role == r] for r in ("train", "valid", "test"))
    X = lambda d: d[HISTORY_COLS]
    y = lambda d: np.log1p(d["r_n"].values)
    R, H = cfg["windows"]["recent_days"], cfg["windows"]["history_days"]
    naive = lambda d: np.log1p(d["h_n"].values / (H - R) * R)   # "same daily rate as before"
    folds = list(forward_folds(tr["quarter"], tq))

    # ---- tune the regularisation strength by forward-chaining, then fit on all training quarters
    table, fitted = [], {}
    table.append({"model": "Naive rule", "alpha": None,
                  **{f"{k}_{s}": v for s, d in (("valid", va), ("test", te)) for k, v in scores(y(d), naive(d)).items()}})
    for name, grid in GRID.items():
        best_a, best = grid[0], np.inf
        if len(grid) > 1:
            for a in grid:
                err = np.mean([scores(y(tr.iloc[v]), make(name, a).fit(X(tr.iloc[f]), y(tr.iloc[f]))
                                      .predict(X(tr.iloc[v])))["rmse"] for f, v in folds])
                if err < best:
                    best, best_a = err, a
        m = make(name, best_a).fit(X(tr), y(tr))
        fitted[name] = (m, best_a)
        row = {"model": name, "alpha": best_a}
        for s, d in (("train", tr), ("valid", va), ("test", te)):
            row.update({f"{k}_{s}": v for k, v in scores(y(d), m.predict(X(d))).items()})
        if name == "Lasso":
            row["zero_coefs"] = int((m[-1].coef_ == 0).sum())
        table.append(row)

    # ---- model complexity study 1: polynomial degree (classic over-fitting curve)
    corr = X(tr).fillna(0).apply(lambda c: abs(np.corrcoef(np.log1p(c.clip(lower=0)) if c.name in COUNT_COLS else c, y(tr))[0, 1]))
    top4 = corr.sort_values(ascending=False).index[:4].tolist()
    degree = []
    for dg in (1, 2, 3, 4, 5):
        m = make_pipeline(prep(top4), PolynomialFeatures(dg, include_bias=False), StandardScaler(),
                          LinearRegression()).fit(tr[top4], y(tr))
        degree.append({"degree": dg, "n_terms": int(m[-1].coef_.size),
                       "rmse_train": scores(y(tr), m.predict(tr[top4]))["rmse"],
                       "rmse_valid": scores(y(va), m.predict(va[top4]))["rmse"]})
    # ---- model complexity study 2: regularisation strength lambda on the full degree-2 model
    lam = []
    for a in np.logspace(-2, 5, 15):
        m = make("Polynomial", a).fit(X(tr), y(tr))
        lam.append({"alpha": float(a), "rmse_train": scores(y(tr), m.predict(X(tr)))["rmse"],
                    "rmse_valid": scores(y(va), m.predict(X(va)))["rmse"]})

    # ---- burst signal, cross-fitted on the training quarters
    a_poly = fitted["Polynomial"][1]
    exp_mu, exp_sd, exp_poly = (pd.Series(np.nan, index=df.index) for _ in range(3))
    for q in tq:                                   # each training quarter is scored by a model that never saw it
        fit, hold = tr[tr.quarter != q], tr[tr.quarter == q]
        exp_poly[hold.index] = make("Polynomial", a_poly).fit(X(fit), y(fit)).predict(X(hold))
        mu, sd = bayes_predict(make("Bayesian").fit(X(fit), y(fit)), X(hold))
        exp_mu[hold.index], exp_sd[hold.index] = mu, sd
    for d in (va, te):                             # later quarters are scored by the full-training model
        exp_poly[d.index] = fitted["Polynomial"][0].predict(X(d))
        mu, sd = bayes_predict(fitted["Bayesian"][0], X(d))
        exp_mu[d.index], exp_sd[d.index] = mu, sd
    actual = np.log1p(df["r_n"])
    df["burst_resid"] = actual - exp_poly
    df["burst_z"] = (actual - exp_mu) / exp_sd
    df.to_parquet(p(cfg, cfg["data"]["processed_dir"], "alert_features.parquet"))
    pd.DataFrame({"alert_id": df.alert_id, "actual_log": actual, "expected_log": exp_mu,
                  "expected_sd": exp_sd}).to_parquet(art(cfg, "expected_activity.parquet"))

    lin = fitted["Ridge"][0]
    out = {"table": table, "degree": degree, "lambda": lam, "top4": top4,
           "ridge_coefs": sorted(({"feature": c, "coef": float(w)} for c, w in zip(
               lin[0][0].get_feature_names_out(), lin[-1].coef_)), key=lambda r: -abs(r["coef"]))[:12],
           "burst_by_class": {s: {"resid_report": float(g[g.y == 1].burst_resid.mean()),
                                  "resid_dismiss": float(g[g.y == 0].burst_resid.mean()),
                                  "z_report": float(g[g.y == 1].burst_z.mean()),
                                  "z_dismiss": float(g[g.y == 0].burst_z.mean())}
                              for s, g in (("train", df[df.role == "train"]), ("valid", df[df.role == "valid"]))},
           "scatter": {"actual": actual[va.index].values[:1500], "expected": exp_mu[va.index].values[:1500],
                       "sd": exp_sd[va.index].values[:1500], "y": va.y.values[:1500]}}
    save_json(out, art(cfg, "regression.json"))
    best = min(table[1:], key=lambda r: r["rmse_valid"])
    print(f"[4] regression | best on validation: {best['model']} (RMSE {best['rmse_valid']:.3f}, R2 {best['r2_valid']:.3f}) "
          f"| naive RMSE {table[0]['rmse_valid']:.3f}")
    return out
