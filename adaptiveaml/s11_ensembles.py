"""Stage 11 - Ensembles: many models, combined.

Everything here keeps the PBL-1 protocol: fit on 2020, choose settings by forward-chaining over the
2020 quarters, never touch the 2021 test quarters until the final evaluation (stage 12).

    Random Forest       bagging of CART trees + random feature subsets at each split
    Bagging             bagging alone (plain CART trees on bootstrap samples)
    XGBoost / LightGBM  gradient boosting: each new tree fits the gradient of the log-loss
    AdaBoost            written out by hand (adaboost.py) so the re-weighting is visible,
                        checked against scikit-learn's implementation
    Bagging vs boosting the bias-variance experiment: the same data resampled, each model refitted,
                        its error split into a bias part and a variance part
    GAM                 generalised additive model: logit = sum of one smooth curve per feature
    Voting committee    equal-weight average of the base models
    Stacking            a meta-model that LEARNS how much to trust each base model, trained on
                        OUT-OF-FOLD predictions so it never sees a base model's in-sample score
    BMA                 Bayesian model averaging, to show why it is not the same thing as an ensemble

Outputs: predictions_ensembles.parquet (scores on 2021), ensembles.json (diagnostics),
shap_test.parquet (XGBoost TreeSHAP contributions for every test alert).
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMClassifier
from scipy.special import logit as _logit
from scipy.stats import spearmanr
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .adaboost import AdaBoostStumps
from .common import CLF_COLS, art, forward_folds, label, load_features, load_json, save_json
from .s04_regression import prep
from .s05_classification import build, platt, raw_score

warnings.filterwarnings("ignore")
AP = average_precision_score


def lg(p):
    return _logit(np.clip(p, 1e-6, 1 - 1e-6))


def load_all(cfg):
    """The PBL-1 features plus the PBL-2 profile and sequence features, one row per alert."""
    F = load_features(cfg)
    C = pd.read_parquet(art(cfg, "clusters.parquet"))
    S = pd.read_parquet(art(cfg, "sequence_features.parquet"))
    cl = [c for c in C.columns if c.startswith(("km_", "gmm_p"))] + ["gmm_density", "gmm_entropy"]
    sq = [c for c in S.columns if c.startswith("hmm_") and c != "hmm_n"]
    df = (F.merge(C[["alert_id"] + cl], on="alert_id", how="left")
           .merge(S[["alert_id"] + sq], on="alert_id", how="left"))
    return df, cl, sq


# --------------------------------------------------------------------------- base learners
def make_rf(params, seed, n=500):
    return RandomForestClassifier(n_estimators=n, class_weight="balanced_subsample", n_jobs=-1,
                                  random_state=seed, oob_score=True, **params)


def make_xgb(params, seed, spw):
    return xgb.XGBClassifier(objective="binary:logistic", tree_method="hist", learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.5, min_child_weight=5, reg_lambda=1.0,
                             scale_pos_weight=spw, n_jobs=-1, random_state=seed, verbosity=0, **params)


def run(cfg: dict) -> dict:
    seed, t_all = cfg["seed"], time.time()
    rng = np.random.default_rng(seed)
    df, cl_cols, sq_cols = load_all(cfg)
    tq = cfg["split"]["train"]
    tr = df[df.role == "train"].reset_index(drop=True)
    va = df[df.role == "valid"].reset_index(drop=True)
    held = df[df.role.isin(["valid", "test"])].reset_index(drop=True)
    te_mask = (held.role == "test").values
    yv, ytr = va.y.values, tr.y.values
    folds = list(forward_folds(tr["quarter"], tq))
    spw = float((ytr == 0).sum() / ytr.sum())
    best = load_json(art(cfg, "classification.json"))["models"]
    out: dict = {"n_train": len(tr), "n_valid": len(va), "n_test": int(te_mask.sum())}
    preds = held[["alert_id", "quarter", "role", "y"]].copy()
    timing = {}

    # one shared preprocessing (log counts -> impute -> scale), fitted on 2020 only.
    # Trees do not need it (they are invariant to monotone transforms) but it keeps every model
    # on literally the same inputs.
    pre = prep(CLF_COLS).fit(tr[CLF_COLS])
    Xtr, Xva, Xho = (pre.transform(d[CLF_COLS]) for d in (tr, va, held))

    # ================================================================ 1. Random Forest
    t0 = time.time()
    grid = [{"max_features": mf, "min_samples_leaf": ml} for mf in ["sqrt", 0.3] for ml in [5, 20]]
    res = []
    for g in grid:
        res.append((np.mean([AP(ytr[v], make_rf(g, seed, 300).fit(Xtr[f], ytr[f]).predict_proba(Xtr[v])[:, 1])
                             for f, v in folds]), g))
    rf_cv, rf_p = max(res, key=lambda r: r[0])
    rf = make_rf(rf_p, seed).fit(Xtr, ytr)
    preds["ens|Random Forest"] = rf.predict_proba(Xho)[:, 1]
    timing["Random Forest"] = time.time() - t0
    mdi = pd.Series(rf.feature_importances_, CLF_COLS).sort_values(ascending=False)
    sub = rng.choice(len(va), 2000, replace=False)
    perm = permutation_importance(rf, Xva[sub], yv[sub], scoring="average_precision", n_repeats=3,
                                  random_state=seed, n_jobs=1)
    perm_s = pd.Series(perm.importances_mean, CLF_COLS).sort_values(ascending=False)
    depths = [e.get_depth() for e in rf.estimators_]
    leaves = [e.get_n_leaves() for e in rf.estimators_]
    out["random_forest"] = {
        "params": rf_p, "cv_pr_auc": rf_cv, "grid": [{"params": g, "cv_pr_auc": a} for a, g in res],
        "n_trees": rf.n_estimators, "oob_pr_auc": float(AP(ytr, rf.oob_decision_function_[:, 1])),
        "tree_depth_mean": float(np.mean(depths)), "tree_leaves_mean": float(np.mean(leaves)),
        "base_learner": "CART tree, Gini impurity, bootstrap sample, random feature subset at every split",
        "mdi": [{"feature": f, "label": label(f), "importance": float(v)} for f, v in mdi.head(20).items()],
        "permutation": [{"feature": f, "label": label(f), "importance": float(v),
                         "sd": float(perm.importances_std[CLF_COLS.index(f)])} for f, v in perm_s.head(15).items()],
        "rank_agreement_mdi_vs_permutation": float(spearmanr(mdi[CLF_COLS], perm_s[CLF_COLS]).correlation),
    }
    print(f"    Random Forest   cv {rf_cv:.3f} {rf_p}  oob {out['random_forest']['oob_pr_auc']:.3f} ({time.time()-t0:.0f}s)")

    # ================================================================ 2. Bagging (no feature subsampling)
    t0 = time.time()
    bag = BaggingClassifier(DecisionTreeClassifier(min_samples_leaf=5, class_weight="balanced", random_state=seed),
                            n_estimators=300, oob_score=True, n_jobs=-1, random_state=seed).fit(Xtr, ytr)
    preds["ens|Bagging (CART)"] = bag.predict_proba(Xho)[:, 1]
    timing["Bagging (CART)"] = time.time() - t0
    single = DecisionTreeClassifier(min_samples_leaf=5, class_weight="balanced", random_state=seed).fit(Xtr, ytr)
    preds["ens|Single deep tree"] = single.predict_proba(Xho)[:, 1]
    out["bagging"] = {"n_trees": 300, "oob_pr_auc": float(AP(ytr, np.nan_to_num(bag.oob_decision_function_[:, 1], nan=0.5)))}

    # ================================================================ 3. XGBoost (+ TreeSHAP) and LightGBM
    t0 = time.time()
    res = []
    for depth in (2, 3, 4):
        aps = np.zeros(3)
        for f, v in folds:
            mdl = make_xgb({"max_depth": depth, "n_estimators": 600}, seed, spw).fit(Xtr[f], ytr[f])
            aps += [AP(ytr[v], mdl.predict_proba(Xtr[v], iteration_range=(0, k))[:, 1]) for k in (100, 300, 600)]
        for k, a in zip((100, 300, 600), aps / len(folds)):
            res.append((float(a), {"max_depth": depth, "n_estimators": k}))
    xgb_cv, xgb_p = max(res, key=lambda r: r[0])
    xg = make_xgb(xgb_p, seed, spw).fit(Xtr, ytr)
    preds["ens|XGBoost"] = xg.predict_proba(Xho)[:, 1]
    timing["XGBoost"] = time.time() - t0
    # TreeSHAP: exact Shapley values for tree ensembles, in log-odds units; they add up to the score
    dm = xgb.DMatrix(Xho[te_mask])
    shap = xg.get_booster().predict(dm, pred_contribs=True)
    margin = xg.get_booster().predict(dm, output_margin=True)
    shap_df = pd.DataFrame(shap[:, :-1].astype("float32"), columns=CLF_COLS)
    shap_df.insert(0, "alert_id", held.alert_id[te_mask].values)
    shap_df["bias"] = shap[:, -1]
    shap_df.to_parquet(art(cfg, "shap_test.parquet"))
    mabs = shap_df[CLF_COLS].abs().mean().sort_values(ascending=False)
    Xte_raw = held.loc[te_mask, CLF_COLS].reset_index(drop=True)
    dep_idx = rng.choice(len(shap_df), min(900, len(shap_df)), replace=False)
    out["xgboost"] = {
        "params": xgb_p, "cv_pr_auc": xgb_cv, "grid": [{"params": g, "cv_pr_auc": a} for a, g in res],
        "fixed": {"learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.5, "min_child_weight": 5,
                  "reg_lambda": 1.0, "scale_pos_weight": round(spw, 2)},
        "shap_additivity_max_error": float(np.abs(shap.sum(1) - margin).max()),
        "shap_global": [{"feature": f, "label": label(f), "mean_abs": float(v)} for f, v in mabs.head(20).items()],
        "shap_dependence": {f: {"label": label(f), "x": Xte_raw[f].values[dep_idx].tolist(),
                                "shap": shap_df[f].values[dep_idx].tolist()} for f in mabs.index[:4]},
    }
    print(f"    XGBoost         cv {xgb_cv:.3f} {xgb_p}  SHAP additivity err {out['xgboost']['shap_additivity_max_error']:.1e} ({time.time()-t0:.0f}s)")

    t0 = time.time()
    lgb = LGBMClassifier(n_estimators=xgb_p["n_estimators"], learning_rate=0.05, num_leaves=2 ** xgb_p["max_depth"],
                         max_depth=xgb_p["max_depth"], subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                         min_child_samples=20, scale_pos_weight=spw, random_state=seed, verbose=-1, n_jobs=-1).fit(Xtr, ytr)
    preds["ens|LightGBM"] = lgb.predict_proba(Xho)[:, 1]
    timing["LightGBM"] = time.time() - t0
    out["lightgbm"] = {"params": {"num_leaves": 2 ** xgb_p["max_depth"], "n_estimators": xgb_p["n_estimators"]},
                       "note": "leaf-wise growth, histogram splits; settings mirrored from the tuned XGBoost"}

    # ================================================================ 4. AdaBoost, by hand
    t0 = time.time()
    M = 300
    ada = AdaBoostStumps(M, seed).fit(Xtr, ytr, watch={"valid": Xva})
    preds["ens|AdaBoost (from scratch)"] = ada.decision_function(Xho)
    timing["AdaBoost (from scratch)"] = time.time() - t0
    sk = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, random_state=seed), n_estimators=M,
                            learning_rate=1.0, random_state=seed).fit(Xtr, ytr)
    sk_s = sk.decision_function(Xho)
    ours = preds["ens|AdaBoost (from scratch)"].values
    trace = ada.trace
    rounds_ap = [{"round": m, "valid_pr_auc": float(AP(yv, ada.watch_path["valid"][m - 1]))}
                 for m in sorted(set([1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 250, len(trace)]))
                 if m <= len(trace)]
    snaps = {}
    for m, w in ada.weight_snapshots.items():
        lw = np.log10(w * len(w))
        h, e = np.histogram(lw, bins=30, range=(-2, 2))
        snaps[str(m)] = {"edges": e.tolist(), "counts_reported": np.histogram(lw[ytr == 1], bins=e)[0].tolist(),
                         "counts_dismissed": np.histogram(lw[ytr == 0], bins=e)[0].tolist()}
    feat_counts = pd.Series([CLF_COLS[r["feature"]] for r in trace]).value_counts()
    hard = np.argsort(-ada.final_weights)[: int(0.05 * len(ytr))]
    out["adaboost"] = {
        "rounds": len(trace), "trace": trace, "valid_by_round": rounds_ap, "weight_snapshots": snaps,
        "stump_features": [{"feature": f, "label": label(f), "rounds": int(c)} for f, c in feat_counts.head(12).items()],
        "hardest_5pct_reported_share": float(ytr[hard].mean()), "base_rate_train": float(ytr.mean()),
        "check_vs_sklearn": {"max_abs_diff": float(np.abs(ours / ours.std() - sk_s / sk_s.std()).max()),
                             "rank_corr": float(spearmanr(ours, sk_s).correlation),
                             "sklearn_test_pr_auc": float(AP(held.y.values[te_mask], sk_s[te_mask])),
                             "ours_test_pr_auc": float(AP(held.y.values[te_mask], ours[te_mask]))},
    }
    preds["ens|AdaBoost (scikit-learn)"] = sk_s
    print(f"    AdaBoost        {len(trace)} rounds, rank corr vs sklearn {out['adaboost']['check_vs_sklearn']['rank_corr']:.4f} ({time.time()-t0:.0f}s)")

    # ================================================================ 5. More trees: bagging vs boosting
    t0 = time.time()
    ks = [1, 2, 5, 10, 20, 50, 100, 200, 300]
    rf_trees = np.stack([e.predict_proba(Xva)[:, 1] for e in rf.estimators_[:300]])
    bag_trees = np.stack([e.predict_proba(Xva)[:, 1] for e in bag.estimators_[:300]])
    xg600 = make_xgb({"max_depth": xgb_p["max_depth"], "n_estimators": 600}, seed, spw).fit(Xtr, ytr)
    curve = []
    for k in ks + [400, 600]:
        row = {"n": k}
        if k <= 300:
            row["Random Forest"] = float(AP(yv, rf_trees[:k].mean(0)))
            row["Bagging (CART)"] = float(AP(yv, bag_trees[:k].mean(0)))
            row["AdaBoost"] = float(AP(yv, ada.watch_path["valid"][min(k, len(trace)) - 1]))
        row["XGBoost"] = float(AP(yv, xg600.predict_proba(Xva, iteration_range=(0, k))[:, 1]))
        curve.append(row)
    out["n_estimators_curve"] = curve

    # bias-variance: refit each model on bootstrap resamples of 2020; split squared error on validation
    R, sub_v = 12, rng.choice(len(va), 2500, replace=False)
    Xv_s, yv_s = Xva[sub_v], yv[sub_v]
    makers = {
        "Single deep tree": lambda s: DecisionTreeClassifier(min_samples_leaf=5, random_state=s),
        "Bagging (CART)": lambda s: BaggingClassifier(DecisionTreeClassifier(min_samples_leaf=5), n_estimators=60, n_jobs=-1, random_state=s),
        "Random Forest": lambda s: RandomForestClassifier(n_estimators=100, min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=s),
        "Single stump": lambda s: DecisionTreeClassifier(max_depth=1, random_state=s),
        "AdaBoost": lambda s: AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=150, random_state=s),
        "XGBoost": lambda s: make_xgb({"max_depth": xgb_p["max_depth"], "n_estimators": xgb_p["n_estimators"]}, s, 1.0),
    }
    P = {k: [] for k in makers}
    for r in range(R):
        bi = rng.integers(0, len(ytr), len(ytr))
        for k, mk in makers.items():
            m_ = mk(seed + r).fit(Xtr[bi], ytr[bi])
            if k == "AdaBoost":     # SAMME probabilities are squashed toward 0.5; calibrate on the bag itself
                d = m_.decision_function(Xtr[bi])
                c = LogisticRegression(C=1e6).fit(d.reshape(-1, 1), ytr[bi])
                P[k].append(c.predict_proba(m_.decision_function(Xv_s).reshape(-1, 1))[:, 1])
            else:
                P[k].append(m_.predict_proba(Xv_s)[:, 1])
    bv = []
    for k, arr in P.items():
        A = np.stack(arr)
        fbar = A.mean(0)
        bv.append({"model": k, "family": "bagging" if k in ("Bagging (CART)", "Random Forest") else
                   "boosting" if k in ("AdaBoost", "XGBoost") else "single",
                   "bias2_plus_noise": float(np.mean((fbar - yv_s) ** 2)), "variance": float(np.mean(A.var(0))),
                   "total": float(np.mean((A - yv_s) ** 2)), "pr_auc_mean": float(np.mean([AP(yv_s, a) for a in A])),
                   "pr_auc_sd": float(np.std([AP(yv_s, a) for a in A]))})
    out["bias_variance"] = {"replicates": R, "n_eval": len(sub_v), "rows": bv,
                            "note": "squared error of predicted probability = (bias^2 + noise) + variance; "
                                    "the noise part (irreducible) is the same for every model"}
    print(f"    bias-variance experiment {R} replicates ({time.time()-t0:.0f}s)")

    # ================================================================ 6. GAM: one smooth curve per feature
    t0 = time.time()
    n_spl = 6 + 3 - 1                               # n_knots + degree - 1 basis functions per feature
    def gam(C):
        return make_pipeline(prep(CLF_COLS), SplineTransformer(n_knots=6, degree=3, extrapolation="constant"),
                             LogisticRegression(C=C, class_weight="balanced", max_iter=4000))
    res = [(np.mean([AP(ytr[v], gam(C).fit(tr[CLF_COLS].iloc[f], ytr[f]).predict_proba(tr[CLF_COLS].iloc[v])[:, 1])
                     for f, v in folds]), C) for C in (0.001, 0.003, 0.01, 0.03)]
    gam_cv, gam_C = max(res)
    G = gam(gam_C).fit(tr[CLF_COLS], ytr)
    preds["ens|GAM (spline)"] = G.predict_proba(held[CLF_COLS])[:, 1]
    timing["GAM (spline)"] = time.time() - t0
    coef = G[-1].coef_[0]
    med = tr[CLF_COLS].median()
    shapes, strength = {}, {}
    basis_tr = G[:-1].transform(tr[CLF_COLS])
    for j, f in enumerate(CLF_COLS):
        cj = coef[j * n_spl:(j + 1) * n_spl]
        contrib_tr = basis_tr[:, j * n_spl:(j + 1) * n_spl] @ cj
        strength[f] = float(contrib_tr.std())
        qs = np.unique(tr[f].quantile(np.linspace(0.01, 0.99, 40)).values)
        grid_df = pd.DataFrame([med.values] * len(qs), columns=CLF_COLS)
        grid_df[f] = qs
        fx = G[:-1].transform(grid_df)[:, j * n_spl:(j + 1) * n_spl] @ cj - contrib_tr.mean()
        shapes[f] = {"x": qs.tolist(), "f": fx.tolist(), "label": label(f)}
    top_g = sorted(strength, key=strength.get, reverse=True)[:12]
    out["gam"] = {"C": gam_C, "cv_pr_auc": gam_cv, "knots": 6, "degree": 3, "basis_per_feature": n_spl,
                  "n_coefficients": int(coef.size),
                  "strength": [{"feature": f, "label": label(f), "sd_logit": strength[f]} for f in top_g],
                  "shapes": {f: shapes[f] for f in top_g}}
    print(f"    GAM             cv {gam_cv:.3f} C={gam_C} ({time.time()-t0:.0f}s)")

    # ================================================================ 7. PBL-1 and PBL-2 base models, refitted
    t0 = time.time()
    pbl1 = ["Logistic Regression", "LDA", "SVM (RBF)"]
    full_models = {}
    for name in pbl1:
        m_ = build(name, CLF_COLS, best[name]["best_params"], seed).fit(tr[CLF_COLS], ytr)
        full_models[name] = m_
        preds[f"ens|{name}"] = raw_score(m_, held[CLF_COLS])
    # PBL-2 "pattern model": what the profiles and sequences alone say (logistic regression on them)
    pat_cols = cl_cols + sq_cols
    pattern = build("Logistic Regression", pat_cols, {"C": 0.1}, seed)
    preds["ens|Pattern model (profiles + sequences)"] = pattern.fit(tr[pat_cols], ytr).predict_proba(held[pat_cols])[:, 1]
    preds["ens|HMM suspicion score"] = held.hmm_llr.fillna(0).values
    # PBL-2's best enriched configuration, for the phase comparison
    both = CLF_COLS + pat_cols
    preds["ens|SVM + profiles + sequences"] = raw_score(
        build("SVM (RBF)", both, best["SVM (RBF)"]["best_params"], seed).fit(tr[both], ytr), held[both])
    preds["ens|XGBoost + profiles + sequences"] = make_xgb(xgb_p, seed, spw).fit(
        prep(both).fit(tr[both]).transform(tr[both]), ytr).predict_proba(prep(both).fit(tr[both]).transform(held[both]))[:, 1]
    print(f"    PBL-1/PBL-2 base models refitted ({time.time()-t0:.0f}s)")

    # ================================================================ 8. Stacking with out-of-fold predictions
    # Level 0: for each 2020 quarter q >= Q2, fit every base model on the quarters BEFORE q and
    # predict q. These out-of-fold scores are what the meta-model learns from: it sees how each
    # base model behaves on alerts it was not trained on, exactly as at test time.
    t0 = time.time()
    def to_meta(name, s):                       # put every base score on a log-odds-like scale
        return s if name in ("SVM (RBF)", "HMM suspicion score") else lg(s)

    def base_fit_predict(name, fit_df, pred_df):
        if name in pbl1:
            return raw_score(build(name, CLF_COLS, best[name]["best_params"], seed).fit(fit_df[CLF_COLS], fit_df.y), pred_df[CLF_COLS])
        if name == "Pattern model (profiles + sequences)":
            return build("Logistic Regression", pat_cols, {"C": 0.1}, seed).fit(fit_df[pat_cols], fit_df.y).predict_proba(pred_df[pat_cols])[:, 1]
        if name == "HMM suspicion score":
            return pred_df.hmm_llr.fillna(0).values
        pp = prep(CLF_COLS).fit(fit_df[CLF_COLS])
        Xf, Xp = pp.transform(fit_df[CLF_COLS]), pp.transform(pred_df[CLF_COLS])
        if name == "Random Forest":
            return make_rf(rf_p, seed, 300).fit(Xf, fit_df.y).predict_proba(Xp)[:, 1]
        if name == "XGBoost":
            s = float((fit_df.y == 0).sum() / fit_df.y.sum())
            return make_xgb(xgb_p, seed, s).fit(Xf, fit_df.y).predict_proba(Xp)[:, 1]
        raise ValueError(name)

    base_names = pbl1 + ["Pattern model (profiles + sequences)", "HMM suspicion score", "Random Forest", "XGBoost"]
    phase_of = {**{n: "PBL-1" for n in pbl1}, "Pattern model (profiles + sequences)": "PBL-2",
                "HMM suspicion score": "PBL-2", "Random Forest": "PBL-3", "XGBoost": "PBL-3"}
    oof_rows = []
    for f, v in folds:
        fit_df, pred_df = tr.iloc[f], tr.iloc[v]
        blk = pd.DataFrame({"quarter": pred_df.quarter.values, "y": pred_df.y.values})
        for n in base_names:
            blk[n] = to_meta(n, base_fit_predict(n, fit_df, pred_df))
        oof_rows.append(blk)
    OOF = pd.concat(oof_rows, ignore_index=True)
    HELD = pd.DataFrame({n: to_meta(n, preds[f"ens|{n}"].values) for n in base_names})
    out["stacking_oof"] = {"rows": len(OOF), "quarters": sorted(OOF.quarter.unique().tolist()),
                           "base_oof_pr_auc": {n: float(AP(OOF.y, OOF[n])) for n in base_names}}

    def meta_model(C):
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))

    def fit_stack(cols):
        # meta regularisation chosen by forward-chaining on the OOF quarters themselves
        qs = sorted(OOF.quarter.unique())
        cands = []
        for C in (0.01, 0.1, 1.0):
            sc = [AP(OOF.y[OOF.quarter == qs[i]], meta_model(C).fit(OOF.loc[OOF.quarter.isin(qs[:i]), cols], OOF.y[OOF.quarter.isin(qs[:i])])
                     .predict_proba(OOF.loc[OOF.quarter == qs[i], cols])[:, 1]) for i in range(1, len(qs))]
            cands.append((float(np.mean(sc)), C))
        cv, C = max(cands)
        mm = meta_model(C).fit(OOF[cols], OOF.y)
        return mm, C, cv

    stacks = {"Stack: PBL-1 models": pbl1,
              "Stack: PBL-1 + PBL-2": pbl1 + ["Pattern model (profiles + sequences)", "HMM suspicion score"],
              "Stack: all phases": base_names}
    out["stacking"] = {"base_models": [{"name": n, "phase": phase_of[n]} for n in base_names], "variants": {}}
    for sname, cols in stacks.items():
        mm, C, cv = fit_stack(cols)
        preds[f"ens|{sname}"] = mm.predict_proba(HELD[cols])[:, 1]
        w = mm[-1].coef_[0]
        boot_w = []
        for _ in range(200):
            bi = rng.integers(0, len(OOF), len(OOF))
            boot_w.append(meta_model(C).fit(OOF.loc[bi, cols], OOF.y.values[bi])[-1].coef_[0])
        boot_w = np.array(boot_w)
        out["stacking"]["variants"][sname] = {
            "C": C, "meta_cv_pr_auc": cv,
            "weights": [{"model": n, "phase": phase_of[n], "weight": float(w[i]),
                         "ci": [float(np.percentile(boot_w[:, i], 2.5)), float(np.percentile(boot_w[:, i], 97.5))]}
                        for i, n in enumerate(cols)]}
    # leave-one-out: how much does the full stack lose without each base model?
    full_cols = stacks["Stack: all phases"]
    loo = []
    yva_h = held.y.values[~te_mask]
    for n in full_cols:
        cols = [c for c in full_cols if c != n]
        mm, _, _ = fit_stack(cols)
        s = mm.predict_proba(HELD[cols])[:, 1]
        loo.append({"removed": n, "phase": phase_of[n], "valid_pr_auc": float(AP(yva_h, s[~te_mask])),
                    "test_scores_key": f"stackloo|{n}"})
        preds[f"stackloo|{n}"] = s
    out["stacking"]["leave_one_out"] = loo
    # diversity: base models that disagree are what a committee can exploit
    corr = HELD.loc[~te_mask, base_names].rank().corr().values
    out["stacking"]["diversity"] = {"models": base_names, "spearman_valid": corr.tolist()}
    print(f"    Stacking        OOF rows {len(OOF)} ({time.time()-t0:.0f}s)")

    # ================================================================ 9. Voting committee (equal weights)
    # every base score z-scored with its own 2020 mean/sd (no labels involved), then averaged
    vote_cols = pbl1 + ["Random Forest", "XGBoost"]
    trs = {"Logistic Regression": lg(full_models["Logistic Regression"].predict_proba(tr[CLF_COLS])[:, 1]),
           "LDA": lg(full_models["LDA"].predict_proba(tr[CLF_COLS])[:, 1]),
           "SVM (RBF)": full_models["SVM (RBF)"].decision_function(tr[CLF_COLS]),
           "Random Forest": lg(np.nan_to_num(rf.oob_decision_function_[:, 1], nan=float(ytr.mean()))), "XGBoost": lg(xg.predict_proba(Xtr)[:, 1])}
    preds["ens|Voting committee"] = np.mean([(HELD[n] - trs[n].mean()) / trs[n].std() for n in vote_cols], axis=0)

    # ================================================================ 10. Bayesian model averaging
    # p(M_k | D) is proportional to p(D | M_k) p(M_k). With equal priors, the weight of each model is
    # its predictive likelihood on held-out data: calibrate on 2021Q1, score the likelihood on 2021Q2.
    t0 = time.time()
    bma_models = ["Logistic Regression", "LDA", "SVM (RBF)", "Random Forest", "XGBoost", "GAM (spline)"]
    q1 = (held.quarter == cfg["split"]["valid"][0]).values
    q2 = (held.quarter == cfg["split"]["valid"][1]).values
    yq2 = held.y.values[q2]
    calp = {}
    for n in bma_models:
        s = preds[f"ens|{n}"].values
        calp[n] = platt(s[q1], held.y.values[q1])(s)
    ll = {n: float(np.sum(yq2 * np.log(calp[n][q2]) + (1 - yq2) * np.log(1 - calp[n][q2]))) for n in bma_models}
    def post(lls):
        a = np.array(lls) - max(lls)
        return np.exp(a) / np.exp(a).sum()
    wts = post([ll[n] for n in bma_models])
    preds["ens|Bayesian model averaging"] = sum(w * calp[n] for w, n in zip(wts, bma_models))
    # how the posterior concentrates as evidence accumulates (averaged over 30 random orderings)
    order_curves = []
    ns = [10, 25, 50, 100, 200, 400, 800, 1600, int(q2.sum())]
    for _ in range(30):
        perm_ = rng.permutation(int(q2.sum()))
        rows_ = []
        for n_ in ns:
            ii = perm_[:n_]
            rows_.append(post([float(np.sum(yq2[ii] * np.log(calp[n][q2][ii]) + (1 - yq2[ii]) * np.log(1 - calp[n][q2][ii])))
                               for n in bma_models]))
        order_curves.append(rows_)
    oc = np.array(order_curves).mean(0)
    out["bma"] = {"models": bma_models, "log_likelihood": ll, "weights": dict(zip(bma_models, wts.tolist())),
                  "evidence_curve": [{"n": n_, **{m: float(oc[i, j]) for j, m in enumerate(bma_models)}} for i, n_ in enumerate(ns)],
                  "n_evidence": int(q2.sum())}
    print(f"    BMA weights " + ", ".join(f"{n} {w:.2f}" for n, w in zip(bma_models, wts)) + f" ({time.time()-t0:.0f}s)")

    # ================================================================ calibrated versions + practical cost
    cal_cols = [c for c in preds.columns if c.startswith("ens|")]
    va_mask = ~te_mask
    for c in cal_cols:
        s = preds[c].values
        preds["encal|" + c[4:]] = platt(s[va_mask], held.y.values[va_mask])(s)
    t_inf = {}
    X1k = Xho[:1000]
    for n, m_ in {"Random Forest": rf, "XGBoost": xg, "LightGBM": lgb, "Bagging (CART)": bag}.items():
        s0 = time.time(); m_.predict_proba(X1k); t_inf[n] = (time.time() - s0)
    for n in pbl1:
        s0 = time.time(); raw_score(full_models[n], held[CLF_COLS].iloc[:1000]); t_inf[n] = time.time() - s0
    out["cost"] = {"fit_seconds": {k: round(v, 1) for k, v in timing.items()},
                   "predict_seconds_per_1000": {k: round(v, 3) for k, v in t_inf.items()}}
    preds.to_parquet(art(cfg, "predictions_ensembles.parquet"))
    save_json(out, art(cfg, "ensembles.json"))
    print(f"[11] ensembles trained in {time.time() - t_all:.0f}s")
    return out
