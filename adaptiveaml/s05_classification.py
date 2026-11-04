"""Stage 5 - Classification: will this alert be reported or dismissed?

Six classifiers are trained under one identical protocol so the comparison is fair:

    Logistic Regression   discriminative: models P(report | x) directly, linear boundary
    Gaussian Naive Bayes  generative: models P(x | class) with independent Gaussians, then Bayes' rule
    LDA                   generative: Gaussians with a shared covariance (Fisher's discriminant)
    K-Nearest Neighbours  non-parametric: "alerts like this one were reported"
    Decision Tree         rule-like splits an investigator can read
    SVM (RBF kernel)      maximum-margin boundary made non-linear by the kernel trick

Protocol
    1. Preprocessing lives INSIDE each pipeline (log counts -> impute -> scale) and is fitted on
       training data only. Nothing is fitted before the split.
    2. Hyper-parameters are tuned by forward-chaining over the 2020 quarters (never random k-fold:
       that would train on the future and validate on the past).
    3. The tuned model is refitted on all of 2020.
    4. Scores are calibrated on the validation quarters (Platt scaling) so that a risk score of
       0.30 really means "about 30% of such alerts get reported".
    5. The operating threshold is chosen on validation and only then applied to the test quarters.

Imbalance: the default is class weighting (or class priors for the generative models). SMOTE and
"no handling" are trained as comparison arms, not assumed to help.
"""
from __future__ import annotations

import time
import warnings

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .bayes_logreg import LaplaceLogReg
from .common import BEHAVIOUR_COLS, BURST_COLS, CLF_COLS, PROFILE_COLS, art, forward_folds, load_features, p, save_json
from .s04_regression import prep as _prep_cols

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# name -> (estimator factory taking class_weight, hyper-parameter grid, supports class_weight?)
MODELS = {
    "Logistic Regression": (lambda cw, s: LogisticRegression(class_weight=cw, max_iter=3000),
                            {"C": [0.01, 0.1, 1, 10]}, True),
    "Naive Bayes": (lambda cw, s: GaussianNB(), {"var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]}, False),
    "LDA": (lambda cw, s: LinearDiscriminantAnalysis(solver="lsqr"), {"shrinkage": ["auto", 0.1, 0.5, 0.9]}, False),
    "K-Nearest Neighbours": (lambda cw, s: KNeighborsClassifier(weights="distance"),
                             {"n_neighbors": [15, 51, 101, 201]}, False),
    "Decision Tree": (lambda cw, s: DecisionTreeClassifier(class_weight=cw, random_state=s),
                      {"max_depth": [3, 5, 8], "min_samples_leaf": [20, 50, 100]}, True),
    "SVM (RBF)": (lambda cw, s: SVC(kernel="rbf", class_weight=cw, cache_size=1000),
                  {"C": [0.3, 1, 3], "gamma": ["scale", 0.01]}, True),
}

def build(name: str, cols: list[str], params: dict, seed: int, imbalance: str = "weights"):
    """One pipeline template for every model: preprocess -> (optional SMOTE) -> classifier."""
    factory, _, _ = MODELS[name]
    est = factory("balanced" if imbalance == "weights" else None, seed).set_params(**params)
    if imbalance == "smote":   # SMOTE sits inside the pipeline, so it only ever sees the fitting fold
        return ImbPipeline([*_prep_cols(cols).steps, ("smote", SMOTE(random_state=seed)), ("model", est)])
    return make_pipeline(_prep_cols(cols), est)


def raw_score(model, X) -> np.ndarray:
    """A ranking score: predicted probability if available, else distance from the SVM hyperplane."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def platt(valid_score, valid_y):
    """Platt scaling: a 1-D logistic regression mapping raw score -> calibrated probability."""
    cal = LogisticRegression(C=1e6, max_iter=2000).fit(valid_score.reshape(-1, 1), valid_y)
    return lambda s: cal.predict_proba(np.asarray(s).reshape(-1, 1))[:, 1]


def run(cfg: dict) -> dict:
    seed, df = cfg["seed"], load_features(cfg)
    tq = cfg["split"]["train"]
    tr, va = df[df.role == "train"], df[df.role == "valid"]
    held = df[df.role.isin(["valid", "test"])]                 # everything from 2021
    folds = list(forward_folds(tr["quarter"], tq))
    preds = held[["alert_id", "quarter", "role", "y"]].copy()
    info: dict = {"models": {}, "features": CLF_COLS}
    no_burst = [c for c in CLF_COLS if c not in BURST_COLS]

    for name, (_, grid, has_cw) in MODELS.items():
        t0 = time.time()
        # ---- forward-chaining search, scored by PR-AUC on the later quarter
        results = []
        for params in ParameterGrid(grid):
            aps = [average_precision_score(tr.y.iloc[v], raw_score(
                build(name, CLF_COLS, params, seed).fit(tr[CLF_COLS].iloc[f], tr.y.iloc[f]), tr[CLF_COLS].iloc[v]))
                for f, v in folds]
            results.append((float(np.mean(aps)), params))
        cv_ap, best = max(results, key=lambda r: r[0])
        # ---- refit on all training quarters, score 2021, calibrate on validation
        model = build(name, CLF_COLS, best, seed).fit(tr[CLF_COLS], tr.y)
        joblib.dump(model, p(cfg, cfg["artifacts_dir"], "models", f"{name}.joblib"))
        s_held = raw_score(model, held[CLF_COLS])
        cal = platt(raw_score(model, va[CLF_COLS]), va.y.values)
        preds[f"raw|{name}"], preds[f"cal|{name}"] = s_held, cal(s_held)
        # ---- ablation: same model without the burst signal
        preds[f"noburst|{name}"] = raw_score(build(name, no_burst, best, seed).fit(tr[no_burst], tr.y), held[no_burst])
        # ---- ablation: feature families (behavioural only / channel-direction profile only)
        for tag, cols in (("behaviour", BEHAVIOUR_COLS), ("profile", PROFILE_COLS)):
            preds[f"only_{tag}|{name}"] = raw_score(build(name, cols, best, seed).fit(tr[cols], tr.y), held[cols])
        # ---- ablation: imbalance strategy
        arms = (["none", "weights", "smote"] if has_cw else ["none", "smote"])
        for arm in arms:
            if arm == ("weights" if has_cw else "none"):
                preds[f"imb_{arm}|{name}"] = s_held          # this is the default arm, already fitted
            else:
                preds[f"imb_{arm}|{name}"] = raw_score(
                    build(name, CLF_COLS, best, seed, arm).fit(tr[CLF_COLS], tr.y), held[CLF_COLS])
        est = model[-1]
        info["models"][name] = {"best_params": best, "cv_pr_auc": cv_ap, "fit_seconds": round(time.time() - t0, 1),
                                "grid": [{"params": pr, "cv_pr_auc": a} for a, pr in results],
                                "default_imbalance": "class weights" if has_cw else "none (threshold only)"}
        if name == "SVM (RBF)":
            info["models"][name]["n_support_vectors"] = int(est.n_support_.sum())
            info["models"][name]["support_vector_share"] = float(est.n_support_.sum() / len(tr))
        if name == "Decision Tree":
            info["models"][name].update(depth=int(est.get_depth()), leaves=int(est.get_n_leaves()))
        print(f"    {name:<22} cv PR-AUC {cv_ap:.3f}  {best}  ({time.time() - t0:.0f}s)")

    # ---- kernel comparison: the same SVM with a linear kernel (is the non-linearity earning its cost?)
    C = info["models"]["SVM (RBF)"]["best_params"]["C"]
    lin = make_pipeline(_prep_cols(CLF_COLS), SVC(kernel="linear", C=min(C, 1), class_weight="balanced", cache_size=1000))
    preds["ref|SVM (linear kernel)"] = raw_score(lin.fit(tr[CLF_COLS], tr.y), held[CLF_COLS])
    # ---- simplest possible rule: rank alerts by recent transaction count alone
    preds["ref|Rule: recent transaction count"] = held["r_n"].values.astype(float)

    # ---- existing approach, re-implemented: the SynthAML paper's 56 summary features
    #      (trained on 2020 exactly as published; library-default settings)
    bench = pd.read_parquet(p(cfg, cfg["data"]["processed_dir"], "benchmark_features.parquet"))
    bcols = [c for c in bench.columns if c not in ("alert_id", "quarter", "role", "y")]
    btr, bheld = bench[bench.role == "train"], bench[bench.role.isin(["valid", "test"])].set_index("alert_id").loc[preds.alert_id]
    base = lambda est: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)
    for bname, est in {
        "Decision Tree": DecisionTreeClassifier(min_samples_leaf=50, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=seed),
        "Logistic Regression": LogisticRegression(max_iter=3000),
        "SVM": SVC(kernel="rbf", cache_size=1000),
        "Multilayer Perceptron": MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=500, random_state=seed),
    }.items():
        preds[f"bench|{bname}"] = raw_score(base(est).fit(btr[bcols], btr.y), bheld[bcols])
    info["benchmark_n_features"] = len(bcols)

    # ---- uncertainty-aware scoring: Bayesian logistic regression (Laplace approximation)
    pre = _prep_cols(CLF_COLS).fit(tr[CLF_COLS])
    blr = LaplaceLogReg(C=info["models"]["Logistic Regression"]["best_params"]["C"]).fit(
        pre.transform(tr[CLF_COLS]), tr.y.values)
    mu, sd, pr_ = blr.predict(pre.transform(held[CLF_COLS]))
    preds["blr_logit"], preds["blr_sd"], preds["blr_prob"] = mu, sd, pr_

    preds.to_parquet(art(cfg, "predictions.parquet"))
    save_json(info, art(cfg, "classification.json"))
    print(f"[5] trained {len(MODELS)} classifiers + ablations + benchmark re-implementation")
    return info
