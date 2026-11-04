"""Stage 8 - Behavioural profiles: dimensionality reduction and clustering.

Stages 1-7 scored each alert on its own. This stage asks a different question without using the
label at all: which groups of alerts behave alike, and do some groups concentrate the reported
alerts? Everything is FITTED ON THE 2020 TRAINING QUARTERS ONLY and then applied, unchanged, to
2021 - the same time boundary as the classifiers.

    1. Standardise the 84 engineered features (same preprocessing as the classifiers).
    2. PCA       eigen-decomposition of the covariance matrix (checked against the SVD); scree plot,
                 cumulative explained variance; keep enough components for 80% of the variance.
       Factor analysis   a few latent factors with readable loadings; varimax rotation shows the
                 unidentifiability of FA (different loadings, identical likelihood).
       Choosing the number of latent dimensions   held-out log-likelihood of PPCA and FA.
    3. K-Means   hard assignment, Euclidean distance; elbow (inertia) and silhouette for K.
    4. GMM / EM  soft assignment; BIC to choose K; the log-likelihood rises at every EM cycle;
                 low density under the mixture = an unusual alert.
    5. Hierarchical clustering on a sample: single / complete / average / Ward linkage under
                 Euclidean, Manhattan and cosine dissimilarity; dendrogram.
    6. Evaluation   silhouette, Davies-Bouldin, purity, NMI and adjusted Rand index against the
                 Report / Dismiss label (used only for evaluation, never for fitting).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture

from .common import BURST_COLS, CLF_COLS, REGISTRY, art, label, load_features, save_json
from .s03_eda import wilson
from .s04_regression import prep

FEATS = [c for c in CLF_COLS if c not in BURST_COLS]          # the 84 engineered features


def purity(labels: np.ndarray, y: np.ndarray) -> float:
    """Share of alerts that belong to their cluster's majority class (0..1, higher = purer)."""
    return float(sum(max((y[labels == c] == 1).sum(), (y[labels == c] == 0).sum()) for c in np.unique(labels)) / len(y))


def quality(X: np.ndarray, labels: np.ndarray, y: np.ndarray, rng, n: int = 4000) -> dict:
    s = rng.choice(len(X), min(n, len(X)), replace=False)
    ok = len(np.unique(labels)) > 1
    return {"silhouette": float(silhouette_score(X[s], labels[s])) if ok else None,
            "davies_bouldin": float(davies_bouldin_score(X, labels)) if ok else None,
            "purity": purity(labels, y), "nmi": float(normalized_mutual_info_score(y, labels)),
            "ari": float(adjusted_rand_score(y, labels))}


def describe(Z: pd.DataFrame, labels: np.ndarray, y: np.ndarray, top: int = 6) -> list[dict]:
    """Profile each cluster: size, report rate (Wilson interval) and the features that set it apart."""
    out = []
    for c in np.unique(labels):
        m = labels == c
        z = Z[m].mean()
        key = z.abs().sort_values(ascending=False).index[:top]
        lo, hi = wilson(int(y[m].sum()), int(m.sum()))
        up = [f for f in key if z[f] > 0][:2]
        down = [f for f in key if z[f] < 0][:1]
        name = " · ".join([f"high {label(f).lower()}" for f in up] + [f"low {label(f).lower()}" for f in down])
        out.append({"cluster": int(c), "size": int(m.sum()), "share": float(m.mean()), "report_rate": float(y[m].mean()),
                    "lo": lo, "hi": hi, "name": name[:90],
                    "signature": [{"feature": f, "label": label(f), "z": float(z[f])} for f in key]})
    return out


def run(cfg: dict) -> dict:
    seed = cfg["seed"]
    rng = np.random.default_rng(seed)
    df = load_features(cfg)
    df = df[df.role.isin(["train", "valid", "test"])].reset_index(drop=True)
    tr = (df.role == "train").values
    y = df.y.values

    # ---- 1. standardise (fitted on training alerts only)
    pre = prep(FEATS).fit(df.loc[tr, FEATS])
    X = pre.transform(df[FEATS])
    Z = pd.DataFrame(X, columns=pre[0].get_feature_names_out())          # z-scores, used for profiles
    Xtr = X[tr]

    # ---- 2a. PCA by eigen-decomposition of the covariance matrix, verified against the SVD
    cov = np.cov(Xtr, rowvar=False)
    eigval, eigvec = np.linalg.eigh(cov)
    eigval, eigvec = eigval[::-1], eigvec[:, ::-1]
    pca_full = PCA(svd_solver="full").fit(Xtr)                            # SVD route
    svd_agrees = bool(np.allclose(eigval[:20], pca_full.explained_variance_[:20], rtol=1e-6))
    ratio = pca_full.explained_variance_ratio_
    d = int(np.searchsorted(np.cumsum(ratio), 0.80) + 1)                # components for 80% of variance
    pca = PCA(n_components=d, random_state=seed).fit(Xtr)
    P = pca.transform(X)
    P3 = PCA(n_components=3, random_state=seed).fit(Xtr).transform(X)    # for the 2-D / 3-D maps
    loadings = [{"component": i + 1, "variance": float(ratio[i]),
                 "top": [{"label": label(Z.columns[j]), "weight": float(pca_full.components_[i, j])}
                         for j in np.argsort(-np.abs(pca_full.components_[i]))[:6]]} for i in range(5)]

    # ---- 2b. choosing the number of latent dimensions: held-out log-likelihood (PPCA and FA)
    Xva = X[(df.role == "valid").values]
    dims = [1, 2, 3, 4, 6, 8, 10, 12, 15, 20]
    dim_sel = []
    for k in dims:
        t0 = time.time()
        fa_k = FactorAnalysis(n_components=k, max_iter=200, tol=1e-3, random_state=seed).fit(Xtr)
        dim_sel.append({"k": k, "ppca": float(PCA(n_components=k).fit(Xtr).score(Xva)), "fa": float(fa_k.score(Xva))})
        print(f"    latent dims {k:>2}: held-out log-lik PPCA {dim_sel[-1]['ppca']:.2f}, FA {dim_sel[-1]['fa']:.2f} ({time.time() - t0:.0f}s)")
    best_fa = max(dim_sel, key=lambda r: r["fa"])["k"]

    # ---- 2c. factor analysis with readable factors, before and after varimax rotation
    nf = 6
    fa_raw = FactorAnalysis(n_components=nf, max_iter=200, tol=1e-3, random_state=seed).fit(Xtr)
    fa_rot = FactorAnalysis(n_components=nf, rotation="varimax", max_iter=200, tol=1e-3, random_state=seed).fit(Xtr)
    factors = [{"factor": i + 1, "top": [{"label": label(Z.columns[j]), "loading": float(fa_rot.components_[i, j])}
                                         for j in np.argsort(-np.abs(fa_rot.components_[i]))[:6]]} for i in range(nf)]
    fa_info = {"n_factors": nf, "factors": factors,
               "loglik_unrotated": float(fa_raw.score(Xtr)), "loglik_rotated": float(fa_rot.score(Xtr)),
               "noise_variance_mean": float(fa_rot.noise_variance_.mean())}
    F = fa_rot.transform(X)

    # ---- 3. K-Means on the PCA space: elbow + silhouette, K chosen by silhouette on training alerts
    s_idx = rng.choice(np.where(tr)[0], min(4000, tr.sum()), replace=False)
    elbow = []
    for k in range(2, 13):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(P[tr])
        elbow.append({"k": k, "inertia": float(km.inertia_),
                      "silhouette": float(silhouette_score(P[s_idx], km.predict(P[s_idx])))})
    K = max((r for r in elbow if 3 <= r["k"] <= 10), key=lambda r: r["silhouette"])["k"]
    km = KMeans(n_clusters=K, n_init=20, random_state=seed).fit(P[tr])
    lab_km = km.predict(P)

    # ---- 4. Gaussian mixture by EM: BIC for the number of components, soft assignments
    bic = []
    for k in range(2, 11):
        g = GaussianMixture(k, covariance_type="full", n_init=2, random_state=seed).fit(P[tr])
        bic.append({"k": k, "bic": float(g.bic(P[tr])), "aic": float(g.aic(P[tr]))})
    G = min(bic, key=lambda r: r["bic"])["k"]
    # EM run one cycle at a time to record the log-likelihood after each E/M cycle
    gmm = GaussianMixture(G, covariance_type="full", max_iter=1, warm_start=True, init_params="kmeans",
                          random_state=seed, tol=0)
    em_curve = []
    for _ in range(60):
        gmm.fit(P[tr])
        em_curve.append(float(gmm.score(P[tr])))
        if len(em_curve) > 2 and abs(em_curve[-1] - em_curve[-2]) < 1e-5:
            break
    resp = gmm.predict_proba(P)                       # E-step responsibilities = soft assignments
    lab_gmm = resp.argmax(1)
    entropy = -(resp * np.log(np.clip(resp, 1e-12, 1))).sum(1)
    dens = -gmm.score_samples(P)                      # negative log density: high = unusual alert

    # ---- 5. hierarchical clustering on a training sample (agglomerative; quadratic memory)
    hs = rng.choice(np.where(tr)[0], min(1500, int(tr.sum())), replace=False)
    Ph = P[hs]
    hier = []
    for metric in ("euclidean", "cityblock", "cosine"):
        D = pdist(Ph, metric=metric)
        for method in ("single", "complete", "average") + (("ward",) if metric == "euclidean" else ()):
            L = linkage(Ph if method == "ward" else D, method=method, metric=metric)
            labs = fcluster(L, K, criterion="maxclust")
            sizes = np.bincount(labs)[1:]
            hier.append({"metric": {"cityblock": "manhattan"}.get(metric, metric), "linkage": method,
                         "cophenetic": float(cophenet(L, D)[0]),
                         "silhouette": float(silhouette_score(Ph, labs, metric=metric)) if len(np.unique(labs)) > 1 else None,
                         "largest_cluster_share": float(sizes.max() / sizes.sum())})
    Lw = linkage(Ph, method="ward")
    dg = dendrogram(Lw, truncate_mode="lastp", p=30, no_plot=True)
    lab_h = fcluster(Lw, K, criterion="maxclust") - 1

    # ---- 6. evaluation on the 2021 test quarters (models fitted on 2020 only)
    te = (df.role == "test").values
    yte = y[te]
    evaluation = {
        "K-Means": {"train": quality(P[tr], lab_km[tr], y[tr], rng), "test": quality(P[te], lab_km[te], yte, rng)},
        "Gaussian mixture": {"train": quality(P[tr], lab_gmm[tr], y[tr], rng), "test": quality(P[te], lab_gmm[te], yte, rng)},
        "Hierarchical (Ward, sample)": {"train": quality(Ph, lab_h, y[hs], rng)},
        "baseline_purity": float(max(y[te].mean(), 1 - y[te].mean())),
        "kmeans_vs_gmm_ari": float(adjusted_rand_score(lab_km, lab_gmm)),
    }
    profiles = {"kmeans": {"train": describe(Z[tr], lab_km[tr], y[tr]), "test": describe(Z[te], lab_km[te], yte)},
                "gmm": {"train": describe(Z[tr], lab_gmm[tr], y[tr]), "test": describe(Z[te], lab_gmm[te], yte)}}

    # ---- per-alert outputs (consumed by stage 10 and by the console)
    tr_rate = pd.Series(y[tr]).groupby(lab_km[tr]).mean()
    out_df = pd.DataFrame({"alert_id": df.alert_id, "role": df.role, "quarter": df.quarter, "y": y,
                           "pc1": P3[:, 0], "pc2": P3[:, 1], "pc3": P3[:, 2], "kmeans": lab_km, "gmm": lab_gmm,
                           "gmm_conf": resp.max(1), "gmm_entropy": entropy, "gmm_density": dens,
                           "cluster_rate": tr_rate.reindex(lab_km).values})
    for k in range(G):
        out_df[f"gmm_p{k}"] = resp[:, k]
    for k in range(K):
        out_df[f"km_{k}"] = (lab_km == k).astype(np.int8)
    for i in range(nf):
        out_df[f"factor_{i + 1}"] = F[:, i]
    out_df.to_parquet(art(cfg, "clusters.parquet"))

    res = {"n_features": len(FEATS), "pca": {"ratio": ratio[:40], "eigenvalues": eigval[:40], "d80": d,
                                              "svd_matches_eigen": svd_agrees, "loadings": loadings},
           "dim_selection": dim_sel, "fa_best_k": best_fa, "fa": fa_info,
           "kmeans": {"K": K, "elbow": elbow}, "gmm": {"K": G, "bic": bic, "em_curve": em_curve},
           "hierarchical": {"sample": len(hs), "table": hier,
                            "dendrogram": {"icoord": dg["icoord"], "dcoord": dg["dcoord"], "leaves": dg["ivl"]}},
           "evaluation": evaluation, "profiles": profiles}
    save_json(res, art(cfg, "clustering.json"))
    print(f"[8] PCA keeps {d} components (80% variance) | K-Means K={K} | GMM K={G} | "
          f"test silhouette {evaluation['K-Means']['test']['silhouette']:.3f} | test purity "
          f"{evaluation['K-Means']['test']['purity']:.3f} (baseline {evaluation['baseline_purity']:.3f})")
    return res
