"""AdaBoost written out by hand (Bishop, Pattern Recognition and Machine Learning, section 14.3).

The library version hides the mechanics; this one records them, round by round, so they can be
shown and checked:

    1. start with equal weights            w_n = 1 / N
    2. for m = 1 .. M
         fit a decision stump y_m(x) to the WEIGHTED data
         weighted error                    eps_m  = sum_n w_n I(y_m(x_n) != t_n) / sum_n w_n
         say of this stump                 alpha_m = ln((1 - eps_m) / eps_m)
         re-weight                         w_n <- w_n exp(alpha_m I(y_m(x_n) != t_n))
       (misclassified alerts get heavier, so the next stump concentrates on them)
    3. committee                           Y_M(x) = sign( sum_m alpha_m y_m(x) )

Why it works: this is forward stage-wise minimisation of the EXPONENTIAL loss
    E = sum_n exp(-t_n f_m(x_n)),   f_m(x) = 1/2 sum_{l<=m} alpha_l y_l(x),   t_n in {-1, +1}
Each round adds the stump and weight alpha_m that reduce E the most given the rounds before it,
and the weights w_n are exactly proportional to exp(-t_n f_{m-1}(x_n)).

With K = 2 classes this is the same algorithm as scikit-learn's AdaBoostClassifier (SAMME), which
the pipeline fits alongside it as a correctness check.
"""
from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier


class AdaBoostStumps:
    def __init__(self, n_rounds: int = 300, seed: int = 0):
        self.n_rounds, self.seed = n_rounds, seed

    def fit(self, X: np.ndarray, y: np.ndarray, watch: dict | None = None, snapshots=(1, 10, 50)):
        """watch: {name: X_other} -> the committee's score on other data is tracked every round."""
        t = 2 * y - 1                                   # targets in {-1, +1}
        n = len(y)
        w = np.full(n, 1.0 / n)
        F = np.zeros(n)                                 # f_m(x_n) on the training data
        self.stumps, self.alpha = [], []
        self.trace = []                                 # one record per round
        self.weight_snapshots = {}
        self.watch_scores = {k: np.zeros(len(v)) for k, v in (watch or {}).items()}
        self.watch_path = {k: [] for k in self.watch_scores}
        for m in range(1, self.n_rounds + 1):
            stump = DecisionTreeClassifier(max_depth=1, random_state=self.seed).fit(X, y, sample_weight=w)
            h = 2 * stump.predict(X) - 1
            miss = h != t
            eps = float(w[miss].sum() / w.sum())
            eps = min(max(eps, 1e-10), 1 - 1e-10)
            alpha = float(np.log((1 - eps) / eps))
            if alpha <= 0:                              # weak learner no better than chance: stop
                break
            w = w * np.exp(alpha * miss)
            w = w / w.sum()
            F += 0.5 * alpha * h
            self.stumps.append(stump)
            self.alpha.append(alpha)
            self.trace.append({
                "round": m, "eps": eps, "alpha": alpha,
                "exp_loss": float(np.mean(np.exp(-t * F))),          # exponential loss (per alert)
                "train_error": float(np.mean(np.sign(F + 1e-12) != t)),
                "max_weight_x_n": float(w.max() * n),                  # heaviest alert, in units of 1/N
                "ess_share": float(1.0 / np.sum(w ** 2) / n),          # effective sample size / N
                "weight_on_reported": float(w[y == 1].sum()),
                "feature": int(stump.tree_.feature[0]), "threshold": float(stump.tree_.threshold[0]),
            })
            if m in snapshots:
                self.weight_snapshots[m] = w.copy()
            for k, Xo in (watch or {}).items():
                self.watch_scores[k] += 0.5 * alpha * (2 * stump.predict(Xo) - 1)
                self.watch_path[k].append(self.watch_scores[k].copy())
        self.weight_snapshots[len(self.alpha)] = w.copy()
        self.final_weights = w
        return self

    def decision_function(self, X: np.ndarray, rounds: int | None = None) -> np.ndarray:
        k = len(self.alpha) if rounds is None else rounds
        return sum(0.5 * a * (2 * s.predict(X) - 1) for a, s in zip(self.alpha[:k], self.stumps[:k]))
