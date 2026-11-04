"""Bayesian logistic regression via the Laplace approximation.

Ordinary logistic regression returns one weight vector. The Bayesian view keeps a DISTRIBUTION
over weights, so each alert gets a score and a measure of how sure the model is about it.

The exact posterior has no closed form, so it is approximated by a Gaussian (Laplace):
    1. find the posterior mode w_MAP  (= L2-regularised logistic regression; the Gaussian prior
       on the weights is exactly the L2 penalty)
    2. take the curvature of the negative log-posterior at the mode:
           H = X^T S X + (1/C) I      with  S = diag(p_i (1 - p_i))
    3. posterior  ~  Normal(w_MAP, H^-1)

For a new alert x the logit is then Gaussian with mean x.w_MAP and variance x^T H^-1 x. A wide
variance means the alert sits in a region the training data says little about.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class LaplaceLogReg:
    def __init__(self, C: float = 1.0):
        self.C = C

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LaplaceLogReg":
        self.map_ = LogisticRegression(C=self.C, max_iter=2000).fit(X, y)
        Xb = np.hstack([X, np.ones((len(X), 1))])              # add the intercept column
        prob = self.map_.predict_proba(X)[:, 1]
        S = prob * (1 - prob)
        prior_precision = np.eye(Xb.shape[1]) / self.C
        prior_precision[-1, -1] = 1e-6                          # the intercept is not regularised
        H = (Xb * S[:, None]).T @ Xb + prior_precision          # Hessian at the mode
        self.cov_ = np.linalg.inv(H)                            # posterior covariance of the weights
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (logit mean, logit std, predictive probability)."""
        Xb = np.hstack([X, np.ones((len(X), 1))])
        mu = self.map_.decision_function(X)
        var = np.einsum("ij,jk,ik->i", Xb, self.cov_, Xb)
        # probit approximation to the logistic-Gaussian integral (Bishop PRML, eq. 4.153)
        prob = 1 / (1 + np.exp(-mu / np.sqrt(1 + np.pi * var / 8)))
        return mu, np.sqrt(var), prob
