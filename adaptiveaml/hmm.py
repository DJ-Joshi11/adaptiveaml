"""A discrete-emission Hidden Markov Model, written out in full so every step can be explained.

    hidden state   z_t in {0..K-1}      a behavioural regime the client is in
    observation    x_t in {0..M-1}      one transaction, encoded as a symbol
    parameters     pi[k]    = p(z_1 = k)                       initial distribution
                   A[j, k]  = p(z_t = k | z_{t-1} = j)          transition matrix (Markov property)
                   B[k, m]  = p(x_t = m | z_t = k)              emission matrix

Algorithms (Bishop, PRML ch. 13)
    forward_backward   alpha-beta recursions with SCALING FACTORS c_t, so long sequences never
                       underflow: alpha_hat_t = alpha_t / prod(c), log p(X) = sum_t log c_t
    fit (Baum-Welch)   maximum likelihood by EM: the E-step computes gamma (state marginals) and
                       xi (pairwise marginals) with forward-backward; the M-step re-estimates
                       pi, A and B from those expected counts
    viterbi            max-sum in log space: the single most probable state path, recovered by
                       backtracking (no scaling needed because logs turn products into sums)

Many sequences are processed at once: sequences are padded to a common length and a mask marks
the real positions, so the recursions run as matrix operations over the whole batch.
"""
from __future__ import annotations

import numpy as np


def pad(seqs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Stack variable-length integer sequences into (B, T) arrays plus a validity mask."""
    T = max(len(s) for s in seqs)
    obs = np.zeros((len(seqs), T), dtype=np.int64)
    mask = np.zeros((len(seqs), T), dtype=bool)
    for i, s in enumerate(seqs):
        obs[i, :len(s)], mask[i, :len(s)] = s, True
    return obs, mask


def batches(seqs: list[np.ndarray], size: int = 400):
    """Yield (indices, obs, mask) for groups of similar-length sequences (keeps padding small)."""
    order = np.argsort([len(s) for s in seqs])
    for i in range(0, len(order), size):
        idx = order[i:i + size]
        obs, mask = pad([seqs[j] for j in idx])
        yield idx, obs, mask


class DiscreteHMM:
    def __init__(self, n_states: int, n_symbols: int, seed: int = 42, pseudo: float = 1e-2):
        self.K, self.M, self.seed, self.pseudo = n_states, n_symbols, seed, pseudo
        rng = np.random.default_rng(seed)
        self.pi = np.full(self.K, 1 / self.K)
        A = np.eye(self.K) * 4 + rng.random((self.K, self.K))       # start "sticky": regimes persist
        self.A = A / A.sum(1, keepdims=True)
        B = rng.random((self.K, self.M)) + 0.5
        self.B = B / B.sum(1, keepdims=True)
        self.history: list[float] = []

    # ------------------------------------------------------------------ forward-backward
    def forward_backward(self, obs: np.ndarray, mask: np.ndarray, need_beta: bool = True):
        """Scaled alpha-beta recursions for a padded batch.
        Returns alpha_hat, beta_hat (B,T,K), log c (B,T) and the emission terms (B,T,K)."""
        Bn, T = obs.shape
        e = self.B.T[obs]                                    # (B,T,K): p(x_t | z_t = k)
        alpha = np.empty((Bn, T, self.K))
        logc = np.zeros((Bn, T))
        a = self.pi * e[:, 0]
        c = a.sum(1, keepdims=True)
        alpha[:, 0] = a / c
        logc[:, 0] = np.log(c[:, 0])
        for t in range(1, T):
            a = (alpha[:, t - 1] @ self.A) * e[:, t]
            c = a.sum(1, keepdims=True)
            m = mask[:, t]
            alpha[:, t] = np.where(m[:, None], a / np.where(c > 0, c, 1), alpha[:, t - 1])
            logc[:, t] = np.where(m, np.log(np.where(c[:, 0] > 0, c[:, 0], 1)), 0.0)
        if not need_beta:
            return alpha, None, logc, e
        beta = np.ones((Bn, T, self.K))
        for t in range(T - 2, -1, -1):
            nxt = mask[:, t + 1]
            b = ((e[:, t + 1] * beta[:, t + 1]) @ self.A.T) / np.exp(logc[:, t + 1])[:, None]
            beta[:, t] = np.where(nxt[:, None], b, beta[:, t + 1])
        return alpha, beta, logc, e

    def loglik(self, seqs: list[np.ndarray]) -> np.ndarray:
        """log p(X) for each sequence, from the scaling factors: sum_t log c_t."""
        out = np.empty(len(seqs))
        for idx, obs, mask in batches(seqs):
            _, _, logc, _ = self.forward_backward(obs, mask, need_beta=False)
            out[idx] = logc.sum(1)
        return out

    def posteriors(self, seqs: list[np.ndarray]) -> list[np.ndarray]:
        """gamma_t(k) = p(z_t = k | X) for each sequence (the state marginals)."""
        out: list = [None] * len(seqs)
        for idx, obs, mask in batches(seqs):
            alpha, beta, _, _ = self.forward_backward(obs, mask)
            g = alpha * beta
            g /= g.sum(2, keepdims=True)
            for r, i in enumerate(idx):
                out[i] = g[r, :mask[r].sum()]
        return out

    # ------------------------------------------------------------------ Baum-Welch (EM)
    def fit(self, seqs: list[np.ndarray], n_iter: int = 30, tol: float = 1e-4) -> "DiscreteHMM":
        n_obs = sum(len(s) for s in seqs)
        prev = -np.inf
        for _ in range(n_iter):
            pi_acc = np.zeros(self.K)
            A_acc = np.zeros((self.K, self.K))
            B_acc = np.zeros((self.K, self.M))
            ll = 0.0
            for _, obs, mask in batches(seqs):
                alpha, beta, logc, e = self.forward_backward(obs, mask)
                ll += logc.sum()
                gamma = alpha * beta                                   # E-step: state marginals
                gamma /= gamma.sum(2, keepdims=True)
                pi_acc += gamma[:, 0].sum(0)
                # xi_t(j,k) = alpha_t(j) A(j,k) e_{t+1}(k) beta_{t+1}(k) / c_{t+1}: pairwise marginals
                both = mask[:, :-1] & mask[:, 1:]
                w = (e[:, 1:] * beta[:, 1:]) / np.exp(logc[:, 1:])[:, :, None]
                A_acc += self.A * np.einsum("btj,btk->jk", alpha[:, :-1] * both[:, :, None], w)
                g = gamma[mask]
                o = obs[mask]
                for k in range(self.K):
                    B_acc[k] += np.bincount(o, weights=g[:, k], minlength=self.M)
            # M-step (small pseudo-counts stop any probability collapsing to exactly zero)
            self.pi = (pi_acc + self.pseudo) / (pi_acc + self.pseudo).sum()
            self.A = (A_acc + self.pseudo) / (A_acc + self.pseudo).sum(1, keepdims=True)
            self.B = (B_acc + self.pseudo) / (B_acc + self.pseudo).sum(1, keepdims=True)
            self.history.append(ll / n_obs)
            if np.isfinite(prev) and abs(ll - prev) / abs(prev) < tol:
                break
            prev = ll
        return self

    def n_params(self) -> int:
        return (self.K - 1) + self.K * (self.K - 1) + self.K * (self.M - 1)

    # ------------------------------------------------------------------ Viterbi (max-sum)
    def viterbi(self, seqs: list[np.ndarray]) -> list[np.ndarray]:
        logA, logB, logpi = np.log(self.A), np.log(self.B), np.log(self.pi)
        out: list = [None] * len(seqs)
        for idx, obs, mask in batches(seqs):
            Bn, T = obs.shape
            le = logB.T[obs]
            delta = logpi + le[:, 0]
            psi = np.zeros((Bn, T, self.K), dtype=np.int16)
            stay = np.broadcast_to(np.arange(self.K, dtype=np.int16), (Bn, self.K))
            for t in range(1, T):
                cand = delta[:, :, None] + logA                     # (B, from, to)
                best = cand.argmax(1)
                new = np.take_along_axis(cand, best[:, None, :], 1)[:, 0] + le[:, t]
                m = mask[:, t]
                delta = np.where(m[:, None], new, delta)
                psi[:, t] = np.where(m[:, None], best, stay)         # padded steps point to themselves
            path = np.empty((Bn, T), dtype=np.int16)
            path[:, -1] = delta.argmax(1)
            for t in range(T - 1, 0, -1):                           # backtracking
                path[:, t - 1] = np.take_along_axis(psi[:, t], path[:, t:t + 1].astype(np.int64), 1)[:, 0]
            for r, i in enumerate(idx):
                out[i] = path[r, :mask[r].sum()]
        return out
