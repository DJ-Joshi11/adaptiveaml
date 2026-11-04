"""Checks the hand-written HMM against brute force on tiny problems (every state path enumerated)."""
import itertools

import numpy as np

from adaptiveaml.hmm import DiscreteHMM


def brute(h, x):
    paths = list(itertools.product(range(h.K), repeat=len(x)))
    p = np.array([h.pi[z[0]] * h.B[z[0], x[0]] * np.prod([h.A[z[t - 1], z[t]] * h.B[z[t], x[t]] for t in range(1, len(x))])
                  for z in paths])
    return p, paths


def test_forward_matches_brute_force_and_viterbi_finds_best_path():
    rng = np.random.default_rng(0)
    h = DiscreteHMM(3, 4, seed=1)
    seqs = [rng.integers(0, 4, n) for n in (1, 2, 5, 6)]
    ll = h.loglik(seqs)
    paths = h.viterbi(seqs)
    gam = h.posteriors(seqs)
    for x, l, v, g in zip(seqs, ll, paths, gam):
        p, all_paths = brute(h, x)
        assert np.isclose(l, np.log(p.sum()))                              # forward + scaling
        assert tuple(v) == all_paths[int(p.argmax())]                      # Viterbi
        marg = np.array([[p[[z[t] == k for z in all_paths]].sum() for k in range(3)] for t in range(len(x))]) / p.sum()
        assert np.allclose(g, marg)                                        # forward-backward marginals


def test_em_never_decreases_likelihood():
    rng = np.random.default_rng(3)
    seqs = [rng.integers(0, 6, rng.integers(20, 60)) for _ in range(40)]
    h = DiscreteHMM(3, 6, seed=2, pseudo=0).fit(seqs, n_iter=15, tol=0)
    assert all(b >= a - 1e-9 for a, b in zip(h.history, h.history[1:]))
