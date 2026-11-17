"""PBL-3 checks: the hand-written AdaBoost is correct, stacking never learns from the future,
and the dashboard export is consistent with the final comparison.  Run with:  pytest -q"""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from adaptiveaml.adaboost import AdaBoostStumps
from adaptiveaml.common import ROOT, forward_folds, load_config

cfg = load_config()


def _toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = (X[:, 0] + 0.5 * X[:, 1] ** 2 + rng.normal(scale=0.8, size=n) > 0.7).astype(int)
    return X, y


def test_adaboost_matches_sklearn():
    X, y = _toy()
    ours = AdaBoostStumps(40, seed=0).fit(X, y)
    sk = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, random_state=0), n_estimators=40, random_state=0).fit(X, y)
    a, b = ours.decision_function(X), sk.decision_function(X)
    assert np.corrcoef(a, b)[0, 1] > 0.999


def test_adaboost_exponential_loss_never_increases():
    X, y = _toy(seed=1)
    tr = AdaBoostStumps(60, seed=0).fit(X, y).trace
    loss = np.array([r["exp_loss"] for r in tr])
    assert np.all(np.diff(loss) <= 1e-12)
    assert all(0 < r["eps"] < 0.5 and r["alpha"] > 0 for r in tr)


def test_adaboost_weights_are_exp_of_margin():
    # after m rounds, w_n is proportional to exp(-t_n f_m(x_n)) (Bishop eq. 14.22 with f = half the vote)
    X, y = _toy(seed=2)
    m = AdaBoostStumps(25, seed=0).fit(X, y)
    t = 2 * y - 1
    w = np.exp(-t * m.decision_function(X))
    assert np.allclose(w / w.sum(), m.final_weights, atol=1e-9)


def test_stacking_folds_only_look_backwards():
    q = pd.Series(np.repeat(cfg["split"]["train"], 50))
    for fit, val in forward_folds(q, cfg["split"]["train"]):
        assert q.iloc[fit].max() < q.iloc[val].min()


FINAL = ROOT / "artifacts" / "final.json"


@pytest.mark.skipif(not FINAL.exists(), reason="run stages 11-12 first")
def test_champion_rule_and_phase_winners():
    f = json.loads(FINAL.read_text())
    if not f["promoted"]:
        assert f["champion"] == f["pbl1_champion"]
        assert f["valid_gap_vs_pbl1"]["ci"][0] <= 0
    names = {r["model"] for r in f["table"]}
    assert {p["model"] for p in f["phases"]} <= names and len(f["table"]) == len(names)


DATA = ROOT / "web" / "public" / "data"


@pytest.mark.skipif(not (DATA / "queue.json").exists(), reason="run stage 13 first")
def test_dashboard_queue_is_a_ranking_of_the_test_quarters():
    q = json.loads((DATA / "queue.json").read_text())
    ranks = sorted(r["rank"] for r in q)
    assert ranks == list(range(1, len(q) + 1))
    assert {r["q"] for r in q} == set(cfg["split"]["test"])
    review = sum(r["band"] == "Review now" for r in q)
    assert review == int(np.ceil(cfg["evaluation"]["review_budget"] * len(q)))
