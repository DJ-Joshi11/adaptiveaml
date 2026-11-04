"""Leakage and protocol checks. Run with:  pytest -q

These turn the project's promises (no look-ahead, no target leakage) into code that fails loudly.
"""
import numpy as np
import pandas as pd
import pytest

from adaptiveaml.common import BURST_COLS, CLF_COLS, HISTORY_COLS, REGISTRY, forward_folds, load_config, split_of, ROOT

cfg = load_config()
FEATURES = ROOT / cfg["data"]["processed_dir"] / "alert_features.parquet"


def test_regression_inputs_are_history_only():
    # the regression target is r_n (last 30 days); no input may come from that window or from the burst signal
    assert all(c.startswith("h_") for c in HISTORY_COLS)
    assert "r_n" not in HISTORY_COLS and not set(BURST_COLS) & set(HISTORY_COLS)


def test_every_feature_is_documented():
    assert set(CLF_COLS) == set(REGISTRY)
    assert all(len(v) == 3 and all(v) for v in REGISTRY.values())


def test_split_quarters_are_disjoint_and_ordered():
    s = cfg["split"]
    assert not (set(s["train"]) & set(s["valid"])) and not (set(s["valid"]) & set(s["test"]))
    assert max(s["train"]) < min(s["valid"]) <= max(s["valid"]) < min(s["test"])
    assert len(split_of(cfg)) == 8


def test_forward_folds_never_validate_on_the_past():
    tq = cfg["split"]["train"]
    q = pd.Series(np.repeat(tq, 5))
    folds = list(forward_folds(q, tq))
    assert len(folds) == len(tq) - 1
    for fit, val in folds:
        assert q.iloc[fit].max() < q.iloc[val].min()
        assert not set(fit) & set(val)


@pytest.mark.skipif(not FEATURES.exists(), reason="run the pipeline first")
def test_feature_table_is_one_row_per_alert_with_burst_signal():
    df = pd.read_parquet(FEATURES)
    assert df.alert_id.is_unique
    assert set(df.role.unique()) <= {"train", "valid", "test", "unused"}
    if "burst_z" in df:                                   # written by stage 4
        scored = df[df.role.isin(["train", "valid", "test"])]
        assert scored[BURST_COLS].notna().all().all()


@pytest.mark.skipif(not (ROOT / "artifacts" / "predictions.parquet").exists(), reason="run the pipeline first")
def test_no_training_alert_is_ever_scored_for_evaluation():
    preds = pd.read_parquet(ROOT / "artifacts" / "predictions.parquet")
    assert set(preds.role.unique()) <= {"valid", "test"}
    assert preds.quarter.min() > max(cfg["split"]["train"])


ART = ROOT / "artifacts"


@pytest.mark.skipif(not (ART / "clusters.parquet").exists(), reason="run stage 8 first")
def test_profiles_cover_every_alert_once():
    c = pd.read_parquet(ART / "clusters.parquet")
    assert c.alert_id.is_unique and set(c.role) <= {"train", "valid", "test"}
    p = c.filter(like="gmm_p").sum(1)
    assert np.allclose(p, 1, atol=1e-6)                       # soft assignments are probabilities


@pytest.mark.skipif(not (ART / "sequence_features.parquet").exists(), reason="run stage 9 first")
def test_sequence_features_are_finite_and_occupancies_sum_to_one():
    s = pd.read_parquet(ART / "sequence_features.parquet")
    occ = s.filter(like="hmm_occ_")
    assert np.allclose(occ.sum(1), 1, atol=1e-6)
    assert np.isfinite(s[["hmm_llr", "hmm_normal_ll"]].values).all()


@pytest.mark.skipif(not (ART / "enrichment.json").exists(), reason="run stage 10 first")
def test_enrichment_baseline_reproduces_stage_6():
    import json
    en = json.loads((ART / "enrichment.json").read_text())
    ev = json.loads((ART / "evaluation.json").read_text())
    stage6 = {r["model"]: r["pr_auc"] for r in ev["table"]}
    for r in en["rows"]:
        if r["variant"] == "Baseline":
            # same models, same split, same alerts; a tiny tolerance absorbs floating-point differences
            # between machines (the artifacts may have been produced on different computers)
            assert abs(r["pr_auc"] - stage6[r["model"]]) < 1e-4
