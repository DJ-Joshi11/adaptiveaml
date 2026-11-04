# AdaptiveAML

Explainable alert triage for anti-money-laundering teams. Scores each investigated alert by its
probability of being reported, orders the analyst queue, and shows the behavioural evidence
behind every score. Built and evaluated on the public SynthAML dataset.

## Quick start (Windows, VS Code terminal)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

1. Download SynthAML (Figshare collection 6504421) and put `synthetic_alerts.csv` and
   `synthetic_transactions.csv` in `data/raw/`.
2. Run the pipeline (a few minutes; the 16M-row read needs roughly 6 GB of free RAM):

```powershell
python run_pipeline.py
```

3. Open the console:

```powershell
streamlit run app/Home.py
```

No dataset yet? Preview the whole thing on a small generated stand-in. The console shows a
warning banner in this mode because the numbers are meaningless:

```powershell
python tests/make_fixture.py 3000
python run_pipeline.py --sample
streamlit run app/Home.py
```

Run the protocol checks with `pytest -q`.

## How the code is organised

| File | Stage | What it does |
| --- | --- | --- |
| `configs/base.yaml` | | Every decision that can change a result: seed, split, windows, review budget |
| `adaptiveaml/common.py` | | Config, the documented feature registry, forward-chaining folds |
| `adaptiveaml/s01_ingest.py` | 1 | CSV to typed Parquet, data-quality report, time relative to the alert |
| `adaptiveaml/s02_features.py` | 2 | 16M transactions to one row per alert (84 documented features: 36 behavioural, 48 channel-direction profile) |
| `adaptiveaml/s03_eda.py` | 3 | Statistical analysis on training quarters only |
| `adaptiveaml/s04_regression.py` | 4 | Expected-activity models and the burst signal |
| `adaptiveaml/s05_classification.py` | 5 | Six classifiers, tuning, calibration, ablation arms, benchmark re-implementation |
| `adaptiveaml/bayes_logreg.py` | 5 | Uncertainty on each score (Laplace approximation) |
| `adaptiveaml/s06_evaluate.py` | 6 | Held-out metrics, bootstrap intervals, significance tests |
| `adaptiveaml/s07_explain.py` | 7 | Global and per-alert explanations |
| `app/` | | The console. Reads `artifacts/` only; never trains anything |
| `tests/test_leakage.py` | | No look-ahead and no target leakage, enforced in code |
| `docs/UNIT_TOPIC_MAPPING.md` | | Course-topic map (documentation only) |

## PBL-II: behavioural profiles and sequence models

Three stages build on the PBL-1 pipeline without changing it:

| File | Stage | What it does |
| --- | --- | --- |
| `adaptiveaml/s08_clustering.py` | 8 | PCA and factor analysis, K-Means, Gaussian mixture (EM), hierarchical clustering, cluster quality |
| `adaptiveaml/hmm.py` | 9 | Hidden Markov model written out in full: scaled forward-backward, Baum-Welch, Viterbi |
| `adaptiveaml/s09_sequences.py` | 9 | Encodes each alert's transactions as a sequence of 72 symbols and fits the HMMs |
| `adaptiveaml/s10_compare.py` | 10 | Retrains the six classifiers with profile and sequence features added (ablation) |
| `tests/test_hmm.py` | | Forward pass, state marginals and Viterbi checked against brute force |

Console pages added: **Behavioural Profiles**, **Sequence Analysis**, **Feature Lab**; Alert Detail and
Alert Queue now show each alert's profile and dominant regime.

Run only the new stages (after stages 1-7 have run once):

```powershell
python run_pipeline.py --stage 8 9 10
```

Every model in stages 8-10 is fitted on 2020 alerts only and applied unchanged to 2021.

## Evaluation protocol

- Train: 2020 Q1-Q4. Tune, calibrate, set threshold: 2021 Q1-Q2. Final evaluation: 2021 Q3-Q4.
- Headline metric: PR-AUC with 95% bootstrap intervals. Accuracy is never reported.
- Operating point: review the top 20% of the queue; cut-off fixed on validation quarters.
- The model in service is chosen on validation quarters, never on the test quarters.

## Known limitations

- SynthAML is synthetic and contains only alerts that were investigated, so this is alert
  triage, not detection over all transactions.
- Alert dates are accurate to the quarter only. Feature windows are anchored on each alert's
  last transaction when timestamps are absolute dates.
- Transaction size is log-transformed and standardised: a relative amount, not currency.
- No client, account or counterparty identifiers exist, so network features and fairness
  testing are not possible on this data.
- The published-benchmark comparison is our re-implementation of the paper's 56-feature recipe.
  To show the paper's own figures, enter them by hand in `configs/published_benchmark.yaml`.

## References

1. Jensen, R. I. T. and Iosifidis, A. (2023). Fighting Money Laundering With Statistics and Machine Learning. IEEE Access. DOI 10.1109/ACCESS.2023.3239549
2. Jensen, R. I. T. et al. (2023). A synthetic data set to benchmark anti-money laundering methods. Scientific Data 10. DOI 10.1038/s41597-023-02569-2
