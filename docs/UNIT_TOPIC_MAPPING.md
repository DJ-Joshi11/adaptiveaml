# Unit-topic mapping

For course evaluation only. None of this wording appears in the product or its code output.

The console referred to below is the React dashboard in `web/` (PBL-3 rebuild). Page names: Overview, Alert Queue, Alert Detail, Cluster Explorer, Sequence Analysis, Model Arena, Ensemble Insights, Data Insights, About & Methodology. PBL-1 and PBL-2 console references (e.g. "Activity Baseline", "Behavioural Profiles", "Feature Lab") now live at: Activity Baseline → Model Arena › Activity baseline; Model Comparison / Benchmark / Model Reasoning → Model Arena tabs; Behavioural Profiles → Cluster Explorer; Feature Lab → Model Arena › Ablations.

| Unit | Topic | Where it appears | How it is used | Evidence |
| --- | --- | --- | --- | --- |
| 1 | Supervised learning: regression and classification | Whole pipeline | Expected-activity regression feeds report/dismiss classification | `s04_regression.py`, `s05_classification.py` |
| 1 | Polynomial curve fitting, over-fitting, regularisation λ | Complexity study | Error vs polynomial degree and vs λ, training against later quarters | `s04_regression.py` (complexity study); console: Activity Baseline > Complexity and regularisation |
| 1 | Bayes' rule: prior, likelihood, posterior | EDA, Naive Bayes, LDA | Base rate updated by one piece of evidence; generative classifiers | `s03_eda.py` (Bayes block); console: Data Insights > Evidence and belief |
| 1 | Discrete and continuous random variables | Feature engineering | Counts and channel as discrete, Size as continuous; entropy of discrete distributions | `s02_features.py` |
| 1 | Gaussian density and fitting | EDA | Per-class Gaussian fits, skew and kurtosis as checks of the assumption | `s03_eda.py` (Gaussian block) |
| 1 | Expectation, variance, covariance | Features, EDA | Mean and spread features; correlation structure | `s02_features.py`, `s03_eda.py` (correlation block) |
| 1 | Conditional probability | EDA, evaluation | Report rate given a behaviour; precision and recall | `s03_eda.py`, `s06_evaluate.py` |
| 2 | Maximum likelihood and least squares | Activity baseline, EDA | Linear regression as Gaussian MLE; MLE of class-conditional Gaussians | `s04_regression.py`, `s03_eda.py` |
| 2 | Robust regression (Huber loss) | Activity baseline | Baseline resistant to very active outlier clients | `s04_regression.py` |
| 2 | Ridge regression (L2) | Activity baseline, logistic regression | Shrinks correlated weights; MAP with a Gaussian prior | `s04_regression.py`, `s05_classification.py` |
| 2 | Bayesian linear regression | Burst z-score | Predictive standard deviation normalises the burst residual | `s04_regression.py` (`bayes_predict`, burst block) |
| 2 | Discriminant functions, Fisher's linear discriminant | EDA, LDA classifier | 1-D separability view; LDA as a classifier | `s03_eda.py` (Fisher block), `s05_classification.py` |
| 2 | Generative vs discriminative models | Classification | Naive Bayes and LDA against logistic regression | `s05_classification.py`; console: Model Comparison > Tuning |
| 2 | Kernel functions and the kernel trick | SVM | Linear vs RBF kernel on the same data | `s05_classification.py` (kernel comparison) |
| 2 | SVM: margin, support vectors, soft margin | SVM | Effect of C and γ, support-vector count | `s05_classification.py`; console: Model Comparison > Tuning |
| 2 | Laplace approximation, Bayesian logistic regression | Confidence tag | Posterior spread on each alert's score flags low-confidence alerts | `bayes_logreg.py` |

## Additional methods (committed in the PBL-1 review)

| Method | Where | Why |
| --- | --- | --- |
| Lasso (L1) | `s04_regression.py` | Sparse baseline; shows which history features are dispensable |
| K-Nearest Neighbours | `s05_classification.py` | Non-parametric reference |
| Decision Tree | `s05_classification.py`, `s07_explain.py` | Readable rules for investigators |
| SMOTE | `s05_classification.py` | Tested as an alternative to class weighting |

## PBL-II: Units 3 and 4

| Unit | Topic | Where it appears | How it is used | Evidence |
| --- | --- | --- | --- | --- |
| 3 | K-Means clustering | `s08_clustering.py`; console: Behavioural Profiles | Hard assignment of alerts to behavioural profiles in the PCA space; K from elbow (inertia) and silhouette | Profiles > How many profiles? |
| 3 | Mixtures of Gaussians | `s08_clustering.py` | Full-covariance Gaussian mixture; soft memberships, assignment confidence, density as an unusualness score | Profiles > How many profiles? (hard vs soft) |
| 3 | EM algorithm (and its alternative view) | `s08_clustering.py`, `hmm.py` | Mixture fitted one E-step / M-step cycle at a time to show the log-likelihood rising; Baum-Welch is EM for HMMs; K-Means as the hard-assignment limit of EM | EM curve on Behavioural Profiles; Baum-Welch curve on Sequence Analysis |
| 3 | Factor analysis | `s08_clustering.py` | Six latent factors with varimax rotation; identical likelihood before and after rotation demonstrates unidentifiability | Profiles > Latent dimensions |
| 3 | PCA (eigen-decomposition, SVD) | `s08_clustering.py` | Covariance eigen-decomposition checked against the SVD; scree plot; 80% variance threshold; 2-D / 3-D maps | Profiles > Latent dimensions, Profile map |
| 3 | Choosing the number of latent dimensions | `s08_clustering.py` | Held-out log-likelihood of probabilistic PCA and FA against the number of dimensions | Profiles > Latent dimensions |
| 3 | Measuring dissimilarity | `s08_clustering.py` | Euclidean, Manhattan and cosine dissimilarity compared in hierarchical clustering | Profiles > Hierarchy |
| 3 | Evaluating clustering output | `s08_clustering.py` | Purity (with its 0.82 trivial floor), silhouette, Davies-Bouldin, mutual information, adjusted Rand index | Profiles > Quality |
| 3 | Hierarchical clustering | `s08_clustering.py` | Agglomerative single / complete / average / Ward linkage, cophenetic correlation, dendrogram; chaining effect of single link | Profiles > Hierarchy |
| 4 | Sequential data, Markov property | `s09_sequences.py` | Each alert's transactions kept in time order; first-order transitions between regimes | Sequence Analysis > Transitions |
| 4 | Hidden Markov models | `hmm.py`, `s09_sequences.py` | 72-symbol discrete emissions (direction × channel × size band × time gap); hidden behavioural regimes | Sequence Analysis > Regimes |
| 4 | Maximum likelihood for the HMM | `hmm.py` (`fit`) | Baum-Welch re-estimation of initial, transition and emission probabilities; states chosen by held-out log-likelihood and BIC | Sequence Analysis > Model fit |
| 4 | Forward-backward algorithm | `hmm.py` (`forward_backward`, `posteriors`) | Sequence likelihoods (suspicion and unusualness scores) and state marginals (regime occupancy) | Sequence Analysis > Suspicion scores |
| 4 | Sum-product algorithm | `hmm.py` | The alpha and beta recursions are the sum-product messages on the HMM chain; gamma and xi are the resulting marginals | Code comments in `hmm.py` |
| 4 | Scaling factors | `hmm.py` | Rescaled alpha / beta with stored scaling factors; log-likelihood = sum of log scaling factors | Sequence Analysis > Model fit |
| 4 | Viterbi algorithm | `hmm.py` (`viterbi`) | Max-sum in log space with backtracking; regime path per alert | Sequence Analysis > Alert timeline; Alert Detail > Patterns |
| 4 | Linear dynamical systems | Sequence Analysis > Model fit | Discussed as the continuous-state counterpart (Kalman filter, particle filters); not implemented because the observations are categorical | Console note |

Brute-force checks of the forward pass, the state marginals and Viterbi are in `tests/test_hmm.py`.


## PBL-3: Unit 5 (ensemble learning) and the final comparison

| Unit | Topic | Where it appears | How it is used | Evidence |
| --- | --- | --- | --- | --- |
| 5 | Combining models: committees | `s11_ensembles.py` (voting) | Equal-weight committee of LR, LDA, SVM, RF, XGBoost on a common z-scored scale | Ensemble Insights › forest plot; Model Arena › Leaderboard (test PR-AUC 0.290) |
| 5 | Bagging (bootstrap aggregation) | `s11_ensembles.py` | 300 CART trees on bootstrap samples, out-of-bag score; variance reduction measured directly | Ensemble Insights › Bagging vs boosting (deep-tree variance 0.090 → bagged 0.0030) |
| 5 | Random forests, CART base learner | `s11_ensembles.py` | Gini CART trees + feature subsampling, tuned by forward-chaining; impurity vs permutation importance; OOB check | Ensemble Insights › Random Forest |
| 5 | Tree-based models / decision trees | `s05_classification.py`, `s11_ensembles.py` | Single tree (PBL-1), stumps (AdaBoost), CART in bagging/forest, shallow trees in boosting | Model Arena › Readable models (tree rules) |
| 5 | Boosting: AdaBoost | `adaboost.py` (written out), `s11_ensembles.py` | ε_m, α_m = ln((1−ε_m)/ε_m), w_n ← w_n exp(α_m I(miss)); exponential-loss trace; weight snapshots; verified against scikit-learn (rank corr 1.0000) | Ensemble Insights › AdaBoost lab; `tests/test_ensembles.py` |
| 5 | Minimising exponential error | `adaboost.py` | Exponential loss recorded each round and shown to fall monotonically; w_n ∝ exp(−t_n f_m(x_n)) checked in tests | AdaBoost lab › loss chart |
| 5 | Gradient boosting (XGBoost, LightGBM) | `s11_ensembles.py` | Depth × rounds tuned by forward-chaining; TreeSHAP per alert (additivity error 8.3e-07) | Ensemble Insights › Gradient boosting; Alert Detail › What drove the score |
| 5 | Bias–variance of bagging vs boosting | `s11_ensembles.py` | 12 bootstrap refits; squared error split into bias²+noise and variance; performance vs number of estimators | Ensemble Insights › Bagging vs boosting |
| 5 | Stacking / conditional mixture of models | `s11_ensembles.py` | Meta logistic regression over 7 base learners from all three reviews, trained on out-of-fold (forward-chaining) scores; weight CIs, ablation ladder, leave-one-out, diversity matrix | Ensemble Insights › Stacking |
| 5 | Bayesian model averaging | `s11_ensembles.py` | Posterior model weights from held-out predictive likelihood; evidence curve shows collapse onto one model; contrasted with ensembles | Ensemble Insights › BMA vs ensembles |
| 5 | Generalised additive models | `s11_ensembles.py` | Cubic B-spline per feature + penalised logistic regression; shape functions | Model Arena › Readable models |
| — | Final comparative analysis | `s12_final.py` | 21 models, bootstrap CIs, paired differences and McNemar vs PBL-1 champion, phase winners chosen on validation, champion–challenger promotion rule, per-quarter stability | Model Arena; Overview |

### Final result (test quarters 2021 Q3–Q4, 3,266 alerts, base rate 0.182)

| Phase | Best on validation | Test PR-AUC [95% CI] | vs PBL-1 SVM |
| --- | --- | --- | --- |
| PBL-1 | SVM (RBF) | 0.283 [0.262, 0.310] | reference |
| PBL-2 | SVM + profiles + sequences | 0.283 [0.262, 0.312] | tied |
| PBL-3 | Bagging (CART) | 0.274 [0.255, 0.299] | tied |

Best test score of any ensemble: Voting committee 0.290 (difference vs SVM +0.007, 95% CI [-0.006, +0.018]). The challenger (Bagging (CART)) was not promoted: its validation lead over the SVM was +0.005 [-0.011, +0.020]. Honest conclusion: ensembles reach the top group but do not beat the PBL-1 SVM by a statistically reliable margin on this data.

## Risk register

| Risk | Level | Mitigation |
| --- | --- | --- |
| Synthetic data may not reflect real banks | Medium | Results framed as a method study; protocol transfers unchanged |
| Weak-signal ceiling (all strong models ≈ 0.28–0.29) | High | Reported honestly; promotion rule prevents shipping noise |
| Drift between 2020 and 2021 | Medium | Time-based split, per-quarter stability, quarterly retraining |
| Selecting the validation winner by luck (21 candidates) | Medium | Champion–challenger rule with paired bootstrap on validation |
| Leakage into the stacker | High if ignored | Out-of-fold, time-forward base predictions only; tests in `test_ensembles.py` |
| Explanation model ≠ ranking model | Low | XGBoost SHAP shown with its rank agreement to the SVM (0.83); exact LR reasons one click away |
| Over-trusting a single score | Medium | Calibration, Bayesian uncertainty band, six-model agreement flag on every alert |
