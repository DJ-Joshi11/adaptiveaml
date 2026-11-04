# Unit-topic mapping

For course evaluation only. None of this wording appears in the product or its code output.

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
