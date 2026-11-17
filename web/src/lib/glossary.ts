// Every metric and method on the dashboard has an entry here. `plain` is what an analyst needs;
// `tech` is the precise definition shown in Data Scientist mode (and on hover in either mode).
export type Term = { title: string; plain: string; tech: string };

export const G: Record<string, Term> = {
  prauc: {
    title: "PR-AUC (average precision)",
    plain: "One number for how well a model pushes the alerts that will be reported to the top of the queue. 1.0 is perfect; random ordering scores the share of alerts that get reported (about 0.18).",
    tech: "Area under the precision-recall curve, computed as average precision AP = Σₙ (Rₙ − Rₙ₋₁)·Pₙ over score thresholds. Preferred over accuracy and ROC-AUC under class imbalance because it ignores true negatives.",
  },
  rocauc: {
    title: "ROC-AUC",
    plain: "The chance that a randomly picked reported alert is scored above a randomly picked dismissed one. 0.5 = coin flip.",
    tech: "Area under TPR vs FPR; equals the Mann-Whitney U statistic / (n₊n₋). Insensitive to class balance, so it can look good when the useful top of the queue is poor.",
  },
  base_rate: {
    title: "Base rate",
    plain: "The share of alerts that were actually reported. Any ranking has to beat this to be useful.",
    tech: "P(y = 1) on the evaluation quarters; the PR-AUC of a random ranking.",
  },
  lift: { title: "Lift", plain: "How many times better than random ordering. 1.6× means the top of our queue is 60% richer in real cases.", tech: "PR-AUC / base rate." },
  recall_budget: {
    title: "Catch rate at 20% review",
    plain: "If analysts open only the top 20% of the queue, the share of all eventually-reported alerts they find.",
    tech: "Recall at k = ⌈0.2·N⌉ by rank (top-k, ties broken by queue order). Rank-based, so every model reviews exactly the same number of alerts.",
  },
  precision_budget: { title: "Hit rate in the reviewed 20%", plain: "Of the alerts analysts open, the share that turn out to be reportable.", tech: "Precision at k = ⌈0.2·N⌉." },
  ci: {
    title: "95% interval",
    plain: "The range the true value plausibly lies in. If two models' ranges overlap a lot, the difference may be luck.",
    tech: "Percentile interval from 1,000 stratified bootstrap resamples of the test alerts (reported and dismissed alerts resampled separately so every resample keeps the class balance).",
  },
  paired: {
    title: "Paired difference",
    plain: "The gap between two models measured on exactly the same reshuffled alert sets, so luck affects both equally. If the range excludes zero, the gap is real.",
    tech: "Δ = AP_A(b) − AP_B(b) for each bootstrap resample b using the same index set; reported as mean Δ with 2.5/97.5 percentiles.",
  },
  mcnemar: {
    title: "McNemar's test",
    plain: "Checks whether two models disagree about which alerts to review more often in one direction than the other.",
    tech: "Exact binomial test on discordant pairs (n₀₁, n₁₀) of the top-20% review decisions; H₀: n₀₁ = n₁₀.",
  },
  brier: { title: "Brier score", plain: "How far predicted probabilities are from what happened. Lower is better.", tech: "Mean (p − y)² of the Platt-calibrated probability." },
  calibration: {
    title: "Calibration",
    plain: "A risk of 0.30 should mean about 30% of such alerts get reported. Calibration fixes the scale without changing the order.",
    tech: "Platt scaling: a 1-D logistic regression from raw score to probability, fitted on the validation quarters (2021 Q1–Q2) only.",
  },
  risk: { title: "Risk score", plain: "The estimated chance this alert will be reported after investigation.", tech: "Champion model score, Platt-calibrated on validation quarters." },
  band: {
    title: "Priority band",
    plain: "Review now = top 20% of the queue. Standard = the rest of the upper half. Low priority = lower half.",
    tech: "Rank-based bands on the test queue: top-k with k = ⌈0.2N⌉, then split at the median risk.",
  },
  agreement: {
    title: "Model agreement",
    plain: "Six different models score every alert. When they agree, the score is more trustworthy; when they disagree, a human should look closer.",
    tech: "Standard deviation of the percentile rank each of the six committee models gives the alert (LR, LDA, SVM, RF, XGBoost, pattern model). Cut at the test-set median and 90th percentile.",
  },
  bayes_band: {
    title: "Uncertainty band",
    plain: "A range for the risk that reflects how much data stood behind similar alerts.",
    tech: "Bayesian logistic regression with a Laplace approximation: posterior N(ŵ, H⁻¹); 95% interval of the logit xᵀw mapped through the sigmoid.",
  },
  shap: {
    title: "SHAP contribution",
    plain: "How much each behaviour pushed this alert's score up or down compared with an average alert.",
    tech: "TreeSHAP (exact Shapley values for the XGBoost ensemble) in log-odds units. Additive: base value + Σ contributions = model margin (checked to < 1e-5).",
  },
  lr_contrib: {
    title: "Logistic regression reasons",
    plain: "The transparent model's own reasons. Exact by construction.",
    tech: "βⱼ·zⱼ, coefficient times the standardised feature value, in log-odds units.",
  },
  rf: {
    title: "Random Forest",
    plain: "Hundreds of decision trees, each trained on a reshuffled copy of the data and allowed to look at only some behaviours at each step. Their votes are averaged.",
    tech: "Bagging of CART trees (Gini impurity) + feature subsampling (max_features) at every split, which decorrelates trees. Prediction = mean of tree probabilities. Variance of the mean ≈ ρσ² + (1−ρ)σ²/B.",
  },
  bagging: {
    title: "Bagging",
    plain: "Train many copies of the same model on reshuffled data and average them. Smooths out a jumpy model.",
    tech: "Bootstrap aggregation: f̂_bag(x) = (1/B) Σ_b f̂*ᵇ(x). Reduces variance; leaves bias roughly unchanged.",
  },
  boosting: {
    title: "Boosting",
    plain: "Build models one after another, each focusing on the mistakes of the ones before.",
    tech: "Forward stage-wise additive modelling f_m = f_{m−1} + ν·h_m, each h_m fitted to the negative gradient (or re-weighted data) of the loss. Mainly reduces bias.",
  },
  adaboost: {
    title: "AdaBoost",
    plain: "Each new tiny decision rule gets to see the alerts the previous rules got wrong with more weight, so it concentrates on the hard cases.",
    tech: "εₘ = Σ wₙ I(yₘ(xₙ) ≠ tₙ)/Σ wₙ; αₘ = ln((1−εₘ)/εₘ); wₙ ← wₙ exp(αₘ I(yₘ(xₙ) ≠ tₙ)). Equivalent to stage-wise minimisation of the exponential loss Σ exp(−tₙ fₘ(xₙ)).",
  },
  xgboost: {
    title: "XGBoost",
    plain: "Gradient-boosted trees: small trees added one by one, each correcting what the sum so far still gets wrong.",
    tech: "Second-order (gradient + Hessian) boosting of regression trees on the log-loss with L2 leaf penalty λ, shrinkage η = 0.05, row/column subsampling; class weight via scale_pos_weight.",
  },
  lightgbm: { title: "LightGBM", plain: "Another gradient-boosting library; grows trees leaf by leaf and is very fast.", tech: "Histogram-based GBDT with leaf-wise growth (num_leaves = 2^depth), settings mirrored from the tuned XGBoost." },
  stacking: {
    title: "Stacking",
    plain: "A small model that learns how much to trust each of the other models, based on how they behaved on alerts they had not seen.",
    tech: "Level-0 base learners produce out-of-fold scores on 2020 Q2–Q4 (forward-chaining); a regularised logistic-regression meta-learner is fitted on those scores; base learners are then refitted on all of 2020 for test-time scoring.",
  },
  oof: {
    title: "Out-of-fold predictions",
    plain: "Scores each model gave to alerts it was not trained on. The stacker only ever learns from these, so it cannot be fooled by a model that memorised its training data.",
    tech: "For quarter q ∈ {Q2, Q3, Q4} of 2020, fit on quarters < q and predict q. Time-respecting, so no future information reaches the meta-learner.",
  },
  voting: { title: "Voting committee", plain: "Every model gets an equal say: their scores are put on the same scale and averaged.", tech: "Mean of z-scored base scores (z using each model's own 2020 score mean/sd, no labels involved)." },
  bma: {
    title: "Bayesian model averaging",
    plain: "Weights models by how believable each is as THE true model. With enough data it ends up trusting just one, which is why it is not the same as an ensemble.",
    tech: "p(y|x,D) = Σₖ p(y|x,Mₖ) p(Mₖ|D), p(Mₖ|D) ∝ p(D|Mₖ)p(Mₖ). Evidence approximated by held-out predictive likelihood on 2021 Q2. Ensembles instead combine models to enlarge the hypothesis space.",
  },
  gam: {
    title: "Generalised additive model",
    plain: "Like logistic regression, but each behaviour gets its own curve instead of a straight line, and you can still read every curve.",
    tech: "logit p(x) = β₀ + Σⱼ fⱼ(xⱼ), each fⱼ a cubic B-spline (6 knots, 8 basis functions) with an L2 penalty tuned by forward-chaining.",
  },
  bias_variance: {
    title: "Bias-variance split",
    plain: "Error comes from being systematically off (bias) or from being jumpy when the data changes slightly (variance). Bagging attacks jumpiness, boosting attacks systematic error.",
    tech: "Over R bootstrap refits: E[(f−y)²] = (f̄−y)² + Var(f); the first term contains the irreducible noise, identical for every model.",
  },
  oob: { title: "Out-of-bag score", plain: "A free accuracy check: each tree is tested on the alerts left out of its reshuffled sample.", tech: "≈37% of rows are out-of-bag per tree; OOB prediction averages only trees that did not see the row. Ignores time order, so it is a sanity check, not the evaluation." },
  mdi: { title: "Impurity importance", plain: "How much each behaviour helped the trees split alerts cleanly.", tech: "Mean decrease in Gini impurity, weighted by node size, averaged over trees. Biased toward high-cardinality continuous features." },
  permutation: { title: "Permutation importance", plain: "Shuffle one behaviour and see how much worse the model gets. Big drop = the model relies on it.", tech: "Decrease in average precision on 2,000 validation alerts after permuting the column, 3 repeats." },
  kmeans: { title: "K-Means", plain: "Splits alerts into K groups of look-alikes, each alert belonging to exactly one group.", tech: "Lloyd's algorithm minimising within-cluster squared Euclidean distance on standardised features; K by silhouette." },
  gmm: { title: "Gaussian mixture", plain: "Soft grouping: an alert can be 70% one profile and 30% another.", tech: "Full-covariance GMM fitted by EM; K by BIC; responsibilities γ(z_k) as soft memberships." },
  pca: { title: "PCA", plain: "Compresses many related behaviours into a few summary directions.", tech: "Eigen-decomposition of the covariance of standardised features (checked against SVD); components ranked by explained variance." },
  hmm: {
    title: "Hidden Markov model",
    plain: "Reads a client's transactions in order and infers hidden 'regimes' of behaviour and how the client moves between them.",
    tech: "Discrete HMM with K = 5 states over 72 symbols; Baum-Welch (scaled forward-backward) for fitting, Viterbi for the most likely state path.",
  },
  llr: {
    title: "Sequence suspicion score",
    plain: "Does this year of transactions look more like past reported alerts or past dismissed ones?",
    tech: "Per-transaction log-likelihood ratio log p(x|HMM_report) − log p(x|HMM_dismiss); outcome HMMs cross-fitted on 2020 halves.",
  },
  temporal: {
    title: "Time-based split",
    plain: "Models learn only from 2020, settings are chosen on early 2021, and the final score is measured on late 2021, exactly as it would be used in real life.",
    tech: "Train 2020 Q1–Q4 · validation 2021 Q1–Q2 (tuning of thresholds, calibration, champion choice) · test 2021 Q3–Q4 (reported once).",
  },
  champion: { title: "Champion", plain: "The model chosen to run the queue. Picked on early-2021 data, never on the test quarters.", tech: "argmax validation PR-AUC over all candidates; test metrics reported afterwards." },
  safe: {
    title: "Safe to deprioritise",
    plain: "The share of alerts at the bottom of the queue that can wait while still keeping 95% of the reportable ones in view.",
    tech: "Cut chosen on validation so recall ≥ 0.95 of the calibrated score; share of test alerts below the cut.",
  },
  eps: { title: "Weighted error εₘ", plain: "How often round m's rule is wrong, counting heavier alerts more.", tech: "εₘ = Σₙ wₙ I(yₘ(xₙ) ≠ tₙ) / Σₙ wₙ." },
  alpha: { title: "Say of the rule αₘ", plain: "How much round m's rule counts in the final vote. Accurate rules count more.", tech: "αₘ = ln((1 − εₘ)/εₘ); in the committee fₘ(x) = ½ Σ αₗ yₗ(x)." },
  exp_loss: { title: "Exponential loss", plain: "The quantity AdaBoost is secretly minimising; it falls every round.", tech: "E = (1/N) Σₙ exp(−tₙ fₘ(xₙ)), tₙ ∈ {−1, +1}." },
  ess: { title: "Effective sample size", plain: "How many alerts still really matter. As boosting focuses on hard cases, fewer alerts carry most of the weight.", tech: "ESS = 1 / Σ wₙ² (weights summing to 1), shown as a share of N." },
};
