# FraudLens — Explainable Machine Learning for Credit Card Fraud Detection & Investigation

**Detect, Explain and Review Suspicious Credit Card Transactions**

Project and interview refresher.

Read sections 1–2 before a quick screening call. Read the rest before a technical interview, then open the [learning notebook](fraud_detection.ipynb) to revisit the actual experiments.

This summary follows the saved notebook outputs, fitted model, and current application. Where the original learning notes contain an oversimplification, use the more precise explanation here.

## 1. Explain the project in 45 seconds

> I built a credit card fraud classification project around a highly imbalanced, anonymized dataset. I explored data quality and feature distributions, compared Logistic Regression, XGBoost, and Gaussian Naive Bayes, and experimented with several validation strategies. My final XGBoost pipeline uses class weighting, randomized hyperparameter search, and a threshold selected from out-of-fold development predictions. On the recorded test partition, it achieved about 79.90% precision and 75.36% recall. I then built FraudLens, a Streamlit investigation workflow where users can explore alerts, inspect SHAP explanations, examine false positives and false negatives, and record simulated review decisions. The notebook preserves the experiments and their limitations, including repeated consultation of the test set.

The strongest story is the sequence of decisions: **understand the data → establish baselines → question the validation → investigate errors → explain predictions → make the results usable.**

## 2. Facts to remember

- **Task:** binary classification; `Class=1` means fraud, `Class=0` means legitimate.
- **Dataset:** 284,807 original transactions; 1,081 exact duplicates removed, leaving 283,726 rows and 473 frauds, approximately **0.167%** of cleaned rows.
- **Split:** chronological by existing row order, 50/50 after deduplication; **141,863 rows each** for training and test. Test contains **211 frauds**.
- **Inputs to the model:** `V1`–`V28`, `Amount`, and derived `Hour`; 30 features in total.
- **Final estimator:** saved XGBoost classifier inside a scikit-learn preprocessing pipeline.
- **Saved threshold:** exactly `np.float64(0.97)` before display formatting; inference uses the loaded value without rounding.
- **Recorded test results:** average precision **0.7821**, ROC-AUC **0.9713**, precision **79.90%**, recall **75.36%**, F1 **0.7756**.
- **Confusion counts:** **159 TP, 40 FP, 52 FN, 141,612 TN**. The model flags 199 transactions, of which 159 are labeled fraud.
- **Demo sample:** 1,000 real test-partition rows, deliberately curated to include model successes and errors. Its metrics are separate from the full benchmark.

Numbers come from [saved metrics](model_artifacts/metrics.json), [supporting provenance](model_artifacts/fraudlens_assets.json), and the notebook. The benchmark is a development result, not a claim about future bank transactions.

## 3. Data understanding and feature engineering

### What the columns mean

`V1`–`V28` are supplied PCA-derived features. I did **not** fit that PCA transformation, and I do not know the original behavioral meaning of each component. `Amount` is transaction amount; `Time` is elapsed seconds from the first transaction in the source data.

The application requires this original feature representation. A different banking dataset cannot become compatible simply by renaming its columns to `V1`, `V2`, and so on.

### What the EDA covered

1. Checked shape, types, missing values, and duplicate rows.
2. Examined severe class imbalance and feature distributions.
3. Inspected feature means, variances, correlations, and extreme standardized values.
4. Compared Amount distributions by class, using a logarithmic plot axis to make the distributions readable.
5. Ranked PCA features using `(fraud median − legitimate median) / overall standard deviation`. The leading separators were **V14, V17, V12, V10, and V16**.
6. Examined feature correlations and the distribution of elapsed-cycle hours by class.

These were exploratory findings, not proof that any feature causes fraud. All 28 PCA features remained in the model. The log axis was a visualization choice; it did not introduce a log-Amount feature into the pipeline.

### The engineered feature

`Hour = floor(Time / 3600) % 24`

The original Time column was then dropped from model inputs. This exposes a repeating 24-hour position, although the integer encoding still has a discontinuity between 23 and 0. Sine/cosine encoding was not implemented.

**Important:** the timestamp origin is not documented as midnight, so Hour 2 does not establish “2 AM.” Separately normalized KDE curves show distributions within each class; they do not directly estimate the probability of fraud given an hour.

Exact duplicates were removed during research. Incoming duplicate transactions are preserved by the app, because distinct input rows need distinct review references. Without real transaction identifiers, identical feature values alone cannot establish whether a live transaction is an accidental duplicate.

## 4. Preprocessing and why the pipeline matters

The fitted pipeline performs:

1. **RobustScaler on Amount and Hour:** subtract the training median, then divide by the training interquartile range.
2. **Pass through V1–V28.**
3. **Apply XGBoost** to obtain scores.
4. **Apply the saved threshold** to convert scores into flags.

`ColumnTransformer` routes columns to transformations; `Pipeline` keeps preprocessing and estimation together. During cross-validation, each cloned pipeline fits preprocessing using its training fold. Inference only transforms data with the already fitted preprocessing.

Robust scaling reduces the influence of extreme values on the scaling statistics. It does not remove outliers. Scaling is useful for the Logistic Regression baseline; tree models generally do not require it. The final XGBoost pipeline retains the original shared preprocessing design.

The fitted transformer outputs **scaled Amount, scaled Hour, then V1–V28**. That differs from the incoming column order and matters critically for labeling SHAP explanations.

Pipeline discipline prevents preprocessing leakage within fitting, but it does not erase the separate limitation that EDA and model comparisons consulted the test data.

## 5. Baselines and what they taught me

The notebook evaluated three pipelines at their default classification decision rule:

- **Logistic Regression:** AP 0.7691; precision 3.20%; recall 90.05%; F1 0.0618. It found 190 frauds but produced **5,744 false positives**. It used `class_weight="balanced"`.
- **Baseline XGBoost:** AP 0.7736; precision 69.75%; recall 78.67%; F1 0.7394. It found 166 frauds with **72 false positives**. It used `scale_pos_weight=576`.
- **Gaussian Naive Bayes:** AP 0.0674; precision 4.82%; recall 82.46%; F1 0.0911. It produced **3,434 false positives**, despite ROC-AUC 0.9652.

XGBoost offered a useful starting tradeoff on this dataset. Boosted trees can capture nonlinear relationships and feature interactions. This experiment does not establish that XGBoost is always best or that a separately threshold-tuned Logistic Regression model could not improve its operating point.

Class weighting changes the training objective so fraud examples receive more weight. The project did **not** use SMOTE, oversampling, or undersampling. The XGBoost weight 576 is a configured value explored during tuning; do not describe it as the exact cleaned-training class ratio.

## 6. The tuning story, including the unsuccessful attempt

### Validation experiments

- **Shuffled KFold:** examined variation in fraud counts across folds.
- **StratifiedKFold:** maintained more consistent class proportions, while still mixing chronology when shuffled.
- **TimeSeriesSplit:** trained on earlier rows and validated on later rows using expanding windows.
- **Final GroupKFold:** five folds grouped by Hour. Each Hour group is held out in one fold, but the folds do not enforce training-before-validation chronology. The configured splitter uses `shuffle=False`. See [GroupKFold's grouping semantics](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).

### First attempt

The first randomized search tested **15 configurations across five TimeSeriesSplit folds**, optimizing average precision. Its best cross-validation average precision (CV AP) was **0.7831**.

The subsequent threshold scan used predictions on the same training rows the fitted model had learned from. It selected **0.57** and reported training F1 **1.0000**. That was an optimistic in-sample result. On the test partition, AP was **0.7654**, F1 **0.7124**, and false positives increased to **89**.

The lesson: a perfect training score is not evidence of generalization. Also, threshold selection cannot explain a change in AP; AP depends on the model's score ranking.

### Second attempt and final model

The revised search used Hour-based GroupKFold, still with 15 configurations and five folds. It additionally searched `gamma`, `min_child_weight`, and class weighting, and widened the row/column sampling ranges. Best CV AP was **0.8508**.

It then used `cross_val_predict(..., method="predict_proba")` for out-of-fold development scores. Each row's score came from a fold model that was not fitted on that row. Scanning thresholds from **0.01 to 0.99** selected **0.97**: recorded development F1 rose from **0.8315** at 0.50 to **0.8760**.

The final saved estimator was the search's best pipeline refitted on the full training partition. It is not an ensemble of the fold models used to generate those development scores.

**Why not use `cross_val_predict` directly with TimeSeriesSplit?** Its validation folds do not cover the earliest training rows. The helper requires a partition that gives every row one held-out prediction. A manual forward-validation loop could still collect predictions for eligible later rows; abandoning temporal validation was not the only option.

**Limits of the improvement:** hyperparameter selection reused the development folds later used for threshold selection, and the test partition had already been inspected. GroupKFold, extra search parameters, and threshold selection changed together, so the experiment does not isolate which change caused each improvement.

### Saved hyperparameters, for a deeper follow-up

Values below are rounded for reading; the serialized model contains the full values:

- `n_estimators=302`, `max_depth=7`, `learning_rate≈0.119799`.
- `gamma≈0.475357`, `min_child_weight=5`.
- `subsample≈0.409196`, `colsample_bytree≈0.562178`.
- `scale_pos_weight=576`, `random_state=42`, `objective="binary:logistic"`.

Depth controls tree complexity; learning rate shrinks each boosting update; row and column subsampling introduce variation. Gamma requires a minimum split-loss reduction. **`min_child_weight` constrains the sum of Hessians in a child, not simply a minimum number of transactions.** These settings can regularize learning but do not guarantee that overfitting disappears. See the [official XGBoost parameter definitions](https://xgboost.readthedocs.io/en/stable/parameter.html).

## 7. Explain the metrics and the tradeoff

- **Precision = TP / (TP + FP):** of flagged transactions, how many are fraud? Here, `159 / 199 ≈ 79.90%`.
- **Recall = TP / (TP + FN):** of actual frauds, how many did the model find? Here, `159 / 211 ≈ 75.36%`.
- **F1 = 2TP / (2TP + FP + FN):** balances precision and recall, but does not encode transaction losses or analyst costs.
- **Average precision:** summarizes precision across recall increases. This is the quantity called PR-AUC in the project; it differs from trapezoidal PR-curve area. See [scikit-learn's AP definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
- **ROC-AUC:** measures ranking discrimination across false-positive and true-positive rates. It remains useful, but a strong value alone does not tell me the workload created by false alerts.

Predicting every cleaned transaction as legitimate would achieve about **99.83% accuracy while finding zero frauds**. That is why accuracy is not the headline metric.

Compared with baseline XGBoost, the final configuration reduced false positives **72 → 40**, while detected frauds fell **166 → 159** and missed frauds rose **45 → 52**. Precision and F1 improved; recall fell. This is a concrete tradeoff to explain in an interview.

Changing the threshold changes flags, precision, recall, and F1. It does not change the underlying scores, AP, or ROC-AUC. Raising a threshold reduces the set of flagged rows; precision is not guaranteed to increase at every step on a finite dataset.

The selected threshold maximizes development F1. It is not established as the best threshold for a bank's financial costs or review capacity. A model score of 0.97 also does not establish a calibrated 97% chance of fraud.

## 8. SHAP and reference statistics

SHAP explains how features contribute to an individual model output relative to the explainer's baseline. For this tree explanation:

`baseline + sum(feature contributions) ≈ raw model score in log-odds`

Positive contributions raise that raw score; negative contributions lower it. The waterfall bars are not percentage-point changes in fraud probability. Global importance averages absolute SHAP contributions over a deterministic sample of at most 1,000 loaded rows.

The upgrade corrected labels using `get_feature_names_out()` from the fitted transformer. Assigning original input names to transformed columns could produce a convincing-looking explanation of the wrong features. Checks verify the feature mapping, reconstruction of raw model scores, and unchanged predictions after explanation computation.

The workspace also compares raw feature values against medians and percentile bands from **141,601 legitimate training transactions**. These statistics are descriptive context, not additional model inputs or a second fraud classifier. Neither SHAP nor unusual percentiles establish causality or reveal what an anonymized PCA feature means in the real world.

## 9. What FraudLens adds

The workflow is **Overview → Transaction Explorer → Investigation Workspace → Model Intelligence**.

- Load the sample or a compatible CSV; validate it before replacing the current analysis.
- Search/filter actual scores and model flags; preserve row identity through sorting and filtering.
- Investigate a selected transaction using its score, SHAP, reference bands, and optional known label.
- Record Approve, Escalate, or Block with notes and revision history.
- Export scored rows and the session audit log; return to the notebook for the research story.

Model outputs, dataset labels, and analyst judgments remain separate. Decisions are simulations: they do not freeze cards, change labels, or retrain the model. Records live in the current session and must be downloaded before it ends.

The curated sample contains **20 TP, 20 FP, 20 FN, and 940 TN**, selected and shuffled with seed 42. Its fraud prevalence is deliberately inflated. Fixed, verified walkthrough cases show one successful detection, one false alarm, and one missed fraud. The persistent warning and exported provenance prevent confusing this demonstration with representative evaluation.

Uploads require `V1`–`V28`, Amount, and Time or Hour; Class is optional. The app checks numeric/finite values, missing required inputs, duplicate headers, and Time/Hour consistency, with limits of **200 MB and 300,000 rows**. Content hashes distinguish changed files even when filenames match. Without usable Class labels, it can report flags but cannot measure precision or recall.

## 10. Architecture and evidence

- [Notebook](fraud_detection.ipynb): EDA, experiments, validation, searches, threshold selection, and learning notes.
- [app.py](app.py): Streamlit interface and navigation.
- [fraudlens/core.py](fraudlens/core.py): validation, saved-model inference, evaluation, and exports.
- [fraudlens/plots.py](fraudlens/plots.py): Matplotlib/seaborn charts and SHAP presentation.
- [fraudlens/records.py](fraudlens/records.py): session decisions, revisions, and audit export.
- [Preparation utility](scripts/prepare_demo_assets.py): reconstructs the original split and creates small assets without fitting anything.
- [Acceptance checks](tests/run_checks.py): artifact hashes, preserved notebook code/outputs, benchmark reproduction, SHAP, input handling, exports, and Streamlit user journeys.

The stack is Python, Streamlit, pandas, NumPy, scikit-learn, XGBoost, SHAP, Matplotlib, seaborn, Joblib, and the notebook's existing SciPy usage. No separate API server, frontend framework, database, or new third-party dependency was introduced for FraudLens.

The upgrade preserved the fitted models, threshold artifacts, training/tuning code, and numerical results. Notebook edits were limited to corrected final SHAP labels/output, future export paths, accuracy notes, and an application appendix.

## 11. Limitations and sensible next steps

Be ready to explain these plainly:

- **Evaluation reuse:** EDA used the full dataset and test results influenced development. A fresh chronological holdout, with separate development tuning, would provide stronger evidence.
- **Validation mismatch:** grouping by Hour is not equivalent to predicting future transactions. A manual forward-validation procedure is a reasonable next experiment.
- **Limited feature meaning:** anonymization prevents merchant, customer, or geographic explanations; compatibility with arbitrary banking data is unproven.
- **Uncalibrated scores and F1-based decisions:** calibration assessment and cost/capacity-aware threshold selection would be future work, using separate validation data.
- **Short historical coverage:** the project has not established performance under later changes in fraud behavior or transaction distributions.
- **Application scope:** session storage and simulated decisions support a portfolio demo. Persistent multi-user case management, access controls, monitoring, and banking integration are not implemented.

These are possible extensions, not features to claim on a resume. My next priority would be stronger evaluation before adding more infrastructure.

## 12. Interview pitfalls to avoid

- **“I performed PCA.”** The dataset supplied the components; inspecting their statistics does not reconstruct or independently certify the original transformation.
- **“Uncorrelated features are independent or have unit variance.”** Neither follows generally. PCA components can have different variances, and Gaussian Naive Bayes makes stronger distributional assumptions.
- **“Logistic Regression cannot work with outliers or correlated inputs.”** They can affect stability, coefficients, and regularization; they are not automatic prohibitions.
- **“Pearson correlation cannot use a binary target.”** With a binary variable it corresponds to point-biserial correlation. The median-gap measure was an exploratory choice, not a mathematical necessity.
- **“GroupKFold eliminated all leakage.”** It separates groups; the chronology and development-reuse caveats remain.
- **“The final model catches more fraud than every baseline.”** Its precision/F1 tradeoff improved over baseline XGBoost while recall decreased.
- **“97% means a verified fraud probability.”** It is a model score threshold; calibration was not established.
- **“Block stops the transaction.”** It only records a simulated decision.

## 13. Resume wording and a five-minute demo

Possible resume bullets, adjusted to the role and the work I can explain confidently:

- Developed an imbalanced credit card fraud classification pipeline comparing three classifiers, with XGBoost tuning and out-of-fold threshold selection; recorded 0.7821 average precision, 79.90% precision, and 75.36% recall on a 141,863-row chronological test partition.
- Built a Streamlit investigation workflow with validated CSV inference, SHAP explanations, training-reference comparisons, simulated analyst review history, and a reproducible walkthrough of successful detections and model errors.

**Demo order:**

1. Open FraudLens and choose **Load Sample and Start Walkthrough**; explain the sample warning.
2. Show the true positive. Separate the model result from the known label and explain one SHAP contribution.
3. Show the false positive and false negative. Discuss review workload and missed frauds.
4. Save a simulated decision and point to the downloadable audit history.
5. Open Model Intelligence, explain the benchmark tradeoff and validation limitation, then show the notebook's unsuccessful and final tuning attempts.

To run the app from the repository root: `.\.venv\Scripts\python.exe -m streamlit run app.py`.

Before an interview, practice answering: **Why this metric? Why this split? What went wrong initially? What does the threshold do? How do explanations stay aligned with preprocessing? What evidence would I need before using this beyond a demo?**
