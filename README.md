# Sentinel — Explainable Credit Card Fraud Intelligence

A Streamlit investigation workspace built around a documented credit card fraud ML project. Explore real transactions, inspect SHAP explanations, examine a false alarm or a missed fraud, and record a simulated analyst decision.

**Start with the [learning notebook](fraud_detection.ipynb).** It preserves the original EDA, baseline models, validation experiments, unsuccessful tuning attempt, final XGBoost search, threshold selection, and saved numerical results.

## Try the application

Validated with Python 3.14 and the package versions in requirements.txt.

    python -m pip install -r requirements.txt
    python -m streamlit run app.py

Choose **Load Sample and Start Walkthrough**. The three short cases show:

1. A true positive: a fraudulent transaction correctly flagged.
2. A false positive: a legitimate transaction flagged by the model.
3. A false negative: a fraudulent transaction below the saved threshold.

Each case has actual model scores, known dataset labels, SHAP contributions, and reference comparisons. The walkthrough ends at Model Intelligence, which connects the application back to the notebook.

The app can run with the bundled small assets; the full dataset is needed only to rebuild them or repeat the original research.

## What the application provides

- **Overview:** active transaction counts, flag rate, review progress, score distribution, and the highest-scoring alerts.
- **Transaction Explorer:** search and filter by generated row ID, score, model result, amount, elapsed time/hour, and review status. Select a row to investigate or export all scored transactions.
- **Investigation Workspace:** risk assessment, SHAP waterfall, legitimate-training reference bands, known dataset label when supplied, and a separate simulated analyst decision.
- **Model Intelligence:** original benchmark metrics, recorded baseline comparison, PR/ROC curves, methodology notes, optional uploaded-data evaluation, and global SHAP.

Approve, Escalate, and Block are simulation records. They do not perform banking actions, modify ground-truth labels, or update the model. Notes and revision history are held in the current session; download the CSV audit log before leaving. Earlier datasets' review events remain in that session's log.

## Curated demonstration sample

> **Curated demonstration sample. Fraud and model-error cases are deliberately overrepresented. This sample does not represent real fraud prevalence.**

The bundled 1,000-row sample contains **20 true positives, 20 false positives, 20 false negatives, and 940 true negatives**. It is drawn from the existing test partition, using seed 42 for selection and shuffling. Selection intentionally uses the known labels and model outcomes.

This makes all three walkthrough cases easy to explore. Sample metrics describe this selected sample only; they do not estimate population fraud prevalence or replace the full benchmark. The app keeps this notice visible across pages and includes provenance in scored-data and decision-log exports.

Generated TXN IDs identify input rows within an analysis. They are not real bank-issued transaction identifiers.

## Supported CSV input

The model requires the original anonymized PCA feature representation:

- All columns V1 through V28.
- Amount.
- Time (non-negative elapsed seconds) or Hour (an integer from 0 to 23).
- Optional Class labels: 0 for legitimate, 1 for fraud.
- Additional columns are retained but excluded from inference.

The saved model's feature list determines the actual inference schema. Matching column names alone cannot establish that a different dataset uses the same PCA representation. Arbitrary banking datasets are not supported.

The app derives Hour as floor(Time / 3600) modulo 24. If both columns are present, they must agree. Hour is an elapsed-cycle feature; the source data does not establish a real wall-clock origin.

Files are limited to **200 MB and 300,000 rows**. Required inputs are checked for missing, non-numeric, and infinite values. Duplicate transactions are preserved during inference. Duplicate column headers and empty files are rejected.

Uploads are identified by content hash. A valid new file replaces the active analysis and clears its selection and filters. A failed upload leaves the prior analysis active. Invalid or missing Class values disable evaluation while predictions and investigations remain available.

CSV exports preserve original columns and add sentinel_ fields; if an original column already uses an output name, the added column receives a numeric suffix.

## Original ML results and preservation

The original pipeline uses RobustScaler on Amount and Hour, passes through V1–V28, then applies the tuned XGBoost classifier.

The saved threshold was verified as **np.float64(0.97) before any display formatting**. The app always uses the loaded scalar directly. The original model, threshold, and saved metrics artifacts are unchanged.

On the original 141,863-row test partition:

- Average precision, labeled PR-AUC: **0.7821377376**
- ROC-AUC: **0.9712782866**
- Precision: **79.90%**
- Recall: **75.36%**
- F1: **0.7756097561**
- True positives: **159**; false positives: **40**
- True negatives: **141,612**; false negatives: **52**

The app's baseline comparison comes from the saved notebook outputs. Uploaded-data and curated-sample evaluation are explicitly separate from this benchmark.

The fitted preprocessing output order is **scaled Amount, scaled Hour, V1–V28**. SHAP labels follow that fitted order. Contributions are in log-odds; raw feature reference comparisons use the original feature units. Scores are not presented as calibrated fraud probabilities.

## Methodology and learning notes

The notebook removes exact duplicate rows and uses a chronological 50/50 split by row count. Its boundary is at elapsed seconds 84,692 / 84,693, not exactly at a calendar-day boundary.

It compares Logistic Regression, XGBoost, and Naive Bayes; examines several validation methods; and ultimately tunes XGBoost using GroupKFold grouped by Hour. Group isolation does not enforce chronological forward validation. The threshold is selected by maximizing F1 over out-of-fold development predictions.

The full dataset was used for EDA, and the test partition was inspected during model development. Hyperparameter selection and threshold evaluation reused the development folds. The reported metrics are educational development results, not an untouched prospective evaluation.

The notebook retains this history. The upgrade adds short accuracy notes, corrects the final SHAP figure's feature labels, aligns future export paths with the app, and appends an application handoff section. It does not change training or tuning code.

## Rebuild the supporting assets

Place the original creditcard.csv in data/ and run:

    python scripts/prepare_demo_assets.py

This inference-only utility reconstructs the notebook's deduplication and split, checks the existing benchmark, and writes:

- data/sample_transactions.csv — the curated demonstration rows.
- model_artifacts/sentinel_assets.json — provenance, fixed walkthrough cases, reference bands, benchmark curve points, and recorded baseline outputs.

Reference bands use only the **141,601 legitimate training transactions**. The utility never fits a model and verifies that original artifact hashes remain unchanged. The notebook's final optional cell invokes the same utility.

The large original CSV remains excluded from Git. The small sample is explicitly included.

## Verification

With the original dataset available:

    python tests/run_checks.py

The checks use Python assertions, standard-library helpers, and the existing Streamlit AppTest tooling. They verify artifact and notebook preservation, the full benchmark, sample provenance, feature mapping and SHAP additivity, input validation, decision revisions, export provenance, the three-case walkthrough, filtered-row selection, and dataset replacement.

No additional testing framework or frontend framework is required.

## Project layout

- fraud_detection.ipynb — the original ML learning record and application handoff.
- app.py — Streamlit entry point and four-page user journey.
- sentinel/ — validation/inference, session records, plots, and styling.
- scripts/prepare_demo_assets.py — reproducible supporting-asset preparation.
- model_artifacts/ — original fitted model, threshold, metrics, and new supporting metadata.
- tests/ — original preservation fingerprints and acceptance checks.
- documenting_process.md — original process notes with an accuracy addendum.
