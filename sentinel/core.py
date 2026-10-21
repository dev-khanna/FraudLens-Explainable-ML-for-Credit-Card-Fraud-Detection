"""Validated inference and evaluation using the original, fitted pipeline."""

from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import io
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, precision_recall_curve,
    roc_auc_score, roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "model_artifacts"
SAMPLE_PATH = ROOT / "data" / "sample_transactions.csv"
MAX_BYTES = 200 * 1024 * 1024
MAX_ROWS = 300_000
NOTEBOOK_URL = (
    "https://github.com/dev-khanna/credit-card-fraud-detection/"
    "blob/main/fraud_detection.ipynb"
)
SAMPLE_NOTICE = (
    "Curated demonstration sample. Fraud and model-error cases are deliberately "
    "overrepresented. This sample does not represent real fraud prevalence."
)


def digest_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def file_digest(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@dataclass
class ModelBundle:
    model: object
    threshold: float
    features: list
    output_names: list
    output_features: list
    model_hash: str
    threshold_hash: str


def load_bundle():
    model_path = ARTIFACTS / "final_xgboost_model.pkl"
    threshold_path = ARTIFACTS / "final_threshold.pkl"
    model = joblib.load(model_path)
    # Convert the saved scalar without rounding or substituting a UI constant.
    threshold = float(joblib.load(threshold_path))
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The saved decision threshold is invalid.")
    features = list(model.feature_names_in_)
    prep = model.named_steps["preprocessor"]
    if list(prep.feature_names_in_) != features:
        raise ValueError("Pipeline and fitted preprocessor feature orders disagree.")
    output_names = list(prep.get_feature_names_out())
    output_features = [name.split("__", 1)[-1] for name in output_names]
    if len(output_features) != len(features) or set(output_features) != set(features):
        raise ValueError("The fitted feature mapping is unsupported.")
    return ModelBundle(
        model, threshold, features, output_names, output_features,
        file_digest(model_path), file_digest(threshold_path),
    )


def load_supporting(bundle):
    support = json.loads((ARTIFACTS / "sentinel_assets.json").read_text(encoding="utf-8"))
    if (
        support["model_sha256"] != bundle.model_hash
        or support["threshold_sha256"] != bundle.threshold_hash
        or support["threshold_hex"] != bundle.threshold.hex()
    ):
        raise ValueError("Supporting data belongs to a different model. Regenerate the demo assets.")
    return support


def read_csv_bytes(payload):
    if len(payload) > MAX_BYTES:
        raise ValueError("The CSV exceeds the 200 MB upload limit.")
    if not payload.strip():
        raise ValueError("The CSV is empty. Include a header and at least one transaction.")
    try:
        text = io.TextIOWrapper(io.BytesIO(payload), encoding="utf-8-sig", newline="")
        header = next(csv.reader(text))
        if not header or any(not name.strip() for name in header):
            raise ValueError("Every CSV column needs a non-empty name.")
        if len(header) != len(set(header)):
            raise ValueError("Duplicate column names are not supported. Give each column a unique name.")
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ParserWarning)
            frame = pd.read_csv(
                io.BytesIO(payload), nrows=MAX_ROWS + 1, encoding="utf-8-sig",
                index_col=False,
            )
    except (UnicodeError, csv.Error, pd.errors.ParserError, pd.errors.ParserWarning, pd.errors.EmptyDataError, StopIteration) as exc:
        raise ValueError("Could not read the CSV. Use a valid UTF-8, comma-separated file.") from exc
    if frame.empty:
        raise ValueError("The CSV has headers but no transactions.")
    if len(frame) > MAX_ROWS:
        raise ValueError("The CSV exceeds the 300,000-transaction limit.")
    return frame.reset_index(drop=True)


def numeric_column(frame, name):
    column = frame[name]
    if pd.api.types.is_bool_dtype(column):
        raise ValueError(f"'{name}' must contain numbers, not boolean values.")
    values = pd.to_numeric(column, errors="coerce")
    invalid = ~np.isfinite(values.to_numpy(dtype=float))
    if invalid.any():
        rows = ", ".join(str(i + 1) for i in np.flatnonzero(invalid)[:5])
        raise ValueError(f"'{name}' contains missing, non-numeric, or infinite values at data rows {rows}.")
    return values


def prepare_features(frame, feature_columns):
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names are not supported.")
    missing = [name for name in feature_columns if name != "Hour" and name not in frame]
    if missing:
        raise ValueError("Missing required model features: " + ", ".join(missing))
    if "Time" not in frame and "Hour" not in frame:
        raise ValueError("Include 'Time' (elapsed seconds) or a compatible 'Hour' column.")
    time = numeric_column(frame, "Time") if "Time" in frame else None
    if time is not None and (time < 0).any():
        raise ValueError("'Time' must be non-negative elapsed seconds.")
    derived = (time // 3600) % 24 if time is not None else None
    if "Hour" in frame:
        hour = numeric_column(frame, "Hour")
        if ((hour < 0) | (hour > 23) | (hour != np.floor(hour))).any():
            raise ValueError("'Hour' must contain whole numbers from 0 to 23.")
        if derived is not None and not np.array_equal(hour.to_numpy(), derived.to_numpy()):
            raise ValueError("'Hour' disagrees with floor(Time / 3600) modulo 24.")
    else:
        hour = derived
    features = pd.DataFrame(index=frame.index)
    for name in feature_columns:
        features[name] = hour if name == "Hour" else numeric_column(frame, name)
    if "Amount" in features and (features["Amount"] < 0).any():
        raise ValueError("'Amount' must be non-negative.")
    return features


@dataclass
class Analysis:
    dataset_id: str
    source: str
    curated: bool
    raw: pd.DataFrame
    features: pd.DataFrame
    scores: np.ndarray
    flags: np.ndarray
    row_ids: list

    @property
    def provenance(self):
        if self.curated:
            return (
                SAMPLE_NOTICE + " Source: the original chronological test partition after "
                "exact deduplication; selection and shuffling seed 42; 20 true positives, "
                "20 false positives, 20 false negatives, and 940 true negatives."
            )
        return "User-uploaded compatible transaction data; labels are not independently verified."


def score_csv(payload, filename, bundle, sample_hash=None):
    frame = read_csv_bytes(payload)
    features = prepare_features(frame, bundle.features)
    scores = bundle.model.predict_proba(features)[:, 1]
    if not np.isfinite(scores).all():
        raise ValueError("The model returned non-finite scores.")
    dataset_id = digest_bytes(payload)
    return Analysis(
        dataset_id, filename, dataset_id == sample_hash, frame, features, scores,
        scores >= bundle.threshold,
        [f"TXN-{i + 1:06d}" for i in range(len(frame))],
    )


def get_labels(frame):
    if "Class" not in frame:
        return None, "No Class column supplied. Scores and investigations are available; evaluation needs ground-truth labels."
    labels = pd.to_numeric(frame["Class"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        return None, "Class must contain a 0 or 1 for every row. Evaluation is unavailable; inference remains available."
    return labels.to_numpy(dtype=int), None


def evaluate_scores(labels, scores, threshold):
    flags = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, flags, labels=[0, 1]).ravel()
    both_classes = len(np.unique(labels)) == 2
    return {
        "average_precision": float(average_precision_score(labels, scores)) if both_classes else None,
        "roc_auc": float(roc_auc_score(labels, scores)) if both_classes else None,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "transactions": int(len(labels)),
    }


def benchmark_metrics(path=None):
    data = json.loads((path or ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
    mapping = {
        "average_precision": "FINAL_PR_AUC", "roc_auc": "FINAL_ROC_AUC",
        "precision": "FINAL_PRECISION", "recall": "FINAL_RECALL", "f1": "FINAL_F1",
        "tn": "FINAL_TN", "fp": "FINAL_FP", "fn": "FINAL_FN", "tp": "FINAL_TP",
    }
    result = {key: data[value] for key, value in mapping.items()}
    result["transactions"] = sum(result[k] for k in ["tn", "fp", "fn", "tp"])
    return result


def curve_data(labels, scores, limit=2000):
    if len(np.unique(labels)) < 2:
        return None
    precision, recall, _ = precision_recall_curve(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)

    def points(x, y):
        indices = np.unique(np.linspace(0, len(x) - 1, min(limit, len(x)), dtype=int))
        return {"x": np.asarray(x)[indices].tolist(), "y": np.asarray(y)[indices].tolist()}

    return {"pr": points(recall, precision), "roc": points(fpr, tpr)}


def explanation_values(bundle, explainer, features):
    transformed = bundle.model.named_steps["preprocessor"].transform(features[bundle.features])
    values = np.asarray(explainer.shap_values(transformed))
    if values.shape != transformed.shape:
        raise ValueError("The explanation shape does not match the fitted feature mapping.")
    return transformed, values


def result_table(analysis, latest=None):
    latest = latest or {}
    table = pd.DataFrame({
        "Row ID": analysis.row_ids,
        "Amount": analysis.features["Amount"].to_numpy(),
        "Risk score": analysis.scores,
        "Model result": np.where(analysis.flags, "Flagged", "Not flagged"),
        "Review": [latest.get(row_id, "Unreviewed") for row_id in analysis.row_ids],
        "_position": np.arange(len(analysis.raw)),
    })
    if "Time" in analysis.raw:
        table["Elapsed seconds"] = pd.to_numeric(analysis.raw["Time"]).to_numpy()
    table["Hour"] = analysis.features["Hour"].to_numpy()
    return table


def scored_export(analysis, threshold, latest=None):
    result = analysis.raw.copy()
    latest = latest or {}
    fields = {
        "sentinel_row_id": analysis.row_ids,
        "sentinel_risk_score": analysis.scores,
        "sentinel_model_result": np.where(analysis.flags, "Flagged", "Not flagged"),
        "sentinel_threshold": threshold,
        "sentinel_review": [latest.get(r, "Unreviewed") for r in analysis.row_ids],
        "sentinel_dataset_sha256": analysis.dataset_id,
        "sentinel_provenance": analysis.provenance,
    }
    for name, value in fields.items():
        # Preserve original columns even if a scored export is uploaded again.
        unique = name
        suffix = 2
        while unique in result:
            unique = f"{name}_{suffix}"
            suffix += 1
        result[unique] = value
    return result.to_csv(index=False).encode("utf-8")
