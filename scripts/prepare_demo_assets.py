"""Build small demo assets from the original dataset and saved model; never fit."""

import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fraudlens.core import (
    ARTIFACTS, SAMPLE_PATH, SAMPLE_NOTICE, benchmark_metrics, curve_data,
    evaluate_scores, file_digest, load_bundle, prepare_features,
)


def reconstruct_partitions(dataset_path):
    raw = pd.read_csv(dataset_path)
    clean = raw.drop_duplicates(keep="first")
    split = len(clean) // 2  # train_test_split(test_size=0.5, shuffle=False)
    return raw, clean, clean.iloc[:split].copy(), clean.iloc[split:].copy()


def recorded_comparisons():
    notebook = json.loads((ROOT / "fraud_detection.ipynb").read_text(encoding="utf-8"))
    rows = []
    for index, name in [(75, "Logistic Regression"), (77, "XGBoost baseline"), (80, "Naive Bayes")]:
        output = "".join(
            "".join(out.get("text", []))
            for out in notebook["cells"][index].get("outputs", [])
        )
        row = {"Model": name}
        for label in ["PR-AUC", "ROC-AUC", "Precision", "Recall", "F1 Score"]:
            match = re.search(re.escape(label) + r":\s+([0-9.]+)", output)
            if match is None:
                raise ValueError(f"Notebook cell {index} is missing its recorded {label}.")
            row[label] = float(match.group(1))
        rows.append(row)
    return rows


def build_assets(dataset_path=None):
    dataset_path = Path(dataset_path or ROOT / "data" / "creditcard.csv")
    bundle = load_bundle()
    protected = [
        ARTIFACTS / "final_xgboost_model.pkl", ARTIFACTS / "final_threshold.pkl",
        ARTIFACTS / "metrics.json", ROOT / "final_xgboost_pipeline.pkl",
        ROOT / "optimal_threshold.pkl",
    ]
    original_hashes = {str(p): file_digest(p) for p in protected}
    raw, clean, train, test = reconstruct_partitions(dataset_path)
    X_train = prepare_features(train, bundle.features)
    X_test = prepare_features(test, bundle.features)
    scores = bundle.model.predict_proba(X_test)[:, 1]
    labels = test["Class"].to_numpy()
    flags = scores >= bundle.threshold
    metrics = evaluate_scores(labels, scores, bundle.threshold)
    expected = benchmark_metrics()
    for key in expected:
        if not np.isclose(metrics[key], expected[key], rtol=1e-12, atol=1e-12):
            raise ValueError(f"Benchmark mismatch for {key}; no assets have been written.")
    buckets = {
        "true_positive": ((labels == 1) & flags, 20),
        "false_positive": ((labels == 0) & flags, 20),
        "false_negative": ((labels == 1) & ~flags, 20),
        "true_negative": ((labels == 0) & ~flags, 940),
    }
    selected = []
    for name, (mask, count) in buckets.items():
        positions = np.flatnonzero(mask)
        if len(positions) < count:
            raise ValueError(f"Not enough {name} rows for the agreed demo composition.")
        selected.extend(pd.Series(positions).sample(n=count, random_state=42).tolist())
    selected = pd.Series(selected).sample(frac=1, random_state=42).to_numpy()
    sample = test.iloc[selected].reset_index(drop=True)
    # Check the exported CSV round trip, not just the in-memory selection.
    sample_csv = sample.to_csv(index=False, float_format="%.17g", lineterminator="\n")
    from io import StringIO
    sample_roundtrip = pd.read_csv(StringIO(sample_csv))
    sample_scores = bundle.model.predict_proba(prepare_features(sample_roundtrip, bundle.features))[:, 1]
    sample_labels = sample_roundtrip["Class"].to_numpy()
    sample_flags = sample_scores >= bundle.threshold
    sample_metrics = evaluate_scores(sample_labels, sample_scores, bundle.threshold)
    assert [sample_metrics[k] for k in ["tp", "fp", "fn", "tn"]] == [20, 20, 20, 940]
    walkthrough = []
    for kind, title, label, flagged in [
        ("true_positive", "A successful detection", 1, True),
        ("false_positive", "A false alarm", 0, True),
        ("false_negative", "A missed fraud", 1, False),
    ]:
        candidates = np.flatnonzero((sample_labels == label) & (sample_flags == flagged))
        position = int(candidates[np.argmax(sample_scores[candidates])])
        walkthrough.append({
            "kind": kind, "title": title, "position": position,
            "row_id": f"TXN-{position + 1:06d}", "label": label,
            "flagged": flagged, "score": float(sample_scores[position]),
            "source_row_index": int(test.index[selected[position]]),
        })
    legitimate = X_train.loc[train["Class"].eq(0)]
    quantiles = legitimate.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
    reference = {
        feature: {key: float(quantiles.loc[q, feature]) for key, q in [
            ("p01", .01), ("p25", .25), ("median", .5), ("p75", .75), ("p99", .99)
        ]} for feature in bundle.features
    }
    support = {
        "schema_version": 1,
        "model_sha256": bundle.model_hash,
        "threshold_sha256": bundle.threshold_hash,
        "threshold_hex": bundle.threshold.hex(),
        "source_dataset_sha256": file_digest(dataset_path),
        "partition_method": "Drop exact duplicates, preserve row order, use first 50% for training and second 50% for test.",
        "raw_rows": len(raw), "clean_rows": len(clean),
        "training_rows": len(train), "test_rows": len(test),
        "last_training_time": float(train["Time"].iloc[-1]),
        "first_test_time": float(test["Time"].iloc[0]),
        "reference_rows": len(legitimate),
        "reference_source": "Legitimate transactions in the original training partition only.",
        "reference": reference,
        "output_feature_order": bundle.output_names,
        "sample_notice": SAMPLE_NOTICE,
        "sample_rows": len(sample),
        "sample_composition": {name: count for name, (_, count) in buckets.items()},
        "sample_seed": 42,
        "sample_source_row_indices": test.index[selected].astype(int).tolist(),
        "walkthrough": walkthrough,
        "benchmark": metrics,
        "benchmark_curves": curve_data(labels, scores),
        "curve_note": "Curves show up to 2,000 original points each; reported metrics use all test rows.",
        "recorded_baselines": recorded_comparisons(),
    }
    SAMPLE_PATH.parent.mkdir(exist_ok=True)
    SAMPLE_PATH.write_text(sample_csv, encoding="utf-8", newline="")
    support["sample_sha256"] = file_digest(SAMPLE_PATH)
    (ARTIFACTS / "fraudlens_assets.json").write_text(
        json.dumps(support, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline=""
    )
    assert original_hashes == {str(p): file_digest(p) for p in protected}
    return support


if __name__ == "__main__":
    assets = build_assets()
    print(f"Prepared {assets['sample_rows']} curated rows and {assets['reference_rows']} legitimate-training reference rows.")
    print("Original models, threshold artifacts, and saved benchmark metrics were not modified.")
