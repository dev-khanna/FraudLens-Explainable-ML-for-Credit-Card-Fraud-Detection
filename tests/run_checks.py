"""Run with python tests/run_checks.py. Uses existing dependencies and assertions."""

import hashlib
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fraudlens import core
from fraudlens.core import (
    SAMPLE_NOTICE, SAMPLE_PATH, benchmark_metrics, digest_bytes, evaluate_scores,
    explanation_values, get_labels, load_bundle, load_supporting, prepare_features,
    read_csv_bytes, result_table, score_csv, scored_export,
)
from fraudlens.records import append_decision, audit_export, latest_decisions
from scripts.prepare_demo_assets import reconstruct_partitions


def check_preservation():
    baseline = json.loads((ROOT / "tests/preservation_baseline.json").read_text())
    h = lambda value: hashlib.sha256(value).hexdigest()
    for name, expected in baseline["artifacts"].items():
        assert h((ROOT / name).read_bytes()) == expected, f"Original artifact changed: {name}"
    notebook = json.loads((ROOT / "fraud_detection.ipynb").read_text(encoding="utf-8"))
    for index, expected in baseline["code_sources"].items():
        assert h("".join(notebook["cells"][int(index)]["source"]).encode()) == expected, f"Original code changed: {index}"
    for index, expected in baseline["code_outputs"].items():
        assert h(json.dumps(notebook["cells"][int(index)].get("outputs", []), sort_keys=True).encode()) == expected, f"Original output changed: {index}"
    assert len(notebook["cells"]) == baseline["original_cell_count"] + 2
    for index in [143, 149, len(notebook["cells"]) - 1]:
        compile("".join(notebook["cells"][index]["source"]), f"notebook cell {index}", "exec")
    print("PASS original artifacts, experiments, numerical outputs, and notebook syntax", flush=True)


def expect_invalid(payload, bundle, contains=None):
    try:
        score_csv(payload, "invalid.csv", bundle)
    except ValueError as exc:
        if contains:
            assert contains in str(exc), str(exc)
        return
    raise AssertionError("Invalid input was accepted")


def check_inference():
    bundle = load_bundle()
    support = load_supporting(bundle)
    payload = SAMPLE_PATH.read_bytes()
    assert b"\r\n" not in payload, "Demo assets must use canonical LF bytes across Git checkouts."
    assert b"\r\n" not in (core.ARTIFACTS / "fraudlens_assets.json").read_bytes()
    analysis = score_csv(payload, "sample.csv", bundle, support["sample_sha256"])
    assert analysis.curated and len(analysis.raw) == 1000
    assert bundle.threshold == 0.97
    assert bundle.threshold.hex() == support["threshold_hex"]
    assert bundle.output_features == ["Amount", "Hour"] + [f"V{i}" for i in range(1, 29)]
    labels, _ = get_labels(analysis.raw)
    metrics = evaluate_scores(labels, analysis.scores, bundle.threshold)
    assert [metrics[k] for k in ["tp", "fp", "fn", "tn"]] == [20, 20, 20, 940]
    for case in support["walkthrough"]:
        pos = case["position"]
        assert labels[pos] == case["label"]
        assert bool(analysis.flags[pos]) == case["flagged"]
        assert analysis.row_ids[pos] == case["row_id"]
        assert np.isclose(analysis.scores[pos], case["score"], rtol=1e-6)
    boundary = np.array([np.nextafter(bundle.threshold, 0), bundle.threshold])
    exact = evaluate_scores(np.array([0, 1]), boundary, bundle.threshold)
    assert exact["tn"] == 1 and exact["tp"] == 1
    before = analysis.scores.copy()
    case_positions = [c["position"] for c in support["walkthrough"]]
    explainer = shap.TreeExplainer(bundle.model.named_steps["model"])
    transformed, values = explanation_values(bundle, explainer, analysis.features.iloc[case_positions])
    prep = bundle.model.named_steps["preprocessor"]
    assert np.array_equal(
        transformed[:, :2],
        prep.named_transformers_["scale"].transform(analysis.features.iloc[case_positions][["Amount", "Hour"]]),
    )
    assert np.array_equal(transformed[:, 2:], analysis.features.iloc[case_positions][bundle.output_features[2:]].to_numpy())
    margins = bundle.model.named_steps["model"].get_booster().predict(xgb.DMatrix(transformed), output_margin=True)
    assert np.allclose(values.sum(axis=1) + explainer.expected_value, margins, rtol=1e-5, atol=1e-5)
    assert np.array_equal(before, bundle.model.predict_proba(analysis.features)[:, 1])
    print("PASS exact threshold, curated cases, fitted feature order, SHAP additivity, unchanged predictions", flush=True)

    small = analysis.raw.head(3).copy()
    csv_bytes = lambda frame: frame.to_csv(index=False).encode()
    expected = score_csv(csv_bytes(small), "same.csv", bundle)
    unlabeled = score_csv(csv_bytes(small.drop(columns="Class")), "same.csv", bundle)
    assert get_labels(unlabeled.raw)[0] is None
    assert np.array_equal(expected.scores, unlabeled.scores)
    hour = small.copy()
    hour["Hour"] = (hour["Time"] // 3600) % 24
    assert np.array_equal(score_csv(csv_bytes(hour.drop(columns="Time")), "hour.csv", bundle).scores, expected.scores)
    duplicate = pd.concat([small, small.iloc[[0]]], ignore_index=True)
    assert len(score_csv(csv_bytes(duplicate), "duplicates.csv", bundle).raw) == 4
    invalid_label = small.copy()
    invalid_label["Class"] = "unknown"
    accepted = score_csv(csv_bytes(invalid_label), "bad_labels.csv", bundle)
    assert get_labels(accepted.raw)[0] is None
    assert np.array_equal(accepted.scores, expected.scores)
    for value in [np.nan, np.inf, -np.inf, "oops"]:
        invalid = small.copy()
        invalid["V1"] = invalid["V1"].astype(object)
        invalid.loc[0, "V1"] = value
        expect_invalid(csv_bytes(invalid), bundle, "V1")
    expect_invalid(csv_bytes(small.drop(columns="Amount")), bundle, "Amount")
    hour.loc[0, "Hour"] = (hour.loc[0, "Hour"] + 1) % 24
    expect_invalid(csv_bytes(hour), bundle, "disagrees")
    hour.loc[0, "Hour"] = 24
    expect_invalid(csv_bytes(hour), bundle, "0 to 23")
    expect_invalid(b"", bundle, "empty")
    expect_invalid(csv_bytes(small.head(0)), bundle, "no transactions")
    expect_invalid(b"V1,V1\n1,2\n", bundle, "Duplicate")
    expect_invalid(b"V1,V2\n1,2,3\n", bundle, "valid UTF-8")
    with patch.object(core, "MAX_BYTES", 10):
        expect_invalid(csv_bytes(small), bundle, "200 MB")
    with patch.object(core, "MAX_ROWS", 2):
        expect_invalid(csv_bytes(small), bundle, "300,000")
    changed = small.copy()
    changed.loc[0, "Amount"] += 1
    replacement = score_csv(csv_bytes(changed), "same.csv", bundle)
    assert replacement.dataset_id != expected.dataset_id
    single = evaluate_scores(np.array([0, 0]), np.array([.1, .2]), bundle.threshold)
    assert single["average_precision"] is None and single["roc_auc"] is None
    assert single["precision"] is None and single["recall"] is None and single["f1"] is None
    events = append_decision([], analysis, case_positions[0], bundle.threshold, "Block", "First review")
    events = append_decision(events, analysis, case_positions[0], bundle.threshold, "Escalate", "Request further review")
    assert len(events) == 2 and events[1]["revision"] == 2
    assert latest_decisions(events, analysis.dataset_id)[analysis.row_ids[case_positions[0]]] == "Escalate"
    assert not latest_decisions(events, replacement.dataset_id)
    audit = pd.read_csv(io.BytesIO(audit_export(events)))
    assert audit["provenance"].eq(analysis.provenance).all()
    assert SAMPLE_NOTICE in analysis.provenance and "test partition" in analysis.provenance
    exported = pd.read_csv(io.BytesIO(scored_export(analysis, bundle.threshold, latest_decisions(events, analysis.dataset_id))))
    assert exported["fraudlens_provenance"].eq(analysis.provenance).all()
    assert np.allclose(exported["fraudlens_risk_score"], analysis.scores)
    assert np.array_equal(before, analysis.scores)
    print("PASS input validation, optional labels, file identity, row preservation, review revisions, provenance exports", flush=True)
    return bundle, support, analysis


def check_full_benchmark(bundle, support):
    _, _, train, test = reconstruct_partitions(ROOT / "data/creditcard.csv")
    features = prepare_features(test, bundle.features)
    scores = bundle.model.predict_proba(features)[:, 1]
    actual = evaluate_scores(test["Class"].to_numpy(), scores, bundle.threshold)
    expected = benchmark_metrics()
    for key in expected:
        assert np.isclose(actual[key], expected[key], rtol=1e-12, atol=1e-12), key
    reference = prepare_features(train, bundle.features).loc[train["Class"].eq(0)]
    assert len(reference) == support["reference_rows"] == 141601
    for feature, stats in support["reference"].items():
        assert np.allclose(
            reference[feature].quantile([.01, .25, .5, .75, .99]).to_numpy(),
            [stats[k] for k in ["p01", "p25", "median", "p75", "p99"]],
        )
    sample = pd.read_csv(SAMPLE_PATH)
    source = test.loc[support["sample_source_row_indices"]].reset_index(drop=True)
    assert np.allclose(sample.to_numpy(), source.to_numpy(), rtol=1e-12, atol=1e-12)
    print("PASS full benchmark 141,863 rows / TP 159 / FP 40 / FN 52; sample provenance; training-only references", flush=True)


def button(app, label):
    return next(item for item in app.button if item.label == label)


def healthy(app):
    assert not app.exception, [e.message for e in app.exception]
    assert not app.error, [e.value for e in app.error]


def check_ui(support):
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=90)
    healthy(app)
    assert app.session_state["analysis"] is None
    button(app, "Load Sample and Start Walkthrough").click().run(timeout=90)
    healthy(app)
    assert app.session_state["walk_step"] == 0
    assert any(SAMPLE_NOTICE in w.value for w in app.warning)
    analysis = app.session_state["analysis"]
    original = analysis.scores.copy()
    decision = next(r for r in app.radio if r.label == "Analyst decision")
    decision.set_value("Block")
    app.text_area[0].set_value("Reviewed SHAP evidence.")
    button(app, "Save simulated decision").click().run(timeout=90)
    healthy(app)
    assert len(app.session_state["audit_events"]) == 1
    next(r for r in app.radio if r.label == "Analyst decision").set_value("Escalate")
    button(app, "Save simulated decision").click().run(timeout=90)
    assert len(app.session_state["audit_events"]) == 2
    assert np.array_equal(original, app.session_state["analysis"].scores)
    for index in [1, 2]:
        app.button(key="walk_next").click().run(timeout=90)
        healthy(app)
        assert app.session_state["walk_step"] == index
        assert any(SAMPLE_NOTICE in w.value for w in app.warning)
    app.button(key="walk_previous").click().run(timeout=90)
    assert app.session_state["walk_step"] == 1
    app.button(key="walk_next").click().run(timeout=90)
    app.button(key="walk_next").click().run(timeout=90)
    healthy(app)
    assert app.session_state["navigation"] == "Model Intelligence"
    app.button(key="global_shap_button").click().run(timeout=90)
    healthy(app)
    assert app.session_state["global_explanation"].shape == (1000, 30)
    app.sidebar.radio[0].set_value("Overview").run(timeout=90)
    healthy(app)
    assert any(SAMPLE_NOTICE in w.value for w in app.warning)
    app.sidebar.radio[0].set_value("Transaction Explorer").run(timeout=90)
    healthy(app)
    token = app.session_state["dataset_generation"]
    app.selectbox(key=f"filter_model_{token}").set_value("Flagged").run(timeout=90)
    table = result_table(analysis, latest_decisions(app.session_state["audit_events"], analysis.dataset_id))
    filtered = table[table["Model result"].eq("Flagged")].sort_values("Risk score", ascending=False, kind="stable")
    selection_key = f"explorer_selection_{token}_{digest_bytes(filtered['_position'].to_numpy().tobytes())[:12]}"
    app.session_state[selection_key] = {"selection": {"rows": [1], "columns": [], "cells": []}}
    app.run(timeout=90)
    # AppTest does not serialize dataframe selections as browser widget events.
    # Supply the same selection with the button event to emulate that payload.
    app.session_state[selection_key] = {"selection": {"rows": [1], "columns": [], "cells": []}}
    app.button(key="inspect_explorer").click().run(timeout=90)
    healthy(app)
    assert app.session_state["selected_position"] == int(filtered.iloc[1]["_position"])
    assert app.session_state["walk_step"] is None

    class Upload(io.BytesIO):
        name = "same.csv"

    data = analysis.raw.head(3).drop(columns="Class").copy()
    with patch("streamlit.file_uploader", return_value=Upload(data.to_csv(index=False).encode())):
        app.run(timeout=90)
        app.button(key="analyze_upload").click().run(timeout=90)
    healthy(app)
    first_id = app.session_state["analysis"].dataset_id
    assert app.session_state["selected_position"] is None
    assert app.session_state["global_explanation"] is None
    assert len(app.session_state["audit_events"]) == 2
    data.loc[0, "Amount"] += 1
    with patch("streamlit.file_uploader", return_value=Upload(data.to_csv(index=False).encode())):
        app.run(timeout=90)
        app.button(key="analyze_upload").click().run(timeout=90)
    healthy(app)
    second_id = app.session_state["analysis"].dataset_id
    assert second_id != first_id
    with patch("streamlit.file_uploader", return_value=Upload(b"unrelated\n1\n")):
        app.run(timeout=90)
        app.button(key="analyze_upload").click().run(timeout=90)
    assert not app.exception
    assert app.error and app.session_state["analysis"].dataset_id == second_id
    app.sidebar.radio[0].set_value("Model Intelligence").run(timeout=90)
    healthy(app)
    assert any("No Class column" in message.value for message in app.info)
    print("PASS Streamlit walkthrough, review revisions, navigation, SHAP, filtered-row identity, upload replacement/failure, unlabeled evaluation", flush=True)


if __name__ == "__main__":
    check_preservation()
    bundle, support, analysis = check_inference()
    check_full_benchmark(bundle, support)
    check_ui(support)
    check_preservation()
    print("All FraudLens acceptance checks passed.", flush=True)
