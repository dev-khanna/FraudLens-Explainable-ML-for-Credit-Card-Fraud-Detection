"""FraudLens: an investigation workflow around the original ML pipeline."""

# VS Code's "Run Python File" executes this file directly. Start Streamlit in
# that case; a normal Streamlit run already has a ScriptRunContext.
if __name__ == "__main__":
    import subprocess
    import sys
    from pathlib import Path

    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ModuleNotFoundError:
        context = None
    else:
        context = get_script_run_ctx(suppress_warning=True)
    if context is None:
        app_path = Path(__file__).resolve()
        venv_python = app_path.parent / ".venv" / "Scripts" / "python.exe"
        python = venv_python if venv_python.is_file() else Path(sys.executable)
        raise SystemExit(subprocess.call([str(python), "-m", "streamlit", "run", str(app_path)]))

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from fraudlens.core import (
    ARTIFACTS, NOTEBOOK_URL, SAMPLE_NOTICE, SAMPLE_PATH, benchmark_metrics,
    curve_data, digest_bytes, evaluate_scores, explanation_values, file_digest,
    get_labels, load_bundle, load_supporting, result_table, score_csv, scored_export,
)
from fraudlens.plots import (
    confusion_figure, global_importance, performance_curves,
    reference_comparison, risk_distribution, waterfall_figure,
)
from fraudlens.records import (
    DECISIONS, append_decision, audit_export, audit_frame, latest_decisions, latest_records,
)
from fraudlens.style import CSS

st.set_page_config(page_title="FraudLens | Credit Card Fraud Investigation", page_icon="◈", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
PAGES = ("Overview", "Transaction Explorer", "Investigation Workspace", "Model Intelligence")


@st.cache_resource
def cached_model(model_hash, threshold_hash):
    return load_bundle()


@st.cache_resource
def cached_explainer(model_hash, _model):
    return shap.TreeExplainer(_model.named_steps["model"])


@st.cache_data(show_spinner=False)
def cached_supporting(asset_hash, model_hash, threshold_hash, _bundle):
    return load_supporting(_bundle)


def initialize_state():
    defaults = {
        "navigation": "Overview", "analysis": None, "audit_events": [],
        "selected_position": None, "walk_step": None, "tour_complete": False,
        "upload_open": False, "local_explanations": {}, "global_explanation": None,
        "dataset_generation": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    pending = st.session_state.pop("pending_page", None)
    if pending:
        st.session_state.navigation = pending


initialize_state()
try:
    bundle = cached_model(
        file_digest(ARTIFACTS / "final_xgboost_model.pkl"),
        file_digest(ARTIFACTS / "final_threshold.pkl"),
    )
except Exception as exc:
    st.error(f"Could not load the original model artifacts: {exc}")
    st.stop()
try:
    support = cached_supporting(
        file_digest(ARTIFACTS / "fraudlens_assets.json"),
        bundle.model_hash, bundle.threshold_hash, bundle,
    )
    support_error = None
except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
    support, support_error = None, str(exc)


def go(page):
    st.session_state.pending_page = page
    st.rerun()


def activate(analysis):
    st.session_state.analysis = analysis
    st.session_state.selected_position = None
    st.session_state.walk_step = None
    st.session_state.tour_complete = False
    st.session_state.local_explanations = {}
    st.session_state.global_explanation = None
    st.session_state.dataset_generation += 1
    st.session_state.upload_open = False


def load_sample(walkthrough=False):
    if support is None:
        st.error("The verified sample assets are unavailable.")
        return
    try:
        payload = SAMPLE_PATH.read_bytes()
        if digest_bytes(payload) != support["sample_sha256"]:
            raise ValueError("Sample integrity check failed. Regenerate the supporting assets.")
        current = st.session_state.analysis
        if current is None or current.dataset_id != support["sample_sha256"]:
            with st.spinner("Scoring 1,000 real sample transactions with the saved model..."):
                analysis = score_csv(payload, "Curated demonstration sample", bundle, support["sample_sha256"])
            activate(analysis)
        if walkthrough:
            open_walk_step(0)
        else:
            go("Overview")
    except (OSError, ValueError) as exc:
        st.error(str(exc))


def open_walk_step(index):
    analysis = st.session_state.analysis
    case = support["walkthrough"][index]
    labels, _ = get_labels(analysis.raw)
    position = case["position"]
    if (
        not analysis.curated or analysis.dataset_id != support["sample_sha256"]
        or labels is None or labels[position] != case["label"]
        or bool(analysis.flags[position]) != case["flagged"]
        or not np.isclose(analysis.scores[position], case["score"], rtol=1e-6)
    ):
        raise ValueError("The walkthrough case does not match the verified model and sample.")
    st.session_state.walk_step = index
    st.session_state.selected_position = position
    st.session_state.tour_complete = False
    go("Investigation Workspace")


def inspect(position):
    st.session_state.selected_position = int(position)
    st.session_state.walk_step = None
    go("Investigation Workspace")


def show_figure(fig):
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def metric_value(value, percent=False):
    if value is None:
        return "Unavailable"
    return f"{value:.2%}" if percent else f"{value:.4f}"


def show_metrics(metrics):
    for column, key, label, percent in zip(
        st.columns(5),
        ["average_precision", "roc_auc", "precision", "recall", "f1"],
        ["PR-AUC (AP)", "ROC-AUC", "Precision", "Recall", "F1"],
        [False, False, True, True, False],
    ):
        column.metric(label, metric_value(metrics[key], percent))


def latest():
    analysis = st.session_state.analysis
    return latest_decisions(st.session_state.audit_events, analysis.dataset_id) if analysis else {}


def download_audit(key):
    events = st.session_state.audit_events
    st.download_button(
        "Download decision history (CSV)", audit_export(events),
        file_name="fraudlens_investigation_history.csv", mime="text/csv",
        disabled=not events, key=key,
    )
    st.caption("Includes all datasets reviewed in this session. Timestamps are labeled UTC. Download before leaving; session history is temporary.")


def input_controls(page):
    analysis = st.session_state.analysis
    expanded = st.session_state.upload_open or (page == "Transaction Explorer" and analysis is None)
    with st.expander("Load your transactions", expanded=expanded):
        st.write("Use transactions in the original dataset's anonymized PCA feature space.")
        st.caption(
            "Required: V1–V28, Amount, and Time (elapsed seconds) or Hour (0–23). "
            "Class is optional. Matching column names alone does not guarantee the same PCA representation. "
            "Limits: 200 MB and 300,000 rows."
        )
        upload = st.file_uploader("Compatible transaction CSV", type=["csv"], max_upload_size=200, key="csv_upload")
        action, sample_action = st.columns([1, 2])
        if action.button("Validate and analyze", type="primary", disabled=upload is None, key="analyze_upload"):
            payload = upload.getvalue()
            current = st.session_state.analysis
            if current is not None and current.dataset_id == digest_bytes(payload):
                st.info("This exact file is already loaded.")
            else:
                try:
                    with st.spinner("Validating and scoring transactions..."):
                        candidate = score_csv(payload, upload.name, bundle, support["sample_sha256"] if support else None)
                    activate(candidate)
                    go("Overview")
                except Exception as exc:
                    st.error(f"Upload could not be analyzed: {exc}")
                    if current is not None:
                        st.info("Your previous dataset remains active. Its name is shown above.")
        if sample_action.button("Use the curated sample", disabled=support is None, key="sample_in_upload"):
            load_sample()
        st.caption("A successful load replaces the active analysis and resets its filters. Earlier decision history stays in this session.")


def overview():
    analysis = st.session_state.analysis
    if analysis is None:
        st.title("FraudLens")
        st.markdown('<div class="project-title">Explainable Machine Learning for Credit Card Fraud Detection &amp; Investigation</div>', unsafe_allow_html=True)
        st.markdown('<div class="project-tagline">Detect, Explain and Review Suspicious Credit Card Transactions</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="hero-copy">Explore real anonymized transactions, investigate model alerts, '
            'and see where machine learning succeeds—and where it needs human judgment.</p>',
            unsafe_allow_html=True,
        )
        start, sample, upload = st.columns([1.5, 1, 1])
        if start.button("Load Sample and Start Walkthrough", type="primary", width="stretch", disabled=support is None):
            load_sample(walkthrough=True)
        if sample.button("Load Sample Dataset", width="stretch", disabled=support is None):
            load_sample()
        if upload.button("Upload Transactions", width="stretch"):
            st.session_state.upload_open = True
            go("Transaction Explorer")
        st.caption(SAMPLE_NOTICE)
        for column, number, title, text in zip(
            st.columns(3), ["01", "02", "03"],
            ["Explore real cases", "Understand the evidence", "Make a review decision"],
            [
                "Start with a successful detection, then examine a false alarm and a missed fraud.",
                "Connect a saved model score to SHAP contributions and legitimate-training reference data.",
                "Record a simulated analyst outcome, export your findings, and explore the learning notebook.",
            ],
        ):
            column.markdown(
                f'<div class="step"><div class="step-num">{number}</div>'
                f'<div class="step-title">{title}</div><div class="step-copy">{text}</div></div>',
                unsafe_allow_html=True,
            )
        st.divider()
        st.subheader("Built on a documented ML learning journey")
        st.write(
            "The notebook records data exploration, baseline models, validation experiments, "
            "XGBoost tuning, threshold selection, and explainability—including unsuccessful attempts."
        )
        left, right = st.columns([1, 3])
        if left.button("Explore Model Intelligence", width="stretch"):
            go("Model Intelligence")
        right.link_button("Read the learning notebook ↗", NOTEBOOK_URL)
        return

    st.title("Overview")
    st.caption("Your active analysis, model alerts, and investigation progress.")
    decisions = latest()
    columns = st.columns(4)
    columns[0].metric("Transactions analyzed", f"{len(analysis.raw):,}")
    columns[1].metric("Flagged for review", f"{int(analysis.flags.sum()):,}")
    columns[2].metric("Flag rate", f"{analysis.flags.mean():.2%}")
    columns[3].metric("Reviewed", f"{len(decisions):,} / {len(analysis.raw):,}")
    st.caption("Flagged means the model score meets the saved threshold. A review decision is a separate human record.")
    chart, guide = st.columns([2, 1])
    with chart:
        st.subheader("Risk score distribution")
        show_figure(risk_distribution(analysis.scores, bundle.threshold))
    with guide:
        with st.container(border=True):
            st.subheader("Three cases. Three lessons.")
            st.write("Walk through a correct detection, a false alarm, and a missed fraud from the curated sample.")
            if st.button("Start the 3-case walkthrough", type="primary", disabled=support is None, width="stretch"):
                load_sample(walkthrough=True)
            st.caption("Loads the curated sample. All three cases use real labels and current model scores.")
            if st.button("Open Transaction Explorer", width="stretch"):
                go("Transaction Explorer")
    st.subheader("Highest-scoring alerts")
    alerts = result_table(analysis, decisions)
    alerts = alerts.loc[analysis.flags].sort_values("Risk score", ascending=False).head(10)
    if alerts.empty:
        st.info("No transactions meet the saved threshold in this dataset.")
    else:
        selection = st.dataframe(
            alerts[["Row ID", "Amount", "Risk score", "Model result", "Review"]].reset_index(drop=True),
            hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
            column_config={"Risk score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=1, format="%.3f")},
            key=f"alert_selection_{st.session_state.dataset_generation}",
        )
        selected = selection.selection.rows
        if st.button("Investigate selected alert", disabled=not selected, key="inspect_alert"):
            inspect(alerts.iloc[selected[0]]["_position"])
    st.caption("TXN IDs are temporary row references generated by this app, not bank-issued identifiers.")
    with st.expander("Export the active analysis"):
        st.download_button(
            "Download scored transactions (CSV)", scored_export(analysis, bundle.threshold, decisions),
            file_name="fraudlens_scored_transactions.csv", mime="text/csv", key="overview_export",
        )
        download_audit("overview_audit")

def explorer():
    st.title("Transaction Explorer")
    st.caption("Filter the active dataset, select a row, and follow its evidence.")
    analysis = st.session_state.analysis
    if analysis is None:
        st.info("Load the curated sample or upload a compatible CSV above to begin.")
        return
    table = result_table(analysis, latest())
    token = f"{st.session_state.dataset_generation}"
    a, b, c, d = st.columns([1.2, 1, 1, 1.5])
    search = a.text_input("Search row ID", placeholder="TXN-000042", key=f"filter_search_{token}")
    prediction = b.selectbox("Model result", ["All", "Flagged", "Not flagged"], key=f"filter_model_{token}")
    review = c.selectbox("Review status", ["All", "Unreviewed", *DECISIONS], key=f"filter_review_{token}")
    score_range = d.slider("Risk score (%)", 0.0, 100.0, (0.0, 100.0), step=.1, key=f"filter_score_{token}")
    with st.expander("Amount and time filters"):
        low, high, period = st.columns(3)
        min_amount = low.number_input("Minimum amount", min_value=0.0, value=0.0, key=f"filter_min_{token}")
        max_amount = high.number_input("Maximum amount", min_value=0.0, value=float(table["Amount"].max()), key=f"filter_max_{token}")
        hours = period.slider("Elapsed-cycle hour", 0, 23, (0, 23), key=f"filter_hour_{token}")
        if "Elapsed seconds" in table:
            start_col, end_col = st.columns(2)
            min_time = start_col.number_input("From elapsed seconds", min_value=0.0, value=0.0, key=f"filter_time_min_{token}")
            max_time = end_col.number_input("To elapsed seconds", min_value=0.0, value=float(table["Elapsed seconds"].max()), key=f"filter_time_max_{token}")
        st.caption("Time is elapsed seconds from the source dataset's first transaction; Hour is not a verified wall-clock hour.")
    mask = (
        table["Row ID"].str.contains(search.strip(), case=False, regex=False)
        & table["Risk score"].between(score_range[0] / 100, score_range[1] / 100)
        & table["Amount"].between(min_amount, max_amount)
        & table["Hour"].between(*hours)
    )
    if prediction != "All":
        mask &= table["Model result"].eq(prediction)
    if review != "All":
        mask &= table["Review"].eq(review)
    if "Elapsed seconds" in table:
        mask &= table["Elapsed seconds"].between(min_time, max_time)
    filtered = table.loc[mask].sort_values("Risk score", ascending=False, kind="stable")
    st.caption(f"{len(filtered):,} of {len(table):,} transactions · scores shown from 0 to 1 · highest first")
    if filtered.empty:
        st.info("No transactions match these filters. Widen the ranges or clear the search.")
    else:
        visible = filtered.drop(columns="_position").reset_index(drop=True)
        # Clear positional selections when filters change; preserve original row identity.
        filter_id = digest_bytes(filtered["_position"].to_numpy().tobytes())[:12]
        selection = st.dataframe(
            visible, hide_index=True, width="stretch", height=420,
            on_select="rerun", selection_mode="single-row", key=f"explorer_selection_{token}_{filter_id}",
            column_config={
                "Risk score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=1, format="%.4f"),
                "Amount": st.column_config.NumberColumn("Amount", format="%.2f"),
                "Hour": st.column_config.NumberColumn("Elapsed-cycle hour", format="%d"),
            },
        )
        selected = selection.selection.rows
        if st.button("Investigate selected transaction", type="primary", disabled=not selected, key="inspect_explorer"):
            inspect(filtered.iloc[selected[0]]["_position"])
    st.caption("Generated row IDs stay attached to their original input row through sorting and filtering.")
    st.divider()
    st.download_button(
        "Download all scored transactions (CSV)", scored_export(analysis, bundle.threshold, latest()),
        file_name="fraudlens_scored_transactions.csv", mime="text/csv", key="explorer_export",
    )
    st.caption("Exports the complete active dataset, including provenance. Filters only affect the on-screen view.")


WALK_TEXT = [
    "The model flagged this transaction and the dataset labels it as fraud. Inspect the strongest contributors below to see which features raised or lowered its score.",
    "The model flagged this transaction, but the dataset labels it as legitimate. A strong model score can still be a false alarm; inspect the evidence before recording a simulated review outcome.",
    "The dataset labels this transaction as fraud, but its score is below the saved threshold. The model missed it. This case makes the precision–recall tradeoff and the limits of model detection tangible.",
]


def investigation():
    st.title("Investigation Workspace")
    analysis = st.session_state.analysis
    position = st.session_state.selected_position
    if analysis is None or position is None:
        st.info("Select a transaction in the Explorer, or start the guided walkthrough.")
        left, right = st.columns(2)
        if left.button("Go to Transaction Explorer", width="stretch"):
            go("Transaction Explorer")
        if right.button("Load Sample and Start Walkthrough", type="primary", disabled=support is None, width="stretch"):
            load_sample(walkthrough=True)
        return
    if not 0 <= position < len(analysis.raw):
        st.error("The selected row is no longer available. Select another transaction.")
        return
    step = st.session_state.walk_step
    if step is not None:
        case = support["walkthrough"][step]
        with st.container(border=True):
            st.caption(f"GUIDED WALKTHROUGH · CASE {step + 1} OF 3")
            st.subheader(case["title"])
            st.write(WALK_TEXT[step])
            previous, next_case, exit_case = st.columns(3)
            if previous.button("Previous case", disabled=step == 0, key="walk_previous", width="stretch"):
                open_walk_step(step - 1)
            next_label = "Next case" if step < 2 else "Finish walkthrough"
            if next_case.button(next_label, type="primary", key="walk_next", width="stretch"):
                if step < 2:
                    open_walk_step(step + 1)
                st.session_state.walk_step = None
                st.session_state.tour_complete = True
                go("Model Intelligence")
            if exit_case.button("Exit walkthrough", key="walk_exit", width="stretch"):
                st.session_state.walk_step = None
                st.rerun()
    row_id = analysis.row_ids[position]
    score = float(analysis.scores[position])
    flagged = bool(analysis.flags[position])
    st.subheader(row_id)
    st.caption(f"Generated reference for input data row {position + 1:,}.")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Risk score", f"{score:.2%}")
    col2.metric("Model result", "Flagged" if flagged else "Not flagged")
    col3.metric("Saved threshold", repr(bundle.threshold))
    col4.metric("Amount", f"{float(analysis.features.iloc[position]['Amount']):,.2f}")
    st.caption("A model score is not a verified probability of fraud. Flagging uses the unrounded score and saved threshold.")
    labels, label_error = get_labels(analysis.raw)
    if labels is not None:
        label = "Fraud" if labels[position] else "Legitimate"
        st.info(f"Known dataset label: {label}. This label is separate from the model result and your analyst decision.")
    elif "Class" in analysis.raw:
        st.warning(label_error)
    cache_key = (analysis.dataset_id, int(position), bundle.model_hash)
    try:
        if cache_key not in st.session_state.local_explanations:
            with st.spinner("Explaining this transaction using the fitted feature order..."):
                explainer = cached_explainer(bundle.model_hash, bundle.model)
                transformed, values = explanation_values(bundle, explainer, analysis.features.iloc[[position]])
                cache = st.session_state.local_explanations
                if len(cache) >= 100:
                    cache.pop(next(iter(cache)))
                cache[cache_key] = (transformed[0], values[0], float(explainer.expected_value))
        transformed, values, base = st.session_state.local_explanations[cache_key]
        order = np.argsort(np.abs(values))[::-1]
        st.subheader("What influenced the model?")
        up = [bundle.output_features[i] for i in order if values[i] > 0][:2]
        down = [bundle.output_features[i] for i in order if values[i] < 0][:2]
        parts = []
        if up:
            parts.append("Strongest upward contributors: " + ", ".join(up) + ".")
        if down:
            parts.append("Strongest downward contributors: " + ", ".join(down) + ".")
        st.write(" ".join(parts))
        st.caption("V1–V28 are anonymized features. Their real-world behavioral meanings are unknown; SHAP describes model contributions, not causes.")
        with st.expander("Detailed SHAP explanation", expanded=step is not None):
            explanation = shap.Explanation(
                values=values, base_values=base, data=transformed, feature_names=bundle.output_features,
            )
            show_figure(waterfall_figure(explanation))
            st.caption(
                "Bars add to the model's raw log-odds, not percentage-point changes. "
                "Amount and Hour values in this plot are transformed by the fitted scaler."
            )
        if support is not None:
            st.subheader("Context from legitimate training transactions")
            top_features = [bundle.output_features[i] for i in order[:5]]
            show_figure(reference_comparison(
                analysis.features.iloc[position], top_features, support["reference"],
            ))
            st.caption(
                f"Reference: {support['reference_rows']:,} legitimate training rows only. "
                "Thin line: 1st–99th percentiles · thick band: 25th–75th · white mark: median · pink diamond: this transaction. "
                "Each axis uses that feature's raw values. Unusual values are descriptive context, not proof of fraud."
            )
    except Exception as exc:
        st.error(f"The explanation could not be generated: {exc}")
        st.caption("The model score and analyst review remain available.")
    st.divider()
    st.subheader("Record a simulated review")
    st.caption("Approve, Escalate, and Block only record your judgment. They do not move money, freeze cards, change labels, or retrain the model.")
    existing = latest_records(st.session_state.audit_events, analysis.dataset_id).get(row_id)
    form_token = f"{analysis.dataset_id[:16]}_{position}"
    if existing:
        st.caption(f"Latest decision: {existing['decision']} · revision {existing['revision']}")
    with st.form(f"review_form_{form_token}"):
        decision = st.radio(
            "Analyst decision", DECISIONS, horizontal=True,
            index=DECISIONS.index(existing["decision"]) if existing else 1, key=f"decision_{form_token}",
        )
        notes = st.text_area(
            "Investigator notes", value=existing["notes"] if existing else "",
            placeholder="What did you observe? What would you investigate next?",
            max_chars=4000, key=f"notes_{form_token}",
        )
        if st.form_submit_button("Save simulated decision", type="primary"):
            st.session_state.audit_events = append_decision(
                st.session_state.audit_events, analysis, position, bundle.threshold, decision, notes,
            )
            st.success("Decision saved for this session. The model output is unchanged.")
    with st.expander("Decision history and export", expanded=bool(st.session_state.audit_events)):
        events = st.session_state.audit_events
        if events:
            st.dataframe(
                audit_frame(events)[["dataset_source", "row_id", "model_result", "decision", "notes", "timestamp_utc", "revision"]],
                hide_index=True, width="stretch",
            )
        else:
            st.caption("No decisions recorded yet.")
        download_audit("investigation_audit")
    with st.expander("Original transaction values"):
        st.dataframe(analysis.raw.iloc[[position]], hide_index=True, width="stretch")
    if st.button("Back to Transaction Explorer"):
        go("Transaction Explorer")

def intelligence():
    st.title("Model Intelligence")
    st.caption("The experiments, benchmark, and limitations behind this application.")
    if st.session_state.tour_complete:
        st.success("Walkthrough complete: you explored a successful detection, a false alarm, and a missed fraud. Follow the notebook to see how the model was developed.")
    st.link_button("Open the complete learning notebook ↗", NOTEBOOK_URL, type="primary")
    with st.expander("The learning journey", expanded=True):
        st.write(
            "**Data quality and exploration → preprocessing → baseline models → "
            "validation experiments → XGBoost tuning → threshold selection → SHAP.**"
        )
        st.write(
            "The notebook preserves the original experiments, including the first tuning attempt "
            "and the later switch to GroupKFold. The application uses the original saved model; "
            "reference comparisons and analyst records are application additions."
        )
    st.subheader("Historical benchmark")
    st.caption("Fixed results on the original 141,863-row test partition. These are independent of the active upload or curated sample.")
    try:
        metrics = benchmark_metrics()
        show_metrics(metrics)
        matrix, context = st.columns([1, 1.2])
        with matrix:
            show_figure(confusion_figure(metrics))
        with context:
            st.write(
                f"The model flagged **{metrics['tp'] + metrics['fp']:,}** test transactions: "
                f"**{metrics['tp']:,}** were labeled fraud and **{metrics['fp']:,}** were legitimate. "
                f"It missed **{metrics['fn']:,}** labeled frauds."
            )
            st.write(f"The saved decision threshold is **{repr(bundle.threshold)}**. It is loaded directly from the original artifact.")
            st.caption("PR-AUC here means average precision (AP). PR and ROC metrics assess score ranking; changing a decision threshold does not change these ranking metrics.")
        if support is not None:
            with st.expander("Benchmark curves and baseline comparison"):
                show_figure(performance_curves(support["benchmark_curves"]))
                st.caption(support["curve_note"])
                comparison = pd.DataFrame([
                    *support["recorded_baselines"],
                    {"Model": "XGBoost tuned", "PR-AUC": metrics["average_precision"],
                     "ROC-AUC": metrics["roc_auc"], "Precision": metrics["precision"],
                     "Recall": metrics["recall"], "F1 Score": metrics["f1"]},
                ])
                st.dataframe(comparison, hide_index=True, width="stretch")
                st.caption("Baseline values are taken from the notebook's saved outputs and retain their recorded precision. Tuned-model values come from the original metrics artifact.")
    except (OSError, ValueError, KeyError) as exc:
        st.warning(f"Benchmark results are unavailable: {exc}. No substitute metrics are displayed.")
    with st.expander("Validation and threshold selection"):
        st.write(
            "Exact duplicate rows were removed before a 50/50 split in original row order. "
            "The boundary falls at elapsed seconds 84,692 / 84,693, rather than exactly at a calendar-day boundary. "
            "Hour is an elapsed 24-hour-cycle feature; the source does not establish a wall-clock origin."
        )
        st.write(
            "The notebook explores KFold, StratifiedKFold, and TimeSeriesSplit, then tunes with "
            "GroupKFold grouped by Hour. Its groups do not overlap between training and validation, "
            "but its folds are not chronological forward-validation folds."
        )
        st.write(
            "The final threshold was selected from out-of-fold development scores by maximizing F1 "
            "over 0.01–0.99. The recorded development F1 increased from 0.8315 at 0.50 to 0.8760 at 0.97."
        )
        st.caption(
            "Limitations: exploratory analysis used the full dataset; the test partition was inspected "
            "during model development. Hyperparameter selection and threshold evaluation reused the development folds. "
            "These results are educational evidence, not an untouched prospective evaluation."
        )
    st.divider()
    st.subheader("Active dataset evaluation")
    analysis = st.session_state.analysis
    if analysis is None:
        st.info("Load a compatible dataset to calculate its own evaluation or global explanation.")
        return
    if analysis.curated:
        st.warning("Curated-sample metrics below describe the deliberately enriched demo only. They do not estimate real-world performance or prevalence.")
    else:
        st.caption("Uses the supplied Class column as ground truth; its reliability cannot be verified by the application.")
    labels, reason = get_labels(analysis.raw)
    if labels is None:
        st.info(reason)
    else:
        current = evaluate_scores(labels, analysis.scores, bundle.threshold)
        show_metrics(current)
        with st.expander("Active dataset confusion matrix and curves"):
            show_figure(confusion_figure(current))
            curves = curve_data(labels, analysis.scores)
            if curves:
                show_figure(performance_curves(curves))
            else:
                st.info("Both ground-truth classes are needed to display meaningful PR/ROC curves and ranking metrics.")
            st.caption("Undefined metrics are shown as unavailable. This evaluation never uses analyst decisions.")
    st.subheader("Global explanation of the active dataset")
    st.caption("Mean absolute SHAP contributions across a reproducible sample of up to 1,000 active rows.")
    if st.button("Generate global SHAP", key="global_shap_button"):
        try:
            with st.spinner("Calculating the global explanation..."):
                sample = analysis.features.sample(n=min(1000, len(analysis.raw)), random_state=42)
                explainer = cached_explainer(bundle.model_hash, bundle.model)
                _, values = explanation_values(bundle, explainer, sample)
                st.session_state.global_explanation = values
        except Exception as exc:
            st.error(f"Global explanation could not be generated: {exc}")
    values = st.session_state.global_explanation
    if values is not None:
        show_figure(global_importance(values, bundle.output_features))
        st.caption("This explains the loaded population. With a curated sample, feature importance may differ from a representative population.")


with st.sidebar:
    st.markdown('<div class="brand">FraudLens<span class="brand-dot">.</span></div><div class="brand-subtitle">DETECT · EXPLAIN · REVIEW</div>', unsafe_allow_html=True)
    st.caption("Educational decision-support demo")
    st.divider()
    page = st.radio("Workspace", PAGES, key="navigation")
    st.divider()
    st.caption("ORIGINAL SAVED MODEL")
    st.write("Tuned XGBoost")
    st.caption(f"Flag threshold: {repr(bundle.threshold)}")
    st.caption("Exact saved scalar; no display rounding is used for inference.")
    st.link_button("Learning notebook ↗", NOTEBOOK_URL, width="stretch")
    st.divider()
    analysis = st.session_state.analysis
    if analysis:
        st.caption("ACTIVE DATASET")
        st.write(analysis.source)
        st.caption(f"{len(analysis.raw):,} rows · {analysis.dataset_id[:12]}")
        if st.button("Load another dataset", width="stretch"):
            st.session_state.upload_open = True
            go("Transaction Explorer")
    st.caption("Review decisions are session-only. Export the audit log before leaving.")

if support_error:
    st.warning(f"Supporting demo/reference assets are unavailable: {support_error}")
analysis = st.session_state.analysis
if analysis is not None:
    st.caption(f"ACTIVE DATASET · {analysis.source} · {len(analysis.raw):,} transactions")
    if analysis.curated:
        st.warning(SAMPLE_NOTICE)
    else:
        st.caption("Uploaded data · predictions use the original PCA feature schema.")

input_controls(page)
if page == "Overview":
    overview()
elif page == "Transaction Explorer":
    explorer()
elif page == "Investigation Workspace":
    investigation()
else:
    intelligence()
st.divider()
st.caption("FraudLens · Explainable Machine Learning for Credit Card Fraud Detection & Investigation")
