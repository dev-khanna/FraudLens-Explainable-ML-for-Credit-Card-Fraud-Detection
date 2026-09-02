import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)


# MODEL ARTIFACT PATHS

MODEL_PATH = "model_artifacts/final_xgboost_model.pkl"
THRESHOLD_PATH = "model_artifacts/final_threshold.pkl"
FEATURES_PATH = "model_artifacts/feature_columns.pkl"


# FINAL DAY-1 TEST RESULTS
# These are the locked results from the notebook.

FINAL_PR_AUC = 0.7910
FINAL_ROC_AUC = 0.9811
FINAL_PRECISION = 0.6411
FINAL_RECALL = 0.7910
FINAL_F1 = 0.7082

FINAL_TN = 139200
FINAL_FP = 89
FINAL_FN = 42
FINAL_TP = 159


# LOAD MODEL ARTIFACTS

@st.cache_resource
def load_artifacts():
    """Load the final trained model and supporting artifacts."""

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not os.path.exists(THRESHOLD_PATH):
        raise FileNotFoundError(
            f"Threshold file not found: {THRESHOLD_PATH}"
        )

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)
    feature_columns = joblib.load(FEATURES_PATH)

    return model, float(threshold), feature_columns


try:
    model, threshold, feature_columns = load_artifacts()

except Exception as e:
    st.error(f"Could not load model artifacts: {e}")
    st.stop()


# SHAP EXPLAINER

@st.cache_resource
def create_shap_explainer(_model):
    """
    Create SHAP TreeExplainer.

    The leading underscore prevents Streamlit from trying
    to hash the sklearn Pipeline.
    """

    xgb_model = _model.named_steps["clf"]

    return shap.TreeExplainer(xgb_model)


explainer = create_shap_explainer(model)


# HELPER FUNCTIONS

def prepare_input(df):
    """
    Prepare uploaded transaction data.

    If hour_of_day is absent, derive it from Time.
    """

    df = df.copy()

    if "hour_of_day" not in df.columns:

        if "Time" in df.columns:

            df["hour_of_day"] = (
                (df["Time"] // 3600) % 24
            )

        else:

            raise ValueError(
                "The uploaded CSV must contain either "
                "'Time' or 'hour_of_day'."
            )

    return df


def validate_input(df):
    """Return required model features that are missing."""

    return [
        column
        for column in feature_columns
        if column not in df.columns
    ]


def validate_numeric(df):
    """Return required features that are not numeric."""

    return [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]


def score_transactions(df):
    """Generate fraud probabilities and predictions."""

    X = df[feature_columns]

    probabilities = model.predict_proba(
        X
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    results = df.copy()

    results["fraud_probability"] = probabilities

    results["fraud_prediction"] = np.where(
        predictions == 1,
        "Fraud",
        "Legitimate"
    )

    return results


# SESSION STATE

if "input_df" not in st.session_state:
    st.session_state.input_df = None

if "results" not in st.session_state:
    st.session_state.results = None

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None


# HEADER

st.title("💳 Credit Card Fraud Detection")

st.markdown(
    """
    ### Tuned XGBoost Fraud Detection System

    Analyze credit card transactions using a tuned XGBoost
    model with threshold optimization and SHAP-based
    explainability.
    """
)

st.divider()


# SIDEBAR

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Fraud Detection",
        "Model Performance",
        "Explainability",
    ]
)


st.sidebar.divider()

st.sidebar.header("Model Configuration")

st.sidebar.metric(
    "Decision Threshold",
    f"{threshold:.2f}"
)

st.sidebar.write(
    "Transactions with a predicted fraud probability "
    "greater than or equal to this threshold are flagged."
)

st.sidebar.divider()

st.sidebar.write("**Model:** Tuned XGBoost")
st.sidebar.write("**Primary metric:** PR-AUC")
st.sidebar.write("**Validation:** Temporal + group-aware CV")

st.sidebar.divider()

st.sidebar.caption(
    "V1–V28 are anonymized PCA-derived features from "
    "the original dataset."
)


# PAGE 1 — FRAUD DETECTION

if page == "Fraud Detection":

    st.header("1. Upload Transactions")

    st.markdown(
        """
        Upload the original credit card dataset. The application
        will automatically derive `hour_of_day` from `Time`.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help=(
            "The CSV should contain V1–V28, Amount, and Time "
            "or hour_of_day."
        )
    )


    # -----------------------------------------------------
    # Process uploaded file
    # -----------------------------------------------------

    if uploaded_file is not None:

        # Only reprocess when a new file is uploaded
        if (
            st.session_state.uploaded_filename
            != uploaded_file.name
        ):

            try:

                input_df = pd.read_csv(
                    uploaded_file
                )

            except Exception as e:

                st.error(
                    f"Could not read CSV: {e}"
                )

                st.stop()


            # ---------------------------------------------
            # Prepare
            # ---------------------------------------------

            try:

                input_df = prepare_input(
                    input_df
                )

            except Exception as e:

                st.error(str(e))
                st.stop()


            # ---------------------------------------------
            # Validate columns
            # ---------------------------------------------

            missing_columns = validate_input(
                input_df
            )

            if missing_columns:

                st.error(
                    "The uploaded CSV is missing "
                    "required model features:"
                )

                st.write(
                    missing_columns
                )

                st.stop()


            # ---------------------------------------------
            # Validate numeric features
            # ---------------------------------------------

            non_numeric_columns = validate_numeric(
                input_df
            )

            if non_numeric_columns:

                st.error(
                    "These model features must be numeric:"
                )

                st.write(
                    non_numeric_columns
                )

                st.stop()


            # ---------------------------------------------
            # Validate missing values
            # ---------------------------------------------

            missing_values = (
                input_df[
                    feature_columns
                ]
                .isna()
                .sum()
            )

            columns_with_missing = (
                missing_values[
                    missing_values > 0
                ]
            )


            if len(columns_with_missing) > 0:

                st.error(
                    "The uploaded data contains "
                    "missing values in:"
                )

                st.write(
                    columns_with_missing
                )

                st.stop()


            # ---------------------------------------------
            # Score dataset
            # ---------------------------------------------

            try:

                results = score_transactions(
                    input_df
                )

            except Exception as e:

                st.error(
                    f"Could not generate predictions: {e}"
                )

                st.stop()


            # ---------------------------------------------
            # Save in session state
            # ---------------------------------------------

            st.session_state.input_df = input_df

            st.session_state.results = results

            st.session_state.uploaded_filename = (
                uploaded_file.name
            )


            st.success(
                f"Successfully loaded "
                f"{len(input_df):,} transactions."
            )

            st.info(
                "The dataset has been processed and is now "
                "available throughout the dashboard."
            )


    # -----------------------------------------------------
    # Check whether data exists
    # -----------------------------------------------------

    results = st.session_state.results


    if results is None:

        st.info(
            "Upload a transaction CSV above to begin."
        )

        st.subheader(
            "Expected model features"
        )

        st.code(
            ", ".join(feature_columns)
        )

        st.stop()


    # =====================================================
    # SUMMARY
    # =====================================================

    st.header("2. Detection Summary")


    total_transactions = len(
        results
    )


    fraud_count = (
        results["fraud_prediction"]
        == "Fraud"
    ).sum()


    legitimate_count = (
        results["fraud_prediction"]
        == "Legitimate"
    ).sum()


    flag_rate = (
        fraud_count / total_transactions
        if total_transactions > 0
        else 0
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Transactions",
            f"{total_transactions:,}"
        )


    with col2:

        st.metric(
            "Flagged as Fraud",
            f"{fraud_count:,}"
        )


    with col3:

        st.metric(
            "Legitimate",
            f"{legitimate_count:,}"
        )


    with col4:

        st.metric(
            "Flag Rate",
            f"{flag_rate:.2%}"
        )


    # =====================================================
    # PROBABILITY DISTRIBUTION
    # =====================================================

    st.subheader(
        "Fraud Probability Distribution"
    )


    fig, ax = plt.subplots(
        figsize=(10, 4)
    )


    ax.hist(
        results["fraud_probability"],
        bins=40
    )


    ax.axvline(
        threshold,
        linestyle="--",
        label=f"Threshold = {threshold:.2f}"
    )


    ax.set_xlabel(
        "Predicted Fraud Probability"
    )

    ax.set_ylabel(
        "Number of Transactions"
    )

    ax.set_title(
        "Distribution of Fraud Risk Scores"
    )

    ax.legend()

    plt.tight_layout()


    st.pyplot(
        fig,
        clear_figure=True
    )


    # =====================================================
    # RESULTS TABLE
    # =====================================================

    st.header("3. Detection Results")


    display_columns = []


    if "Time" in results.columns:

        display_columns.append(
            "Time"
        )


    if "Amount" in results.columns:

        display_columns.append(
            "Amount"
        )


    display_columns.extend(
        [
            "fraud_probability",
            "fraud_prediction",
        ]
    )


    display_df = results[
        display_columns
    ].copy()


    display_df[
        "fraud_probability"
    ] = (
        display_df[
            "fraud_probability"
        ]
        .map(
            lambda x: f"{x:.2%}"
        )
    )


    st.dataframe(
        display_df,
        use_container_width=True,
        height=450
    )


    # =====================================================
    # DOWNLOAD
    # =====================================================

    st.subheader(
        "Download Predictions"
    )


    download_csv = results.to_csv(
        index=False
    )


    st.download_button(
        label="⬇️ Download Scored Transactions",
        data=download_csv,
        file_name="fraud_detection_results.csv",
        mime="text/csv"
    )


# PAGE 2 — MODEL PERFORMANCE

elif page == "Model Performance":

    st.header(
        "📊 Model Performance"
    )


    st.markdown(
        """
        Final performance of the tuned XGBoost model on the
        untouched Day-1 test set.
        """
    )


    # =====================================================
    # MAIN METRICS
    # =====================================================

    col1, col2, col3, col4, col5 = st.columns(5)


    with col1:

        st.metric(
            "PR-AUC",
            f"{FINAL_PR_AUC:.4f}"
        )


    with col2:

        st.metric(
            "ROC-AUC",
            f"{FINAL_ROC_AUC:.4f}"
        )


    with col3:

        st.metric(
            "Precision",
            f"{FINAL_PRECISION:.2%}"
        )


    with col4:

        st.metric(
            "Recall",
            f"{FINAL_RECALL:.2%}"
        )


    with col5:

        st.metric(
            "F1 Score",
            f"{FINAL_F1:.4f}"
        )


    st.divider()


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    st.subheader(
        "Confusion Matrix"
    )


    confusion_df = pd.DataFrame(
        [
            [FINAL_TN, FINAL_FP],
            [FINAL_FN, FINAL_TP],
        ],
        index=[
            "Actual Legitimate",
            "Actual Fraud",
        ],
        columns=[
            "Predicted Legitimate",
            "Predicted Fraud",
        ]
    )


    st.dataframe(
        confusion_df,
        use_container_width=True
    )


    st.markdown(
        f"""
        **Final test interpretation**

        The model correctly detected **{FINAL_TP} fraud cases**
        and missed **{FINAL_FN}**.

        It incorrectly flagged **{FINAL_FP} legitimate
        transactions** as fraud.
        """
    )


    # =====================================================
    # THRESHOLD
    # =====================================================

    st.subheader(
        "Decision Threshold"
    )


    st.metric(
        "Selected Threshold",
        f"{threshold:.2f}"
    )


    st.write(
        f"""
        At this threshold, the final Day-1 test performance
        was **{FINAL_PRECISION:.2%} precision** and
        **{FINAL_RECALL:.2%} recall**.
        """
    )


    st.caption(
        "The threshold was selected using out-of-fold "
        "development predictions and then evaluated once "
        "on the untouched Day-1 test set."
    )


    # =====================================================
    # MODEL COMPARISON
    # =====================================================

    st.subheader(
        "Model Comparison"
    )


    comparison_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "XGBoost Baseline",
            "XGBoost Tuned",
        ],
        "PR-AUC": [
            0.7559,
            0.7761,
            FINAL_PR_AUC,
        ],
        "Precision": [
            0.0363,
            0.5364,
            FINAL_PRECISION,
        ],
        "Recall": [
            0.8806,
            0.8060,
            FINAL_RECALL,
        ],
        "F1": [
            0.0698,
            0.6441,
            FINAL_F1,
        ],
    })


    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )


# PAGE 3 — EXPLAINABILITY

elif page == "Explainability":

    st.header(
        "🔎 Model Explainability"
    )


    st.markdown(
        """
        SHAP explains how individual features influence the
        model's predictions.

        The transaction dataset is shared automatically from
        the Fraud Detection page — no second upload is needed.
        """
    )


    # =====================================================
    # CHECK FOR UPLOADED DATA
    # =====================================================

    results = st.session_state.results
    input_df = st.session_state.input_df


    if results is None or input_df is None:

        st.info(
            "No transaction dataset is loaded yet."
        )

        st.write(
            "Go to **Fraud Detection** and upload your CSV. "
            "Once uploaded, the same dataset will be "
            "available here automatically."
        )

        st.stop()


    # =====================================================
    # GLOBAL SHAP
    # =====================================================

    st.subheader(
        "Global Feature Importance"
    )


    st.caption(
        "Mean absolute SHAP value across a sample of "
        "uploaded transactions."
    )


    # Use up to 10,000 rows so the dashboard remains responsive.
    sample_size = min(
        len(input_df),
        10000
    )


    shap_input = input_df[
        feature_columns
    ].sample(
        sample_size,
        random_state=42
    )


    prep = model.named_steps[
        "prep"
    ]


    transformed_data = prep.transform(
        shap_input
    )


    shap_values = explainer.shap_values(
        transformed_data
    )


    transformed_feature_names = (
        prep.get_feature_names_out()
    )


    shap_importance = pd.DataFrame({
        "Feature": transformed_feature_names,
        "Mean Absolute SHAP": (
            np.abs(shap_values)
            .mean(axis=0)
        )
    })


    shap_importance = (
        shap_importance
        .sort_values(
            "Mean Absolute SHAP",
            ascending=False
        )
    )


    # -----------------------------------------------------
    # Plot top 15
    # -----------------------------------------------------

    top_features = (
        shap_importance
        .head(15)
        .sort_values(
            "Mean Absolute SHAP"
        )
    )


    fig, ax = plt.subplots(
        figsize=(10, 7)
    )


    ax.barh(
        top_features["Feature"],
        top_features["Mean Absolute SHAP"]
    )


    ax.set_xlabel(
        "Mean Absolute SHAP Value"
    )


    ax.set_ylabel(
        "Feature"
    )


    ax.set_title(
        "Top Features Driving Model Predictions"
    )


    plt.tight_layout()


    st.pyplot(
        fig,
        clear_figure=True
    )


    # -----------------------------------------------------
    # Importance table
    # -----------------------------------------------------

    st.dataframe(
        shap_importance.head(15),
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # INDIVIDUAL TRANSACTION
    # =====================================================

    st.divider()

    st.subheader(
        "Individual Transaction Explanation"
    )


    selected_index = st.selectbox(
        "Select a transaction to explain",

        options=list(
            results.index
        ),

        format_func=lambda x: (
            f"Row {x} — "
            f"{results.loc[x, 'fraud_prediction']} — "
            f"{results.loc[x, 'fraud_probability']:.2%}"
        )
    )


    selected_input = input_df.loc[
        [selected_index],
        feature_columns
    ]


    selected_probability = (
        results.loc[
            selected_index,
            "fraud_probability"
        ]
    )


    selected_prediction = (
        results.loc[
            selected_index,
            "fraud_prediction"
        ]
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Fraud Probability",
            f"{selected_probability:.2%}"
        )


    with col2:

        if selected_prediction == "Fraud":

            st.error(
                "🚨 FRAUD"
            )

        else:

            st.success(
                "✅ LEGITIMATE"
            )


    # =====================================================
    # SHAP WATERFALL
    # =====================================================

    transformed_transaction = (
        prep.transform(
            selected_input
        )
    )


    transaction_shap_values = (
        explainer.shap_values(
            transformed_transaction
        )
    )


    shap_explanation = shap.Explanation(
        values=transaction_shap_values[0],
        base_values=explainer.expected_value,
        data=transformed_transaction[0],
        feature_names=transformed_feature_names,
    )


    st.subheader(
        "Why did the model make this prediction?"
    )


    fig = plt.figure(
        figsize=(10, 7)
    )


    shap.plots.waterfall(
        shap_explanation,
        max_display=10,
        show=False
    )


    st.pyplot(
        fig,
        clear_figure=True
    )


    st.caption(
        "Positive SHAP values push the prediction toward "
        "fraud; negative values push it toward legitimate."
    )


# FOOTER

st.divider()

st.caption(
    "Credit Card Fraud Detection | "
    "Tuned XGBoost | Threshold Optimization | "
    "SHAP Explainability"
)