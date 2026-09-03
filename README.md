# Credit Card Fraud Detection

An end-to-end Machine Learning project designed to detect fraudulent credit card transactions on highly imbalanced data. The project spans from Exploratory Data Analysis (EDA) and robust modeling to deploying an interactive Streamlit dashboard for real-time inference and model explainability.

## Tech Stack
**Machine Learning:** Python, scikit-learn, XGBoost, SHAP (SHapley Additive exPlanations)  
**Deployment:** Streamlit, Pandas, NumPy, Matplotlib

## Key Deliverables & Performance Metrics
* **High-Performance Classification:** Engineered a tuned XGBoost model that achieved a **0.7910 PR-AUC** and **0.9811 ROC-AUC** on the untouched Day-1 test set (~139,500 transactions).
* **Massive Precision Improvement:** Improved precision from **3.6%** (Logistic Regression baseline) to **64.1%** (Tuned XGBoost), drastically reducing false positives (only 89 false positives out of 139,200 legitimate transactions) while maintaining a strong **79.1% Recall** (catching 159 out of 201 fraud cases). 
* **Optimized F1-Score:** Increased the overall F1-Score from **6.98%** to **70.82%** through rigorous hyperparameter tuning and algorithmic class-weight adjustments.
* **Production-Ready Dashboard:** Developed an interactive Streamlit application capable of instantly scoring large transaction datasets and outputting actionable fraud probabilities.

## Core Workflow & Features

### 1. Data Processing & Pipeline Engineering
* **Feature Engineering:** Automatically derived time-based features (e.g., `hour_of_day`) from raw transaction timestamps to capture temporal fraud patterns.
* **Scikit-Learn Pipelines:** Encapsulated data scaling and the XGBoost estimator into a unified Pipeline (`model.named_steps`), preventing data leakage and streamlining inference.
* **Automated Validation:** The Streamlit app includes strict validation checks to handle missing values, enforce numeric data types, and ensure all 28 PCA-derived features are present.

### 2. Modeling & Advanced Validation
* **Imbalanced Data Handling:** Utilized built-in algorithmic class-weight adjustments rather than relying purely on sampling techniques, ensuring the model learns the true minority class distribution.
* **Temporal & Group-Aware Cross-Validation:** Implemented advanced CV strategies that respect the chronological order of transactions, ensuring performance metrics reflect real-world, forward-looking scenarios.
* **Threshold Tuning:** Shifted from the default 0.5 decision threshold to an optimized threshold based on out-of-fold predictions to maximize PR-AUC.

### 3. Machine Learning Explainability (SHAP)
* **Global Interpretability:** Used `shap.TreeExplainer` to calculate Mean Absolute SHAP values, identifying the top PCA components driving the model's behavior across the entire dataset.
* **Local Interpretability:** Integrated SHAP Waterfall plots into the dashboard to explicitly explain individual transaction predictions, showing exactly which features pushed a specific transaction toward "Fraud" or "Legitimate".

## 💻 Running the Dashboard
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py`
3. Upload a transaction CSV to view real-time scoring, probability distributions, and SHAP explanations.