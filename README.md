# Credit Card Fraud Detection

An end-to-end Machine Learning project designed to detect fraudulent credit card transactions on highly imbalanced data. The project spans from Exploratory Data Analysis (EDA) and robust modeling to deploying an interactive Streamlit dashboard for real-time inference and model explainability.

# Pipeline

```mermaid
flowchart TD
    A["EDA"] --> B["Data Preprocessing"]
    B --> C["Model Training & Advanced Validation"]
    C --> D["Cross Validation"]
    D --> E["Hyperparameter Tuning"]
    E --> F["Threshold Optimization"]
    F --> G["Final Evaluation"]
    G --> Z["Explainability & Deployment"]

    A -.-> A1
    subgraph EDA_SUB["EDA Breakdown"]
        direction TB
        A1["Data Quality Check"] --> A2["Univariate Analysis"] --> A3["Bivariate & Multinomial Analysis"] --> A4["Feature Engineering"]
    end

    B -.-> B1
    subgraph PREP_SUB["Data Preprocessing Breakdown"]
        direction TB
        B1["Train-Test Split"] --> B2["Scaling"] --> B3["Encoding"]
    end

    C -.-> C1
    subgraph MODEL_SUB["Model Training Breakdown"]
        direction TB
        C1["Build ML Pipelines for Each Model"] --> C2["Evaluate Each Model Using Appropriate Metrics"] --> C3["Select the Ideal Model"]
    end

    D -.-> D1
    subgraph CV_SUB["Cross Validation Breakdown"]
        direction TB
        D1["Trial Multiple CV Strategies"] --> D2["Check: Data Leakage & Fraud-per-Fold Consistency"] --> D3["Select Strategy with 0 Leakage & Consistent Fraud Ratio"]
    end

    E -.-> E1
    subgraph HPT_SUB["Hyperparameter Tuning Breakdown"]
        direction TB
        E1["Define Candidate Hyperparameter Combinations"] --> E2["Test Each Combination via Chosen CV Technique"] --> E3["Select Optimal Combination"]
    end

    F -.-> F1
    subgraph THRESH_SUB["Threshold Optimization Breakdown"]
        direction TB
        F1["Loop Thresholds 1% to 99%"] --> F2["Calculate F1-score at Each Threshold"] --> F3["Keep Threshold with Max F1-score"]
    end

    G -.-> G1
    subgraph FINAL_SUB["Final Evaluation Breakdown"]
        direction TB
        G1["Apply Optimal Threshold & Hyperparameters"] --> G2["Compute Final Model Metrics"]
    end

    C2 -.-> M1
    G2 -.-> M1
    subgraph METRICS_SUB["Metrics Used"]
        direction TB
        M1["PR-AUC"] --- M2["ROC-AUC"] --- M3["F1 Score"] --- M4["Recall"] --- M5["Precision"]
    end

    Z -.-> Z1
    subgraph DEPLOY_SUB["Explainability & Deployment Breakdown"]
        direction TB
        Z1["SHAP Explainability"] --> Z2["Streamlit Dashboard"]
    end

    classDef main fill:#4A90D9,stroke:#1B3B6F,color:#ffffff,font-weight:bold,stroke-width:2px;
    classDef eda fill:#FFE0B2,stroke:#E65100,color:#000000;
    classDef prep fill:#C8E6C9,stroke:#2E7D32,color:#000000;
    classDef model fill:#F8BBD0,stroke:#AD1457,color:#000000;
    classDef cv fill:#B3E5FC,stroke:#01579B,color:#000000;
    classDef hpt fill:#E1BEE7,stroke:#6A1B9A,color:#000000;
    classDef thresh fill:#FFCDD2,stroke:#B71C1C,color:#000000;
    classDef final fill:#FFF59D,stroke:#F9A825,color:#000000;
    classDef metrics fill:#B2DFDB,stroke:#00695C,color:#000000;
    classDef deploy fill:#D7CCC8,stroke:#4E342E,color:#000000;

    class A,B,C,D,E,F,G,Z main;
    class A1,A2,A3,A4 eda;
    class B1,B2,B3 prep;
    class C1,C2,C3 model;
    class D1,D2,D3 cv;
    class E1,E2,E3 hpt;
    class F1,F2,F3 thresh;
    class G1,G2 final;
    class M1,M2,M3,M4,M5 metrics;
    class Z1,Z2 deploy;
```

## My Journey: The Process Timeline

Here is the step-by-step story of how I built this fraud detection engine:

### 1. Data Quality Inspection
Before feeding data to a machine, we must ensure it is pristine. We ran data quality checks and identified over 1,000 duplicate transactions. Why is this bad? If the model sees the same transaction in the training set and the test set, it simply memorizes the answer instead of learning the pattern, artificially inflating our test metrics. We removed these duplicates immediately to protect the integrity of our test.

### 2. Univariate & Bivariate Analysis
We analyzed each feature in isolation and then compared them against our target variable (Fraud vs Legitimate). 
- **The Imbalance Issue:** We discovered that a staggering 99.83% of transactions were legitimate. Because of this, we realized that standard "Accuracy" would be a completely useless metric (a broken model could just guess "Legitimate" every time and be 99.83% accurate!). We pivoted to tracking PR-AUC, Precision, Recall, and F1-Score.
- **Isolating Predictors:** By calculating the Standardized Mean Difference (Effect Size) of our PCA features, we identified which variables naturally separated fraudsters from innocent customers (e.g., V10, V14, V17) and discarded weak predictors like transaction `Amount`.

### 3. Feature Engineering
Machine learning models only understand repeating patterns. We extracted an `hour_of_day` column from the raw, linear `Time` seconds counter. After plotting the densities, we discovered a massive pattern: while the raw *number* of frauds peaks at 11 AM, the actual *probability* of a transaction being fraudulent severely spikes at 2 AM. By feeding this engineered hour feature into our model, we allowed it to "see" this crucial temporal pattern.

### 4. Data Preprocessing & Leakage Prevention
Machine learning models are just calculators; they require mathematically sterile numbers. 
- **Train/Test Splitting:** We split our dataset 50/50 across the two days of data to see if our model could learn day 1 and accurately predict day 2.
- **Scaling:** Features with massive spreads (like Amount) were scaled down to prevent the algorithm from assigning them too much weight. Most importantly, we strictly applied the Golden Rule of Scaling: we *never* fitted our scaler on the test data, completely preventing data leakage.

### 5. Model Training & Advanced Validation
We tested Logistic Regression and XGBoost against baseline models like Naive Bayes. XGBoost emerged as the clear winner due to its ability to handle complex, non-linear patterns.
To ensure our model wouldn't overfit, we utilized `GroupKFold` cross-validation grouped by the hour of the day. This rigorously tested our model on unseen time blocks, ensuring it was genuinely learning fraud behaviors and not just memorizing the clock.

### 6. Threshold Optimization & Deployment
Finally, we realized that the default 50% cutoff line for flagging fraud was too generic. We tested every possible probability threshold from 1% to 99% and identified the exact cutoff point that maximized our F1-score. We wrapped our finalized pipeline and threshold into a Streamlit dashboard, creating a dual-audience application that translates complex SHAP mathematics into plain-English explanations for non-technical users.


## The Deliverables & Real-World Impact

We engineered a highly tuned XGBoost pipeline and wrapped it in a production-ready application. When deployed against an untouched Day-1 test dataset containing **141,863 total transactions**, the results were exceptional:

* **Massive Precision Improvement (79.89%):** Out of 141,652 innocent customers, our algorithm only accidentally froze the cards of **40** people (False Positives). This means when this model blocks a credit card, there is an **80% chance** that the person is an actual thief.
* **Exceptional Recall (75.35%):** Out of the 211 actual thieves attempting fraud in the dataset, our system successfully caught **159** of them.
* **Balanced F1-Score (77.56%):** We achieved an incredibly strong balance between catching thieves and protecting innocent customers, a massive upgrade from baseline algorithms that often score below 10%.
* **High-Performance Classification:** The model achieved a **0.7821 PR-AUC** (Precision-Recall Area Under Curve) and a **0.9712 ROC-AUC**, proving it is exceptionally confident in its decision boundaries on highly imbalanced data.
* **Dual-Audience Dashboard:** The primary deliverable is an interactive Streamlit application. It doesn't just output predictions; it utilizes SHAP (SHapley Additive exPlanations) to show exactly *why* it blocked a specific transaction, complete with dropdown expanders that translate the machine learning math into plain English for business stakeholders.

## Running the Dashboard
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py`
3. Upload a transaction CSV to view real-time scoring, probability distributions, and SHAP explanations.