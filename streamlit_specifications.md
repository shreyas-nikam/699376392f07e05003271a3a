
# Streamlit Application Specification: Bias Mitigation in Credit Scoring

## 1. Application Overview

### Purpose of the Application

The **Apex Credit Solutions Bias Mitigation Dashboard** is designed for CFA Charterholders and Investment Professionals, like Sri Krishnamurthy, a Senior Risk Analyst, to effectively implement, evaluate, and compare various bias mitigation strategies in credit scoring models. The application addresses critical regulatory and ethical concerns arising from disparate impact identified in existing models, particularly unfair loan approval rates for certain demographic groups. Its primary goal is to guide users through a practical workflow, from detecting bias to actively mitigating it, ensuring compliance with fairness regulations (e.g., Equal Credit Opportunity Act - ECOA) while managing the trade-offs between model accuracy and fairness. The application also facilitates generating a compliance-ready report.

### High-Level Story Flow of the Application

The application unfolds as a structured case study, following Sri Krishnamurthy's journey at Apex Credit Solutions:

1.  **Introduction & Persona Setup:** The application begins by introducing Sri Krishnamurthy and the business problem: an existing credit model exhibits disparate impact against "Group B."
2.  **Data Generation & Setup:** Sri generates a synthetic, biased credit dataset to simulate the real-world scenario.
3.  **Baseline Model & Bias Quantification:** He trains a standard (biased) credit model and evaluates its initial performance and fairness using key metrics (AUC, DIR, SPD, EOD) to establish a baseline. This step quantifies the extent of the bias.
4.  **Implementing Mitigation Strategies:** Sri then systematically applies four distinct bias mitigation strategies, covering different intervention points in the ML pipeline:
    *   **Pre-processing (Sample Reweighting):** Adjusting sample weights to give underrepresented groups more influence.
    *   **Pre-processing (Proxy Feature Removal):** Identifying and removing features that act as proxies for sensitive attributes.
    *   **In-processing (Fairness-Constrained Training):** Retraining the model with explicit fairness constraints (e.g., Demographic Parity) using `fairlearn`.
    *   **Post-processing (Group-Specific Threshold Calibration):** Adjusting decision thresholds for different groups to achieve fairness targets.
5.  **Comparative Analysis & Visualization:** After applying each strategy, Sri compares their impact on accuracy and fairness using a five-way comparison table, an Accuracy-Fairness Pareto Frontier plot, Approval Rate bar charts, and an AUC Cost bar chart.
6.  **Strategic Recommendation & Compliance:** Based on the analysis, Sri evaluates the trade-offs, discusses legal nuances (especially for post-processing), and uses a structured decision framework to formulate a strategic recommendation for Apex Credit Solutions, culminating in a compliance-ready report template.

This workflow simulates a real-world decision-making process for financial professionals grappling with AI ethics and regulatory compliance.

## 2. Code Requirements

### Import Statement

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from source import * # Assumes source.py is available in the Python path or current directory
```

### `st.session_state` Design

The following `st.session_state` keys will be used to maintain application state, store intermediate data, and preserve results across different interactions and simulated pages:

*   `st.session_state.initialized` (bool): `True` after initial data generation and baseline model training. Prevents re-running these expensive operations. Default: `False`.
*   `st.session_state.current_page` (str): Stores the currently selected page from the sidebar. Default: `'Home'`.
*   `st.session_state.X_train`, `st.session_state.X_test`, `st.session_state.y_train`, `st.session_state.y_test`, `st.session_state.sensitive_train`, `st.session_state.sensitive_test`, `st.session_state.feature_cols` (pd.DataFrame, pd.Series, list): Stores the generated dataset components and feature names. Initialized once by `generate_biased_credit_data()` when `initialized` is `False`.
*   `st.session_state.baseline_results` (dict): Stores the evaluation metrics of the baseline model. Initialized by `evaluate_model()`.
*   `st.session_state.baseline_model` (XGBClassifier): Stores the trained baseline model.
*   `st.session_state.y_pred_base`, `st.session_state.y_prob_base` (np.ndarray): Stores predictions and probabilities from the baseline model on the test set.
*   `st.session_state.privileged_group`, `st.session_state.unprivileged_group` (str): Stores the identified privileged and unprivileged groups from the baseline results.
*   `st.session_state.reweighted_results` (dict): Stores evaluation metrics for the sample reweighting strategy. Initialized by `evaluate_model()`.
*   `st.session_state.reweighted_model` (XGBClassifier): Stores the trained reweighted model.
*   `st.session_state.proxy_removed_results` (dict): Stores evaluation metrics for the proxy feature removal strategy. Initialized by `evaluate_model()`.
*   `st.session_state.proxy_removed_model` (XGBClassifier): Stores the trained model after proxy removal.
*   `st.session_state.fair_dp_results` (dict): Stores evaluation metrics for the fairness-constrained training (Demographic Parity). Initialized by `evaluate_model()`.
*   `st.session_state.fair_dp_model` (ExponentiatedGradient): Stores the trained fair model (Demographic Parity).
*   `st.session_state.fair_eod_results` (dict): Stores evaluation metrics for the fairness-constrained training (Equalized Odds). Initialized by `evaluate_model()`.
*   `st.session_state.fair_eod_model` (ExponentiatedGradient): Stores the trained fair model (Equalized Odds).
*   `st.session_state.threshold_calibrated_results` (dict): Stores evaluation metrics for the group-specific threshold calibration strategy. Initialized by `evaluate_model()`.
*   `st.session_state.all_results_df` (pd.DataFrame): Stores a consolidated DataFrame of all mitigation results for comparison. Updated before the "Comprehensive Comparison" page is rendered.

### Application Structure and Flow

The application simulates a multi-page experience using a `st.sidebar.selectbox` widget to control conditional rendering based on `st.session_state.current_page`.

#### Sidebar Navigation

**Initialization:**
At the very top of `app.py`, before any page logic, initialize `st.session_state` keys if they don't exist:
```python
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'
# Initialize all other session state variables to None or empty dicts/dataframes
# This ensures a clean state upon initial load or full rerun.
# ... (e.g., st.session_state.X_train = None, st.session_state.baseline_results = None, etc.)
```

**Sidebar Code:**
```python
with st.sidebar:
    st.image("https://www.quantuniversity.com/static/assets/logo/QuantUniversity_logomark.png", width=150) # Example logo
    st.markdown("---")
    st.markdown("## Navigation")
    pages = [
        "Home",
        "1. Setup & Data Generation",
        "2. Baseline Model Performance",
        "3. Strategy 1: Sample Reweighting",
        "4. Strategy 2: Proxy Feature Removal",
        "5. Strategy 3: Fairness-Constrained Training",
        "6. Strategy 4: Group-Specific Threshold Calibration",
        "7. Comprehensive Comparison & Recommendation",
        "8. Compliance Report"
    ]
    st.session_state.current_page = st.selectbox("Go to", pages, index=pages.index(st.session_state.current_page))
```

#### Page: `Home`

**Conditional Rendering Logic:**
```python
if st.session_state.current_page == 'Home':
    # ... render Home page content ...
```

**Markdown Content:**
```python
st.title("🛡️ Apex Credit Solutions: Bias Mitigation Dashboard")
st.markdown(f"")
st.markdown(f"**Persona:** Sri Krishnamurthy, CFA, Senior Risk Analyst at Apex Credit Solutions.")
st.markdown(f"**Organization:** Apex Credit Solutions, a leading financial institution providing consumer credit.")
st.markdown(f"")
st.markdown(f"As a CFA charterholder, my primary responsibility at Apex Credit Solutions is to ensure the integrity and fairness of our credit scoring models. Recent internal audits, stemming from our D4-T2-C1 bias detection lab, have revealed a concerning issue: our existing credit model exhibits disparate impact, with **\"Group B\" receiving significantly lower loan approval rates compared to \"Group A,\" despite similar financial profiles.** This not only raises ethical concerns but also exposes Apex Credit Solutions to significant regulatory and reputational risks under acts like the Equal Credit Opportunity Act (ECOA).")
st.markdown(f"")
st.markdown(f"My task today is to implement and evaluate various bias mitigation strategies. We need to move beyond merely detecting bias to actively treating it. This dashboard documents my journey through several intervention points in the machine learning pipeline—pre-processing, in-processing, and post-processing—to reduce this disparate impact. The goal is to compare the trade-offs between accuracy and fairness for each strategy and ultimately recommend the most suitable approach for our organization, ensuring we comply with fairness regulations and uphold our ethical obligations.")
st.markdown(f"")
st.markdown(f"This exercise is critical for two reasons:")
st.markdown(f"1.  **Regulatory Compliance:** Avoiding potential legal penalties and ensuring our models meet the standards set by financial regulators.")
st.markdown(f"2.  **Ethical Lending:** Upholding Apex Credit Solutions' commitment to fair and equitable treatment of all applicants, strengthening trust and reputation.")
st.markdown(f"---")
st.info("Click 'Start Analysis' to begin generating the synthetic credit data and establish the baseline.")
```

**Widgets and Interaction with `source.py`:**
```python
if st.button("Start Analysis"):
    # Reset all relevant session state variables to ensure a fresh run
    st.session_state.initialized = False
    st.session_state.X_train = None
    st.session_state.baseline_results = None
    st.session_state.reweighted_results = None
    st.session_state.proxy_removed_results = None
    st.session_state.fair_dp_results = None
    st.session_state.fair_eod_results = None
    st.session_state.threshold_calibrated_results = None
    st.session_state.current_page = "1. Setup & Data Generation"
    st.rerun() # Trigger a rerun to navigate to the next page
```

#### Page: `1. Setup & Data Generation`

**Conditional Rendering Logic:**
```python
elif st.session_state.current_page == '1. Setup & Data Generation':
    # ... render content ...
```

**Markdown Content:**
```python
st.header("1. Setup and Data Generation")
st.markdown(f"Before diving into bias mitigation, we need to set up our environment and generate a synthetic dataset that simulates the biased credit scoring scenario identified in our previous analysis. This dataset will feature demographic information and a sensitive attribute, allowing us to replicate and address the detected bias.")
st.subheader("1.1 Generating a Synthetic Biased Credit Dataset")
st.markdown(f"As identified in D4-T2-C1, our credit model exhibited bias against a specific demographic group. To proceed with mitigation, I need a similar synthetic dataset. This dataset will contain typical credit features, a sensitive attribute (`'race_group'`), and a target variable (`'loan_approved'`), with an intentional bias where 'Group_B' has a lower approval rate.")
```

**Widgets and Interaction with `source.py`:**
```python
if not st.session_state.initialized:
    with st.spinner("Generating biased credit data..."):
        X_train, X_test, y_train, y_test, sensitive_train, sensitive_test, feature_cols = generate_biased_credit_data()
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.sensitive_train = sensitive_train
        st.session_state.sensitive_test = sensitive_test
        st.session_state.feature_cols = feature_cols
        st.session_state.initialized = True # Data is now generated for the first time
    st.success("Dataset generated successfully!")

st.markdown(f"Dataset generated with `{len(st.session_state.X_train)}` training samples and `{len(st.session_state.X_test)}` test samples.")
st.markdown(f"Features: `{st.session_state.feature_cols}`")
st.markdown(f"Sensitive attribute distribution in training: ")
st.write(st.session_state.sensitive_train.value_counts(normalize=True))
st.markdown(f"Sensitive attribute distribution in test: ")
st.write(st.session_state.sensitive_test.value_counts(normalize=True))
st.dataframe(st.session_state.X_train.head())
st.markdown("---")
st.info("Data is ready. Proceed to the next step to establish the baseline.")
if st.button("Proceed to Baseline Model"):
    st.session_state.current_page = "2. Baseline Model Performance"
    st.rerun()
```

#### Page: `2. Baseline Model Performance`

**Conditional Rendering Logic:**
```python
elif st.session_state.current_page == '2. Baseline Model Performance':
    # ... render content ...
```

**Markdown Content:**
```python
st.header("2. Baseline Model Performance and Fairness Evaluation")
st.markdown(f"My first step is to establish a clear baseline. This involves training our standard credit model (an `XGBClassifier`) on the original, biased data and then meticulously evaluating its performance and fairness. This baseline will serve as the benchmark against which all mitigation strategies will be measured. It’s crucial to quantify the extent of the bias before attempting to fix it.")
st.markdown(f"")
st.markdown(f"To ensure consistency across all evaluations, I'll define a comprehensive function, `evaluate_model`, which computes both accuracy metrics (AUC, F1-score) and fairness metrics (Disparate Impact Ratio (DIR), Statistical Parity Difference (SPD), Equal Opportunity Difference (EOD)). The Disparate Impact Ratio is particularly important as it quantifies the ratio of favorable outcomes for the unprivileged group to the privileged group. According to the Four-fifths rule, this ratio should ideally be $\geq 0.80$.")
st.markdown(r"")
st.markdown(r"**Disparate Impact Ratio (DIR):** This metric measures the ratio of the favorable outcome rate (e.g., loan approval rate) for the unprivileged group to the privileged group. A value of 1 indicates perfect fairness. Regulations often require this ratio to be at least 0.80 (the Four-fifths Rule).")
st.markdown(r"$$ \text{{DIR}} = \frac{{P(\hat{{Y}}=1 | A=unprivileged)}}{{P(\hat{{Y}}=1 | A=privileged)}} $$")
st.markdown(r"where $P(\hat{{Y}}=1 | A=g)$ is the probability of a positive prediction (loan approval) for group $g$.")
st.markdown(r"")
st.markdown(r"**Statistical Parity Difference (SPD):** This metric measures the difference in the favorable outcome rates between the privileged and unprivileged groups. An SPD of 0 indicates perfect fairness.")
st.markdown(r"$$ \text{{SPD}} = P(\hat{{Y}}=1 | A=unprivileged) - P(\hat{{Y}}=1 | A=privileged) $$")
st.markdown(r"where $P(\hat{{Y}}=1 | A=g)$ is the probability of a positive prediction (loan approval) for group $g$.")
st.markdown(r"")
st.markdown(r"**Equal Opportunity Difference (EOD):** This metric measures the difference in true positive rates (FNR gap in the provided text, but EOD is standard for true positive rate difference) between the privileged and unprivileged groups among individuals who truly deserve the positive outcome. An EOD of 0 indicates perfect fairness, meaning both groups have an equal chance of being correctly approved if they should be.")
st.markdown(r"$$ \text{{EOD}} = P(\hat{{Y}}=1 | Y=1, A=unprivileged) - P(\hat{{Y}}=1 | Y=1, A=privileged) $$")
st.markdown(r"In the provided context, the FNR (False Negative Rate) gap is calculated as $FNR_{{unprivileged}} - FNR_{{privileged}}$. The EOD is typically defined using True Positive Rate (TPR), where $TPR = 1 - FNR$. So, an EOD of 0 for TPRs implies $TPR_{{privileged}} = TPR_{{unprivileged}}$, which is equivalent to $FNR_{{privileged}} = FNR_{{unprivileged}}$. Thus, the FNR gap in the code directly measures a form of equal opportunity.")
st.markdown(f"")
```

**Widgets and Interaction with `source.py`:**
```python
if not st.session_state.get('X_train'):
    st.warning("Please generate data first from '1. Setup & Data Generation' page.")
    if st.button("Go to Setup & Data Generation"):
        st.session_state.current_page = "1. Setup & Data Generation"
        st.rerun()
else:
    if not st.session_state.get('baseline_results'):
        with st.spinner("Training baseline model and evaluating..."):
            baseline_model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
            baseline_model.fit(st.session_state.X_train[st.session_state.feature_cols], st.session_state.y_train)
            
            y_pred_base = baseline_model.predict(st.session_state.X_test[st.session_state.feature_cols])
            y_prob_base = baseline_model.predict_proba(st.session_state.X_test[st.session_state.feature_cols])[:, 1]
            
            baseline_results = evaluate_model(st.session_state.y_test, y_pred_base, y_prob_base, st.session_state.sensitive_test, 'Baseline (Biased)')
            
            st.session_state.baseline_model = baseline_model
            st.session_state.y_pred_base = y_pred_base
            st.session_state.y_prob_base = y_prob_base
            st.session_state.baseline_results = baseline_results
            st.session_state.privileged_group = baseline_results['privileged_group']
            st.session_state.unprivileged_group = baseline_results['unprivileged_group']
        st.success("Baseline model trained and evaluated!")

    st.subheader("--- Baseline Model Evaluation ---")
    st.write(pd.DataFrame([st.session_state.baseline_results]).T.rename(columns={0: 'Value'})) # Display as a DataFrame
    
    st.subheader("2.1 Interpretation of Baseline Results")
    st.markdown(f"The baseline evaluation provides a quantitative snapshot of our current credit model's performance and, critically, its fairness. With a Disparate Impact Ratio (DIR) of `{st.session_state.baseline_results['dir']:.3f}`, our model clearly `{st.session_state.baseline_results['four_fifths']}` the Four-fifths Rule. This indicates a substantial disparity in approval rates between the privileged (`{st.session_state.privileged_group}`) and unprivileged (`{st.session_state.unprivileged_group}`) groups. The Statistical Parity Difference (SPD) of `{st.session_state.baseline_results['spd']:.4f}` and Equal Opportunity Difference (EOD) of `{st.session_state.baseline_results['eod']:.4f}` further confirm this bias, showing unequal outcomes and opportunities for deserving applicants across groups.")
    st.markdown(f"")
    st.markdown(f"For Apex Credit Solutions, this means:")
    st.markdown(f"*   **High Regulatory Risk:** The current model is non-compliant and could lead to significant fines and legal action.")
    st.markdown(f"*   **Reputational Damage:** Continued use of such a model erodes public trust and damages our brand.")
    st.markdown(f"*   **Ethical Obligation:** We are ethically bound to address this bias and ensure equitable treatment.")
    st.markdown(f"")
    st.markdown(f"This stark reality underscores the urgency of implementing effective bias mitigation strategies.")
    st.markdown("---")
    if st.button("Explore Mitigation Strategy 1: Sample Reweighting"):
        st.session_state.current_page = "3. Strategy 1: Sample Reweighting"
        st.rerun()
```

#### Page: `3. Strategy 1: Sample Reweighting`

**Conditional Rendering Logic:**
```python
elif st.session_state.current_page == '3. Strategy 1: Sample Reweighting':
    # ... render content ...
```

**Markdown Content:**
```python
st.header("3. Strategy 1: Pre-Processing - Sample Reweighting")
st.markdown(f"One of the simplest yet effective pre-processing techniques is sample reweighting. The idea is to adjust the influence of individual data points during model training. If certain sensitive groups or group-outcome combinations are underrepresented or disadvantaged in the training data, we can assign them higher weights. This ensures the model learns equally from all segments of the population, thereby reducing bias.")
st.markdown(r"For each combination of group $g$ and label $l$ (e.g., 'Group B' and 'loan_approved=0'), the weight $W_{{g,l}}$ is calculated as:")
st.markdown(r"$$ W_{{g,l}} = \frac{{N}}{{K \cdot N_{{g,l}}}} $$")
st.markdown(r"where:")
st.markdown(r"*   $N$ is the total number of samples in the training dataset.")
st.markdown(r"*   $K$ is the total number of unique (group, label) combinations.")
st.markdown(r"*   $N_{{g,l}}$ is the count of samples belonging to group $g$ with label $l$.")
st.markdown(f"")
st.markdown(f"This method ensures that each (group, label) intersection contributes equally to the loss function during training, helping to correct both class imbalance and group imbalance simultaneously. This is analogous to `class_weight='balanced'` in scikit-learn, but extended to group-outcome intersections.")
```

**Widgets and Interaction with `source.py`:**
```python
if not st.session_state.get('baseline_results'):
    st.warning("Please complete the '2. Baseline Model Performance' step first.")
    if st.button("Go to Baseline Model Performance"):
        st.session_state.current_page = "2. Baseline Model Performance"
        st.rerun()
else:
    if not st.session_state.get('reweighted_results'):
        with st.spinner("Applying sample reweighting and retraining model..."):
            fairness_weights = compute_fairness_weights(st.session_state.y_train, st.session_state.sensitive_train)
            st.info(f"Weight range for reweighting: {fairness_weights.min():.2f} to {fairness_weights.max():.2f}")
            
            reweighted_model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
            reweighted_model.fit(st.session_state.X_train[st.session_state.feature_cols], st.session_state.y_train, sample_weight=fairness_weights)
            
            y_pred_rw = reweighted_model.predict(st.session_state.X_test[st.session_state.feature_cols])
            y_prob_rw = reweighted_model.predict_proba(st.session_state.X_test[st.session_state.feature_cols])[:, 1]
            reweighted_results = evaluate_model(st.session_state.y_test, y_pred_rw, y_prob_rw, st.session_state.sensitive_test, 'Reweighted')
            
            st.session_state.reweighted_model = reweighted_model
            st.session_state.reweighted_results = reweighted_results
        st.success("Sample reweighting applied and model re-evaluated!")

    st.subheader("--- Reweighted Model Evaluation ---")
    st.write(pd.DataFrame([st.session_state.reweighted_results]).T.rename(columns={0: 'Value'}))

    st.subheader("3.1 Interpretation of Reweighting Results")
    st.markdown(f"After applying sample reweighting, I observe a noticeable improvement in our fairness metrics. The Disparate Impact Ratio (DIR) has increased to `{st.session_state.reweighted_results['dir']:.3f}`, which means it `{st.session_state.reweighted_results['four_fifths']}` the 0.80 threshold for the Four-fifths Rule. This indicates that the approval rates for `{st.session_state.unprivileged_group}` are now more comparable to `{st.session_state.privileged_group}`. The Statistical Parity Difference (SPD) of `{st.session_state.reweighted_results['spd']:.4f}` and Equal Opportunity Difference (EOD) of `{st.session_state.reweighted_results['eod']:.4f}` also show reductions, suggesting a fairer distribution of outcomes.")
    st.markdown(f"")
    st.markdown(f"However, this improvement in fairness typically comes with a slight trade-off in accuracy. The AUC for the reweighted model is `{st.session_state.reweighted_results['auc']:.4f}`, compared to the baseline AUC of `{st.session_state.baseline_results['auc']:.4f}`. As a CFA, I view this accuracy cost as the \"premium\" we pay for regulatory compliance insurance. It's a prudent risk management decision: a small reduction in overall predictive power prevents potentially massive regulatory penalties and reputational damage. This reweighting strategy is considered a \"low-regret\" option due to its minimal code changes and general legal acceptance.")
    st.markdown("---")
    if st.button("Explore Mitigation Strategy 2: Proxy Feature Removal"):
        st.session_state.current_page = "4. Strategy 2: Proxy Feature Removal"
        st.rerun()
```

#### Page: `4. Strategy 2: Proxy Feature Removal`

**Conditional Rendering Logic:**
```python
elif st.session_state.current_page == '4. Strategy 2: Proxy Feature Removal':
    # ... render content ...
```

**Markdown Content:**
```python
st.header("4. Strategy 2: Pre-Processing - Proxy Feature Removal")
st.markdown(f"Another pre-processing approach involves identifying and removing features that, while not explicitly sensitive attributes, act as proxies for them. These proxy features can inadvertently encode demographic information, leading to indirect discrimination. In our D4-T2-C1 analysis, we found that 'revolving_utilization' and 'home_ownership_encoded' showed strong correlations with 'race_group'. Removing these might break the indirect link to the sensitive attribute and reduce bias.")
st.warning("**Practitioner Warning:** Removing proxy features is the bluntest mitigation tool. While simple, it can be counterproductive if the proxy carries genuine predictive information, losing legitimate signal along with the bias. Worse, the model might compensate by shifting weight to other, less obvious proxies, a phenomenon known as \"proxy proliferation.\" Always re-run a full fairness audit after removal to confirm the bias has actually decreased.")
```

**Widgets and Interaction with `source.py`:**
```python
if not st.session_state.get('baseline_results'):
    st.warning("Please complete the '2. Baseline Model Performance' step first.")
    if st.button("Go to Baseline Model Performance"):
        st.session_state.current_page = "2. Baseline Model Performance"
        st.rerun()
else:
    proxy_features = ['revolving_utilization', 'home_ownership_encoded'] # Hardcoded as per source.py
    st.markdown(f"Based on prior analysis, the identified proxy features are: `{proxy_features}`")

    if not st.session_state.get('proxy_removed_results'):
        with st.spinner("Removing proxy features and retraining model..."):
            proxy_removed_results, proxy_removed_model = retrain_without_proxies(
                st.session_state.X_train, st.session_state.y_train, st.session_state.X_test, st.session_state.y_test,
                proxy_features, st.session_state.sensitive_test, st.session_state.feature_cols
            )
            st.session_state.proxy_removed_model = proxy_removed_model
            st.session_state.proxy_removed_results = proxy_removed_results
        st.success("Proxy features removed and model re-evaluated!")

    st.subheader("--- Proxy Removed Model Evaluation ---")
    st.write(pd.DataFrame([st.session_state.proxy_removed_results]).T.rename(columns={0: 'Value'}))

    st.subheader("4.1 Interpretation of Proxy Feature Removal Results")
    st.markdown(f"The impact of proxy feature removal on both fairness and accuracy is usually more unpredictable than reweighting. For this strategy, the DIR is `{st.session_state.proxy_removed_results['dir']:.3f}` (`{st.session_state.proxy_removed_results['four_fifths']}` the Four-fifths Rule) and AUC is `{st.session_state.proxy_removed_results['auc']:.4f}`. While it might improve fairness by reducing the indirect influence of sensitive attributes, it often leads to a more significant drop in overall model accuracy if the removed features were genuinely predictive. It's crucial to ensure that the fairness improvement is substantial enough to justify the accuracy loss.")
    st.markdown(f"")
    st.markdown(f"In this scenario, I observe the changes in DIR, SPD, and EOD. This highlights the trade-off. For regulators, this approach is easy to explain (\"we removed features that correlate with protected attributes\"), but from a technical standpoint, the risk of proxy proliferation means this approach requires careful validation and should not be assumed to fully eliminate bias.")
    st.markdown("---")
    if st.button("Explore Mitigation Strategy 3: Fairness-Constrained Training"):
        st.session_state.current_page = "5. Strategy 3: Fairness-Constrained Training"
        st.rerun()
```

#### Page: `5. Strategy 3: Fairness-Constrained Training`

**Conditional Rendering Logic:**
```python
elif st.session_state.current_page == '5. Strategy 3: Fairness-Constrained Training':
    # ... render content ...
```

**Markdown Content:**
```python
st.header("5. Strategy 3: In-Processing - Fairness-Constrained Training")
st.markdown(f"In-processing methods directly incorporate fairness considerations into the model's training algorithm. This is often considered the most principled approach as it allows the model to find the optimal balance between accuracy and fairness. `fairlearn`'s `ExponentiatedGradient` algorithm is a powerful tool that achieves this by solving a constrained optimization problem. It ensures that the model meets specific fairness criteria (like Demographic Parity or Equalized Odds) while maintaining the highest possible accuracy.")
st.markdown(r"The `ExponentiatedGradient` algorithm works by finding a randomized classifier that minimizes the standard loss function subject to fairness constraints. For example, for Demographic Parity, the objective is to minimize the model's loss $L(\theta)$ such that the Disparate Impact Ratio is maintained above a certain threshold (e.g., $0.80$):")
st.markdown(r"$$ \min_{{\theta}} L(\theta) \quad \text{{subject to}} \quad \text{{DIR}}(\theta) \geq 0.80 $$")
st.markdown(r"where $\theta$ represents the model parameters. This is achieved by finding the Lagrangian dual of the unconstrained problem, effectively adding a \"price\" for unfairness that the algorithm trades off against accuracy.")
st.markdown(f"")
```

**Widgets and Interaction with `source.py`:**
```python
if not st.session_state.get('baseline_results'):
    st.warning("Please complete the '2. Baseline Model Performance' step first.")
    if st.button("Go to Baseline Model Performance"):
        st.session_state.current_page = "2. Baseline Model Performance"
        st.rerun()
else:
    st.subheader("Select Fairness Constraint")
    constraint_choice = st.selectbox(
        "Choose a fairness constraint for `ExponentiatedGradient`:",
        ('demographic_parity', 'equalized_odds'),
        key='fairness_constraint_selection'
    )

    if st.button(f"Train Model with {constraint_choice.replace('_', ' ').title()} Constraint"):
        with st.spinner(f"Training fairness-constrained model with {constraint_choice.replace('_', ' ').title()}..."):
            fair_results, fair_model = train_fair_model(
                st.session_state.X_train[st.session_state.feature_cols], st.session_state.y_train,
                st.session_state.X_test[st.session_state.feature_cols], st.session_state.y_test,
                st.session_state.sensitive_train, st.session_state.sensitive_test,
                constraint=constraint_choice
            )
            if constraint_choice == 'demographic_parity':
                st.session_state.fair_dp_model = fair_model
                st.session_state.fair_dp_results = fair_results
            else: # equalized_odds
                st.session_state.fair_eod_model = fair_model
                st.session_state.fair_eod_results = fair_results
        st.success(f"Fairness-constrained model ({constraint_choice.replace('_', ' ').title()}) trained and evaluated!")

    if st.session_state.get('fair_dp_results') and constraint_choice == 'demographic_parity':
        st.subheader("--- Fairness-Constrained (Demographic Parity) Model Evaluation ---")
        st.write(pd.DataFrame([st.session_state.fair_dp_results]).T.rename(columns={0: 'Value'}))
        st.markdown(f"")
        st.markdown(f"The results from fairness-constrained training generally demonstrate a more balanced trade-off between accuracy and fairness compared to pre-processing methods. By optimizing for both simultaneously, `ExponentiatedGradient` aims to achieve the best possible accuracy while meeting the specified fairness constraints. I observe strong improvements in DIR (`{st.session_state.fair_dp_results['dir']:.3f}`), SPD (`{st.session_state.fair_dp_results['spd']:.4f}`), and EOD (`{st.session_state.fair_dp_results['eod']:.4f}`), often leading to full compliance with regulations like the Four-fifths Rule. The AUC drop is typically more controlled (`{st.session_state.fair_dp_results['auc']:.4f}`) than with blunt feature removal.")
        st.markdown(f"")
        st.markdown(f"This is the most principled approach, offering a robust solution to bias. However, it requires specialized algorithms and understanding of various fairness definitions (Demographic Parity vs. Equalized Odds). For Apex Credit Solutions, this method is ideal for long-term strategic model development, where we can invest the time to build a truly fair model.")

    if st.session_state.get('fair_eod_results') and constraint_choice == 'equalized_odds':
        st.subheader("--- Fairness-Constrained (Equalized Odds) Model Evaluation ---")
        st.write(pd.DataFrame([st.session_state.fair_eod_results]).T.rename(columns={0: 'Value'}))
        st.markdown(f"")
        st.markdown(f"Similar to Demographic Parity, the Equalized Odds constraint also aims to balance fairness and accuracy. For this constraint, the DIR is `{st.session_state.fair_eod_results['dir']:.3f}`, SPD is `{st.session_state.fair_eod_results['spd']:.4f}`, EOD is `{st.session_state.fair_eod_results['eod']:.4f}`, and AUC is `{st.session_state.fair_eod_results['auc']:.4f}`. This method is effective when the goal is to ensure equal true positive rates across groups, addressing potential issues where a deserving individual from one group is less likely to be approved than a deserving individual from another.")
        st.markdown(f"")
        st.markdown(f"This principled approach requires careful consideration of the specific fairness definition that aligns with regulatory and ethical objectives. For Apex Credit Solutions, understanding the nuances between Demographic Parity and Equalized Odds is crucial for selecting the most appropriate long-term strategy.")
    st.markdown("---")
    if st.button("Explore Mitigation Strategy 4: Group-Specific Threshold Calibration"):
        st.session_state.current_page = "6. Strategy 4: Group-Specific Threshold Calibration"
        st.rerun()
```

#### Page: `6. Strategy 4: Group-Specific Threshold Calibration`

**Conditional Rendering Logic:**
```python
elif st.session_state.current_page == '6. Strategy 4: Group-Specific Threshold Calibration':
    # ... render content ...
```

**Markdown Content:**
```python
st.header("6. Strategy 4: Post-Processing - Group-Specific Threshold Calibration")
st.markdown(f"Post-processing involves adjusting the model's outputs after predictions have been made, without altering the underlying model itself. A common technique is group-specific threshold calibration, where different decision thresholds are applied to different sensitive groups to achieve a target fairness level. For instance, if `{st.session_state.unprivileged_group}` has a lower approval rate, we might lower their approval threshold while keeping `{st.session_state.privileged_group}`'s threshold fixed.")
st.markdown(r"The objective here is to find group-specific thresholds that lead to a desired Disparate Impact Ratio. We can achieve this by keeping the advantaged group's threshold fixed (e.g., at $0.5$) and searching for the disadvantaged group's threshold that equalizes approval rates to achieve the target DIR. This search can be performed using scalar optimization techniques.")
st.markdown(r"")
st.warning("**Practitioner Warning:** Post-processing, especially group-specific thresholds, is legally controversial. Explicitly using protected attributes to set different decision thresholds can be viewed as discriminatory itself by some legal interpretations (\"reverse discrimination\"). While faster to deploy and easier to reverse, it requires careful legal review and approval from our firm's legal counsel. It is often best used as an interim fix while a more principled in-processing solution is being developed.")
```

**Widgets and Interaction with `source.py`:**
```python
if not st.session_state.get('baseline_results'):
    st.warning("Please complete the '2. Baseline Model Performance' step first.")
    if st.button("Go to Baseline Model Performance"):
        st.session_state.current_page = "2. Baseline Model Performance"
        st.rerun()
else:
    st.subheader("Calibrate Thresholds for Fairness")
    target_dir_input = st.slider(
        "Select Target Disparate Impact Ratio (DIR) for Calibration:",
        min_value=0.7, max_value=1.0, value=0.85, step=0.01,
        help="Aim for a DIR of at least 0.80 to satisfy the Four-fifths Rule."
    )

    if st.button("Calibrate Group-Specific Thresholds"):
        with st.spinner(f"Calibrating group-specific thresholds for a target DIR of {target_dir_input:.2f}..."):
            threshold_calibrated_results = calibrate_group_thresholds(
                st.session_state.y_test, st.session_state.y_prob_base,
                st.session_state.sensitive_test, target_dir=target_dir_input
            )
            st.session_state.threshold_calibrated_results = threshold_calibrated_results
        st.success("Thresholds calibrated and model re-evaluated!")

    if st.session_state.get('threshold_calibrated_results'):
        st.subheader("--- Threshold Calibrated Model Evaluation ---")
        st.write(pd.DataFrame([st.session_state.threshold_calibrated_results]).T.rename(columns={0: 'Value'}))
        st.markdown(f"**Calibrated Thresholds:** {st.session_state.threshold_calibrated_results.get('thresholds', 'N/A')}")

        st.subheader("6.1 Interpretation of Group-Specific Threshold Calibration Results")
        st.markdown(f"Threshold calibration can be highly effective in closing the fairness gap quickly. I observe a significant improvement in the Disparate Impact Ratio (DIR) to `{st.session_state.threshold_calibrated_results['dir']:.3f}`, often meeting or exceeding our target, and corresponding reductions in SPD (`{st.session_state.threshold_calibrated_results['spd']:.4f}`) and EOD (`{st.session_state.threshold_calibrated_results['eod']:.4f}`). The crucial advantage of post-processing is that it doesn't require retraining the model, making it exceptionally fast to implement and revert. This means the AUC remains unchanged or changes only marginally (`{st.session_state.threshold_calibrated_results['auc']:.4f}`) depending on how the initial probability is interpreted for evaluation, as we are only changing the decision boundary, not the model's underlying scores.")
        st.markdown(f"")
        st.markdown(f"For Apex Credit Solutions, this is a valuable tool for rapid response to detected bias or as an interim solution. However, the legal and ethical implications of using group membership to differentiate decision thresholds demand thorough consultation with our legal department before deployment.")
    st.markdown("---")
    if st.button("View Comprehensive Comparison and Recommendation"):
        st.session_state.current_page = "7. Comprehensive Comparison & Recommendation"
        st.rerun()
```

#### Page: `7. Comprehensive Comparison & Recommendation`

**Conditional Rendering Logic:**
```python
elif st.session_state.current_page == '7. Comprehensive Comparison & Recommendation':
    # ... render content ...
```

**Markdown Content:**
```python
st.header("7. Comprehensive Comparison and Mitigation Selection Guide")
st.markdown(f"Having explored four distinct bias mitigation strategies, it's time to consolidate our findings into a comprehensive comparison. As a CFA, my role now is to synthesize these results, analyze the accuracy-fairness trade-offs, and recommend the most appropriate strategy for Apex Credit Solutions, considering regulatory requirements, operational feasibility, and ethical considerations.")
st.markdown(f"")
```

**Widgets and Interaction with `source.py` (and plotting functions):**
```python
if not st.session_state.get('threshold_calibrated_results') or not st.session_state.get('fair_dp_results') or not st.session_state.get('reweighted_results') or not st.session_state.get('proxy_removed_results'): # Ensure all strategies have been run
    st.warning("Please complete all mitigation strategies (pages 3-6) before viewing the comprehensive comparison.")
    if st.button("Go to Strategy 1"):
        st.session_state.current_page = "3. Strategy 1: Sample Reweighting"
        st.rerun()
else:
    # Collect all results into a DataFrame
    all_results_list = [st.session_state.baseline_results]
    if st.session_state.get('reweighted_results'):
        all_results_list.append(st.session_state.reweighted_results)
    if st.session_state.get('proxy_removed_results'):
        all_results_list.append(st.session_state.proxy_removed_results)
    if st.session_state.get('fair_dp_results'):
        all_results_list.append(st.session_state.fair_dp_results)
    if st.session_state.get('fair_eod_results'): # Include if EOD was trained
        all_results_list.append(st.session_state.fair_eod_results)
    if st.session_state.get('threshold_calibrated_results'):
        all_results_list.append(st.session_state.threshold_calibrated_results)
    
    st.session_state.all_results_df = pd.DataFrame(all_results_list)

    st.subheader("--- FIVE-WAY MITIGATION COMPARISON ---")
    metrics_display_cols = ['label', 'auc', 'f1', 'dir', 'spd', 'eod', 'four_fifths']
    st.dataframe(st.session_state.all_results_df[metrics_display_cols])

    st.subheader("--- ACCURACY COST OF EACH MITIGATION ---")
    baseline_auc = st.session_state.baseline_results['auc']
    baseline_dir = st.session_state.baseline_results['dir']

    auc_cost_data = []
    for index, row in st.session_state.all_results_df.iterrows():
        if row['label'] != 'Baseline (Biased)':
            auc_cost = baseline_auc - row['auc']
            dir_gain = row['dir'] - baseline_dir
            auc_cost_data.append({'Strategy': row['label'], 'AUC Cost': auc_cost, 'DIR Gain': dir_gain})
    df_auc_cost_summary = pd.DataFrame(auc_cost_data)
    st.dataframe(df_auc_cost_summary.set_index('Strategy'))

    st.subheader("7.1 Visualizing Mitigation Impacts")
    st.markdown("Visualizations are key to understanding the trade-offs and communicating our findings to stakeholders, including risk committees and legal counsel.")

    st.markdown("#### 7.1.1 Accuracy-Fairness Pareto Frontier")
    st.markdown(f"This scatter plot illustrates the trade-off between accuracy (AUC) and fairness (DIR) for each strategy. It helps identify Pareto-optimal strategies—those where you cannot improve fairness without sacrificing accuracy, or vice-versa. The Four-fifths Rule threshold line is critical for visual compliance assessment.")
    fig_pareto, ax_pareto = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='dir', y='auc', hue='label', data=st.session_state.all_results_df, s=150, style='label', ax=ax_pareto)
    ax_pareto.axvline(x=0.80, color='r', linestyle='--', label='Four-fifths Rule (0.80 DIR)')
    ax_pareto.set_title('Accuracy-Fairness Pareto Frontier (AUC vs. DIR)')
    ax_pareto.set_xlabel('Disparate Impact Ratio (DIR)')
    ax_pareto.set_ylabel('Area Under ROC Curve (AUC)')
    ax_pareto.set_xlim(0.5, 1.1)
    ax_pareto.set_ylim(0.7, 0.9)
    ax_pareto.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_pareto.grid(True)
    st.pyplot(fig_pareto)
    plt.close(fig_pareto) # Close the figure to prevent display issues

    st.markdown("#### 7.1.2 Before/After Approval Rate Bars for Sensitive Groups")
    st.markdown(f"These grouped bar charts visually demonstrate how each strategy impacts the approval rates for different sensitive groups, explicitly showing the gap closure.")
    privileged_group = st.session_state.privileged_group
    unprivileged_group = st.session_state.unprivileged_group

    df_plot_rates = pd.DataFrame(columns=['Strategy', 'Group', 'Approval Rate'])
    for _, row in st.session_state.all_results_df.iterrows():
        rates = row['approval_rates']
        df_plot_rates = pd.concat([df_plot_rates, pd.DataFrame([
            {'Strategy': row['label'], 'Group': privileged_group, 'Approval Rate': rates.get(privileged_group, 0)},
            {'Strategy': row['label'], 'Group': unprivileged_group, 'Approval Rate': rates.get(unprivileged_group, 0)}
        ])], ignore_index=True)

    fig_rates, ax_rates = plt.subplots(figsize=(12, 7))
    sns.barplot(x='Strategy', y='Approval Rate', hue='Group', data=df_plot_rates, palette='viridis', ax=ax_rates)
    ax_rates.set_title(f'Approval Rates by Strategy for {privileged_group} and {unprivileged_group}')
    ax_rates.set_ylabel('Approval Rate')
    ax_rates.set_xlabel('Mitigation Strategy')
    ax_rates.set_xticklabels(ax_rates.get_xticklabels(), rotation=45, ha='right')
    ax_rates.set_ylim(0, df_plot_rates['Approval Rate'].max() * 1.1)
    ax_rates.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(fig_rates)
    st.pyplot(fig_rates)
    plt.close(fig_rates)

    st.markdown("#### 7.1.3 AUC Cost Bar Chart")
    st.markdown(f"This bar chart clearly illustrates the accuracy cost of each mitigation strategy relative to the biased baseline.")
    fig_auc_cost, ax_auc_cost = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Strategy', y='AUC Cost', data=df_auc_cost_summary, palette='magma', ax=ax_auc_cost)
    ax_auc_cost.set_title('Accuracy Cost (AUC Reduction) per Mitigation Strategy')
    ax_auc_cost.set_ylabel('AUC Reduction (Baseline AUC - Strategy AUC)')
    ax_auc_cost.set_xlabel('Mitigation Strategy')
    ax_auc_cost.set_xticklabels(ax_auc_cost.get_xticklabels(), rotation=45, ha='right')
    ax_auc_cost.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(fig_auc_cost)
    st.pyplot(fig_auc_cost)
    plt.close(fig_auc_cost)

    st.subheader("7.2 Mitigation Selection Guide and Recommendation")
    st.markdown(f"Based on the quantitative metrics and visualizations, I can now formulate a strategic recommendation for Apex Credit Solutions.")
    
    # Static table as markdown
    st.markdown("""
| Strategy | Best When | AUC Cost | Complexity | Legal Risk |
| :-------------------- | :------------------------------------------------------------------------------------------------------------------------- | :------- | :--------- | :--------- |
| **Reweighting** | Mild to moderate bias, quick fix needed, low-regret. | 0.5-2% | Low | None |
| **Proxy Feature Removal** | Clear proxy identified, legally required to remove. | 1-4% | Low | None |
| **Fair Constraints** | Moderate-severe bias, principled solution needed, long-term strategy. | 1-3% | Medium | None |
| **Threshold Calib.** | Interim fix while retraining, fast deployment, temporary measure, with legal approval. | 0-1% | Low | High |
""")

    st.markdown(f"**My Recommendation for Apex Credit Solutions:**")
    st.markdown(f"Given the critical nature of credit decisions and the high regulatory stakes:")
    st.markdown(f"1.  **Immediate Action (Interim Fix):** Implement **Sample Reweighting**. It offers a rapid improvement in fairness with a manageable accuracy trade-off, is legally low-risk, and requires minimal operational changes. This addresses the most urgent compliance failure immediately.")
    st.markdown(f"    *   *Rationale:* This is our \"low-regret\" strategy, providing acceptable fairness gains quickly while we work on a more robust solution.")
    st.markdown(f"2.  **Strategic Long-Term Solution:** Initiate development and validation for a model using **Fairness-Constrained Training** (e.g., Demographic Parity). This in-processing approach offers the most principled and robust solution by optimizing for both accuracy and fairness simultaneously, leading to better compliance stability.")
    st.markdown(f"    *   *Rationale:* While requiring more development time, this strategy yields the most ethically sound and resilient model. The \"cost\" of mitigation (1-3% AUC) is a prudent \"premium\" for regulatory compliance insurance, preventing a potential $50M regulatory penalty.")
    st.markdown(f"3.  **Conditional Use (Contingency):** **Group-Specific Threshold Calibration** should be considered only as an emergency interim measure if immediate compliance is paramount and the other options are not feasible in time. This would require explicit legal review and approval due to its controversial nature, as it directly uses sensitive attributes in decision-making.")
    st.markdown(f"4.  **Avoidance (Unless necessary):** **Proxy Feature Removal** should generally be avoided unless legally mandated or if proxies are demonstrably non-predictive. Its unpredictable impact on accuracy and the risk of proxy proliferation make it less desirable for critical credit models.")
    st.markdown("---")
    if st.button("Generate Compliance Report"):
        st.session_state.current_page = "8. Compliance Report"
        st.rerun()
```

#### Page: `8. Compliance Report`

**Conditional Rendering Logic:**
```python
elif st.session_state.current_page == '8. Compliance Report':
    # ... render content ...
```

**Markdown Content:**
```python
st.header("8. Bias Mitigation Compliance Report")
st.markdown(f"This report summarizes the findings and recommendations for addressing bias in Apex Credit Solutions' credit scoring model. This document serves as a compliance-ready record for internal audits and regulatory bodies.")
st.markdown(f"")
st.markdown(f"---")
st.subheader("Compliance Documentation for Risk Committee:")
st.markdown(f"*   **Bias Detected:** Disparate Impact against `{st.session_state.unprivileged_group}` (Baseline DIR: `{st.session_state.baseline_results['dir']:.3f}`, failing Four-fifths Rule).")

# Dynamically pick the best strategy for the report based on recommendation (Reweighting for interim, Fair DP for strategic)
# Ensure these session state variables exist before accessing them
interim_strategy_label = "Sample Reweighting"
interim_auc_cost = (st.session_state.baseline_results['auc'] - st.session_state.reweighted_results['auc']) if st.session_state.get('reweighted_results') else 0.0
interim_dir = st.session_state.reweighted_results['dir'] if st.session_state.get('reweighted_results') else st.session_state.baseline_results['dir']

strategic_strategy_label = "Fairness-Constrained Training (Demographic Parity)"
strategic_auc_cost = (st.session_state.baseline_results['auc'] - st.session_state.fair_dp_results['auc']) if st.session_state.get('fair_dp_results') else 0.0
strategic_dir = st.session_state.fair_dp_results['dir'] if st.session_state.get('fair_dp_results') else st.session_state.baseline_results['dir']

st.markdown(f"*   **Mitigation Applied:** {interim_strategy_label} (Interim) and {strategic_strategy_label} (Strategic).")
st.markdown(f"*   **Impact on Accuracy/Fairness (Interim - {interim_strategy_label}):** AUC: `{st.session_state.reweighted_results['auc']:.4f}` (cost: `{interim_auc_cost:+.4f}`), DIR: `{interim_dir:.3f}` (passing Four-fifths Rule).")
st.markdown(f"*   **Impact on Accuracy/Fairness (Strategic - {strategic_strategy_label}):** AUC: `{st.session_state.fair_dp_results['auc']:.4f}` (cost: `{strategic_auc_cost:+.4f}`), DIR: `{strategic_dir:.3f}` (passing Four-fifths Rule).")
st.markdown(f"*   **Trade-off Approved By:** [Signatures of Head of Risk, Head of Compliance, Legal Counsel].")
st.markdown(f"")
st.markdown(f"This structured approach ensures that Apex Credit Solutions addresses the detected bias pragmatically and ethically, balancing business needs with regulatory demands and our commitment to fair lending.")
st.markdown(f"")
st.markdown(f"---")
st.markdown(f"**Further Discussion Points for Financial Professionals:**")
st.markdown(f"1.  **Deployment Pragmatism (D1):** The \"best\" mitigation is the one deployed. Prioritize fast, acceptable fixes (like reweighting) while developing principled, long-term solutions (like fairness-constrained training).")
st.markdown(f"2.  **Post-Processing Legal Nuance (D2):** Group-specific thresholds, while effective, raise ethical questions about treating individuals differently based on group membership, requiring extensive legal review.")
st.markdown(f"3.  **Reweighting as 'Low-Regret' (D3):** Sample reweighting is often the first, legally uncontroversial, and operationally simple intervention.")
st.markdown(f"")
st.info("This concludes the bias mitigation analysis. You can navigate back through the sidebar to review any section.")
```

**Widgets and Interaction with `source.py`:**
```python
# No explicit interactions on this page beyond navigation
```
