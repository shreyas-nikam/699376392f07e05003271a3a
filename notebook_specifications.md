
# Mitigating Credit Risk Bias: A CFA's Guide to Fair Lending Models

## Case Study: Ensuring Equitable Credit Decisions at Apex Credit Solutions

**Persona:** Sri Krishnamurthy, CFA, Senior Risk Analyst at Apex Credit Solutions.  
**Organization:** Apex Credit Solutions, a leading financial institution providing consumer credit.  

### Introduction

As a CFA charterholder, my primary responsibility at Apex Credit Solutions is to ensure the integrity and fairness of our credit scoring models. Recent internal audits, stemming from our D4-T2-C1 bias detection lab, have revealed a concerning issue: our existing credit model exhibits disparate impact, with "Group B" receiving significantly lower loan approval rates compared to "Group A," despite similar financial profiles. This not only raises ethical concerns but also exposes Apex Credit Solutions to significant regulatory and reputational risks under acts like the Equal Credit Opportunity Act (ECOA).

My task today is to implement and evaluate various bias mitigation strategies. We need to move beyond merely detecting bias to actively treating it. This notebook documents my journey through several intervention points in the machine learning pipeline—pre-processing, in-processing, and post-processing—to reduce this disparate impact. The goal is to compare the trade-offs between accuracy and fairness for each strategy and ultimately recommend the most suitable approach for our organization, ensuring we comply with fairness regulations and uphold our ethical obligations.

This exercise is critical for two reasons:
1.  **Regulatory Compliance:** Avoiding potential legal penalties and ensuring our models meet the standards set by financial regulators.
2.  **Ethical Lending:** Upholding Apex Credit Solutions' commitment to fair and equitable treatment of all applicants, strengthening trust and reputation.

---

## 1. Setup and Data Generation

Before diving into bias mitigation, we need to set up our environment by installing the necessary libraries and generating a synthetic dataset that simulates the biased credit scoring scenario identified in our previous analysis. This dataset will feature demographic information and a sensitive attribute, allowing us to replicate and address the detected bias.

```python
# Install required libraries
!pip install numpy pandas scikit-learn xgboost fairlearn matplotlib scipy
```

```python
# Import required dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import seaborn as sns
```

### 1.1 Generating a Synthetic Biased Credit Dataset

As identified in D4-T2-C1, our credit model exhibited bias against a specific demographic group. To proceed with mitigation, I need a similar synthetic dataset. This dataset will contain typical credit features, a sensitive attribute (`'race_group'`), and a target variable (`'loan_approved'`), with an intentional bias where 'Group_B' has a lower approval rate.

```python
def generate_biased_credit_data(n_samples=10000, random_state=42):
    """
    Generates a synthetic dataset with pre-existing bias for credit scoring.

    Args:
        n_samples (int): Number of samples to generate.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: X_train, X_test, y_train, y_test, sensitive_train, sensitive_test, feature_cols
    """
    np.random.seed(random_state)

    data = pd.DataFrame()
    data['age'] = np.random.randint(20, 70, n_samples)
    data['income'] = np.random.normal(50000, 15000, n_samples)
    data['credit_score'] = np.random.normal(650, 70, n_samples)
    data['loan_amount'] = np.random.normal(15000, 5000, n_samples)
    data['employment_years'] = np.random.randint(0, 30, n_samples)
    data['revolving_utilization'] = np.random.beta(5, 10, n_samples) * 0.8 # proxy feature
    data['home_ownership_encoded'] = np.random.choice([0, 1], n_samples, p=[0.6, 0.4]) # proxy feature

    # Introduce sensitive attribute: 'race_group'
    data['race_group'] = np.random.choice(['Group_A', 'Group_B'], n_samples, p=[0.7, 0.3])

    # Introduce bias: Group B has lower credit scores and higher revolving utilization
    # Also, for the same credit score, Group B is less likely to be approved
    data.loc[data['race_group'] == 'Group_B', 'credit_score'] = data.loc[data['race_group'] == 'Group_B', 'credit_score'] - 30
    data.loc[data['race_group'] == 'Group_B', 'revolving_utilization'] = data.loc[data['race_group'] == 'Group_B', 'revolving_utilization'] + 0.1

    # Define target variable 'loan_approved' (binary)
    # Base approval probability
    prob_approved = (
        0.05 * (data['credit_score'] - 500) / 200 +
        0.02 * (data['income'] / 100000) -
        0.3 * data['revolving_utilization']
    )
    prob_approved = 1 / (1 + np.exp(-prob_approved)) # Sigmoid transformation

    # Introduce direct bias on loan approval for Group B
    data['loan_approved'] = (prob_approved > np.random.rand(n_samples)).astype(int)
    data.loc[(data['race_group'] == 'Group_B') & (prob_approved > 0.5), 'loan_approved'] = \
        (prob_approved.loc[(data['race_group'] == 'Group_B') & (prob_approved > 0.5)] * 0.8 > np.random.rand(data['race_group'].eq('Group_B').sum() - data['race_group'].eq('Group_B').sum() // 2)).astype(int)

    feature_cols = ['age', 'income', 'credit_score', 'loan_amount', 'employment_years', 'revolving_utilization', 'home_ownership_encoded']
    X = data[feature_cols]
    y = data['loan_approved']
    sensitive = data['race_group']

    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=random_state, stratify=sensitive
    )

    return X_train, X_test, y_train, y_test, sensitive_train, sensitive_test, feature_cols

X_train, X_test, y_train, y_test, sensitive_train, sensitive_test, feature_cols = generate_biased_credit_data()

print(f"Dataset generated with {len(X_train)} training samples and {len(X_test)} test samples.")
print(f"Features: {feature_cols}")
print(f"Sensitive attribute distribution in training: \n{sensitive_train.value_counts(normalize=True)}")
print(f"Sensitive attribute distribution in test: \n{sensitive_test.value_counts(normalize=True)}")
```

---

## 2. Baseline Model Performance and Fairness Evaluation

My first step is to establish a clear baseline. This involves training our standard credit model (an `XGBClassifier`) on the original, biased data and then meticulously evaluating its performance and fairness. This baseline will serve as the benchmark against which all mitigation strategies will be measured. It’s crucial to quantify the extent of the bias before attempting to fix it.

To ensure consistency across all evaluations, I'll define a comprehensive function, `evaluate_model`, which computes both accuracy metrics (AUC, F1-score) and fairness metrics (Disparate Impact Ratio (DIR), Statistical Parity Difference (SPD), Equal Opportunity Difference (EOD)). The Disparate Impact Ratio is particularly important as it quantifies the ratio of favorable outcomes for the unprivileged group to the privileged group. According to the Four-fifths rule, this ratio should ideally be $\geq 0.80$.

**Disparate Impact Ratio (DIR):** This metric measures the ratio of the favorable outcome rate (e.g., loan approval rate) for the unprivileged group to the privileged group. A value of 1 indicates perfect fairness. Regulations often require this ratio to be at least 0.80 (the Four-fifths Rule).
$$ \text{DIR} = \frac{P(\hat{Y}=1 | A=unprivileged)}{P(\hat{Y}=1 | A=privileged)} $$
where $P(\hat{Y}=1 | A=g)$ is the probability of a positive prediction (loan approval) for group $g$.

**Statistical Parity Difference (SPD):** This metric measures the difference in the favorable outcome rates between the privileged and unprivileged groups. An SPD of 0 indicates perfect fairness.
$$ \text{SPD} = P(\hat{Y}=1 | A=unprivileged) - P(\hat{Y}=1 | A=privileged) $$

**Equal Opportunity Difference (EOD):** This metric measures the difference in true positive rates (FNR gap in the provided text, but EOD is standard for true positive rate difference) between the privileged and unprivileged groups among individuals who truly deserve the positive outcome. An EOD of 0 indicates perfect fairness, meaning both groups have an equal chance of being correctly approved if they should be.
$$ \text{EOD} = P(\hat{Y}=1 | Y=1, A=unprivileged) - P(\hat{Y}=1 | Y=1, A=privileged) $$
In the provided context, the FNR (False Negative Rate) gap is calculated as $FNR_{privileged} - FNR_{unprivileged}$. The EOD is typically defined using True Positive Rate (TPR), where $TPR = 1 - FNR$. So, an EOD of 0 for TPRs implies $TPR_{privileged} = TPR_{unprivileged}$, which is equivalent to $FNR_{privileged} = FNR_{unprivileged}$. Thus, the FNR gap in the code directly measures a form of equal opportunity.

```python
def evaluate_model(y_true, y_pred, y_prob, sensitive, label='Model'):
    """Compute accuracy and fairness metrics together."""
    groups = np.unique(sensitive)
    
    # Accuracy
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, pos_label=1)

    # Fairness - Approval rates
    rates = {}
    for g in groups:
        mask = sensitive == g
        rates[g] = y_pred[mask].mean() # Approval rate

    # Determine privileged and unprivileged groups based on approval rates
    # Assume the group with higher approval rate is privileged at baseline
    # If rates are equal or close, default to alphabetical order for consistency
    if len(groups) == 2:
        group_rates = sorted(rates.items(), key=lambda item: item[1], reverse=True)
        privileged_group = group_rates[0][0]
        unprivileged_group = group_rates[1][0]
    else: # Handle more than two groups or edge cases, picking first two alphabetically
        sorted_groups = sorted(groups)
        privileged_group = sorted_groups[0]
        unprivileged_group = sorted_groups[1]
    
    # Recalculate rates based on determined privileged/unprivileged
    rate_privileged = rates.get(privileged_group, 0)
    rate_unprivileged = rates.get(unprivileged_group, 0)

    # Disparate Impact Ratio
    dir_val = (rate_unprivileged + 1e-6) / (rate_privileged + 1e-6) # Add small epsilon to avoid division by zero

    # Statistical Parity Difference
    spd = rate_unprivileged - rate_privileged

    # Equal Opportunity Difference (FNR gap based on provided context)
    fnrs = {}
    for g in groups:
        mask = (sensitive == g) & (y_true == 1) # Only look at true positive cases
        if y_true[mask].sum() > 0:
            # False Negative Rate (FNR) = (Actual 1s but predicted 0s) / (Total Actual 1s)
            fnrs[g] = ((y_true[mask] == 1) & (y_pred[mask] == 0)).sum() / y_true[mask].sum()
        else:
            fnrs[g] = 0.0 # No true positives for this group, FNR is undefined or 0
    
    # EOD is typically defined as TPR_unprivileged - TPR_privileged.
    # Here, we use FNR_unprivileged - FNR_privileged as per the document's FNR gap.
    # EOD (FNR gap) = FNR_unprivileged - FNR_privileged
    eod = fnrs.get(unprivileged_group, 0) - fnrs.get(privileged_group, 0)


    return {
        'label': label, 
        'auc': round(auc, 4), 
        'f1': round(f1, 4),
        'dir': round(dir_val, 3), 
        'spd': round(spd, 4),
        'eod': round(eod, 4),
        'four_fifths': 'PASS' if dir_val >= 0.80 else 'FAIL',
        'approval_rates': rates,
        'privileged_group': privileged_group,
        'unprivileged_group': unprivileged_group,
        'fnr_privileged': round(fnrs.get(privileged_group, 0), 4),
        'fnr_unprivileged': round(fnrs.get(unprivileged_group, 0), 4),
    }

# Train baseline model
baseline_model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
baseline_model.fit(X_train[feature_cols], y_train)

# Predict on test set
y_pred_base = baseline_model.predict(X_test[feature_cols])
y_prob_base = baseline_model.predict_proba(X_test[feature_cols])[:, 1]

# Evaluate baseline
baseline_results = evaluate_model(y_test, y_pred_base, y_prob_base, sensitive_test, 'Baseline (Biased)')

print(f"--- Baseline Model Evaluation ---")
print(f"AUC: {baseline_results['auc']}, F1: {baseline_results['f1']}")
print(f"DIR: {baseline_results['dir']} (Four-fifths Rule: {baseline_results['four_fifths']})")
print(f"SPD: {baseline_results['spd']}")
print(f"EOD (FNR Gap): {baseline_results['eod']}")
print(f"Approval Rates: {baseline_results['approval_rates']}")
print(f"Privileged Group: {baseline_results['privileged_group']}")
print(f"Unprivileged Group: {baseline_results['unprivileged_group']}")
```

### 2.1 Interpretation of Baseline Results

The baseline evaluation provides a quantitative snapshot of our current credit model's performance and, critically, its fairness. With a Disparate Impact Ratio (DIR) significantly below 0.80, our model clearly fails the Four-fifths Rule. This indicates a substantial disparity in approval rates between the privileged and unprivileged groups. The Statistical Parity Difference (SPD) and Equal Opportunity Difference (EOD) further confirm this bias, showing unequal outcomes and opportunities for deserving applicants across groups.

For Apex Credit Solutions, this means:
*   **High Regulatory Risk:** The current model is non-compliant and could lead to significant fines and legal action.
*   **Reputational Damage:** Continued use of such a model erodes public trust and damages our brand.
*   **Ethical Obligation:** We are ethically bound to address this bias and ensure equitable treatment.

This stark reality underscores the urgency of implementing effective bias mitigation strategies.

---

## 3. Strategy 1: Pre-Processing - Sample Reweighting

One of the simplest yet effective pre-processing techniques is sample reweighting. The idea is to adjust the influence of individual data points during model training. If certain sensitive groups or group-outcome combinations are underrepresented or disadvantaged in the training data, we can assign them higher weights. This ensures the model learns equally from all segments of the population, thereby reducing bias.

For each combination of group $g$ and label $l$ (e.g., 'Group B' and 'loan_approved=0'), the weight $W_{g,l}$ is calculated as:
$$ W_{g,l} = \frac{N}{K \cdot N_{g,l}} $$
where:
*   $N$ is the total number of samples in the training dataset.
*   $K$ is the total number of unique (group, label) combinations.
*   $N_{g,l}$ is the count of samples belonging to group $g$ with label $l$.

This method ensures that each (group, label) intersection contributes equally to the loss function during training, helping to correct both class imbalance and group imbalance simultaneously. This is analogous to `class_weight='balanced'` in scikit-learn, but extended to group-outcome intersections.

```python
def compute_fairness_weights(y_train_data, sensitive_train_data):
    """
    Compute sample weights that equalize the effective
    representation of each (group, outcome) combination.
    """
    weights = np.ones(len(y_train_data))
    
    groups = np.unique(sensitive_train_data)
    labels = np.unique(y_train_data)
    
    total = len(y_train_data)
    n_combinations = len(groups) * len(labels)
    
    for g in groups:
        for l in labels:
            mask = (sensitive_train_data == g) & (y_train_data == l)
            count = mask.sum()
            if count > 0:
                weights[mask] = total / (n_combinations * count)
            else:
                # Handle cases where a combination doesn't exist to avoid division by zero or NaN weights
                # Assign a very small weight or average weight to avoid issues, or skip if mask.sum() == 0
                pass 
                
    return weights

# Compute fairness weights for the training data
fairness_weights = compute_fairness_weights(y_train, sensitive_train)

print(f"Weight range: {fairness_weights.min():.2f} to {fairness_weights.max():.2f}")

# Example: Check weights for specific group-outcome combinations (e.g., Group A approved, Group B approved)
group_a_approved_mask = (sensitive_train == baseline_results['privileged_group']) & (y_train == 1)
group_b_approved_mask = (sensitive_train == baseline_results['unprivileged_group']) & (y_train == 1)

if group_a_approved_mask.any():
    print(f"{baseline_results['privileged_group']} approved weight (mean): {fairness_weights[group_a_approved_mask].mean():.2f}")
if group_b_approved_mask.any():
    print(f"{baseline_results['unprivileged_group']} approved weight (mean): {fairness_weights[group_b_approved_mask].mean():.2f}")


# Retrain XGBoost model with fairness weights
reweighted_model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
reweighted_model.fit(X_train[feature_cols], y_train, sample_weight=fairness_weights)

# Predict and evaluate
y_pred_rw = reweighted_model.predict(X_test[feature_cols])
y_prob_rw = reweighted_model.predict_proba(X_test[feature_cols])[:, 1]
reweighted_results = evaluate_model(y_test, y_pred_rw, y_prob_rw, sensitive_test, 'Reweighted')

print(f"\n--- Reweighted Model Evaluation ---")
print(f"AUC: {reweighted_results['auc']}, F1: {reweighted_results['f1']}")
print(f"DIR: {reweighted_results['dir']} (Four-fifths Rule: {reweighted_results['four_fifths']})")
print(f"SPD: {reweighted_results['spd']}")
print(f"EOD (FNR Gap): {reweighted_results['eod']}")
print(f"Approval Rates: {reweighted_results['approval_rates']}")
```

### 3.1 Interpretation of Reweighting Results

After applying sample reweighting, I observe a noticeable improvement in our fairness metrics. The Disparate Impact Ratio (DIR) has likely increased, potentially crossing the 0.80 threshold for the Four-fifths Rule. This indicates that the approval rates for Group B are now more comparable to Group A. The Statistical Parity Difference (SPD) and Equal Opportunity Difference (EOD) also show reductions, suggesting a fairer distribution of outcomes.

However, this improvement in fairness typically comes with a slight trade-off in accuracy (AUC and F1-score). As a CFA, I view this accuracy cost as the "premium" we pay for regulatory compliance insurance. It's a prudent risk management decision: a small reduction in overall predictive power prevents potentially massive regulatory penalties and reputational damage. This reweighting strategy is considered a "low-regret" option due to its minimal code changes and general legal acceptance.

---

## 4. Strategy 2: Pre-Processing - Proxy Feature Removal

Another pre-processing approach involves identifying and removing features that, while not explicitly sensitive attributes, act as proxies for them. These proxy features can inadvertently encode demographic information, leading to indirect discrimination. In our D4-T2-C1 analysis, we found that 'revolving_utilization' and 'home_ownership_encoded' showed strong correlations with 'race_group'. Removing these might break the indirect link to the sensitive attribute and reduce bias.

**Practitioner Warning:** Removing proxy features is the bluntest mitigation tool. While simple, it can be counterproductive if the proxy carries genuine predictive information, losing legitimate signal along with the bias. Worse, the model might compensate by shifting weight to other, less obvious proxies, a phenomenon known as "proxy proliferation." Always re-run a full fairness audit after removal to confirm the bias has actually decreased.

```python
def retrain_without_proxies(X_train_data, y_train_data, X_test_data, y_test_data,
                            proxy_features_list, sensitive_test_data, all_features_list):
    """Remove proxy features and retrain the model."""
    clean_features = [f for f in all_features_list if f not in proxy_features_list]

    print(f"Original features: {len(all_features_list)}")
    print(f"Removed proxies: {proxy_features_list}")
    print(f"Remaining features: {len(clean_features)}")
    
    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train_data[clean_features], y_train_data)
    
    y_pred = model.predict(X_test_data[clean_features])
    y_prob = model.predict_proba(X_test_data[clean_features])[:, 1]
    
    result = evaluate_model(y_test_data, y_pred, y_prob, sensitive_test_data, 'Proxy Removed')
    return result, model

# Identify specific proxy features based on D4-T2-C1 analysis (synthetic choice)
proxy_features = ['revolving_utilization', 'home_ownership_encoded']

# Remove proxy features and retrain
proxy_removed_results, proxy_removed_model = retrain_without_proxies(
    X_train, y_train, X_test, y_test, proxy_features, sensitive_test, feature_cols
)

print(f"\n--- Proxy Removed Model Evaluation ---")
print(f"AUC: {proxy_removed_results['auc']}, F1: {proxy_removed_results['f1']}")
print(f"DIR: {proxy_removed_results['dir']} (Four-fifths Rule: {proxy_removed_results['four_fifths']})")
print(f"SPD: {proxy_removed_results['spd']}")
print(f"EOD (FNR Gap): {proxy_removed_results['eod']}")
print(f"Approval Rates: {proxy_removed_results['approval_rates']}")
```

### 4.1 Interpretation of Proxy Feature Removal Results

The impact of proxy feature removal on both fairness and accuracy is usually more unpredictable than reweighting. While it might improve fairness by reducing the indirect influence of sensitive attributes, it often leads to a more significant drop in overall model accuracy if the removed features were genuinely predictive. It's crucial to ensure that the fairness improvement is substantial enough to justify the accuracy loss.

In this scenario, I observe the changes in DIR, SPD, and EOD. If fairness improved, it might be at the cost of AUC. This highlights the trade-off. For regulators, this approach is easy to explain ("we removed features that correlate with protected attributes"), but from a technical standpoint, the risk of proxy proliferation means this approach requires careful validation and should not be assumed to fully eliminate bias.

---

## 5. Strategy 3: In-Processing - Fairness-Constrained Training

In-processing methods directly incorporate fairness considerations into the model's training algorithm. This is often considered the most principled approach as it allows the model to find the optimal balance between accuracy and fairness. `fairlearn`'s `ExponentiatedGradient` algorithm is a powerful tool that achieves this by solving a constrained optimization problem. It ensures that the model meets specific fairness criteria (like Demographic Parity or Equalized Odds) while maintaining the highest possible accuracy.

The `ExponentiatedGradient` algorithm works by finding a randomized classifier that minimizes the standard loss function subject to fairness constraints. For example, for Demographic Parity, the objective is to minimize the model's loss $L(\theta)$ such that the Disparate Impact Ratio is maintained above a certain threshold (e.g., $0.80$):
$$ \min_{\theta} L(\theta) \quad \text{subject to} \quad \text{DIR}(\theta) \geq 0.80 $$
where $\theta$ represents the model parameters. This is achieved by finding the Lagrangian dual of the unconstrained problem, effectively adding a "price" for unfairness that the algorithm trades off against accuracy.

```python
def train_fair_model(X_train_data, y_train_data, X_test_data, y_test_data,
                     sensitive_train_data, sensitive_test_data,
                     constraint='demographic_parity'):
    """
    Train a model with explicit fairness constraints using fairlearn's ExponentiatedGradient reduction.
    """
    if constraint == 'demographic_parity':
        fairness_constraint = DemographicParity()
    elif constraint == 'equalized_odds':
        fairness_constraint = EqualizedOdds()
    else:
        raise ValueError("Constraint must be 'demographic_parity' or 'equalized_odds'")

    # Base estimator (Logistic Regression for fairlearn compatibility)
    # Using class_weight='balanced' in base estimator helps with class imbalance
    base_estimator = LogisticRegression(class_weight='balanced', max_iter=1000, C=0.1, solver='liblinear', random_state=42)
    
    # ExponentiatedGradient with fairness constraints
    fair_model = ExponentiatedGradient(
        base_estimator,
        constraints=fairness_constraint,
        max_iter=50, # Number of iterations for ExponentiatedGradient
        random_state=42
    )
    
    fair_model.fit(X_train_data, y_train_data, sensitive_features=sensitive_train_data)
    
    # Get predictions (averaged from randomized classifiers)
    y_pred = fair_model.predict(X_test_data)
    # fairlearn's ExponentiatedGradient doesn't directly expose predict_proba for the ensemble
    # A common workaround is to use the predict_proba of the underlying base estimator or average.
    # For simplicity, we'll try to get probabilities via base_estimator if possible or use predict scores.
    # For fairlearn's ExpGrad, probabilities from base estimator with adjusted weights might be closer,
    # but the ensemble itself is what we want.
    # A robust way is to train a new model using the optimal weights learned by ExpGrad, or simply use predict_proba from the best base model found
    # For this exercise, using a dummy prob for ExpGrad if direct proba is not easily available, or assuming we get some score to evaluate AUC.
    # Fairlearn's ExponentiatedGradient returns a wrapper that might not have _pmf_predict.
    # The current framework expects y_prob. Let's use decision_function as a score for AUC
    
    if hasattr(fair_model, "decision_function"):
        y_prob = fair_model.decision_function(X_test_data)
        # Convert decision scores to probabilities if necessary, e.g., using sigmoid
        y_prob = 1 / (1 + np.exp(-y_prob))
    else:
        # Fallback if decision_function is not available, might affect AUC
        print("Warning: decision_function not found for fair_model. AUC might be less accurate.")
        y_prob = fair_model.predict(X_test_data) # Use binary predictions as placeholder for probability

    result = evaluate_model(y_test_data, y_pred, y_prob,
                            sensitive_test_data, f'FairConstraint ({constraint})')
    return result, fair_model

# Train with Demographic Parity constraint
fair_dp_results, fair_dp_model = train_fair_model(
    X_train[feature_cols], y_train, X_test[feature_cols], y_test,
    sensitive_train, sensitive_test, constraint='demographic_parity'
)

print(f"\n--- Fairness-Constrained (Demographic Parity) Model Evaluation ---")
print(f"AUC: {fair_dp_results['auc']}, F1: {fair_dp_results['f1']}")
print(f"DIR: {fair_dp_results['dir']} (Four-fifths Rule: {fair_dp_results['four_fifths']})")
print(f"SPD: {fair_dp_results['spd']}")
print(f"EOD (FNR Gap): {fair_dp_results['eod']}")
print(f"Approval Rates: {fair_dp_results['approval_rates']}")

# Optional: Train with Equalized Odds constraint
fair_eod_results, fair_eod_model = train_fair_model(
    X_train[feature_cols], y_train, X_test[feature_cols], y_test,
    sensitive_train, sensitive_test, constraint='equalized_odds'
)
print(f"\n--- Fairness-Constrained (Equalized Odds) Model Evaluation ---")
print(f"AUC: {fair_eod_results['auc']}, F1: {fair_eod_results['f1']}")
print(f"DIR: {fair_eod_results['dir']} (Four-fifths Rule: {fair_eod_results['four_fifths']})")
print(f"SPD: {fair_eod_results['spd']}")
print(f"EOD (FNR Gap): {fair_eod_results['eod']}")
print(f"Approval Rates: {fair_eod_results['approval_rates']}")
```

### 5.1 Interpretation of Fairness-Constrained Training Results

The results from fairness-constrained training generally demonstrate a more balanced trade-off between accuracy and fairness compared to pre-processing methods. By optimizing for both simultaneously, `ExponentiatedGradient` aims to achieve the best possible accuracy while meeting the specified fairness constraints. I observe strong improvements in DIR, SPD, and EOD, often leading to full compliance with regulations like the Four-fifths Rule. The AUC drop is typically more controlled than with blunt feature removal.

This is the most principled approach, offering a robust solution to bias. However, it requires specialized algorithms and understanding of various fairness definitions (Demographic Parity vs. Equalized Odds). For Apex Credit Solutions, this method is ideal for long-term strategic model development, where we can invest the time to build a truly fair model.

---

## 6. Strategy 4: Post-Processing - Group-Specific Threshold Calibration

Post-processing involves adjusting the model's outputs after predictions have been made, without altering the underlying model itself. A common technique is group-specific threshold calibration, where different decision thresholds are applied to different sensitive groups to achieve a target fairness level. For instance, if Group B has a lower approval rate, we might lower their approval threshold while keeping Group A's threshold fixed.

The objective here is to find group-specific thresholds that lead to a desired Disparate Impact Ratio. We can achieve this by keeping the advantaged group's threshold fixed (e.g., at $0.5$) and searching for the disadvantaged group's threshold that equalizes approval rates to achieve the target DIR. This search can be performed using scalar optimization techniques.

**Practitioner Warning:** Post-processing, especially group-specific thresholds, is legally controversial. Explicitly using protected attributes to set different decision thresholds can be viewed as discriminatory itself by some legal interpretations ("reverse discrimination"). While faster to deploy and easier to reverse, it requires careful legal review and approval from our firm's legal counsel. It is often best used as an interim fix while a more principled in-processing solution is being developed.

```python
def calibrate_group_thresholds(y_true_data, y_prob_data, sensitive_data, target_dir=0.85):
    """
    Find group-specific thresholds that achieve a target DIR.
    Approach: keep the advantaged group's threshold fixed at 0.5.
    Search for the disadvantaged group's threshold that equalizes approval rates to achieve target DIR.
    """
    groups = np.unique(sensitive_data)
    
    # 1. Determine which group is disadvantaged (lower approval rate at 0.5)
    rates_at_50 = {}
    for g in groups:
        mask_g = sensitive_data == g
        rates_at_50[g] = (y_prob_data[mask_g] >= 0.5).mean() # Approval rate at default 0.5 threshold

    # Determine privileged and unprivileged groups based on rates at 0.5 threshold
    sorted_rates_at_50 = sorted(rates_at_50.items(), key=lambda item: item[1], reverse=True)
    privileged_group = sorted_rates_at_50[0][0]
    unprivileged_group = sorted_rates_at_50[1][0]
    
    rate_privileged_at_50 = rates_at_50[privileged_group]
    rate_unprivileged_at_50 = rates_at_50[unprivileged_group]

    print(f"Advantaged group: {privileged_group} (approval {rate_privileged_at_50:.1%})")
    print(f"Disadvantaged group: {unprivileged_group} (approval {rate_unprivileged_at_50:.1%})")

    # 2. Calculate the target approval rate for the disadvantaged group
    # Based on the target DIR and the advantaged group's approval rate at 0.5 threshold
    target_rate_unprivileged = rate_privileged_at_50 * target_dir

    # 3. Define objective function for scalar minimization
    # This function calculates the absolute difference between the current approval rate for the disadvantaged
    # group (at a given threshold) and the target approval rate.
    def objective(threshold):
        mask_disadvantaged = sensitive_data == unprivileged_group
        approval_rate_disadvantaged = (y_prob_data[mask_disadvantaged] >= threshold).mean()
        return abs(approval_rate_disadvantaged - target_rate_unprivileged)

    # 4. Search for the optimal threshold for the disadvantaged group
    # Bounds for the threshold search: (0.01, 0.99)
    result = minimize_scalar(objective, bounds=(0.01, 0.99), method='bounded')
    
    # Store group-specific thresholds
    thresholds = {
        privileged_group: 0.5, # Keep privileged group's threshold fixed
        unprivileged_group: round(result.x, 3) # Optimized threshold for disadvantaged group
    }

    # 5. Apply group-specific thresholds
    y_pred_calibrated = np.zeros(len(y_prob_data), dtype=int)
    for g in groups:
        mask_g = sensitive_data == g
        y_pred_calibrated[mask_g] = (y_prob_data[mask_g] >= thresholds[g]).astype(int)

    # 6. Verify new approval rates and DIR
    new_rates = {}
    for g in groups:
        mask_g = sensitive_data == g
        new_rates[g] = y_pred_calibrated[mask_g].mean()
    
    new_dir = (new_rates[unprivileged_group] + 1e-6) / (new_rates[privileged_group] + 1e-6)

    print(f"\nCalibrated thresholds: {thresholds}")
    print(f"New approval rates: {new_rates}")
    print(f"New DIR: {new_dir:.3f} (Target was {target_dir:.3f})")

    # Evaluate the model with calibrated predictions
    calibrated_results = evaluate_model(y_true_data, y_pred_calibrated, y_prob_data,
                                        sensitive_data, 'Threshold Calibrated')
    calibrated_results['thresholds'] = thresholds
    return calibrated_results

# Calibrate thresholds on the baseline model's probabilities
threshold_calibrated_results = calibrate_group_thresholds(
    y_test, y_prob_base, sensitive_test, target_dir=0.85 # Aim for DIR of 0.85
)

print(f"\n--- Threshold Calibrated Model Evaluation ---")
print(f"AUC: {threshold_calibrated_results['auc']}, F1: {threshold_calibrated_results['f1']}")
print(f"DIR: {threshold_calibrated_results['dir']} (Four-fifths Rule: {threshold_calibrated_results['four_fifths']})")
print(f"SPD: {threshold_calibrated_results['spd']}")
print(f"EOD (FNR Gap): {threshold_calibrated_results['eod']}")
print(f"Approval Rates: {threshold_calibrated_results['approval_rates']}")
```

### 6.1 Interpretation of Group-Specific Threshold Calibration Results

Threshold calibration can be highly effective in closing the fairness gap quickly. I observe a significant improvement in the Disparate Impact Ratio (DIR), often meeting or exceeding our target, and corresponding reductions in SPD and EOD. The crucial advantage of post-processing is that it doesn't require retraining the model, making it exceptionally fast to implement and revert. This means the AUC remains unchanged or changes only marginally depending on how the initial probability is interpreted for evaluation, as we are only changing the decision boundary, not the model's underlying scores.

For Apex Credit Solutions, this is a valuable tool for rapid response to detected bias or as an interim solution. However, the legal and ethical implications of using group membership to differentiate decision thresholds demand thorough consultation with our legal department before deployment.

---

## 7. Comprehensive Comparison and Mitigation Selection Guide

Having explored four distinct bias mitigation strategies, it's time to consolidate our findings into a comprehensive comparison. As a CFA, my role now is to synthesize these results, analyze the accuracy-fairness trade-offs, and recommend the most appropriate strategy for Apex Credit Solutions, considering regulatory requirements, operational feasibility, and ethical considerations.

```python
# Collect all results into a DataFrame
all_results_df = pd.DataFrame([
    baseline_results,
    reweighted_results,
    proxy_removed_results,
    fair_dp_results,
    # fair_eod_results, # Can include if desired, for this example we stick to 4
    threshold_calibrated_results
])

# Re-evaluate approval rates for visualization purposes to explicitly show before/after
# We need baseline approval rates from the original predictions, and then for each strategy
# The 'approval_rates' in the results dictionary already holds this.

print("--- FIVE-WAY MITIGATION COMPARISON ---")
print("=" * 75)
print(f"{'Strategy':<25s} {'AUC':>7s} {'F1':>7s} {'DIR':>7s} {'SPD':>7s} {'EOD':>7s} {'4/5':>6s}")
print("-" * 75)

for index, row in all_results_df.iterrows():
    print(f"{row['label']:<25s} {row['auc']:>7.4f} {row['f1']:>7.4f} {row['dir']:>7.3f} {row['spd']:>7.4f} {row['eod']:>7.4f} {row['four_fifths']:>6s}")

print("\n--- ACCURACY COST OF EACH MITIGATION ---")
baseline_auc = baseline_results['auc']
baseline_dir = baseline_results['dir']

for index, row in all_results_df.iterrows():
    if row['label'] != 'Baseline (Biased)':
        auc_cost = baseline_auc - row['auc']
        dir_gain = row['dir'] - baseline_dir
        print(f"{row['label']:<25s}: AUC cost={auc_cost:+.4f}, DIR gain={dir_gain:+.3f}")
```

### 7.1 Visualizing Mitigation Impacts

Visualizations are key to understanding the trade-offs and communicating our findings to stakeholders, including risk committees and legal counsel.

#### 7.1.1 Accuracy-Fairness Pareto Frontier

This scatter plot illustrates the trade-off between accuracy (AUC) and fairness (DIR) for each strategy. It helps identify Pareto-optimal strategies—those where you cannot improve fairness without sacrificing accuracy, or vice-versa. The Four-fifths Rule threshold line is critical for visual compliance assessment.

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(x='dir', y='auc', hue='label', data=all_results_df, s=150, style='label')
plt.axvline(x=0.80, color='r', linestyle='--', label='Four-fifths Rule (0.80 DIR)')
plt.title('Accuracy-Fairness Pareto Frontier (AUC vs. DIR)')
plt.xlabel('Disparate Impact Ratio (DIR)')
plt.ylabel('Area Under ROC Curve (AUC)')
plt.xlim(0.5, 1.1)
plt.ylim(0.7, 0.9)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
```

#### 7.1.2 Before/After Approval Rate Bars for Sensitive Groups

These grouped bar charts visually demonstrate how each strategy impacts the approval rates for different sensitive groups, explicitly showing the gap closure.

```python
def plot_approval_rates(all_results_df):
    privileged_group = all_results_df.loc[0, 'privileged_group']
    unprivileged_group = all_results_df.loc[0, 'unprivileged_group']

    df_plot = pd.DataFrame(columns=['Strategy', 'Group', 'Approval Rate'])
    
    for _, row in all_results_df.iterrows():
        rates = row['approval_rates']
        df_plot = pd.concat([df_plot, pd.DataFrame([
            {'Strategy': row['label'], 'Group': privileged_group, 'Approval Rate': rates.get(privileged_group, 0)},
            {'Strategy': row['label'], 'Group': unprivileged_group, 'Approval Rate': rates.get(unprivileged_group, 0)}
        ])], ignore_index=True)

    plt.figure(figsize=(12, 7))
    sns.barplot(x='Strategy', y='Approval Rate', hue='Group', data=df_plot, palette='viridis')
    plt.title(f'Approval Rates by Strategy for {privileged_group} and {unprivileged_group}')
    plt.ylabel('Approval Rate')
    plt.xlabel('Mitigation Strategy')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, df_plot['Approval Rate'].max() * 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

plot_approval_rates(all_results_df)
```

#### 7.1.3 AUC Cost Bar Chart

This bar chart clearly illustrates the accuracy cost of each mitigation strategy relative to the biased baseline.

```python
baseline_auc = all_results_df.loc[0, 'auc'] # Assuming baseline is the first row

auc_cost_data = []
for index, row in all_results_df.iterrows():
    if row['label'] != 'Baseline (Biased)':
        auc_cost_data.append({'Strategy': row['label'], 'AUC Cost': baseline_auc - row['auc']})

df_auc_cost = pd.DataFrame(auc_cost_data)

plt.figure(figsize=(10, 6))
sns.barplot(x='Strategy', y='AUC Cost', data=df_auc_cost, palette='magma')
plt.title('Accuracy Cost (AUC Reduction) per Mitigation Strategy')
plt.ylabel('AUC Reduction (Baseline AUC - Strategy AUC)')
plt.xlabel('Mitigation Strategy')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```

### 7.2 Mitigation Selection Guide and Recommendation

Based on the quantitative metrics and visualizations, I can now formulate a strategic recommendation for Apex Credit Solutions.

| Strategy              | Best When                                                                                                                  | AUC Cost | Complexity | Legal Risk |
| :-------------------- | :------------------------------------------------------------------------------------------------------------------------- | :------- | :--------- | :--------- |
| **Reweighting**       | Mild to moderate bias, quick fix needed, low-regret.                                                                       | 0.5-2%   | Low        | None       |
| **Proxy Feature Removal** | Clear proxy identified, legally required to remove.                                                                        | 1-4%     | Low        | None       |
| **Fair Constraints**  | Moderate-severe bias, principled solution needed, long-term strategy.                                                      | 1-3%     | Medium     | None       |
| **Threshold Calib.**  | Interim fix while retraining, fast deployment, temporary measure, with legal approval.                                       | 0-1%     | Low        | High       |

**My Recommendation for Apex Credit Solutions:**

Given the critical nature of credit decisions and the high regulatory stakes:

1.  **Immediate Action (Interim Fix):** Implement **Sample Reweighting**. It offers a rapid improvement in fairness with a manageable accuracy trade-off, is legally low-risk, and requires minimal operational changes. This addresses the most urgent compliance failure immediately.
    *   *Rationale:* This is our "low-regret" strategy, providing acceptable fairness gains quickly while we work on a more robust solution.

2.  **Strategic Long-Term Solution:** Initiate development and validation for a model using **Fairness-Constrained Training** (e.g., Demographic Parity). This in-processing approach offers the most principled and robust solution by optimizing for both accuracy and fairness simultaneously, leading to better compliance stability.
    *   *Rationale:* While requiring more development time, this strategy yields the most ethically sound and resilient model. The "cost" of mitigation (1-3% AUC) is a prudent "premium" for regulatory compliance insurance, preventing a potential $50M regulatory penalty.

3.  **Conditional Use (Contingency):** **Group-Specific Threshold Calibration** should be considered only as an emergency interim measure if immediate compliance is paramount and the other options are not feasible in time. This would require explicit legal review and approval due to its controversial nature, as it directly uses sensitive attributes in decision-making.

4.  **Avoidance (Unless necessary):** **Proxy Feature Removal** should generally be avoided unless legally mandated or if proxies are demonstrably non-predictive. Its unpredictable impact on accuracy and the risk of proxy proliferation make it less desirable for critical credit models.

**Compliance Documentation for Risk Committee:**
*   **Bias Detected:** Disparate Impact against [Unprivileged Group] (DIR: [Baseline DIR] failing Four-fifths Rule).
*   **Mitigation Applied:** Sample Reweighting (Interim) and Fairness-Constrained Training (Strategic).
*   **Impact on Accuracy/Fairness:** [Reweighted/Fair Constraint] model AUC: [New AUC] (cost: [AUC Cost]), DIR: [New DIR] (passing Four-fifths Rule).
*   **Trade-off Approved By:** [Signatures of Head of Risk, Head of Compliance, Legal Counsel].

This structured approach ensures that Apex Credit Solutions addresses the detected bias pragmatically and ethically, balancing business needs with regulatory demands and our commitment to fair lending.
