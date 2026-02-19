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
    data['revolving_utilization'] = np.random.beta(
        5, 10, n_samples) * 0.8  # proxy feature
    data['home_ownership_encoded'] = np.random.choice(
        [0, 1], n_samples, p=[0.6, 0.4])  # proxy feature

    # Introduce sensitive attribute: 'race_group'
    data['race_group'] = np.random.choice(
        ['Group_A', 'Group_B'], n_samples, p=[0.7, 0.3])

    # Introduce bias: Group B has lower credit scores and higher revolving utilization
    # Also, for the same credit score, Group B is less likely to be approved
    data.loc[data['race_group'] == 'Group_B',
             'credit_score'] = data.loc[data['race_group'] == 'Group_B', 'credit_score'] - 30
    data.loc[data['race_group'] == 'Group_B', 'revolving_utilization'] = data.loc[data['race_group']
                                                                                  == 'Group_B', 'revolving_utilization'] + 0.1

    # Define target variable 'loan_approved' (binary)
    # Base approval probability
    prob_approved = (
        0.05 * (data['credit_score'] - 500) / 200 +
        0.02 * (data['income'] / 100000) -
        0.3 * data['revolving_utilization']
    )
    prob_approved = 1 / (1 + np.exp(-prob_approved))  # Sigmoid transformation

    # Introduce direct bias on loan approval for Group B
    data['loan_approved'] = (
        prob_approved > np.random.rand(n_samples)).astype(int)
    mask_group_b_and_approved = (
        data['race_group'] == 'Group_B') & (prob_approved > 0.5)
    num_to_bias = mask_group_b_and_approved.sum()
    if num_to_bias > 0:  # Ensure there are values to process
        data.loc[mask_group_b_and_approved, 'loan_approved'] = \
            (prob_approved.loc[mask_group_b_and_approved]
             * 0.8 > np.random.rand(num_to_bias)).astype(int)

    feature_cols = ['age', 'income', 'credit_score', 'loan_amount',
                    'employment_years', 'revolving_utilization', 'home_ownership_encoded']
    X = data[feature_cols]
    y = data['loan_approved']
    sensitive = data['race_group']

    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=random_state, stratify=sensitive
    )

    return X_train, X_test, y_train, y_test, sensitive_train, sensitive_test, feature_cols


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
        rates[g] = y_pred[mask].mean()  # Approval rate

    # Determine privileged and unprivileged groups based on approval rates
    # Assume the group with higher approval rate is privileged at baseline
    # If rates are equal or close, default to alphabetical order for consistency
    if len(groups) == 2:
        group_rates = sorted(
            rates.items(), key=lambda item: item[1], reverse=True)
        privileged_group = group_rates[0][0]
        unprivileged_group = group_rates[1][0]
    else:  # Handle more than two groups or edge cases, picking first two alphabetically
        sorted_groups = sorted(groups)
        privileged_group = sorted_groups[0]
        unprivileged_group = sorted_groups[1]

    # Recalculate rates based on determined privileged/unprivileged
    rate_privileged = rates.get(privileged_group, 0)
    rate_unprivileged = rates.get(unprivileged_group, 0)

    # Disparate Impact Ratio
    # Add small epsilon to avoid division by zero
    dir_val = (rate_unprivileged + 1e-6) / (rate_privileged + 1e-6)

    # Statistical Parity Difference
    spd = rate_unprivileged - rate_privileged

    # Equal Opportunity Difference (FNR gap based on provided context)
    fnrs = {}
    for g in groups:
        # Only look at true positive cases
        mask = (sensitive == g) & (y_true == 1)
        if y_true[mask].sum() > 0:
            # False Negative Rate (FNR) = (Actual 1s but predicted 0s) / (Total Actual 1s)
            fnrs[g] = ((y_true[mask] == 1) & (y_pred[mask] == 0)
                       ).sum() / y_true[mask].sum()
        else:
            fnrs[g] = 0.0  # No true positives for this group, FNR is undefined or 0

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


def train_and_evaluate_baseline(X_train, y_train, X_test, y_test, sensitive_test, feature_cols, random_state=42):
    """Trains a baseline XGBoost model and evaluates it."""
    baseline_model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                   random_state=random_state, use_label_encoder=False, eval_metric='logloss')
    baseline_model.fit(X_train[feature_cols], y_train)

    y_pred_base = baseline_model.predict(X_test[feature_cols])
    y_prob_base = baseline_model.predict_proba(X_test[feature_cols])[:, 1]

    baseline_results = evaluate_model(
        y_test, y_pred_base, y_prob_base, sensitive_test, 'Baseline (Biased)')
    return baseline_model, y_pred_base, y_prob_base, baseline_results


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
            # if count is 0, weights remain 1 (default), which is fine.

    return weights


def train_and_evaluate_reweighted(X_train, y_train, X_test, y_test, sensitive_train, sensitive_test, feature_cols, random_state=42):
    """Trains an XGBoost model with fairness weights and evaluates it."""
    fairness_weights = compute_fairness_weights(y_train, sensitive_train)

    reweighted_model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                     random_state=random_state, use_label_encoder=False, eval_metric='logloss')
    reweighted_model.fit(X_train[feature_cols],
                         y_train, sample_weight=fairness_weights)

    y_pred_rw = reweighted_model.predict(X_test[feature_cols])
    y_prob_rw = reweighted_model.predict_proba(X_test[feature_cols])[:, 1]
    reweighted_results = evaluate_model(
        y_test, y_pred_rw, y_prob_rw, sensitive_test, 'Reweighted')
    return reweighted_model, y_pred_rw, y_prob_rw, reweighted_results


def retrain_without_proxies(X_train_data, y_train_data, X_test_data, y_test_data,
                            proxy_features_list, sensitive_test_data, all_features_list, random_state=42):
    """Remove proxy features and retrain the model."""
    clean_features = [
        f for f in all_features_list if f not in proxy_features_list]

    print(f"Original features: {len(all_features_list)}")
    print(f"Removed proxies: {proxy_features_list}")
    print(f"Remaining features: {len(clean_features)}\n")

    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=random_state,
                          use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train_data[clean_features], y_train_data)

    y_pred = model.predict(X_test_data[clean_features])
    y_prob = model.predict_proba(X_test_data[clean_features])[:, 1]

    result = evaluate_model(y_test_data, y_pred, y_prob,
                            sensitive_test_data, 'Proxy Removed')
    return result, model


def train_fair_model(X_train_data, y_train_data, X_test_data, y_test_data,
                     sensitive_train_data, sensitive_test_data,
                     constraint='demographic_parity', random_state=42):
    """
    Train a model with explicit fairness constraints using fairlearn's ExponentiatedGradient reduction.
    """
    if constraint == 'demographic_parity':
        fairness_constraint = DemographicParity()
    elif constraint == 'equalized_odds':
        fairness_constraint = EqualizedOdds()
    else:
        raise ValueError(
            "Constraint must be 'demographic_parity' or 'equalized_odds'")

    # Base estimator (Logistic Regression for fairlearn compatibility)
    # Using class_weight='balanced' in base estimator helps with class imbalance
    base_estimator = LogisticRegression(
        class_weight='balanced', max_iter=1000, C=0.1, solver='liblinear', random_state=random_state)

    # ExponentiatedGradient with fairness constraints
    fair_model = ExponentiatedGradient(
        base_estimator,
        constraints=fairness_constraint,
        max_iter=50,  # Number of iterations for ExponentiatedGradient
    )

    fair_model.fit(X_train_data, y_train_data,
                   sensitive_features=sensitive_train_data)

    # Get predictions (averaged from randomized classifiers)
    y_pred = fair_model.predict(X_test_data)

    # For fairlearn's ExponentiatedGradient, _pmf_predict returns probabilities
    # This method only takes X data, not sensitive_features.
    # Note: Accessing private methods like _pmf_predict is generally discouraged,
    # but it's often the only way to get probabilities from ExpGrad directly.
    y_prob = fair_model._pmf_predict(X_test_data)[:, 1]

    result = evaluate_model(y_test_data, y_pred, y_prob,
                            sensitive_test_data, f'FairConstraint ({constraint.replace("_", " ").title()})')
    return result, fair_model


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
        # Approval rate at default 0.5 threshold
        rates_at_50[g] = (y_prob_data[mask_g] >= 0.5).mean()

    # Determine privileged and unprivileged groups based on rates at 0.5 threshold
    sorted_rates_at_50 = sorted(
        rates_at_50.items(), key=lambda item: item[1], reverse=True)
    privileged_group = sorted_rates_at_50[0][0]
    unprivileged_group = sorted_rates_at_50[1][0]

    rate_privileged_at_50 = rates_at_50[privileged_group]
    # rate_unprivileged_at_50 = rates_at_50[unprivileged_group] # Not directly used for target calculation

    print(
        f"Advantaged group: {privileged_group} (approval {rate_privileged_at_50:.1%})")
    print(
        f"Disadvantaged group: {unprivileged_group} (approval {rates_at_50[unprivileged_group]:.1%})")

    # 2. Calculate the target approval rate for the disadvantaged group
    # Based on the target DIR and the advantaged group's approval rate at 0.5 threshold
    target_rate_unprivileged = rate_privileged_at_50 * target_dir

    # 3. Define objective function for scalar minimization
    # This function calculates the absolute difference between the current approval rate for the disadvantaged
    # group (at a given threshold) and the target approval rate.
    def objective(threshold):
        mask_disadvantaged = sensitive_data == unprivileged_group
        approval_rate_disadvantaged = (
            y_prob_data[mask_disadvantaged] >= threshold).mean()
        return abs(approval_rate_disadvantaged - target_rate_unprivileged)

    # 4. Search for the optimal threshold for the disadvantaged group
    # Bounds for the threshold search: (0.01, 0.99)
    result = minimize_scalar(objective, bounds=(0.01, 0.99), method='bounded')

    # Store group-specific thresholds
    thresholds = {
        privileged_group: 0.5,  # Keep privileged group's threshold fixed
        # Optimized threshold for disadvantaged group
        unprivileged_group: round(result.x, 3)
    }

    # 5. Apply group-specific thresholds
    y_pred_calibrated = np.zeros(len(y_prob_data), dtype=int)
    for g in groups:
        mask_g = sensitive_data == g
        y_pred_calibrated[mask_g] = (
            y_prob_data[mask_g] >= thresholds[g]).astype(int)

    # 6. Verify new approval rates and DIR
    new_rates = {}
    for g in groups:
        mask_g = sensitive_data == g
        new_rates[g] = y_pred_calibrated[mask_g].mean()

    new_dir = (new_rates[unprivileged_group] + 1e-6) / \
        (new_rates[privileged_group] + 1e-6)

    print(f"\nCalibrated thresholds: {thresholds}")
    print(f"New approval rates: {new_rates}")
    print(f"New DIR: {new_dir:.3f} (Target was {target_dir:.3f})")

    # Evaluate the model with calibrated predictions
    calibrated_results = evaluate_model(y_true_data, y_pred_calibrated, y_prob_data,
                                        sensitive_data, 'Threshold Calibrated')
    # Add thresholds to results for display
    calibrated_results['thresholds'] = thresholds
    return calibrated_results


def print_evaluation_summary(results_dict):
    """Prints a formatted summary of a model's evaluation results."""
    print(f"--- {results_dict['label']} Model Evaluation ---")
    print(f"AUC: {results_dict['auc']}, F1: {results_dict['f1']}")
    print(
        f"DIR: {results_dict['dir']} (Four-fifths Rule: {results_dict['four_fifths']})")
    print(f"SPD: {results_dict['spd']}")
    print(f"EOD (FNR Gap): {results_dict['eod']}")
    print(f"Approval Rates: {results_dict['approval_rates']}")
    if 'thresholds' in results_dict:
        print(f"Calibrated Thresholds: {results_dict['thresholds']}")
    print("\n")


def print_comparison_table(all_results_df):
    """Prints a comparison table of all mitigation strategies."""
    print("--- FIVE-WAY MITIGATION COMPARISON ---")
    print("=" * 75)
    print(f"{'Strategy':<25s} {'AUC':>7s} {'F1':>7s} {'DIR':>7s} {'SPD':>7s} {'EOD':>7s} {'4/5':>6s}")
    print("-" * 75)

    for _, row in all_results_df.iterrows():
        print(f"{row['label']:<25s} {row['auc']:>7.4f} {row['f1']:>7.4f} {row['dir']:>7.3f} {row['spd']:>7.4f} {row['eod']:>7.4f} {row['four_fifths']:>6s}")
    print("\n")


def print_accuracy_cost_table(all_results_df):
    """Prints the accuracy cost of each mitigation strategy compared to baseline."""
    print("--- ACCURACY COST OF EACH MITIGATION ---")
    baseline_auc = all_results_df[all_results_df['label']
                                  == 'Baseline (Biased)']['auc'].iloc[0]
    baseline_dir = all_results_df[all_results_df['label']
                                  == 'Baseline (Biased)']['dir'].iloc[0]

    for _, row in all_results_df.iterrows():
        if row['label'] != 'Baseline (Biased)':
            auc_cost = baseline_auc - row['auc']
            dir_gain = row['dir'] - baseline_dir
            print(
                f"{row['label']:<25s}: AUC cost={auc_cost:+.4f}, DIR gain={dir_gain:+.3f}")
    print("\n")


def plot_accuracy_fairness_frontier(all_results_df):
    """Plots the Accuracy-Fairness Pareto Frontier (AUC vs. DIR)."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='dir', y='auc', hue='label',
                    data=all_results_df, s=150, style='label')
    plt.axvline(x=0.80, color='r', linestyle='--',
                label='Four-fifths Rule (0.80 DIR)')
    plt.title('Accuracy-Fairness Pareto Frontier (AUC vs. DIR)')
    plt.xlabel('Disparate Impact Ratio (DIR)')
    plt.ylabel('Area Under ROC Curve (AUC)')
    plt.xlim(0.5, 1.1)
    # Adjust y-axis limits dynamically based on the data
    min_auc = all_results_df['auc'].min()
    max_auc = all_results_df['auc'].max()
    plt.ylim(min_auc * 0.95, max_auc * 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_approval_rates(all_results_df):
    """Plots approval rates for privileged and unprivileged groups across strategies."""
    # Ensure there's a baseline result to get group names
    if all_results_df.empty:
        print("No results to plot approval rates.")
        return

    privileged_group = all_results_df.loc[0, 'privileged_group']
    unprivileged_group = all_results_df.loc[0, 'unprivileged_group']

    plot_data = []
    for _, row in all_results_df.iterrows():
        rates = row['approval_rates']
        plot_data.append({'Strategy': row['label'], 'Group': privileged_group,
                         'Approval Rate': rates.get(privileged_group, 0)})
        plot_data.append({'Strategy': row['label'], 'Group': unprivileged_group,
                         'Approval Rate': rates.get(unprivileged_group, 0)})
    df_plot = pd.DataFrame(plot_data)

    plt.figure(figsize=(12, 7))
    sns.barplot(x='Strategy', y='Approval Rate',
                hue='Group', data=df_plot, palette='viridis')
    plt.title(
        f'Approval Rates by Strategy for {privileged_group} and {unprivileged_group}')
    plt.ylabel('Approval Rate')
    plt.xlabel('Mitigation Strategy')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, df_plot['Approval Rate'].max() * 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_auc_cost(all_results_df):
    """Plots the AUC reduction (cost) for each mitigation strategy."""
    if all_results_df.empty or 'Baseline (Biased)' not in all_results_df['label'].values:
        print("No baseline results or dataframe is empty to plot AUC cost.")
        return

    baseline_auc = all_results_df[all_results_df['label']
                                  == 'Baseline (Biased)']['auc'].iloc[0]

    auc_cost_data = []
    for _, row in all_results_df.iterrows():
        if row['label'] != 'Baseline (Biased)':
            auc_cost_data.append(
                {'Strategy': row['label'], 'AUC Cost': baseline_auc - row['auc']})

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


def run_mitigation_strategies(X_train, X_test, y_train, y_test, sensitive_train, sensitive_test, feature_cols,
                              random_state=42, target_dir_threshold_calibration=0.85):
    """
    Runs all defined mitigation strategies and collects their evaluation results.
    Returns a DataFrame containing results for all strategies.
    """
    all_results = []

    print("--- Running Baseline Model ---")
    baseline_model, y_pred_base, y_prob_base, baseline_results = train_and_evaluate_baseline(
        X_train, y_train, X_test, y_test, sensitive_test, feature_cols, random_state
    )
    all_results.append(baseline_results)
    print_evaluation_summary(baseline_results)

    print("--- Running Reweighting Strategy ---")
    reweighted_model, y_pred_rw, y_prob_rw, reweighted_results = train_and_evaluate_reweighted(
        X_train, y_train, X_test, y_test, sensitive_train, sensitive_test, feature_cols, random_state
    )
    all_results.append(reweighted_results)
    print_evaluation_summary(reweighted_results)

    print("--- Running Proxy Feature Removal Strategy ---")
    # Defined based on data generation logic
    proxy_features = ['revolving_utilization', 'home_ownership_encoded']
    proxy_removed_results, proxy_removed_model = retrain_without_proxies(
        X_train, y_train, X_test, y_test, proxy_features, sensitive_test, feature_cols, random_state
    )
    all_results.append(proxy_removed_results)
    print_evaluation_summary(proxy_removed_results)

    print("--- Running Fairness-Constrained (Demographic Parity) Strategy ---")
    fair_dp_results, fair_dp_model = train_fair_model(
        X_train[feature_cols], y_train, X_test[feature_cols], y_test,
        sensitive_train, sensitive_test, constraint='demographic_parity', random_state=random_state
    )
    all_results.append(fair_dp_results)
    print_evaluation_summary(fair_dp_results)

    # Optional: Equalized Odds constraint (included in original notebook, but commented out for final table)
    print("--- Running Fairness-Constrained (Equalized Odds) Strategy ---")
    fair_eod_results, fair_eod_model = train_fair_model(
        X_train[feature_cols], y_train, X_test[feature_cols], y_test,
        sensitive_train, sensitive_test, constraint='equalized_odds', random_state=random_state
    )
    all_results.append(fair_eod_results)
    print_evaluation_summary(fair_eod_results)

    print("--- Running Threshold Calibration Strategy (on Baseline Probs) ---")
    # The original notebook calibrated on y_prob_base, which is appropriate for a post-processing step
    threshold_calibrated_results = calibrate_group_thresholds(
        y_test, y_prob_base, sensitive_test, target_dir=target_dir_threshold_calibration
    )
    all_results.append(threshold_calibrated_results)
    print_evaluation_summary(threshold_calibrated_results)

    return pd.DataFrame(all_results)


def main(n_samples=10000, random_state=42, target_dir_threshold_calibration=0.85, show_plots=True):
    """
    Main function to execute the credit scoring bias mitigation pipeline.

    Args:
        n_samples (int): Number of samples for data generation.
        random_state (int): Seed for reproducibility.
        target_dir_threshold_calibration (float): Target Disparate Impact Ratio for threshold calibration.
        show_plots (bool): Whether to display plots.

    Returns:
        pd.DataFrame: A DataFrame containing evaluation results for all mitigation strategies.
    """
    print("Initializing credit scoring bias mitigation pipeline...\n")

    print("Generating biased credit data...")
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test, feature_cols = generate_biased_credit_data(
        n_samples, random_state)
    print(
        f"Dataset generated with {len(X_train)} training samples and {len(X_test)} test samples.")
    print(f"Features: {feature_cols}")
    print(
        f"Sensitive attribute distribution in training: \n{sensitive_train.value_counts(normalize=True)}")
    print(
        f"Sensitive attribute distribution in test: \n{sensitive_test.value_counts(normalize=True)}\n")

    print("Executing mitigation strategies and evaluations...")
    all_results_df = run_mitigation_strategies(
        X_train, X_test, y_train, y_test, sensitive_train, sensitive_test, feature_cols,
        random_state, target_dir_threshold_calibration
    )

    print_comparison_table(all_results_df)
    print_accuracy_cost_table(all_results_df)

    if show_plots:
        print("Generating plots...")
        plot_accuracy_fairness_frontier(all_results_df)
        plot_approval_rates(all_results_df)
        plot_auc_cost(all_results_df)

    print("Pipeline execution complete.")
    return all_results_df


if __name__ == "__main__":
    results_df = main()
