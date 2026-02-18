# QuLab: Lab 41: Mitigating Bias in Credit Scoring

![QuantUniversity Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Title: Apex Credit Solutions: Bias Mitigation Dashboard

This project, "QuLab: Lab 41: Mitigating Bias via Reweighting," implements a comprehensive Streamlit application designed to explore and evaluate various bias mitigation strategies in a credit scoring model. Developed from the persona of Sri Krishnamurthy, CFA, a Senior Risk Analyst at Apex Credit Solutions, the dashboard guides users through detecting, quantifying, and actively treating bias to ensure regulatory compliance and ethical lending practices.

The application addresses a critical issue: a credit model exhibiting disparate impact where one demographic group receives significantly lower loan approval rates. It systematically demonstrates pre-processing, in-processing, and post-processing techniques, comparing their trade-offs between model accuracy and fairness.

## Features

This interactive Streamlit application provides a structured workflow for bias mitigation:

*   **Home/Introduction**: Sets the context, introduces the persona, outlines the problem (biased credit model), and the project's objectives (regulatory compliance, ethical lending).
*   **1. Setup & Data Generation**:
    *   Generates a synthetic, intentionally biased credit dataset with features, a sensitive attribute (`'race_group'`), and a target variable (`'loan_approved'`).
    *   Displays data statistics and a sample of the generated data.
*   **2. Baseline Model Performance**:
    *   Trains an `XGBClassifier` on the original biased data to establish a performance benchmark.
    *   Evaluates the baseline model using both accuracy metrics (AUC, F1-score) and crucial fairness metrics (Disparate Impact Ratio (DIR), Statistical Parity Difference (SPD), Equal Opportunity Difference (EOD)).
    *   Provides interpretation of the baseline, highlighting non-compliance with the Four-fifths Rule.
*   **3. Strategy 1: Pre-Processing - Sample Reweighting**:
    *   Applies sample weights during training to give more influence to underrepresented or disadvantaged group-outcome combinations.
    *   Retrains the `XGBClassifier` with these fairness weights and re-evaluates its performance and fairness.
    *   Discusses the accuracy-fairness trade-off and legal implications.
*   **4. Strategy 2: Pre-Processing - Proxy Feature Removal**:
    *   Identifies and removes features (`'revolving_utilization'`, `'home_ownership_encoded'`) identified as proxies for the sensitive attribute.
    *   Retrains the model on the reduced feature set and re-evaluates.
    *   Highlights the practitioner warning about potential loss of predictive power and proxy proliferation.
*   **5. Strategy 3: In-Processing - Fairness-Constrained Training**:
    *   Utilizes `fairlearn`'s `ExponentiatedGradient` algorithm to train a model that optimizes both accuracy and fairness simultaneously.
    *   Allows selection between **Demographic Parity** and **Equalized Odds** as fairness constraints.
    *   Trains and evaluates the fairness-constrained model, demonstrating a principled approach to bias mitigation.
*   **6. Strategy 4: Post-Processing - Group-Specific Threshold Calibration**:
    *   Adjusts decision thresholds for different sensitive groups *after* the model has made predictions.
    *   Enables selection of a target Disparate Impact Ratio (DIR) for calibration.
    *   Re-evaluates the outcomes based on the new thresholds.
    *   Provides a strong practitioner warning regarding the legal controversy of this approach.
*   **7. Comprehensive Comparison & Recommendation**:
    *   Aggregates results from all mitigation strategies into a comparative table.
    *   Visualizes trade-offs with:
        *   **Accuracy-Fairness Pareto Frontier (AUC vs. DIR)**: To understand optimal balance points.
        *   **Before/After Approval Rate Bars**: Showing impact on sensitive group approval rates.
        *   **AUC Cost Bar Chart**: Quantifying accuracy reduction for each strategy.
    *   Offers a strategic recommendation for Apex Credit Solutions, balancing regulatory compliance, operational feasibility, and ethical considerations.
*   **8. Bias Mitigation Compliance Report**:
    *   Summarizes key findings, applied mitigations, and their impact in a compliance-ready format for stakeholders.
    *   Includes "Further Discussion Points for Financial Professionals" for deeper insights.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your_username/your_repository_name.git
    cd your_repository_name
    ```
    (Replace `your_username/your_repository_name.git` with the actual repository URL)

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

    If a `requirements.txt` is not provided, you can create one or install the dependencies directly:

    ```bash
    pip install streamlit pandas numpy matplotlib seaborn xgboost fairlearn scikit-learn
    ```
    *(Note: `fairlearn` depends on `scikit-learn`.)*

## Usage

To run the Streamlit application:

1.  **Activate your virtual environment (if you created one):**

    ```bash
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

2.  **Navigate to the project directory (if not already there):**

    ```bash
    cd path/to/your/project
    ```

3.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

    This command will open the application in your default web browser. If it doesn't open automatically, look for a URL in your terminal (usually `http://localhost:8501`).

### Basic Usage Instructions:

*   **Navigate through the lab:** Use the sidebar dropdown menu to move between different sections of the analysis (e.g., "Home", "1. Setup & Data Generation", "2. Baseline Model Performance", etc.).
*   **Follow the prompts:** Click buttons like "Start Analysis", "Proceed to Baseline Model", "Explore Mitigation Strategy 1", etc., to advance through the workflow and trigger computations.
*   **Interpret results:** Review the displayed dataframes, metrics, and plots to understand the impact of each mitigation strategy on accuracy and fairness.
*   **Experiment:** For fairness-constrained training and threshold calibration, you can interact with selection boxes and sliders to observe different outcomes.

## Project Structure

The project is organized as follows:

```
.
├── app.py                  # Main Streamlit application script
├── source.py               # Contains helper functions for data generation, model training, and evaluation
├── requirements.txt        # List of Python dependencies
└── README.md               # This README file
```

*   `app.py`: This is the core Streamlit application. It defines the layout, navigation, and orchestrates the calls to functions defined in `source.py` to display the interactive dashboard. It manages the Streamlit session state to preserve information across user interactions.
*   `source.py`: This file encapsulates the heavy lifting of the project, including:
    *   `generate_biased_credit_data()`: Creates the synthetic dataset.
    *   `evaluate_model()`: Calculates both accuracy and fairness metrics.
    *   `compute_fairness_weights()`: Determines sample weights for the reweighting strategy.
    *   `retrain_without_proxies()`: Implements the proxy feature removal strategy.
    *   `train_fair_model()`: Implements fairness-constrained training using `fairlearn`.
    *   `calibrate_group_thresholds()`: Implements group-specific threshold calibration.

## Technology Stack

*   **Python**: The core programming language.
*   **Streamlit**: For building the interactive web application and user interface.
*   **Pandas**: For data manipulation and analysis.
*   **NumPy**: For numerical operations.
*   **Matplotlib** & **Seaborn**: For data visualization and plotting results.
*   **XGBoost**: For training the gradient boosting classification models.
*   **Fairlearn**: An open-source toolkit for assessing and improving fairness of AI systems, used for fairness-constrained training.
*   **Scikit-learn**: Underlying machine learning library, used for various utilities and potentially as a dependency for other libraries.

## Contributing

Contributions are welcome! If you have suggestions for improving the lab, fixing bugs, or adding new features, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
5.  Push to the branch (`git push origin feature/AmazingFeature`).
6.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, feedback, or further information, please contact:

*   **QuantUniversity:** [www.quantuniversity.com](https://www.quantuniversity.com)
*   **Email:** info@quantuniversity.com (or your specific contact)

---
*Generated based on the provided Streamlit application code.*