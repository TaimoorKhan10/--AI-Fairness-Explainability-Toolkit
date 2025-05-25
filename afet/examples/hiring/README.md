# Hiring Analysis Example

This example demonstrates how to use the AI Fairness and Explainability Toolkit (AFET) to analyze and compare machine learning models for a hiring prediction task. The example includes synthetic data generation, model training, fairness evaluation, and visualization.

## Overview

The hiring analysis example covers the following key aspects:

1. **Synthetic Data Generation**: Creates a realistic hiring dataset with potential biases.
2. **Model Training**: Trains multiple machine learning models for hiring prediction.
3. **Fairness Evaluation**: Evaluates models using various fairness metrics.
4. **Visualization**: Generates visualizations to compare model performance and fairness.
5. **Advanced Analysis**: Performs detailed fairness analysis using advanced metrics.

## Requirements

Before running this example, ensure you have the following dependencies installed:

- Python 3.7+
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- tqdm
- scikit-posthocs (for statistical testing)

You can install all required packages using:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tqdm scikit-posthocs
```

## Running the Example

1. Navigate to the examples/hiring directory:

```bash
cd afet/examples/hiring
```

2. Run the hiring analysis script:

```bash
python hiring_analysis.py
```

## Example Output

The script will generate several outputs:

1. **Model Comparison Table**: Displays performance and fairness metrics for all trained models.
2. **ROC Curves**: Visualizes the ROC curves for model comparison.
3. **Fairness Metrics**: Shows detailed fairness analysis for each model.

## Key Features Demonstrated

### 1. Data Generation

- Creates a synthetic hiring dataset with realistic features (age, education, experience, etc.)
- Introduces synthetic bias for demonstration purposes
- Handles preprocessing of both numerical and categorical features

### 2. Model Training

- Trains multiple models (Random Forest, Gradient Boosting, Logistic Regression)
- Uses scikit-learn's Pipeline for clean preprocessing
- Handles class imbalance in the target variable

### 3. Fairness Evaluation

- Calculates various fairness metrics:
  - Demographic Parity Difference
  - Equalized Odds Difference
  - Equal Opportunity Difference
  - Statistical Parity Difference
- Performs statistical significance testing
- Supports both binary and multi-class classification

### 4. Visualization

- Generates ROC curves for model comparison
- Creates bar plots for fairness metric comparison
- Visualizes model performance across different demographic groups

## Customization

You can customize the example by:

1. **Using Your Own Data**:
   Replace the synthetic data generation with your own dataset by modifying the `load_and_preprocess_data` method to load from a CSV file.

2. **Adding New Models**:
   Add new models to the `train_models` method by including them in the `models` dictionary.

3. **Customizing Metrics**:
   Modify the `evaluate_fairness` method to include additional fairness metrics or performance measures.

## Advanced Usage

### Hyperparameter Tuning

To improve model performance, you can add hyperparameter tuning using scikit-learn's `GridSearchCV` or `RandomizedSearchCV`.

### Cross-Validation

The example includes cross-validation support in the `compare_models` method. You can adjust the number of folds and other parameters as needed.

### Fairness Mitigation

To address fairness issues, you can integrate fairness-aware algorithms or post-processing techniques such as:
- Reject Option Classification
- Calibrated Equalized Odds
- Exponentiated Gradient Reduction

## Troubleshooting

1. **Import Errors**:
   Ensure all required packages are installed in your Python environment.

2. **Memory Issues**:
   For large datasets, consider reducing the number of samples or using more memory-efficient models.

3. **Convergence Warnings**:
   Some models like Logistic Regression might show convergence warnings. You can increase the `max_iter` parameter if needed.

## License

This example is part of the AI Fairness and Explainability Toolkit (AFET) and is available under the MIT License.

## Contributing

Contributions to improve this example are welcome! Please feel free to submit issues or pull requests with your enhancements.
