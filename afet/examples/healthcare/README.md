# Patient Readmission Prediction with Fairness Analysis

This example demonstrates how to use the AI Fairness and Explainability Toolkit (AFET) to analyze and compare machine learning models for predicting patient readmissions with a focus on fairness across different demographic groups.

## Overview

The patient readmission prediction example covers the following key aspects:

1. **Synthetic Data Generation**: Creates a realistic healthcare dataset with potential biases related to race, insurance status, and other factors.
2. **Model Training**: Trains multiple machine learning models for readmission prediction.
3. **Fairness Evaluation**: Evaluates models using various fairness metrics.
4. **Bias Analysis**: Analyzes the impact of sensitive attributes (like race) on model predictions.
5. **Feature Importance**: Identifies which features are most influential in the model's predictions.

## Features

- **Synthetic Data Generation**:
  - Realistic patient demographics and medical features
  - Built-in biases based on race and insurance status
  - Customizable data generation parameters

- **Model Training**:
  - Multiple algorithms (Random Forest, Gradient Boosting, Logistic Regression)
  - Automated preprocessing pipeline
  - Class imbalance handling

- **Fairness Analysis**:
  - Demographic parity difference
  - Equalized odds difference
  - Impact analysis by race/ethnicity
  - Statistical significance testing

- **Visualization**:
  - Fairness metric comparisons
  - Readmission rate disparities
  - Feature importance analysis

## Requirements

- Python 3.7+
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- tqdm
- scikit-posthocs

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AI-Fairness-Explainability-Toolkit.git
   cd AI-Fairness-Explainability-Toolkit
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Navigate to the healthcare example directory:
   ```bash
   cd afet/examples/healthcare
   ```

2. Run the patient readmission analysis:
   ```bash
   python patient_readmission.py
   ```

## Example Output

The script will generate several outputs:

1. **Model Comparison Table**: Shows performance and fairness metrics for all trained models
2. **Fairness Metrics Visualization**: Displays demographic parity and equalized odds differences
3. **Race Impact Analysis**: Visualizes readmission rates by race for both actual and predicted outcomes
4. **Feature Importance**: Shows which features are most influential in the model's predictions

## Customization

### Using Your Own Data

To use your own dataset instead of the synthetic data:

1. Prepare a CSV file with your data
2. Update the `load_and_preprocess_data` method to load your file:
   ```python
   analysis.load_and_preprocess_data("path/to/your/patient_data.csv")
   ```

### Adding New Models

To add a new model to the comparison:

1. Import your model class
2. Add it to the `models` dictionary in the `train_models` method:
   ```python
   models = {
       'YourModel': YourModelClass(),
       # ... existing models
   }
   ```

### Adjusting Fairness Metrics

To modify the fairness analysis:

1. Update the `evaluate_fairness` method
2. Add or modify metrics in the `evaluate_model` call

## Interpreting Results

- **Demographic Parity Difference**: Measures the difference in readmission rates between groups
  - Values close to 0 indicate fairer outcomes
  - Positive values indicate bias in favor of the majority group

- **Equalized Odds Difference**: Measures differences in true positive and false positive rates
  - Values close to 0 indicate fairer outcomes
  - Larger absolute values indicate greater disparities

## License

This example is part of the AI Fairness and Explainability Toolkit (AFET) and is available under the MIT License.

## Contributing

Contributions to improve this example are welcome! Please feel free to submit issues or pull requests with your enhancements.
