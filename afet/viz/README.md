# Interactive Visualization Module

This module provides interactive visualization tools for exploring model fairness and explainability using Plotly.

## Features

1. **FairnessDashboard**: A comprehensive dashboard for comparing model fairness across different sensitive groups.
   - ROC curves by sensitive group
   - Precision-Recall curves
   - Fairness metrics comparison
   - Feature importance visualization

2. **ThresholdAnalysis**: Interactive analysis of fairness-accuracy trade-offs across different classification thresholds.

3. **Confusion Matrices**: Side-by-side comparison of confusion matrices for multiple models.

## Installation

```bash
pip install plotly pandas scikit-learn
```

## Usage

### 1. Fairness Dashboard

```python
from afet.viz import FairnessDashboard

# Initialize dashboard
dashboard = FairnessDashboard(
    model_names=['Model1', 'Model2'],
    sensitive_attr='gender'
)

# Create dashboard
fig = dashboard.create_dashboard(
    y_true=y_test,
    y_pred={
        'Model1': y_pred1,
        'Model2': y_pred2
    },
    y_prob={
        'Model1': y_prob1,
        'Model2': y_prob2
    },
    sensitive=X_test['gender'].values,
    feature_names=feature_names,
    feature_importances={
        'Model1': importances1,
        'Model2': importances2
    }
)

# Show dashboard
fig.show()
```

### 2. Threshold Analysis

```python
from afet.viz import ThresholdAnalysis

# Create threshold analysis
fig = ThresholdAnalysis.plot_threshold_analysis(
    y_true=y_test,
    y_prob=y_prob,
    sensitive=X_test['gender'].values,
    model_name='MyModel'
)

# Show figure
fig.show()
```

### 3. Confusion Matrices

```python
from afet.viz import plot_confusion_matrices

# Create confusion matrices
fig = plot_confusion_matrices(
    y_true=y_test,
    y_pred_dict={
        'Model1': y_pred1,
        'Model2': y_pred2
    },
    model_names=['Model1', 'Model2'],
    class_names=['Negative', 'Positive']
)

# Show figure
fig.show()
```

## Example

See the [full pipeline example](../../examples/full_pipeline_example.py) for a complete demonstration of how to use these visualization tools in a fairness-aware machine learning pipeline.

## Interactive Features

- Hover over data points to see detailed information
- Toggle traces on/off by clicking on the legend
- Zoom, pan, and download plots
- Save visualizations as HTML files for sharing

## Dependencies

- Python 3.7+
- plotly
- pandas
- scikit-learn
- numpy
