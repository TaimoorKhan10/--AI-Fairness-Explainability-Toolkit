Quickstart
=========

This guide will help you get started with the AI Fairness Toolkit.

Basic Usage
----------

```python
from afet import FairnessAnalyzer, BiasMitigator, ModelExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
sensitive_features = pd.Series(['A'] * 500 + ['B'] * 500)

# Initialize and train a model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Initialize the Fairness Analyzer
analyzer = FairnessAnalyzer()

# Analyze model fairness
fairness_report = analyzer.analyze(
    model=model,
    X=X,
    y_true=y,
    sensitive_features=sensitive_features
)

# View the fairness report
print(fairness_report.summary())
```

Key Features
-----------

### Fairness Analysis

```python
# Get detailed fairness metrics
metrics = fairness_report.get_metrics()
print(metrics)

# Visualize fairness metrics
fairness_report.visualize()
```

### Bias Mitigation

```python
# Initialize bias mitigator
mitigator = BiasMitigator()

# Apply pre-processing mitigation
X_mitigated, y_mitigated = mitigator.fit_resample(
    X, y, 
    sensitive_features=sensitive_features
)
```

### Model Explainability

```python
# Initialize explainer
explainer = ModelExplainer()

# Generate feature importance
importance = explainer.explain(model, X, feature_names=[f"feature_{i}" for i in range(10)])

# Visualize feature importance
explainer.plot_feature_importance(importance)
```
