# Loan Approval System Tutorial

This tutorial demonstrates how to use AFET to evaluate and improve a loan approval system. We'll use the synthetic loan dataset to showcase the toolkit's capabilities.

## 1. Load and Prepare Data

```python
from afet.datasets import SyntheticLoanDataset

# Initialize dataset generator
dataset = SyntheticLoanDataset(n_samples=10000)

# Generate dataset
data = dataset.generate_dataset()
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

# Get sensitive features
sensitive_features = X_test['gender']
```

## 2. Train a Baseline Model

```python
from sklearn.ensemble import RandomForestClassifier

# Train a simple model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
```

## 3. Evaluate Fairness

```python
from afet.core.fairness_metrics import FairnessMetrics

# Initialize fairness metrics
demographics = FairnessMetrics(
    protected_attribute='gender',
    favorable_label=1,
    unfavorable_label=0
)

# Get predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate fairness metrics
metrics = demographics.get_comprehensive_metrics(
    y_pred=y_pred,
    y_true=y_test,
    sensitive_features=sensitive_features
)

print("Fairness Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

## 4. Model Explainability

```python
from afet.core.explainability import ModelExplainer

# Initialize explainability
explainer = ModelExplainer(
    model=model,
    feature_names=X_test.columns.tolist(),
    class_names=['0', '1'],
    training_data=X_train.values
)

# Get global feature importance
feature_importance = explainer.explain_global_shap(X_test.values)

# Explain a specific instance
instance_idx = 0
instance = X_test.iloc[instance_idx].values
explanations = explainer.get_all_explanations(
    instance=instance,
    data=X_test.values
)
```

## 5. Apply Fairness Mitigation

```python
from afet.core.fairness_mitigation import FairnessMitigator

# Initialize mitigator
mitigator = FairnessMitigator(
    protected_attribute='gender',
    favorable_label=1,
    unfavorable_label=0
)

# Apply mitigation
mitigated_results = mitigator.mitigation_pipeline(
    X=X_train,
    y=y_train,
    sensitive_features=X_train['gender'],
    estimator=model,
    preprocessing=True
)

# Evaluate mitigated model
mitigated_metrics = demographics.get_comprehensive_metrics(
    y_pred=mitigated_results['predictions'],
    y_true=y_test,
    sensitive_features=sensitive_features
)
```

## 6. Compare Models

```python
from afet.core.model_comparison import ModelComparator

# Initialize comparator
comparator = ModelComparator(
    protected_attribute='gender',
    favorable_label=1,
    unfavorable_label=0
)

# Compare models
models = {
    'baseline': model,
    'mitigated': mitigated_results['model']
}

comparison = comparator.compare_models(
    models,
    X_test,
    y_test,
    sensitive_features
)

# Generate comparison report
report = comparator.create_comparison_report(
    models,
    X_test,
    y_test,
    sensitive_features
)
```

## 7. Visualize Results

```python
from afet.core.advanced_visualizations import AdvancedVisualizations

# Initialize visualizations
visualizer = AdvancedVisualizations()

# Create fairness dashboard
fairness_fig = visualizer.create_fairness_dashboard(
    metrics,
    sensitive_features
)

# Create model comparison plot
comparison_fig = visualizer.create_model_comparison_plot(
    comparison,
    'accuracy'
)

# Create fairness tradeoff plot
tradeoff_fig = visualizer.create_fairness_tradeoff_plot(
    report['summary'],
    sensitive_features
)
```

## 8. Interpret Results

1. **Fairness Metrics**
   - Compare demographic parity and equal opportunity
   - Analyze predictive parity across groups
   - Evaluate calibration error

2. **Model Performance**
   - Compare accuracy and ROC AUC
   - Analyze feature importance
   - Review instance-level explanations

3. **Mitigation Impact**
   - Assess changes in fairness metrics
   - Evaluate performance tradeoffs
   - Document findings

## Best Practices

1. **Data Analysis**
   - Examine data distribution
   - Check for imbalances
   - Document findings

2. **Model Selection**
   - Consider multiple models
   - Evaluate tradeoffs
   - Document decisions

3. **Fairness Monitoring**
   - Regularly evaluate metrics
   - Track changes over time
   - Document trends

4. **Documentation**
   - Maintain clear records
   - Document decisions
   - Share findings
