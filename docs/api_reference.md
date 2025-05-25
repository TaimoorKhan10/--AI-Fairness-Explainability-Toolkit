# API Reference

## Core Components

### Fairness Metrics

```python
from afet.core.fairness_metrics import FairnessMetrics

# Initialize fairness metrics
demographics = FairnessMetrics(
    protected_attribute='gender',
    favorable_label=1,
    unfavorable_label=0
)

# Calculate metrics
metrics = demographics.get_comprehensive_metrics(
    y_pred=model_predictions,
    y_true=true_labels,
    sensitive_features=sensitive_features
)
```

### Explainability

```python
from afet.core.explainability import ModelExplainer

# Initialize explainability
explainer = ModelExplainer(
    model=model,
    feature_names=feature_names,
    class_names=['0', '1'],
    training_data=training_data
)

# Get explanations
explanations = explainer.get_all_explanations(
    instance=data_instance,
    data=training_data
)
```

### Fairness Mitigation

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
    X=training_data,
    y=labels,
    sensitive_features=sensitive_features,
    estimator=model,
    preprocessing=True
)
```

## Dashboard API

### Endpoints

1. **Train Model**
   - POST `/api/models/train`
   - Request Body:
     ```json
     {
         "model_type": "classification",
         "features": {"feature1": "value1"},
         "target": "target_column",
         "sensitive_features": ["gender", "race"]
     }
     ```

2. **Evaluate Fairness**
   - POST `/api/fairness/evaluate`
   - Request Body:
     ```json
     {
         "model_name": "model_1",
         "data_id": "data_1",
         "metrics": ["demographic_parity", "equalized_odds"]
     }
     ```

3. **Get Explanations**
   - POST `/api/explain`
   - Request Body:
     ```json
     {
         "model_name": "model_1",
         "instance_id": "instance_1",
         "explanation_type": "shap"
     }
     ```

4. **Apply Mitigation**
   - POST `/api/mitigate`
   - Request Body:
     ```json
     {
         "model_name": "model_1",
         "data_id": "data_1",
         "strategy": "reweighing",
         "parameters": {"epsilon": 0.1}
     }
     ```

## Model Registry

```python
from afet.models.model_registry import registry

# Register a new model
registry.register_model(
    model=model,
    model_id="model_1",
    metadata={
        "model_type": "classification",
        "features": ["feature1", "feature2"],
        "target": "target_column",
        "sensitive_features": ["gender", "race"]
    }
)

# Get model list
models = registry.list_models()

# Delete a model
registry.delete_model("model_1")
```

## Configuration

```python
from afet.core.config import config

# Get configuration
threshold = config.get('fairness.default_threshold')

# Set configuration
config.set('fairness.default_threshold', 0.1)
```

## Logging

```python
from afet.core.logging_setup import logger

# Log messages
logger.info("Starting model training")
logger.warning("Low accuracy detected")
logger.error("Training failed")
```

## Best Practices

1. **Data Preparation**
   - Clean and preprocess data
   - Handle missing values
   - Encode categorical variables
   - Scale features appropriately

2. **Model Selection**
   - Choose appropriate model type
   - Consider computational resources
   - Document model architecture

3. **Fairness Evaluation**
   - Evaluate multiple metrics
   - Consider different groups
   - Document findings

4. **Model Deployment**
   - Monitor performance
   - Track fairness metrics
   - Maintain documentation

## Error Handling

```python
try:
    # Model training
    model.fit(X, y)
except ValueError as e:
    logger.error(f"Training failed: {str(e)}")
except Exception as e:
    logger.error(f"Unexpected error: {str(e)}")
```

## Performance Optimization

1. **Data Processing**
   - Use efficient data structures
   - Batch processing
   - Parallel computing

2. **Model Training**
   - Early stopping
   - Hyperparameter optimization
   - Model caching

3. **Prediction**
   - Batch predictions
   - Model serialization
   - Caching results

## Security Considerations

1. **Data Protection**
   - Encrypt sensitive data
   - Access controls
   - Audit logging

2. **API Security**
   - Authentication
   - Rate limiting
   - Input validation

3. **Model Protection**
   - Model encryption
   - Access controls
   - Version management
