"""
Tests for fairness metrics
"""

import pytest
import numpy as np
import pandas as pd
from afet.core.fairness_metrics import FairnessMetrics
from afet.core.fairness_mitigation import FairnessMitigator


def test_fairness_metrics():
    """
    Test fairness metrics calculations
    """
    # Create test data
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    sensitive_features = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    
    # Initialize fairness metrics
    metrics = FairnessMetrics(
        protected_attribute='gender',
        favorable_label=1,
        unfavorable_label=0
    )
    
    # Calculate metrics
    results = metrics.get_comprehensive_metrics(
        y_pred=y_pred,
        y_true=y_true,
        sensitive_features=sensitive_features
    )
    
    # Check results
    assert 'demographic_parity_0' in results
    assert 'demographic_parity_1' in results
    assert 'disparate_impact' in results
    assert 'statistical_parity_difference' in results

def test_fairness_mitigation():
    """
    Test fairness mitigation
    """
    # Create test data
    X = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    })
    y = np.array([1, 0, 1, 0, 1, 0])
    sensitive_features = np.array([0, 0, 1, 1, 0, 1])
    
    # Initialize mitigator
    mitigator = FairnessMitigator(
        protected_attribute='gender',
        favorable_label=1,
        unfavorable_label=0
    )
    
    # Apply mitigation
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    
    mitigated_results = mitigator.mitigation_pipeline(
        X=X,
        y=y,
        sensitive_features=sensitive_features,
        estimator=model,
        preprocessing=True
    )
    
    # Check results
    assert 'model' in mitigated_results
    assert 'predictions' in mitigated_results
    assert 'evaluation' in mitigated_results

def test_edge_cases():
    """
    Test edge cases for fairness metrics
    """
    # Test with empty arrays
    metrics = FairnessMetrics(
        protected_attribute='gender',
        favorable_label=1,
        unfavorable_label=0
    )
    
    with pytest.raises(ValueError):
        metrics.get_comprehensive_metrics(
            y_pred=np.array([]),
            y_true=np.array([]),
            sensitive_features=np.array([])
        )
    
    # Test with mismatched array sizes
    with pytest.raises(ValueError):
        metrics.get_comprehensive_metrics(
            y_pred=np.array([1, 0]),
            y_true=np.array([1, 0, 1]),
            sensitive_features=np.array([0, 1])
        )

def test_performance():
    """
    Test performance with large datasets
    """
    # Create large test data
    np.random.seed(42)
    n_samples = 10000
    
    X = pd.DataFrame({
        'feature1': np.random.normal(size=n_samples),
        'feature2': np.random.normal(size=n_samples)
    })
    y = np.random.binomial(1, 0.5, n_samples)
    sensitive_features = np.random.binomial(1, 0.5, n_samples)
    
    # Initialize metrics
    metrics = FairnessMetrics(
        protected_attribute='gender',
        favorable_label=1,
        unfavorable_label=0
    )
    
    # Calculate metrics (should complete in reasonable time)
    results = metrics.get_comprehensive_metrics(
        y_pred=y,
        y_true=y,
        sensitive_features=sensitive_features
    )
    
    assert len(results) > 0

def test_integration():
    """
    Test integration of metrics and mitigation
    """
    # Create test data
    X = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    })
    y = np.array([1, 0, 1, 0, 1, 0])
    sensitive_features = np.array([0, 0, 1, 1, 0, 1])
    
    # Initialize metrics and mitigator
    metrics = FairnessMetrics(
        protected_attribute='gender',
        favorable_label=1,
        unfavorable_label=0
    )
    
    mitigator = FairnessMitigator(
        protected_attribute='gender',
        favorable_label=1,
        unfavorable_label=0
    )
    
    # Train model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    # Get initial metrics
    y_pred = model.predict(X)
    initial_metrics = metrics.get_comprehensive_metrics(
        y_pred=y_pred,
        y_true=y,
        sensitive_features=sensitive_features
    )
    
    # Apply mitigation
    mitigated_results = mitigator.mitigation_pipeline(
        X=X,
        y=y,
        sensitive_features=sensitive_features,
        estimator=model,
        preprocessing=True
    )
    
    # Get mitigated metrics
    mitigated_metrics = metrics.get_comprehensive_metrics(
        y_pred=mitigated_results['predictions'],
        y_true=y,
        sensitive_features=sensitive_features
    )
    
    # Check if mitigation improved fairness
    assert mitigated_metrics['demographic_parity_difference'] <= initial_metrics['demographic_parity_difference']
