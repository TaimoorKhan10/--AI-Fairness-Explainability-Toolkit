"""
Tests for the visualization module.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Import visualization components to test
from afet.viz import (
    FairnessDashboard,
    ThresholdAnalysis,
    plot_confusion_matrices
)


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    # Create synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    
    # Create synthetic sensitive attribute
    sensitive = np.random.choice(['A', 'B'], size=1000, p=[0.7, 0.3])
    
    # Create train-test split
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]
    sensitive_train, sensitive_test = sensitive[:800], sensitive[800:]
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Get predictions and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return {
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'sensitive_test': sensitive_test,
        'feature_importances': model.feature_importances_,
        'feature_names': [f'feature_{i}' for i in range(10)]
    }

def test_fairness_dashboard_creation(sample_data):
    """Test creation of FairnessDashboard."""
    dashboard = FairnessDashboard(
        model_names=['TestModel'],
        sensitive_attr='test_attr'
    )
    
    fig = dashboard.create_dashboard(
        y_true=sample_data['y_test'],
        y_pred={'TestModel': sample_data['y_pred']},
        y_prob={'TestModel': sample_data['y_prob']},
        sensitive=sample_data['sensitive_test'],
        feature_names=sample_data['feature_names'],
        feature_importances={'TestModel': sample_data['feature_importances']}
    )
    
    assert fig is not None
    assert len(fig.data) > 0  # Should have at least one trace


def test_threshold_analysis(sample_data):
    """Test threshold analysis visualization."""
    fig = ThresholdAnalysis.plot_threshold_analysis(
        y_true=sample_data['y_test'],
        y_prob=sample_data['y_prob'],
        sensitive=sample_data['sensitive_test'],
        model_name='TestModel'
    )
    
    assert fig is not None
    assert len(fig.data) > 0


def test_plot_confusion_matrices(sample_data):
    """Test confusion matrix visualization."""
    fig = plot_confusion_matrices(
        y_true=sample_data['y_test'],
        y_pred_dict={
            'Model1': sample_data['y_pred'],
            'Model2': 1 - sample_data['y_pred']  # Invert predictions for testing
        },
        model_names=['Model1', 'Model2']
    )
    
    assert fig is not None
    assert len(fig.data) == 2  # One heatmap per model
