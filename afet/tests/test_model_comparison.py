"""
Tests for the ModelComparator class in afet.core.model_comparison
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Import the class to test
from afet.core.model_comparison import ModelComparator

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Fixture for test data
@pytest.fixture(scope="module")
def test_data():
    """Generate test data for model comparison."""
    # Generate synthetic data with a binary target and a sensitive attribute
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        weights=[0.7, 0.3],
        random_state=RANDOM_STATE
    )
    
    # Add a sensitive attribute (e.g., gender)
    sensitive = np.random.choice(['A', 'B'], size=len(y), p=[0.6, 0.4])
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    
    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'sens_train': sens_train, 'sens_test': sens_test
    }

# Fixture for models
@pytest.fixture(scope="module")
def models():
    """Initialize models for testing."""
    return {
        'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE),
        'GradientBoosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
        'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    }

# Fixture for model comparator
@pytest.fixture(scope="module")
def model_comparator():
    """Initialize model comparator for testing."""
    return ModelComparator(
        protected_attribute='gender',
        favorable_label=1,
        unfavorable_label=0,
        random_state=RANDOM_STATE
    )

def test_model_comparator_initialization(model_comparator):
    """Test that the ModelComparator initializes correctly."""
    assert model_comparator is not None
    assert model_comparator.protected_attribute == 'gender'
    assert model_comparator.favorable_label == 1
    assert model_comparator.unfavorable_label == 0
    assert model_comparator.random_state == RANDOM_STATE

def test_evaluate_model(model_comparator, test_data, models):
    """Test the evaluate_model method."""
    # Test with a single model
    model_name = list(models.keys())[0]
    model = models[model_name]
    
    # Train the model
    model.fit(test_data['X_train'], test_data['y_train'])
    
    # Evaluate the model
    metrics = model_comparator.evaluate_model(
        model=model,
        X=test_data['X_test'],
        y_true=test_data['y_test'],
        sensitive_features=test_data['sens_test'],
        model_name=model_name
    )
    
    # Check that metrics were calculated
    assert 'accuracy' in metrics
    assert 'roc_auc' in metrics
    assert 'demographic_parity_diff' in metrics
    assert 'equal_odds_diff' in metrics
    assert metrics['model'] == model_name

def test_compare_models(model_comparator, test_data, models):
    """Test the compare_models method."""
    # Train and compare models
    results_df = model_comparator.compare_models(
        models=models,
        X=test_data['X_train'],
        y_true=test_data['y_train'],
        sensitive_features=test_data['sens_train'],
        n_splits=3  # Use fewer splits for faster testing
    )
    
    # Check that results are as expected
    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) == len(models)
    assert set(results_df['model'].values) == set(models.keys())
    assert 'accuracy' in results_df.columns
    assert 'roc_auc' in results_df.columns
    assert 'demographic_parity_diff' in results_df.columns

def test_plot_metric_comparison(model_comparator, test_data, models):
    """Test the plot_metric_comparison method."""
    import matplotlib.pyplot as plt
    
    # First, evaluate some models
    for name, model in models.items():
        model.fit(test_data['X_train'], test_data['y_train'])
        model_comparator.evaluate_model(
            model=model,
            X=test_data['X_test'],
            y_true=test_data['y_test'],
            sensitive_features=test_data['sens_test'],
            model_name=name
        )
    
    # Test plotting
    fig = model_comparator.plot_metric_comparison(metric='accuracy')
    
    # Check that the figure was created
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_roc_curve(model_comparator, test_data, models):
    """Test the plot_roc_curve method."""
    import matplotlib.pyplot as plt
    
    # Train models
    trained_models = {}
    for name, model in models.items():
        model.fit(test_data['X_train'], test_data['y_train'])
        trained_models[name] = model
    
    # Test plotting
    fig = model_comparator.plot_roc_curve(
        models=trained_models,
        X=test_data['X_test'],
        y_true=test_data['y_test']
    )
    
    # Check that the figure was created
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_fairness_metrics_calculation(model_comparator, test_data):
    """Test that fairness metrics are calculated correctly."""
    # Train a simple model
    model = LogisticRegression(random_state=RANDOM_STATE)
    model.fit(test_data['X_train'], test_data['y_train'])
    
    # Get predictions
    y_pred = model.predict(test_data['X_test'])
    
    # Evaluate with the comparator
    metrics = model_comparator.evaluate_model(
        model=model,
        X=test_data['X_test'],
        y_true=test_data['y_test'],
        sensitive_features=test_data['sens_test'],
        model_name='test_model'
    )
    
    # Check that fairness metrics are within expected ranges
    assert -1 <= metrics['demographic_parity_diff'] <= 1
    assert -1 <= metrics['equal_odds_diff'] <= 1
    assert -1 <= metrics['equal_opportunity_diff'] <= 1
    assert -1 <= metrics['selection_rate_diff'] <= 1

def test_cross_validation_splits(model_comparator, test_data, models):
    """Test that cross-validation works with the specified number of splits."""
    n_splits = 3
    results_df = model_comparator.compare_models(
        models={k: v for k, v in list(models.items())[:2]},  # Just test with 2 models for speed
        X=test_data['X_train'],
        y_true=test_data['y_train'],
        sensitive_features=test_data['sens_train'],
        n_splits=n_splits
    )
    
    # Check that we have results for each model
    assert len(results_df) == 2
    
    # Check that the results include the expected metrics
    expected_metrics = ['accuracy', 'roc_auc', 'demographic_parity_diff', 'equal_odds_diff']
    for metric in expected_metrics:
        assert metric in results_df.columns

def test_handling_of_single_class(model_comparator, test_data):
    """Test handling of edge case where predictions contain only one class."""
    # Create a model that always predicts the same class
    from sklearn.dummy import DummyClassifier
    model = DummyClassifier(strategy='constant', constant=1)
    model.fit(test_data['X_train'], test_data['y_train'])
    
    # This should not raise an exception
    metrics = model_comparator.evaluate_model(
        model=model,
        X=test_data['X_test'],
        y_true=test_data['y_test'],
        sensitive_features=test_data['sens_test'],
        model_name='dummy_model'
    )
    
    # Check that metrics were still calculated
    assert 'accuracy' in metrics
    assert 'demographic_parity_diff' in metrics
