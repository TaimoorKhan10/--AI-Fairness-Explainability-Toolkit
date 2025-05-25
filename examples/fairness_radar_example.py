"""
Fairness Radar Plot Example

This example demonstrates how to use the FairnessRadar visualization component
to compare fairness metrics across different demographic groups and models.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Import AFET components
from afet.viz import plot_fairness_radar
from afet.core.bias_mitigation import Reweighing

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def generate_synthetic_data(n_samples=5000):
    """Generate synthetic data with known biases."""
    # Generate demographic features
    age = np.random.normal(45, 15, n_samples).astype(int)
    age = np.clip(age, 18, 90)
    
    # Generate race/ethnicity with potential bias
    race = np.random.choice(
        ['White', 'Black', 'Hispanic', 'Asian', 'Other'],
        size=n_samples,
        p=[0.6, 0.13, 0.18, 0.06, 0.03]
    )
    
    # Generate income with race-based differences
    income = np.zeros(n_samples)
    income[race == 'White'] = np.random.normal(75000, 20000, np.sum(race == 'White'))
    income[race == 'Black'] = np.random.normal(45000, 15000, np.sum(race == 'Black'))
    income[race == 'Hispanic'] = np.random.normal(50000, 18000, np.sum(race == 'Hispanic'))
    income[race == 'Asian'] = np.random.normal(80000, 25000, np.sum(race == 'Asian'))
    income[race == 'Other'] = np.random.normal(55000, 20000, np.sum(race == 'Other'))
    income = np.clip(income, 20000, 200000).astype(int)
    
    # Generate credit score (300-850)
    credit_score = np.random.normal(650, 100, n_samples)
    credit_score = np.clip(credit_score, 300, 850).astype(int)
    
    # Generate debt-to-income ratio (0.1 to 0.8)
    dti = np.random.beta(2, 5, n_samples) * 0.7 + 0.1
    
    # Generate target (loan approval) with some bias
    prob_approval = (
        0.3 +
        0.4 * (credit_score / 850) +
        0.2 * (1 - dti) +
        0.1 * (income / 200000) -
        0.15 * (race == 'Black') -
        0.1 * (race == 'Hispanic') +
        np.random.normal(0, 0.1, n_samples)  # Add some noise
    )
    prob_approval = np.clip(prob_approval, 0, 1)
    approved = (np.random.random(n_samples) < prob_approval).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'age': age,
        'race': race,
        'income': income,
        'credit_score': credit_score,
        'debt_to_income': dti,
        'approved': approved
    })
    
    return data


def main():
    """Run the fairness radar example."""
    print("=== Fairness Radar Plot Example ===\n")
    
    # Generate synthetic data
    print("Generating synthetic data...")
    data = generate_synthetic_data()
    
    # Split data into features and target
    X = data.drop(columns=['approved'])
    y = data['approved']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    
    # Preprocess numerical features
    scaler = StandardScaler()
    numerical_cols = ['age', 'income', 'credit_score', 'debt_to_income']
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Train baseline models
    print("Training models...")
    models = {}
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X_train_scaled, y_train)
    models['RandomForest'] = rf
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_scaled, y_train)
    models['LogisticRegression'] = lr
    
    # Apply bias mitigation - Reweighing
    print("Applying bias mitigation...")
    reweigher = Reweighing(sensitive_attr='race', target_attr='approved')
    sample_weights = reweigher.fit_transform(X_train, y_train)
    
    # Train models with reweighing
    rf_reweighted = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf_reweighted.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    models['RandomForest_Reweighted'] = rf_reweighted
    
    lr_reweighted = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr_reweighted.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    models['LogisticRegression_Reweighted'] = lr_reweighted
    
    # Generate predictions
    print("Generating predictions...")
    y_pred = {}
    for name, model in models.items():
        y_pred[name] = model.predict(X_test_scaled)
    
    # Create fairness radar plot
    print("Creating fairness radar plot...")
    fig = plot_fairness_radar(
        y_true=y_test.values,
        y_pred=y_pred,
        sensitive_features=X_test['race'].values,
        title="Fairness Metrics Comparison Across Demographic Groups"
    )
    
    # Save the plot
    fig.write_html("fairness_radar_plot.html")
    print("Fairness radar plot saved to 'fairness_radar_plot.html'")
    
    # Create a more focused plot with specific models and groups
    print("Creating focused fairness radar plot...")
    focused_fig = plot_fairness_radar(
        y_true=y_test.values,
        y_pred={k: y_pred[k] for k in ['RandomForest', 'RandomForest_Reweighted']},
        sensitive_features=X_test['race'].values,
        group_names=['White', 'Black', 'Hispanic'],
        title="Impact of Reweighing on Fairness Metrics"
    )
    
    # Save the focused plot
    focused_fig.write_html("focused_fairness_radar_plot.html")
    print("Focused fairness radar plot saved to 'focused_fairness_radar_plot.html'")
    
    print("\n=== Example complete! ===")


if __name__ == "__main__":
    main()
