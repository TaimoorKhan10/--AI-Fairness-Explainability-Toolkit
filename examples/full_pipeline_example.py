"""
Complete Fairness Pipeline Example

This example demonstrates a complete fairness-aware machine learning pipeline,
including data loading, preprocessing, model training, bias mitigation, and
interactive visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix
)

# Import AFET components
from afet.core.model_comparison import ModelComparator
from afet.core.fairness_metrics import FairnessMetrics
from afet.core.bias_mitigation import (
    PrejudiceRemover, Reweighing, CalibratedEqualizedOddsPostprocessing
)
from afet.viz import FairnessDashboard, ThresholdAnalysis, plot_confusion_matrices

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_credit_data():
    """Load and preprocess the credit dataset."""
    # For demonstration, we'll use a synthetic dataset
    # In practice, you would load your actual dataset here
    print("Generating synthetic credit data...")
    
    # Generate synthetic data with known biases
    n_samples = 5000
    
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
    
    # Generate loan amount (5k to 1M)
    loan_amount = np.random.lognormal(10, 0.8, n_samples).astype(int)
    loan_amount = np.clip(loan_amount, 5000, 1000000)
    
    # Generate employment length (0-30 years)
    employment_length = np.random.exponential(5, n_samples).astype(int)
    employment_length = np.clip(employment_length, 0, 30)
    
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
        'loan_amount': loan_amount,
        'employment_length': employment_length,
        'approved': approved
    })
    
    return data


def preprocess_data(data, target_col, sensitive_col):
    """Preprocess the data and split into train/test sets."""
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    
    # Define preprocessing for numerical and categorical features
    numeric_features = ['age', 'income', 'credit_score', 'debt_to_income',
                       'loan_amount', 'employment_length']
    categorical_features = ['race']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Fit and transform the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after preprocessing
    feature_names = (
        numeric_features +
        list(preprocessor.named_transformers_['cat'].named_steps['onehot']
             .get_feature_names_out(categorical_features))
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_processed': X_train_processed,
        'X_test_processed': X_test_processed,
        'sensitive_train': X_train[sensitive_col].values,
        'sensitive_test': X_test[sensitive_col].values,
        'preprocessor': preprocessor,
        'feature_names': feature_names
    }


def train_models(X_train, y_train):
    """Train multiple models for comparison."""
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        ),
        'LogisticRegression': LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight='balanced',
            solver='liblinear'
        )
    }
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
    
    return models


def apply_bias_mitigation(models, X_train, y_train, sensitive):
    """Apply bias mitigation techniques to the models."""
    mitigated_models = {}
    
    # 1. Reweighing (pre-processing)
    print("\nApplying Reweighing...")
    reweigher = Reweighing(sensitive_attr='race', target_attr='approved')
    sample_weights = reweigher.fit_transform(X_train, y_train)
    
    for name, model in models.items():
        print(f"Retraining {name} with sample weights...")
        model.fit(X_train, y_train, sample_weight=sample_weights)
        mitigated_models[f"{name}_reweighted"] = model
    
    # 2. Prejudice Remover (in-processing)
    print("\nTraining Prejudice Remover...")
    pr_model = PrejudiceRemover(eta=1.0, sensitive_cols=[X_train.columns.get_loc('race')])
    pr_model.fit(X_train.values, y_train, sensitive=X_train['race'].values)
    mitigated_models["PrejudiceRemover"] = pr_model
    
    # 3. Post-processing (Calibrated Equalized Odds)
    print("\nApplying Calibrated Equalized Odds...")
    for name in ['RandomForest', 'LogisticRegression']:
        y_prob = models[name].predict_proba(X_train)[:, 1]
        post_processor = CalibratedEqualizedOddsPostprocessing(cost_constraint='weighted')
        post_processor.fit(y_train, y_prob, sensitive=X_train['race'].values)
        mitigated_models[f"{name}_postprocessed"] = {
            'model': models[name],
            'post_processor': post_processor
        }
    
    return mitigated_models


def evaluate_models(models, X_test, y_test, sensitive_test, preprocessor):
    """Evaluate models and return predictions and metrics."""
    results = []
    y_preds = {}
    y_probs = {}
    
    for name, model in models.items():
        if 'postprocessed' in name:
            # Handle post-processed models
            y_prob = model['model'].predict_proba(X_test)[:, 1]
            y_pred = model['post_processor'].predict(y_prob, sensitive_test)
            y_prob = model['model'].predict_proba(X_test)[:, 1]  # Keep original probabilities
        else:
            # Handle regular models
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        y_preds[name] = y_pred
        y_probs[name] = y_prob
        
        # Calculate metrics
        fm = FairnessMetrics(
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sensitive_test
        )
        
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_prob) if y_prob is not None else None,
            'Demographic Parity': fm.demographic_parity_difference(),
            'Equal Opportunity': fm.equal_opportunity_difference(),
            'Average Odds': fm.average_odds_difference()
        })
    
    return pd.DataFrame(results), y_preds, y_probs


def main():
    """Run the complete fairness pipeline."""
    print("=== Fairness-Aware Machine Learning Pipeline ===\n")
    
    # 1. Load and preprocess data
    print("1. Loading and preprocessing data...")
    data = load_credit_data()
    processed = preprocess_data(data, 'approved', 'race')
    
    # 2. Train baseline models
    print("\n2. Training baseline models...")
    models = train_models(processed['X_train_processed'], processed['y_train'])
    
    # 3. Apply bias mitigation techniques
    print("\n3. Applying bias mitigation techniques...")
    mitigated_models = apply_bias_mitigation(
        models,
        processed['X_train'],
        processed['y_train'],
        processed['sensitive_train']
    )
    
    # Combine all models
    all_models = {**models, **mitigated_models}
    
    # 4. Evaluate all models
    print("\n4. Evaluating models...")
    results_df, y_preds, y_probs = evaluate_models(
        all_models,
        processed['X_test_processed'],
        processed['y_test'],
        processed['sensitive_test'],
        processed['preprocessor']
    )
    
    # Print results
    print("\n=== Model Comparison ===")
    print(results_df.round(4).to_string(index=False))
    
    # 5. Create interactive visualizations
    print("\n5. Creating interactive visualizations...")
    
    # Filter to include only original and reweighted models for clarity
    model_names = ['RandomForest', 'LogisticRegression', 
                  'RandomForest_reweighted', 'LogisticRegression_reweighted',
                  'PrejudiceRemover']
    
    # Create dashboard
    dashboard = FairnessDashboard(
        model_names=model_names,
        sensitive_attr='race'
    )
    
    # Prepare data for dashboard
    y_preds_filtered = {k: v for k, v in y_preds.items() if k in model_names}
    y_probs_filtered = {k: v for k, v in y_probs.items() if k in model_names}
    
    # Get feature importances for RandomForest
    feature_importances = {}
    if 'RandomForest' in models:
        feature_importances['RandomForest'] = models['RandomForest'].feature_importances_
    if 'RandomForest_reweighted' in mitigated_models:
        feature_importances['RandomForest_reweighted'] = mitigated_models['RandomForest_reweighted'].feature_importances_
    
    # Create dashboard
    fig = dashboard.create_dashboard(
        y_true=processed['y_test'],
        y_pred=y_preds_filtered,
        y_prob=y_probs_filtered,
        sensitive=processed['sensitive_test'],
        feature_names=processed['feature_names'],
        feature_importances=feature_importances if feature_importances else None
    )
    
    # Save dashboard to HTML
    fig.write_html("fairness_dashboard.html")
    print("\nDashboard saved to 'fairness_dashboard.html'")
    
    # Create threshold analysis for RandomForest
    if 'RandomForest' in models:
        print("\nCreating threshold analysis for RandomForest...")
        threshold_fig = ThresholdAnalysis.plot_threshold_analysis(
            y_true=processed['y_test'],
            y_prob=y_probs['RandomForest'],
            sensitive=processed['sensitive_test'],
            model_name='RandomForest'
        )
        threshold_fig.write_html("threshold_analysis.html")
        print("Threshold analysis saved to 'threshold_analysis.html'")
    
    # Create confusion matrices
    print("\nCreating confusion matrices...")
    confusion_fig = plot_confusion_matrices(
        y_true=processed['y_test'],
        y_pred_dict={
            'Original': y_preds['RandomForest'],
            'Reweighted': y_preds['RandomForest_reweighted'],
            'PrejudiceRemover': y_preds.get('PrejudiceRemover', None)
        },
        model_names=['Original', 'Reweighted', 'PrejudiceRemover'],
        class_names=['Rejected', 'Approved']
    )
    confusion_fig.write_html("confusion_matrices.html")
    print("Confusion matrices saved to 'confusion_matrices.html'")
    
    print("\n=== Pipeline complete! ===")


if __name__ == "__main__":
    main()
