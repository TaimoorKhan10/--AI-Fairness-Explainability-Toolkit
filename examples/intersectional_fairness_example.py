"""
Intersectional Fairness Analysis Example

This example demonstrates how to use the intersectional fairness metrics and visualizations
to analyze bias across multiple protected attributes.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Import AFET components
from afet.core.intersectional_fairness import IntersectionalFairnessMetrics
from afet.viz.intersectional_heatmap import plot_intersectional_heatmap, plot_multi_metric_heatmaps
from afet.datasets.synthetic_generator import generate_credit_dataset, generate_hiring_dataset, generate_healthcare_dataset
from afet.core.bias_mitigation import Reweighing

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def main():
    """Run the intersectional fairness analysis example."""
    print("=== Intersectional Fairness Analysis Example ===\n")
    
    # Generate synthetic data with multiple protected attributes
    print("Generating synthetic hiring data with controlled bias...")
    data = generate_hiring_dataset(
        n_samples=5000,
        bias_level=0.3,  # Moderate bias level
        include_gender=True,
        include_age=True,
        include_race=True,
        random_state=RANDOM_STATE
    )
    
    print(f"Generated dataset with {len(data)} samples")
    print("\nData summary:")
    print(data.head())
    
    # Print distribution of protected attributes
    print("\nDistribution of protected attributes:")
    for attr in ['gender', 'age_group', 'race']:
        print(f"\n{attr.capitalize()} distribution:")
        print(data[attr].value_counts(normalize=True).round(3) * 100)
    
    # Print hiring rates by protected attribute
    print("\nHiring rates by protected attribute:")
    for attr in ['gender', 'age_group', 'race']:
        print(f"\n{attr.capitalize()} hiring rates:")
        print(data.groupby(attr)['hired'].mean().round(3) * 100)
    
    # Split data into features and target
    X = data.drop(columns=['hired'])
    y = data['hired']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    
    # Preprocess numerical features
    scaler = StandardScaler()
    numerical_cols = ['experience', 'skills', 'interview_score']
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Train a Random Forest model
    print("\nTraining Random Forest model...")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X_train_scaled[numerical_cols + ['education']], y_train)
    
    # Generate predictions
    y_pred_rf = rf.predict(X_test_scaled[numerical_cols + ['education']])
    
    # Analyze intersectional fairness
    print("\nAnalyzing intersectional fairness...")
    
    # Create protected features dictionary for intersectional analysis
    protected_features = {
        'gender': X_test['gender'].values,
        'race': X_test['race'].values,
        'age_group': X_test['age_group'].values
    }
    
    # Calculate intersectional fairness metrics
    ifm = IntersectionalFairnessMetrics(
        y_true=y_test.values,
        y_pred=y_pred_rf,
        protected_features=protected_features
    )
    
    # Print group metrics
    print("\nIntersectional group metrics:")
    group_metrics = ifm.group_metrics()
    print(group_metrics[['Group', 'Size', 'Accuracy', 'Selection_Rate', 'TPR', 'FPR']].round(3))
    
    # Print disparity metrics
    print("\nDisparity metrics:")
    for metric in ['Selection_Rate', 'TPR', 'FPR', 'Accuracy']:
        disparity = ifm.intersectional_disparity(metric)
        print(f"{metric} disparity: {disparity['max_disparity']:.3f} (min: {disparity['min_value']:.3f}, max: {disparity['max_value']:.3f})")
    
    # Create visualizations
    print("\nCreating intersectional fairness visualizations...")
    
    # Create heatmap for selection rate by gender and race
    selection_rate_fig = plot_intersectional_heatmap(
        y_true=y_test.values,
        y_pred=y_pred_rf,
        protected_features=protected_features,
        primary_attr='gender',
        secondary_attr='race',
        metric='Selection_Rate',
        title="Selection Rate by Gender and Race"
    )
    
    # Save the heatmap
    selection_rate_fig.write_html("selection_rate_heatmap.html")
    print("Selection rate heatmap saved to 'selection_rate_heatmap.html'")
    
    # Create multi-metric heatmaps for gender and age
    multi_metric_fig = plot_multi_metric_heatmaps(
        y_true=y_test.values,
        y_pred=y_pred_rf,
        protected_features=protected_features,
        primary_attr='race',
        secondary_attr='age_group',
        metrics=['Selection_Rate', 'TPR', 'FPR', 'Accuracy'],
        title="Fairness Metrics by Race and Age"
    )
    
    # Save the multi-metric heatmaps
    multi_metric_fig.write_html("multi_metric_heatmaps.html")
    print("Multi-metric heatmaps saved to 'multi_metric_heatmaps.html'")
    
    # Apply bias mitigation and compare results
    print("\nApplying bias mitigation...")
    
    # Apply Reweighing to mitigate bias
    reweigher = Reweighing(sensitive_attr='gender', target_attr='hired')
    sample_weights = reweigher.fit_transform(X_train, y_train)
    
    # Train a new model with reweighing
    rf_reweighted = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf_reweighted.fit(X_train_scaled[numerical_cols + ['education']], y_train, sample_weight=sample_weights)
    
    # Generate predictions with the reweighted model
    y_pred_reweighted = rf_reweighted.predict(X_test_scaled[numerical_cols + ['education']])
    
    # Calculate intersectional fairness metrics for the reweighted model
    ifm_reweighted = IntersectionalFairnessMetrics(
        y_true=y_test.values,
        y_pred=y_pred_reweighted,
        protected_features=protected_features
    )
    
    # Compare disparity metrics before and after mitigation
    print("\nDisparity metrics comparison (before vs. after mitigation):")
    for metric in ['Selection_Rate', 'TPR', 'FPR', 'Accuracy']:
        disparity_before = ifm.intersectional_disparity(metric)['max_disparity']
        disparity_after = ifm_reweighted.intersectional_disparity(metric)['max_disparity']
        change = ((disparity_after - disparity_before) / disparity_before) * 100
        print(f"{metric} disparity: {disparity_before:.3f} → {disparity_after:.3f} ({change:.1f}% change)")
    
    # Create comparison visualization
    print("\nCreating comparison visualization...")
    
    # Compare selection rates before and after mitigation
    comparison_fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Before Mitigation", "After Mitigation"]
    )
    
    # Add heatmaps for original model
    heatmap_before = plot_intersectional_heatmap(
        y_true=y_test.values,
        y_pred=y_pred_rf,
        protected_features=protected_features,
        primary_attr='gender',
        secondary_attr='race',
        metric='Selection_Rate'
    )
    
    # Add heatmaps for mitigated model
    heatmap_after = plot_intersectional_heatmap(
        y_true=y_test.values,
        y_pred=y_pred_reweighted,
        protected_features=protected_features,
        primary_attr='gender',
        secondary_attr='race',
        metric='Selection_Rate'
    )
    
    # Combine the figures
    for trace in heatmap_before.data:
        comparison_fig.add_trace(trace, row=1, col=1)
    
    for trace in heatmap_after.data:
        comparison_fig.add_trace(trace, row=1, col=2)
    
    # Update layout
    comparison_fig.update_layout(
        title="Selection Rate Before and After Bias Mitigation",
        height=500,
        width=1000
    )
    
    # Save the comparison visualization
    comparison_fig.write_html("mitigation_comparison.html")
    print("Mitigation comparison visualization saved to 'mitigation_comparison.html'")
    
    print("\n=== Example complete! ===")


# Import for the comparison visualization
from plotly.subplots import make_subplots

if __name__ == "__main__":
    main()
