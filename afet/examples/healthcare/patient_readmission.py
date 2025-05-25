"""
Patient Readmission Prediction with Fairness Analysis

This example demonstrates how to use the AI Fairness and Explainability Toolkit (AFET)
to analyze and compare machine learning models for predicting patient readmissions
with a focus on fairness across different demographic groups.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, 
    precision_recall_curve, average_precision_score
)

# Import AFET components
from afet.core.model_comparison import ModelComparator
from afet.core.fairness_metrics import FairnessMetrics
from afet.core.advanced_fairness_metrics import AdvancedFairnessMetrics

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
plt.style.use('seaborn-v0_8-whitegrid')

class PatientReadmissionAnalysis:
    """A class to demonstrate fairness analysis in patient readmission prediction."""
    
    def __init__(self):
        """Initialize the patient readmission analysis."""
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.sensitive_train = None
        self.sensitive_test = None
        self.models = {}
        self.preprocessor = None
        self.comparator = None
        self.feature_names = None
    
    def load_and_preprocess_data(self, data_path: str = None):
        """
        Load and preprocess the patient readmission data.
        
        Args:
            data_path: Path to the dataset file. If None, generates synthetic data.
        """
        if data_path:
            self.data = pd.read_csv(data_path)
        else:
            self._generate_synthetic_data()
        
        self._preprocess_data()
    
    def _generate_synthetic_data(self, n_samples: int = 10000):
        """Generate synthetic patient readmission data with potential biases."""
        np.random.seed(RANDOM_STATE)
        
        # Generate demographic features
        age = np.random.normal(65, 15, n_samples).astype(int)
        age = np.clip(age, 18, 100)
        
        # Generate race/ethnicity with potential bias
        race = np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], 
                              size=n_samples, 
                              p=[0.6, 0.13, 0.18, 0.06, 0.03])
        
        # Generate gender
        gender = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.48, 0.52])
        
        # Generate insurance status with race-based differences
        insurance = np.zeros(n_samples, dtype=object)
        for r in ['White', 'Black', 'Hispanic', 'Asian', 'Other']:
            mask = (race == r)
            if r == 'White':
                insurance[mask] = np.random.choice(
                    ['Private', 'Medicare', 'Medicaid', 'None'], 
                    size=mask.sum(),
                    p=[0.6, 0.25, 0.1, 0.05]
                )
            elif r == 'Black':
                insurance[mask] = np.random.choice(
                    ['Private', 'Medicare', 'Medicaid', 'None'], 
                    size=mask.sum(),
                    p=[0.3, 0.2, 0.4, 0.1]
                )
            else:  # Hispanic, Asian, Other
                insurance[mask] = np.random.choice(
                    ['Private', 'Medicare', 'Medicaid', 'None'], 
                    size=mask.sum(),
                    p=[0.4, 0.15, 0.3, 0.15]
                )
        
        # Generate medical features
        num_medications = np.random.poisson(8, n_samples)
        num_procedures = np.random.poisson(2, n_samples)
        num_diagnoses = np.random.poisson(5, n_samples)
        time_in_hospital = np.random.lognormal(1.5, 0.4, n_samples).astype(int)
        time_in_hospital = np.clip(time_in_hospital, 1, 14)
        
        # Generate lab results with some correlation with readmission
        glucose_avg = np.random.normal(120, 30, n_samples)
        a1c = np.random.normal(6.5, 1.5, n_samples)
        
        # Generate readmission target with bias
        base_prob = 0.3
        prob_readmit = base_prob + \
                      -0.1 * (insurance == 'Private') + \
                      0.15 * (insurance == 'None') + \
                      0.1 * (race == 'Black') + \
                      0.07 * (race == 'Hispanic') + \
                      0.05 * (num_medications / 20) + \
                      0.1 * (time_in_hospital > 7) + \
                      0.05 * ((glucose_avg > 140) | (a1c > 6.5))
        
        # Add some random noise
        prob_readmit += np.random.normal(0, 0.05, n_samples)
        prob_readmit = np.clip(prob_readmit, 0, 1)
        
        # Generate readmission outcome
        readmitted = (np.random.random(n_samples) < prob_readmit).astype(int)
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'age': age,
            'gender': gender,
            'race': race,
            'insurance': insurance,
            'num_medications': num_medications,
            'num_procedures': num_procedures,
            'num_diagnoses': num_diagnoses,
            'time_in_hospital': time_in_hospital,
            'glucose_avg': glucose_avg,
            'a1c': a1c,
            'readmitted': readmitted
        })
    
    def _preprocess_data(self):
        """Preprocess the data and split into train/test sets."""
        X = self.data.drop('readmitted', axis=1)
        y = self.data['readmitted']
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
        )
        
        # Define preprocessing for numerical and categorical features
        numeric_features = ['age', 'num_medications', 'num_procedures', 
                          'num_diagnoses', 'time_in_hospital', 'glucose_avg', 'a1c']
        categorical_features = ['gender', 'race', 'insurance']
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Store feature names for later use
        self.feature_names = (
            numeric_features +
            list(self.preprocessor.named_transformers_['cat'].named_steps['onehot']
                 .get_feature_names_out(categorical_features))
        )
        
        # Fit and transform the training data
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Store processed data
        self.X_train = X_train_processed
        self.X_test = X_test_processed
        self.y_train = y_train.values
        self.y_test = y_test.values
        self.sensitive_train = X_train['race'].values
        self.sensitive_test = X_test['race'].values
    
    def train_models(self):
        """Train multiple models for comparison."""
        # Define models
        models = {
            'RandomForest': RandomForestClassifier(
                random_state=RANDOM_STATE, 
                n_estimators=100,
                class_weight='balanced'
            ),
            'GradientBoosting': GradientBoostingClassifier(
                random_state=RANDOM_STATE,
                n_estimators=100
            ),
            'LogisticRegression': LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000,
                class_weight='balanced',
                solver='liblinear'
            )
        }
        
        # Train each model
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
    
    def evaluate_fairness(self):
        """Evaluate and compare models for fairness."""
        if not self.models:
            raise ValueError("No models trained. Call train_models() first.")
        
        # Initialize model comparator
        self.comparator = ModelComparator(
            protected_attribute='race',
            favorable_label=1,
            unfavorable_label=0,
            random_state=RANDOM_STATE
        )
        
        # Evaluate each model
        results = []
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            # Get predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = self.comparator.evaluate_model(
                model=model,
                X=self.X_test,
                y_true=self.y_test,
                sensitive_features=self.sensitive_test,
                model_name=name
            )
            
            # Add to results
            results.append(metrics)
        
        # Convert to DataFrame for better display
        results_df = pd.DataFrame(results)
        print("\nModel Comparison Results:")
        print(results_df[['model', 'accuracy', 'roc_auc', 'demographic_parity_diff', 'equal_odds_diff']])
        
        return results_df
    
    def plot_fairness_metrics(self, results_df):
        """Plot fairness metrics across different models."""
        plt.figure(figsize=(14, 6))
        
        # Plot demographic parity difference
        plt.subplot(1, 2, 1)
        sns.barplot(x='model', y='demographic_parity_diff', data=results_df)
        plt.title('Demographic Parity Difference by Model')
        plt.xticks(rotation=45)
        plt.axhline(0, color='black', linestyle='--')
        plt.ylabel('Difference in Positive Rate')
        
        # Plot equal odds difference
        plt.subplot(1, 2, 2)
        sns.barplot(x='model', y='equal_odds_diff', data=results_df)
        plt.title('Equalized Odds Difference by Model')
        plt.xticks(rotation=45)
        plt.axhline(0, color='black', linestyle='--')
        plt.ylabel('Difference in TPR - Difference in FPR')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_race_impact(self):
        """Analyze the impact of race on model predictions."""
        if not self.models:
            raise ValueError("No models trained. Call train_models() first.")
        
        # Use the best model (based on ROC AUC)
        model_name = max(
            [(name, roc_auc_score(self.y_test, model.predict_proba(self.X_test)[:, 1])) 
             for name, model in self.models.items()],
            key=lambda x: x[1]
        )[0]
        model = self.models[model_name]
        
        # Get test data with original features
        X_test_df = self.data.drop('readmitted', axis=1).iloc[-len(self.y_test):]
        X_test_df['predicted_prob'] = model.predict_proba(self.X_test)[:, 1]
        X_test_df['readmitted'] = self.y_test
        
        # Plot readmission rates by race
        plt.figure(figsize=(14, 6))
        
        # Actual readmission rates
        plt.subplot(1, 2, 1)
        sns.barplot(x='race', y='readmitted', data=X_test_df, ci=None)
        plt.title('Actual Readmission Rate by Race')
        plt.xticks(rotation=45)
        plt.ylabel('Readmission Rate')
        
        # Predicted readmission rates
        plt.subplot(1, 2, 2)
        sns.barplot(x='race', y='predicted_prob', data=X_test_df, ci=None)
        plt.title('Predicted Readmission Probability by Race')
        plt.xticks(rotation=45)
        plt.ylabel('Predicted Probability')
        
        plt.tight_layout()
        plt.show()
        
        return X_test_df
    
    def feature_importance_analysis(self):
        """Analyze and visualize feature importance."""
        if not self.models:
            raise ValueError("No models trained. Call train_models() first.")
        
        # Get the best model
        model_name = max(
            [(name, roc_auc_score(self.y_test, model.predict_proba(self.X_test)[:, 1])) 
             for name, model in self.models.items()],
            key=lambda x: x[1]
        )[0]
        model = self.models[model_name]
        
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            print("Model does not support feature importance analysis.")
            return
        
        # Create a DataFrame for visualization
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
        plt.title(f'Top 15 Feature Importances - {model_name}')
        plt.tight_layout()
        plt.show()
        
        return feature_importance


def main():
    """Run the patient readmission analysis pipeline."""
    print("Starting patient readmission analysis...")
    analysis = PatientReadmissionAnalysis()
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    analysis.load_and_preprocess_data()
    
    # Train models
    print("\nTraining models...")
    analysis.train_models()
    
    # Evaluate fairness
    print("\nEvaluating model fairness...")
    results_df = analysis.evaluate_fairness()
    
    # Plot fairness metrics
    print("\nVisualizing fairness metrics...")
    analysis.plot_fairness_metrics(results_df)
    
    # Analyze race impact
    print("\nAnalyzing race impact on predictions...")
    analysis.analyze_race_impact()
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    analysis.feature_importance_analysis()
    
    print("\nAnalysis complete!")
    return results_df


if __name__ == "__main__":
    results = main()
