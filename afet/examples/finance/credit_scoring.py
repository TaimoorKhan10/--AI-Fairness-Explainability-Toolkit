"""
Credit Scoring with Fairness Analysis

This example demonstrates how to use the AI Fairness and Explainability Toolkit (AFET)
to analyze and compare machine learning models for credit scoring with a focus on fairness.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score

# Import AFET components
from afet.core.model_comparison import ModelComparator
from afet.core.fairness_metrics import FairnessMetrics
from afet.core.advanced_fairness_metrics import AdvancedFairnessMetrics

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class CreditScoringAnalysis:
    """A class to demonstrate fairness analysis in credit scoring."""
    
    def __init__(self):
        """Initialize the credit scoring analysis."""
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
    
    def load_and_preprocess_data(self, data_path: str = None):
        """
        Load and preprocess the credit scoring data.
        
        Args:
            data_path: Path to the dataset file. If None, generates synthetic data.
        """
        if data_path:
            self.data = pd.read_csv(data_path)
        else:
            self._generate_synthetic_data()
        
        self._preprocess_data()
    
    def _generate_synthetic_data(self, n_samples: int = 10000):
        """Generate synthetic credit scoring data with potential biases."""
        np.random.seed(RANDOM_STATE)
        
        # Generate demographic features
        age = np.random.normal(45, 15, n_samples).astype(int)
        age = np.clip(age, 18, 90)
        
        # Generate race/ethnicity with potential bias
        race = np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], 
                              size=n_samples, 
                              p=[0.6, 0.13, 0.18, 0.06, 0.03])
        
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
            0.1 * (race == 'Hispanic')
        )
        prob_approval = np.clip(prob_approval, 0, 1)
        approved = (np.random.random(n_samples) < prob_approval).astype(int)
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'age': age,
            'race': race,
            'income': income,
            'credit_score': credit_score,
            'debt_to_income': dti,
            'loan_amount': loan_amount,
            'employment_length': employment_length,
            'approved': approved
        })
    
    def _preprocess_data(self):
        """Preprocess the data and split into train/test sets."""
        X = self.data.drop('approved', axis=1)
        y = self.data['approved']
        
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
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
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
            'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100),
            'GradientBoosting': GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=100),
            'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
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
    
    def plot_disparities(self, results_df):
        """Plot fairness metric disparities across models."""
        plt.figure(figsize=(12, 6))
        
        # Plot demographic parity difference
        plt.subplot(1, 2, 1)
        sns.barplot(x='model', y='demographic_parity_diff', data=results_df)
        plt.title('Demographic Parity Difference')
        plt.xticks(rotation=45)
        plt.axhline(0, color='black', linestyle='--')
        
        # Plot equal odds difference
        plt.subplot(1, 2, 2)
        sns.barplot(x='model', y='equal_odds_diff', data=results_df)
        plt.title('Equalized Odds Difference')
        plt.xticks(rotation=45)
        plt.axhline(0, color='black', linestyle='--')
        
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
        X_test_df = self.data.drop('approved', axis=1).iloc[-len(self.y_test):]
        X_test_df['predicted_prob'] = model.predict_proba(self.X_test)[:, 1]
        X_test_df['approved'] = self.y_test
        
        # Plot approval rates by race
        plt.figure(figsize=(12, 6))
        
        # Actual approval rates
        plt.subplot(1, 2, 1)
        sns.barplot(x='race', y='approved', data=X_test_df, ci=None)
        plt.title('Actual Approval Rate by Race')
        plt.xticks(rotation=45)
        
        # Predicted approval rates
        plt.subplot(1, 2, 2)
        sns.barplot(x='race', y='predicted_prob', data=X_test_df, ci=None)
        plt.title('Predicted Approval Probability by Race')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return X_test_df


def main():
    """Run the credit scoring analysis pipeline."""
    print("Starting credit scoring analysis...")
    analysis = CreditScoringAnalysis()
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    analysis.load_and_preprocess_data()
    
    # Train models
    print("\nTraining models...")
    analysis.train_models()
    
    # Evaluate fairness
    print("\nEvaluating model fairness...")
    results_df = analysis.evaluate_fairness()
    
    # Plot disparities
    print("\nVisualizing fairness metrics...")
    analysis.plot_disparities(results_df)
    
    # Analyze race impact
    print("\nAnalyzing race impact on predictions...")
    analysis.analyze_race_impact()
    
    print("\nAnalysis complete!")
    return results_df


if __name__ == "__main__":
    results = main()
