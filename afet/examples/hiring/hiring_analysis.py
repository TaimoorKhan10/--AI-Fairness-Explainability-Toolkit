"""
Hiring Analysis Example

This example demonstrates how to use the AI Fairness and Explainability Toolkit (AFET)
to analyze and compare machine learning models for a hiring prediction task.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Import AFET components
from afet.core.model_comparison import ModelComparator
from afet.core.fairness_metrics import FairnessMetrics
from afet.core.advanced_fairness_metrics import AdvancedFairnessMetrics

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class HiringAnalysis:
    """
    A class to demonstrate fairness analysis in hiring predictions.
    """
    
    def __init__(self):
        """Initialize the hiring analysis with default settings."""
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
        Load and preprocess the hiring dataset.
        
        Args:
            data_path: Path to the dataset file. If None, generates synthetic data.
        """
        if data_path:
            # Load data from file if path is provided
            self.data = pd.read_csv(data_path)
        else:
            # Generate synthetic hiring data
            self._generate_synthetic_data()
        
        # Preprocess the data
        self._preprocess_data()
    
    def _generate_synthetic_data(self, n_samples: int = 5000):
        """
        Generate synthetic hiring data for demonstration purposes.
        
        Args:
            n_samples: Number of samples to generate
        """
        np.random.seed(RANDOM_STATE)
        
        # Generate demographic features
        gender = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.6, 0.4])
        age = np.random.normal(35, 10, n_samples).astype(int)
        age = np.clip(age, 18, 70)
        
        # Generate education level (1-5, where 5 is highest)
        education = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1])
        
        # Generate experience (years)
        experience = np.random.normal(10, 5, n_samples).astype(int)
        experience = np.clip(experience, 0, 40)
        
        # Generate skills assessment (0-100)
        skills = np.random.normal(70, 15, n_samples).astype(int)
        skills = np.clip(skills, 0, 100)
        
        # Generate interview scores (0-100)
        interview = np.random.normal(75, 10, n_samples).astype(int)
        interview = np.clip(interview, 0, 100)
        
        # Create synthetic bias: women have slightly lower interview scores
        interview[gender == 'Female'] = np.clip(interview[gender == 'Female'] - 5, 0, 100)
        
        # Generate target variable (1 = hired, 0 = not hired)
        # Base probability with some bias against women
        prob = 0.3 + 0.5 * (skills / 100) + 0.3 * (interview / 100) - 0.1 * (gender == 'Female')
        prob = np.clip(prob, 0, 1)
        hired = (np.random.random(n_samples) < prob).astype(int)
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'gender': gender,
            'age': age,
            'education': education,
            'experience': experience,
            'skills_assessment': skills,
            'interview_score': interview,
            'hired': hired
        })
    
    def _preprocess_data(self):
        """Preprocess the data and split into train/test sets."""
        # Define features and target
        X = self.data.drop('hired', axis=1)
        y = self.data['hired']
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
        )
        
        # Define preprocessing for numerical and categorical features
        numeric_features = ['age', 'experience', 'skills_assessment', 'interview_score']
        categorical_features = ['gender', 'education']
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
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
        self.sensitive_train = X_train['gender'].values
        self.sensitive_test = X_test['gender'].values
    
    def train_models(self):
        """Train multiple models for comparison."""
        # Define models
        models = {
            'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE),
            'GradientBoosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
            'LogisticRegression': Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', LogisticRegression(random_state=RANDOM_STATE))
            ])
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
            protected_attribute='gender',
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
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models."""
        if not self.models:
            raise ValueError("No models trained. Call train_models() first.")
        
        fig = self.comparator.plot_roc_curve(
            models=self.models,
            X=self.X_test,
            y_true=self.y_test
        )
        plt.title('ROC Curves - Hiring Prediction Models')
        plt.show()
        return fig
    
    def analyze_fairness_metrics(self):
        """Perform detailed fairness analysis using AdvancedFairnessMetrics."""
        if not self.models:
            raise ValueError("No models trained. Call train_models() first.")
        
        # Initialize advanced fairness metrics
        afm = AdvancedFairnessMetrics(
            protected_attribute='gender',
            favorable_label=1,
            unfavorable_label=0
        )
        
        # Analyze each model
        for name, model in self.models.items():
            print(f"\nAnalyzing fairness for {name}...")
            
            # Get predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate predictive parity
            print("\nPredictive Parity:")
            pp_metrics = afm.calculate_predictive_parity(
                y_pred=y_pred,
                y_true=self.y_test,
                sensitive_features=self.sensitive_test
            )
            print(pp_metrics)
            
            # Calculate calibration metrics if probabilities are available
            if y_pred_proba is not None:
                print("\nCalibration Metrics:")
                cal_metrics = afm.calculate_calibration(
                    y_pred_proba=y_pred_proba,
                    y_true=self.y_test,
                    sensitive_features=self.sensitive_test
                )
                print(cal_metrics)
            
            # Calculate distributional fairness
            if y_pred_proba is not None:
                print("\nDistributional Fairness (KS Test):")
                dist_metrics = afm.calculate_kolmogorov_smirnov(
                    y_pred_proba=y_pred_proba,
                    sensitive_features=self.sensitive_test
                )
                print(dist_metrics)


def main():
    """Run the hiring analysis pipeline."""
    # Initialize analysis
    print("Starting hiring analysis...")
    analysis = HiringAnalysis()
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    analysis.load_and_preprocess_data()
    
    # Train models
    print("\nTraining models...")
    analysis.train_models()
    
    # Evaluate fairness
    print("\nEvaluating model fairness...")
    results_df = analysis.evaluate_fairness()
    
    # Plot ROC curves
    print("\nGenerating ROC curves...")
    analysis.plot_roc_curves()
    
    # Perform detailed fairness analysis
    print("\nPerforming detailed fairness analysis...")
    analysis.analyze_fairness_metrics()
    
    print("\nAnalysis complete!")
    return results_df


if __name__ == "__main__":
    results = main()
