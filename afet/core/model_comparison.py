"""
Model Comparison and Evaluation Module for AFET

This module provides comprehensive tools for comparing machine learning models
across multiple dimensions including performance, fairness, and robustness.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    precision_recall_curve, roc_curve, auc, confusion_matrix, classification_report,
    average_precision_score, brier_score_loss, log_loss
)
from fairlearn.metrics import (
    demographic_parity_difference, equalized_odds_difference,
    equal_opportunity_difference, selection_rate_difference,
    false_positive_rate_difference, false_negative_rate_difference
)
from scipy import stats
import scikit_posthocs as sp
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Set plot style
plt.style.use('seaborn')
sns.set_palette('colorblind')


class ModelComparator:
    """
    Comprehensive model comparison and evaluation utility.
    
    This class provides methods to evaluate and compare machine learning models
    across multiple dimensions including performance metrics, fairness metrics,
    and statistical significance testing.
    
    Args:
        protected_attribute: Name of the protected attribute for fairness analysis
        favorable_label: Label value considered favorable (default: 1)
        unfavorable_label: Label value considered unfavorable (default: 0)
        random_state: Random seed for reproducibility (default: 42)
    """
    
    def __init__(self, 
                 protected_attribute: str,
                 favorable_label: int = 1,
                 unfavorable_label: int = 0,
                 random_state: int = 42):
        self.protected_attribute = protected_attribute
        self.favorable_label = favorable_label
        self.unfavorable_label = unfavorable_label
        self.random_state = random_state
        np.random.seed(random_state)
        self.results = {}
        self.metrics_history = []
        
    def evaluate_model(self, 
                      model: BaseEstimator,
                      X: Union[pd.DataFrame, np.ndarray],
                      y_true: Union[pd.Series, np.ndarray],
                      sensitive_features: Union[pd.Series, np.ndarray],
                      model_name: str = None,
                      sample_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate a machine learning model across multiple performance and fairness metrics.
        
        Args:
            model: Trained model implementing predict() and predict_proba() methods
            X: Feature matrix for evaluation
            y_true: True target values
            sensitive_features: Protected attribute values for fairness analysis
            model_name: Optional name for the model (default: 'model_<index>')
            sample_weights: Optional sample weights for metrics calculation
            
        Returns:
            Dictionary containing comprehensive evaluation metrics
            
        Example:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
            >>> sensitive = np.random.choice(['A', 'B'], size=1000)
            >>> model = RandomForestClassifier().fit(X, y)
            >>> comparator = ModelComparator(protected_attribute='group')
            >>> metrics = comparator.evaluate_model(model, X, y, sensitive, 'RandomForest')
        """
        # Convert inputs to numpy arrays if they're pandas objects
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y_true, (pd.Series, pd.DataFrame)):
            y_true = y_true.values
        if isinstance(sensitive_features, (pd.Series, pd.DataFrame)):
            sensitive_features = sensitive_features.values
            
        # Generate default model name if not provided
        if model_name is None:
            model_name = f'model_{len(self.results) + 1}'
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Handle binary and multi-class classification
        try:
            y_pred_proba = model.predict_proba(X)
            if y_pred_proba.shape[1] == 2:  # Binary classification
                y_pred_proba = y_pred_proba[:, 1]
        except (AttributeError, IndexError):
            y_pred_proba = None
        
        # Calculate performance metrics
        metrics = {
            'model': model_name,
            'accuracy': accuracy_score(y_true, y_pred, sample_weight=sample_weights),
            'precision': precision_score(y_true, y_pred, zero_division=0, sample_weight=sample_weights),
            'recall': recall_score(y_true, y_pred, zero_division=0, sample_weight=sample_weights),
            'f1_score': f1_score(y_true, y_pred, zero_division=0, sample_weight=sample_weights),
            'log_loss': log_loss(y_true, y_pred_proba if y_pred_proba is not None else y_pred, 
                               sample_weight=sample_weights) if y_pred_proba is not None else None,
        }
        
        # Add ROC-AUC if probability predictions are available
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, sample_weight=sample_weights)
                metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba, 
                                                                sample_weight=sample_weights)
                metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba, 
                                                        sample_weight=sample_weights)
            except (ValueError, TypeError):
                pass
        
        # Calculate fairness metrics
        fairness_metrics = self._calculate_fairness_metrics(
            y_true, y_pred, sensitive_features, sample_weights
        )
        metrics.update(fairness_metrics)
        
        # Store results
        self.results[model_name] = metrics
        self.metrics_history.append(metrics.copy())
        
        return metrics
    
    def compare_models(self, 
                      models: Dict[str, BaseEstimator],
                      X: Union[pd.DataFrame, np.ndarray],
                      y_true: Union[pd.Series, np.ndarray],
                      sensitive_features: Union[pd.Series, np.ndarray],
                      sample_weights: Optional[np.ndarray] = None,
                      n_splits: int = 5) -> pd.DataFrame:
        """
        Compare multiple models using cross-validation and return comprehensive metrics.
        
        Args:
            models: Dictionary of model names to model instances
            X: Feature matrix for evaluation
            y_true: True target values
            sensitive_features: Protected attribute values for fairness analysis
            sample_weights: Optional sample weights
            n_splits: Number of cross-validation folds (default: 5)
            
        Returns:
            DataFrame containing metrics for all models across all folds
            
        Example:
            >>> from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            >>> from sklearn.model_selection import train_test_split
            >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            >>> models = {
            ...     'RandomForest': RandomForestClassifier(),
            ...     'GradientBoosting': GradientBoostingClassifier()
            ... }
            >>> comparator = ModelComparator(protected_attribute='group')
            >>> results = comparator.compare_models(models, X_test, y_test, sensitive_test)
        """
        from sklearn.model_selection import StratifiedKFold
        
        # Initialize results storage
        all_results = []
        
        # Convert to numpy arrays if pandas objects
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y_true, (pd.Series, pd.DataFrame)):
            y_true = y_true.values
        if isinstance(sensitive_features, (pd.Series, pd.DataFrame)):
            sensitive_features = sensitive_features.values
        
        # Set up cross-validation
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                           random_state=self.random_state)
        
        # Evaluate each model with cross-validation
        for model_name, model in tqdm(models.items(), desc="Evaluating models"):
            fold_metrics = []
            
            for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_true)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y_true[train_idx], y_true[test_idx]
                
                # Get sample weights for this fold if provided
                fold_weights = None
                if sample_weights is not None:
                    fold_weights = sample_weights[train_idx]
                
                # Train model
                model.fit(X_train, y_train, sample_weight=fold_weights)
                
                # Evaluate on test fold
                metrics = self.evaluate_model(
                    model=model,
                    X=X_test,
                    y_true=y_test,
                    sensitive_features=sensitive_features[test_idx],
                    model_name=model_name,
                    sample_weight=sample_weights[test_idx] if sample_weights is not None else None
                )
                metrics['fold'] = fold + 1
                fold_metrics.append(metrics)
            
            # Aggregate results across folds
            fold_df = pd.DataFrame(fold_metrics)
            mean_metrics = fold_df.mean(numeric_only=True).to_dict()
            mean_metrics['model'] = model_name
            mean_metrics['std_accuracy'] = fold_df['accuracy'].std()
            all_results.append(mean_metrics)
        
        # Create and return results DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Add statistical significance markers
        if len(models) > 1:
            self._add_statistical_significance(results_df)
        
        return results_df
    
    def _calculate_fairness_metrics(self,
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 sensitive_features: np.ndarray,
                                 sample_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive fairness metrics for model predictions.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            sensitive_features: Protected attribute values
            sample_weights: Optional sample weights
            
        Returns:
            Dictionary of fairness metrics
        """
        metrics = {}
        
        # Calculate basic fairness metrics
        try:
            metrics.update({
                'demographic_parity_diff': demographic_parity_difference(
                    y_true, y_pred, sensitive_features=sensitive_features,
                    sample_weight=sample_weights
                ),
                'equal_odds_diff': equalized_odds_difference(
                    y_true, y_pred, sensitive_features=sensitive_features,
                    sample_weight=sample_weights
                ),
                'equal_opportunity_diff': equal_opportunity_difference(
                    y_true, y_pred, sensitive_features=sensitive_features,
                    sample_weight=sample_weights
                ),
                'selection_rate_diff': selection_rate_difference(
                    y_true, y_pred, sensitive_features=sensitive_features,
                    sample_weight=sample_weights
                )
            })
        except Exception as e:
            print(f"Warning: Could not calculate all fairness metrics: {str(e)}")
        
        return metrics
    
    def _add_statistical_significance(self, results_df: pd.DataFrame) -> None:
        """
        Add statistical significance markers to results DataFrame.
        
        Args:
            results_df: DataFrame containing model comparison results
        """
        # Get numeric columns for significance testing
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns
        
        # Perform pairwise statistical tests
        for col in numeric_cols:
            if col == 'fold':
                continue
                
            # Get values for all models
            values = [results_df[results_df['model'] == model][col].values 
                     for model in results_df['model'].unique()]
            
            # Perform Kruskal-Wallis test for multiple comparisons
            if len(values) > 2:
                _, p_value = stats.kruskal(*values)
                if p_value < 0.05:
                    # Perform post-hoc Dunn's test
                    dunn_results = sp.posthoc_dunn(values, p_adjust='holm')
                    # Store results in DataFrame
                    # (implementation depends on how you want to display results)
                    pass
            
            # For two models, use paired t-test or Wilcoxon signed-rank test
            elif len(values) == 2:
                if len(values[0]) > 1:  # Multiple samples per model
                    _, p_value = stats.ttest_rel(values[0], values[1])
                    # Add significance marker
                    results_df[f'{col}_sig'] = p_value < 0.05
    
    def plot_metric_comparison(self, 
                              metric: str = 'roc_auc', 
                              title: str = None,
                              figsize: Tuple[int, int] = (10, 6),
                              **kwargs) -> plt.Figure:
        """
        Create a bar plot comparing models on a specific metric.
        
        Args:
            metric: Name of the metric to plot
            title: Plot title (default: 'Comparison of {metric}')
            figsize: Figure size (width, height)
            **kwargs: Additional arguments passed to seaborn.barplot
            
        Returns:
            Matplotlib Figure object
        """
        if not hasattr(self, 'results') or not self.results:
            raise ValueError("No model results available. Run evaluate_model() first.")
            
        # Create DataFrame from results
        df = pd.DataFrame(self.results).T.reset_index()
        
        # Set up plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create bar plot
        sns.barplot(
            x='index', 
            y=metric, 
            data=df, 
            ax=ax,
            **kwargs
        )
        
        # Customize plot
        ax.set_title(title or f'Model Comparison: {metric.upper()}')
        ax.set_xlabel('Model')
        ax.set_ylabel(metric.upper())
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def plot_roc_curve(self, 
                      models: Dict[str, BaseEstimator],
                      X: np.ndarray,
                      y_true: np.ndarray,
                      figsize: Tuple[int, int] = (8, 8)) -> plt.Figure:
        """
        Plot ROC curves for multiple models.
        
        Args:
            models: Dictionary of model names to model instances
            X: Feature matrix
            y_true: True labels
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure object with ROC curves
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot diagonal line for random classifier
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        
        # Plot ROC curve for each model
        for name, model in models.items():
            try:
                y_pred_proba = model.predict_proba(X)[:, 1]
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
            except Exception as e:
                print(f"Could not generate ROC curve for {name}: {str(e)}")
        
        # Customize plot
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curves')
        ax.legend(loc='lower right')
        plt.tight_layout()
        
        return fig
            significance[metric] = p_value < alpha
            
        return significance
    
    def create_comparison_report(self, 
                               models: Dict[str, BaseEstimator],
                               X: pd.DataFrame,
                               y_true: pd.Series,
                               sensitive_features: pd.Series) -> Dict:
        """
        Create comprehensive model comparison report
        """
        # Get comparison results
        results_df = self.compare_models(models, X, y_true, sensitive_features)
        
        # Calculate summary statistics
        summary = {
            'best_model_performance': results_df.loc[results_df['accuracy'].idxmax()]['model'],
            'best_model_fairness': results_df.loc[results_df['demographic_parity'].idxmin()]['model'],
            'performance_range': {
                'accuracy': {
                    'min': results_df['accuracy'].min(),
                    'max': results_df['accuracy'].max(),
                    'mean': results_df['accuracy'].mean()
                },
                'roc_auc': {
                    'min': results_df['roc_auc'].min(),
                    'max': results_df['roc_auc'].max(),
                    'mean': results_df['roc_auc'].mean()
                }
            },
            'fairness_range': {
                'demographic_parity': {
                    'min': results_df['demographic_parity'].min(),
                    'max': results_df['demographic_parity'].max(),
                    'mean': results_df['demographic_parity'].mean()
                },
                'equalized_odds': {
                    'min': results_df['equalized_odds'].min(),
                    'max': results_df['equalized_odds'].max(),
                    'mean': results_df['equalized_odds'].mean()
                }
            }
        }
        
        # Compare all pairs of models
        pairwise_comparison = {}
        for i, model1_name in enumerate(models):
            for model2_name in list(models.keys())[i + 1:]:
                model1_metrics = results_df[results_df['model'] == model1_name].iloc[0].to_dict()
                model2_metrics = results_df[results_df['model'] == model2_name].iloc[0].to_dict()
                
                sig_results = self.statistical_significance(model1_metrics, model2_metrics)
                pairwise_comparison[f'{model1_name}_vs_{model2_name}'] = sig_results
                
        return {
            'results': results_df,
            'summary': summary,
            'pairwise_comparison': pairwise_comparison
        }
