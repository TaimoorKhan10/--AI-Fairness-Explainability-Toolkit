"""
Intersectional Fairness Metrics

This module provides metrics for evaluating fairness across intersectional demographic groups,
which is important for understanding how multiple protected attributes interact.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from sklearn.metrics import confusion_matrix

class IntersectionalFairnessMetrics:
    """
    A class for computing fairness metrics across intersectional demographic groups.
    
    Intersectional fairness considers how multiple protected attributes (e.g., race and gender)
    interact to affect model outcomes, rather than considering each protected attribute in isolation.
    """
    
    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_features: Dict[str, np.ndarray]
    ):
        """
        Initialize the IntersectionalFairnessMetrics with prediction data and protected attributes.
        
        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels
        y_pred : np.ndarray
            Predicted labels
        protected_features : Dict[str, np.ndarray]
            Dictionary mapping protected attribute names to their values for each instance
        """
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.protected_features = protected_features
        
        # Create intersectional groups
        self._create_intersectional_groups()
    
    def _create_intersectional_groups(self):
        """
        Create intersectional groups based on combinations of protected attributes.
        """
        # Create a DataFrame with all protected features
        df = pd.DataFrame(self.protected_features)
        
        # Create a new column for intersectional groups
        df['intersectional_group'] = df.apply(
            lambda row: '_'.join(str(row[col]) for col in df.columns),
            axis=1
        )
        
        self.intersectional_groups = df['intersectional_group'].values
        self.unique_groups = np.unique(self.intersectional_groups)
    
    def _get_group_indices(self, group: str) -> np.ndarray:
        """
        Get indices for a specific intersectional group.
        
        Parameters
        ----------
        group : str
            Intersectional group name
            
        Returns
        -------
        np.ndarray
            Boolean mask for the specified group
        """
        return self.intersectional_groups == group
    
    def group_confusion_matrices(self) -> Dict[str, np.ndarray]:
        """
        Compute confusion matrices for each intersectional group.
        
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping group names to their confusion matrices
        """
        result = {}
        
        for group in self.unique_groups:
            group_mask = self._get_group_indices(group)
            
            if np.sum(group_mask) > 0:
                cm = confusion_matrix(
                    self.y_true[group_mask],
                    self.y_pred[group_mask],
                    labels=[0, 1]
                )
                result[group] = cm
        
        return result
    
    def group_metrics(self) -> pd.DataFrame:
        """
        Compute various fairness metrics for each intersectional group.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing metrics for each group
        """
        results = []
        
        for group in self.unique_groups:
            group_mask = self._get_group_indices(group)
            
            if np.sum(group_mask) > 0:
                y_true_group = self.y_true[group_mask]
                y_pred_group = self.y_pred[group_mask]
                
                # Calculate metrics
                tn, fp, fn, tp = confusion_matrix(
                    y_true_group, y_pred_group, labels=[0, 1]
                ).ravel()
                
                # Avoid division by zero
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                
                # Selection rate (% of positive predictions)
                selection_rate = (tp + fp) / len(y_true_group)
                
                # Accuracy
                accuracy = (tp + tn) / len(y_true_group)
                
                results.append({
                    'Group': group,
                    'Size': np.sum(group_mask),
                    'Accuracy': accuracy,
                    'Selection_Rate': selection_rate,
                    'TPR': tpr,  # True Positive Rate (Recall)
                    'TNR': tnr,  # True Negative Rate
                    'FPR': fpr,  # False Positive Rate
                    'FNR': fnr,  # False Negative Rate
                    'PPV': ppv,  # Positive Predictive Value (Precision)
                    'NPV': npv   # Negative Predictive Value
                })
        
        return pd.DataFrame(results)
    
    def intersectional_disparity(self, metric: str = 'Selection_Rate') -> Dict[str, float]:
        """
        Calculate the disparity in a specified metric across intersectional groups.
        
        Parameters
        ----------
        metric : str, optional
            The metric to calculate disparity for, by default 'Selection_Rate'
            
        Returns
        -------
        Dict[str, float]
            Dictionary with disparity metrics
        """
        group_metrics = self.group_metrics()
        
        if len(group_metrics) <= 1:
            return {'max_disparity': 0.0, 'variance': 0.0}
        
        # Calculate disparities
        metric_values = group_metrics[metric].values
        max_val = np.max(metric_values)
        min_val = np.min(metric_values)
        
        return {
            'max_disparity': max_val - min_val,
            'variance': np.var(metric_values),
            'min_value': min_val,
            'max_value': max_val,
            'mean': np.mean(metric_values)
        }
    
    def equal_opportunity_difference(self) -> float:
        """
        Calculate the maximum difference in true positive rates across groups.
        
        Returns
        -------
        float
            Maximum difference in true positive rates
        """
        return self.intersectional_disparity('TPR')['max_disparity']
    
    def demographic_parity_difference(self) -> float:
        """
        Calculate the maximum difference in selection rates across groups.
        
        Returns
        -------
        float
            Maximum difference in selection rates
        """
        return self.intersectional_disparity('Selection_Rate')['max_disparity']
    
    def equalized_odds_difference(self) -> float:
        """
        Calculate the maximum difference in the sum of FPR and FNR across groups.
        
        Returns
        -------
        float
            Maximum difference in equalized odds
        """
        group_metrics = self.group_metrics()
        
        if len(group_metrics) <= 1:
            return 0.0
        
        # Calculate equalized odds for each group (average of TPR and TNR)
        group_metrics['EO'] = (group_metrics['TPR'] + group_metrics['TNR']) / 2
        
        eo_values = group_metrics['EO'].values
        return np.max(eo_values) - np.min(eo_values)
    
    def group_fairness_report(self) -> pd.DataFrame:
        """
        Generate a comprehensive fairness report for all intersectional groups.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing detailed fairness metrics for each group
        """
        metrics = self.group_metrics()
        
        # Add disparity from average
        for col in ['Selection_Rate', 'TPR', 'TNR', 'FPR', 'FNR', 'PPV', 'NPV']:
            avg = metrics[col].mean()
            metrics[f'{col}_disparity'] = metrics[col] - avg
        
        return metrics
