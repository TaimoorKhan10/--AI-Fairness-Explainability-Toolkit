"""
Core fairness metrics implementation for AFET
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import BinaryLabelDataset


class FairnessMetrics:
    """
    Main class for calculating fairness metrics across different groups
    """
    
    def __init__(self, 
                 protected_attribute: str, 
                 favorable_label: int = 1,
                 unfavorable_label: int = 0):
        """
        Initialize fairness metrics calculator
        
        Args:
            protected_attribute: Name of the protected attribute column
            favorable_label: Label value considered favorable
            unfavorable_label: Label value considered unfavorable
        """
        self.protected_attribute = protected_attribute
        self.favorable_label = favorable_label
        self.unfavorable_label = unfavorable_label
        
    def calculate_demographic_parity(self, 
                                   y_pred: np.ndarray, 
                                   y_true: np.ndarray, 
                                   sensitive_features: np.ndarray) -> Dict[str, float]:
        """
        Calculate demographic parity metrics
        """
        metrics = {}
        unique_groups = np.unique(sensitive_features)
        
        for group in unique_groups:
            group_mask = sensitive_features == group
            group_pred = y_pred[group_mask]
            metrics[f'demographic_parity_{group}'] = np.mean(group_pred == self.favorable_label)
            
        return metrics
    
    def calculate_equal_opportunity(self, 
                                  y_pred: np.ndarray, 
                                  y_true: np.ndarray, 
                                  sensitive_features: np.ndarray) -> Dict[str, float]:
        """
        Calculate equal opportunity metrics
        """
        metrics = {}
        unique_groups = np.unique(sensitive_features)
        
        for group in unique_groups:
            group_mask = sensitive_features == group
            group_pred = y_pred[group_mask]
            group_true = y_true[group_mask]
            
            true_positives = np.sum((group_pred == self.favorable_label) & 
                                  (group_true == self.favorable_label))
            total_positives = np.sum(group_true == self.favorable_label)
            
            metrics[f'equal_opportunity_{group}'] = true_positives / total_positives if total_positives > 0 else 0
            
        return metrics
    
    def calculate_disparate_impact(self, 
                                 y_pred: np.ndarray, 
                                 y_true: np.ndarray, 
                                 sensitive_features: np.ndarray) -> float:
        """
        Calculate disparate impact ratio
        """
        unique_groups = np.unique(sensitive_features)
        if len(unique_groups) < 2:
            raise ValueError("At least two groups are required for disparate impact calculation")
            
        group_metrics = self.calculate_demographic_parity(y_pred, y_true, sensitive_features)
        
        # Get the maximum and minimum demographic parity values
        max_value = max(group_metrics.values())
        min_value = min(group_metrics.values())
        
        return min_value / max_value
    
    def calculate_statistical_parity_difference(self, 
                                               y_pred: np.ndarray, 
                                               y_true: np.ndarray, 
                                               sensitive_features: np.ndarray) -> float:
        """
        Calculate statistical parity difference
        """
        unique_groups = np.unique(sensitive_features)
        if len(unique_groups) < 2:
            raise ValueError("At least two groups are required for statistical parity calculation")
            
        group_metrics = self.calculate_demographic_parity(y_pred, y_true, sensitive_features)
        
        # Get the maximum and minimum demographic parity values
        max_value = max(group_metrics.values())
        min_value = min(group_metrics.values())
        
        return max_value - min_value
    
    def get_comprehensive_metrics(self, 
                                y_pred: np.ndarray, 
                                y_true: np.ndarray, 
                                sensitive_features: np.ndarray) -> Dict[str, float]:
        """
        Get all fairness metrics in one go
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
        
        # Fairness metrics
        metrics.update(self.calculate_demographic_parity(y_pred, y_true, sensitive_features))
        metrics.update(self.calculate_equal_opportunity(y_pred, y_true, sensitive_features))
        metrics['disparate_impact'] = self.calculate_disparate_impact(y_pred, y_true, sensitive_features)
        metrics['statistical_parity_difference'] = self.calculate_statistical_parity_difference(y_pred, y_true, sensitive_features)
        
        return metrics
