"""
Core fairness metrics implementation for AFET
"""

from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import BinaryLabelDataset
import logging

logger = logging.getLogger(__name__)


class FairnessMetrics:
    """
    Main class for calculating fairness metrics across different groups
    """
    
    def __init__(self, 
                 protected_attribute: str, 
                 favorable_label: int = 1,
                 unfavorable_label: int = 0,
                 error_tolerance: float = 1e-6):
        """
        Initialize fairness metrics calculator
        
        Args:
            protected_attribute: Name of the protected attribute column
            favorable_label: Label value considered favorable
            unfavorable_label: Label value considered unfavorable
            error_tolerance: Small value to prevent division by zero
        """
        self.protected_attribute = protected_attribute
        self.favorable_label = favorable_label
        self.unfavorable_label = unfavorable_label
        self.error_tolerance = error_tolerance
        
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
        Calculate equal opportunity metrics (true positive rate parity)
        
        Args:
            y_pred: Predicted labels
            y_true: True labels
            sensitive_features: Protected attribute values
            
        Returns:
            Dictionary of equal opportunity metrics by group
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
            
            # Prevent division by zero
            if total_positives <= self.error_tolerance:
                logger.warning(f"No positive samples for group {group}, setting equal_opportunity to 0")
                metrics[f'equal_opportunity_{group}'] = 0
            else:
                metrics[f'equal_opportunity_{group}'] = true_positives / total_positives
            
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
    
    def calculate_equalized_odds(self,
                               y_pred: np.ndarray,
                               y_true: np.ndarray,
                               sensitive_features: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """
        Calculate equalized odds metric (true positive and false positive rate parity)
        
        Args:
            y_pred: Predicted labels
            y_true: True labels
            sensitive_features: Protected attribute values
            
        Returns:
            Dictionary of equalized odds metrics by group (TPR, FPR)
        """
        metrics = {}
        unique_groups = np.unique(sensitive_features)
        
        for group in unique_groups:
            group_mask = sensitive_features == group
            group_pred = y_pred[group_mask]
            group_true = y_true[group_mask]
            
            # Calculate confusion matrix values
            tn, fp, fn, tp = confusion_matrix(group_true, group_pred, labels=[self.unfavorable_label, self.favorable_label]).ravel()
            
            # True positive rate (sensitivity)
            positives = tp + fn
            tpr = tp / positives if positives > self.error_tolerance else 0
            
            # False positive rate (1 - specificity)
            negatives = tn + fp
            fpr = fp / negatives if negatives > self.error_tolerance else 0
            
            metrics[f'equalized_odds_{group}'] = (tpr, fpr)
            
        return metrics
        
    def calculate_treatment_equality(self,
                                    y_pred: np.ndarray,
                                    y_true: np.ndarray,
                                    sensitive_features: np.ndarray) -> Dict[str, float]:
        """
        Calculate treatment equality (ratio of false negatives to false positives)
        
        Args:
            y_pred: Predicted labels
            y_true: True labels
            sensitive_features: Protected attribute values
            
        Returns:
            Dictionary of treatment equality metrics by group
        """
        metrics = {}
        unique_groups = np.unique(sensitive_features)
        
        for group in unique_groups:
            group_mask = sensitive_features == group
            group_pred = y_pred[group_mask]
            group_true = y_true[group_mask]
            
            # Calculate confusion matrix values
            tn, fp, fn, tp = confusion_matrix(group_true, group_pred, labels=[self.unfavorable_label, self.favorable_label]).ravel()
            
            # Treatment equality: ratio of FN to FP
            if fp <= self.error_tolerance:
                logger.warning(f"No false positives for group {group}, treatment equality undefined")
                metrics[f'treatment_equality_{group}'] = np.nan
            else:
                metrics[f'treatment_equality_{group}'] = fn / fp
            
        return metrics
    
    def get_comprehensive_metrics(self, 
                                y_pred: np.ndarray, 
                                y_true: np.ndarray, 
                                sensitive_features: np.ndarray) -> Dict[str, Union[float, Tuple[float, float]]]:
        """
        Get all fairness metrics in one go
        
        Args:
            y_pred: Predicted labels
            y_true: True labels
            sensitive_features: Protected attribute values
            
        Returns:
            Dictionary of all calculated metrics
        """
        metrics = {}
        
        try:
            # Basic metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
            
            # Fairness metrics
            metrics.update(self.calculate_demographic_parity(y_pred, y_true, sensitive_features))
            metrics.update(self.calculate_equal_opportunity(y_pred, y_true, sensitive_features))
            metrics.update(self.calculate_equalized_odds(y_pred, y_true, sensitive_features))
            metrics.update(self.calculate_treatment_equality(y_pred, y_true, sensitive_features))
            metrics['disparate_impact'] = self.calculate_disparate_impact(y_pred, y_true, sensitive_features)
            metrics['statistical_parity_difference'] = self.calculate_statistical_parity_difference(y_pred, y_true, sensitive_features)
        except Exception as e:
            logger.error(f"Error calculating fairness metrics: {str(e)}")
            raise
        
        return metrics
