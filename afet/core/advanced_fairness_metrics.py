"""
Advanced fairness metrics implementation
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from scipy.stats import ks_2samp


class AdvancedFairnessMetrics:
    """
    Class for calculating advanced fairness metrics
    """
    
    def __init__(self, 
                 protected_attribute: str, 
                 favorable_label: int = 1,
                 unfavorable_label: int = 0):
        """
        Initialize advanced fairness metrics
        """
        self.protected_attribute = protected_attribute
        self.favorable_label = favorable_label
        self.unfavorable_label = unfavorable_label
        
    def calculate_predictive_parity(self, 
                                  y_pred: np.ndarray, 
                                  y_true: np.ndarray, 
                                  sensitive_features: np.ndarray) -> Dict[str, float]:
        """
        Calculate predictive parity metrics
        """
        metrics = {}
        unique_groups = np.unique(sensitive_features)
        
        for group in unique_groups:
            group_mask = sensitive_features == group
            group_pred = y_pred[group_mask]
            group_true = y_true[group_mask]
            
            # Calculate positive predictive value
            true_positives = np.sum((group_pred == self.favorable_label) & 
                                  (group_true == self.favorable_label))
            all_positives = np.sum(group_pred == self.favorable_label)
            ppv = true_positives / all_positives if all_positives > 0 else 0
            
            metrics[f'positive_predictive_value_{group}'] = ppv
            
        return metrics
    
    def calculate_calibration(self, 
                            y_pred_proba: np.ndarray, 
                            y_true: np.ndarray, 
                            sensitive_features: np.ndarray,
                            n_bins: int = 10) -> Dict[str, float]:
        """
        Calculate calibration metrics across groups
        """
        metrics = {}
        unique_groups = np.unique(sensitive_features)
        
        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        
        for group in unique_groups:
            group_mask = sensitive_features == group
            group_proba = y_pred_proba[group_mask]
            group_true = y_true[group_mask]
            
            bin_metrics = []
            for i in range(n_bins):
                bin_mask = (group_proba >= bins[i]) & (group_proba < bins[i + 1])
                if np.sum(bin_mask) > 0:
                    bin_proba = np.mean(group_proba[bin_mask])
                    bin_true = np.mean(group_true[bin_mask])
                    bin_metrics.append(abs(bin_proba - bin_true))
            
            metrics[f'calibration_error_{group}'] = np.mean(bin_metrics)
            
        return metrics
    
    def calculate_kolmogorov_smirnov(self, 
                                    y_pred_proba: np.ndarray, 
                                    sensitive_features: np.ndarray) -> Dict[str, float]:
        """
        Calculate Kolmogorov-Smirnov test for probability distributions
        """
        metrics = {}
        unique_groups = np.unique(sensitive_features)
        
        # Calculate KS statistic between all pairs of groups
        for i, group1 in enumerate(unique_groups):
            for group2 in unique_groups[i + 1:]:
                group1_mask = sensitive_features == group1
                group2_mask = sensitive_features == group2
                
                ks_stat, _ = ks_2samp(
                    y_pred_proba[group1_mask],
                    y_pred_proba[group2_mask]
                )
                
                metrics[f'ks_stat_{group1}_vs_{group2}'] = ks_stat
                
        return metrics
    
    def calculate_group_fairness(self, 
                                y_pred: np.ndarray, 
                                y_true: np.ndarray, 
                                sensitive_features: np.ndarray) -> Dict[str, float]:
        """
        Calculate group fairness metrics
        """
        metrics = {}
        unique_groups = np.unique(sensitive_features)
        
        # Calculate metrics for each group
        for group in unique_groups:
            group_mask = sensitive_features == group
            group_pred = y_pred[group_mask]
            group_true = y_true[group_mask]
            
            # Accuracy
            metrics[f'accuracy_{group}'] = accuracy_score(group_true, group_pred)
            
            # ROC AUC
            metrics[f'roc_auc_{group}'] = roc_auc_score(group_true, group_pred)
            
            # Demographic parity
            metrics[f'demographic_parity_{group}'] = np.mean(group_pred == self.favorable_label)
            
            # Equal opportunity
            true_positives = np.sum((group_pred == self.favorable_label) & 
                                  (group_true == self.favorable_label))
            total_positives = np.sum(group_true == self.favorable_label)
            metrics[f'equal_opportunity_{group}'] = true_positives / total_positives if total_positives > 0 else 0
            
        return metrics
    
    def calculate_fairness_metrics(self, 
                                 y_pred: np.ndarray, 
                                 y_pred_proba: np.ndarray, 
                                 y_true: np.ndarray, 
                                 sensitive_features: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive fairness metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics.update(self.calculate_group_fairness(y_pred, y_true, sensitive_features))
        
        # Predictive parity
        metrics.update(self.calculate_predictive_parity(y_pred, y_true, sensitive_features))
        
        # Calibration
        metrics.update(self.calculate_calibration(y_pred_proba, y_true, sensitive_features))
        
        # Distribution comparison
        metrics.update(self.calculate_kolmogorov_smirnov(y_pred_proba, sensitive_features))
        
        return metrics
