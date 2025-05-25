"""
Fairness mitigation strategies implementation for AFET
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from fairlearn.reductions import ExponentiatedGradient, GridSearch
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset


class FairnessMitigator:
    """
    Class for implementing various fairness mitigation strategies
    """
    
    def __init__(self, 
                 protected_attribute: str,
                 favorable_label: int = 1,
                 unfavorable_label: int = 0):
        """
        Initialize fairness mitigator
        """
        self.protected_attribute = protected_attribute
        self.favorable_label = favorable_label
        self.unfavorable_label = unfavorable_label
        
    def reweighing(self, 
                  X: pd.DataFrame, 
                  y: pd.Series, 
                  sensitive_features: pd.Series) -> pd.DataFrame:
        """
        Apply reweighing preprocessing technique
        """
        # Convert to AIF360 dataset format
        dataset = BinaryLabelDataset(
            df=pd.concat([X, y, sensitive_features], axis=1),
            label_names=[y.name],
            protected_attribute_names=[self.protected_attribute]
        )
        
        # Apply reweighing
        reweighing = Reweighing()
        reweighed_dataset = reweighing.fit_transform(dataset)
        
        # Convert back to DataFrame
        reweighed_df = reweighed_dataset.convert_to_dataframe()[0]
        return reweighed_df.drop([y.name], axis=1)
    
    def exponentiated_gradient(self, 
                             estimator: BaseEstimator,
                             X: pd.DataFrame, 
                             y: pd.Series, 
                             sensitive_features: pd.Series,
                             constraints: str = 'demographic_parity') -> BaseEstimator:
        """
        Apply Exponentiated Gradient reduction method
        """
        mitigator = ExponentiatedGradient(
            estimator=estimator,
            constraints=constraints,
            eps=0.1
        )
        
        mitigator.fit(X, y, sensitive_features=sensitive_features)
        return mitigator
    
    def grid_search(self, 
                   estimator: BaseEstimator,
                   X: pd.DataFrame, 
                   y: pd.Series, 
                   sensitive_features: pd.Series,
                   constraints: str = 'demographic_parity') -> BaseEstimator:
        """
        Apply Grid Search reduction method
        """
        mitigator = GridSearch(
            estimator=estimator,
            constraints=constraints,
            grid_size=10
        )
        
        mitigator.fit(X, y, sensitive_features=sensitive_features)
        return mitigator
    
    def evaluate_mitigation(self, 
                          y_true: pd.Series,
                          y_pred: pd.Series,
                          sensitive_features: pd.Series,
                          metrics: List[str] = ['demographic_parity', 'equalized_odds']) -> Dict[str, float]:
        """
        Evaluate fairness metrics after mitigation
        """
        results = {}
        
        # Calculate demographic parity difference
        if 'demographic_parity' in metrics:
            dp_diff = demographic_parity_difference(
                y_true,
                y_pred,
                sensitive_features=sensitive_features
            )
            results['demographic_parity_difference'] = dp_diff
        
        # Calculate equalized odds difference
        if 'equalized_odds' in metrics:
            eo_diff = equalized_odds_difference(
                y_true,
                y_pred,
                sensitive_features=sensitive_features
            )
            results['equalized_odds_difference'] = eo_diff
        
        return results
    
    def mitigation_pipeline(self, 
                          X: pd.DataFrame, 
                          y: pd.Series, 
                          sensitive_features: pd.Series,
                          estimator: BaseEstimator,
                          preprocessing: bool = True,
                          reduction_method: str = 'exponentiated_gradient') -> Dict:
        """
        Complete fairness mitigation pipeline
        """
        # Apply preprocessing if requested
        if preprocessing:
            X = self.reweighing(X, y, sensitive_features)
        
        # Apply reduction method
        if reduction_method == 'exponentiated_gradient':
            mitigated_model = self.exponentiated_gradient(
                estimator,
                X,
                y,
                sensitive_features
            )
        else:
            mitigated_model = self.grid_search(
                estimator,
                X,
                y,
                sensitive_features
            )
        
        # Make predictions
        y_pred = mitigated_model.predict(X)
        
        # Evaluate results
        evaluation = self.evaluate_mitigation(
            y_true=y,
            y_pred=y_pred,
            sensitive_features=sensitive_features
        )
        
        return {
            'model': mitigated_model,
            'predictions': y_pred,
            'evaluation': evaluation
        }
