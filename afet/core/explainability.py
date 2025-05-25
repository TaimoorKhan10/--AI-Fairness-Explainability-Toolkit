"""
Core explainability tools for AFET
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show


class ModelExplainer:
    """
    Main class for model explainability
    """
    
    def __init__(self, 
                 model,
                 feature_names: List[str],
                 class_names: List[str],
                 training_data: np.ndarray):
        """
        Initialize the model explainer
        
        Args:
            model: Trained machine learning model
            feature_names: List of feature names
            class_names: List of class names
            training_data: Training data used for explanations
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.training_data = training_data
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        self.ebm_explainer = None
        
        # Create explainers
        self._initialize_explainers()
    
    def _initialize_explainers(self):
        """Initialize all explainers"""
        # SHAP Explainer
        self.shap_explainer = shap.Explainer(self.model.predict_proba)
        
        # LIME Explainer
        self.lime_explainer = LimeTabularExplainer(
            self.training_data,
            feature_names=self.feature_names,
            class_names=self.class_names,
            discretize_continuous=True
        )
        
        # EBM Explainer
        self.ebm_explainer = ExplainableBoostingClassifier()
        self.ebm_explainer.fit(self.training_data, self.model.predict(self.training_data))
    
    def explain_instance_shap(self, 
                            instance: np.ndarray,
                            num_samples: int = 1000) -> Dict[str, float]:
        """
        Explain a single instance using SHAP
        
        Args:
            instance: Single data instance to explain
            num_samples: Number of samples for SHAP approximation
            
        Returns:
            Dictionary of feature importance values
        """
        shap_values = self.shap_explainer(instance.reshape(1, -1),
                                        max_evals=num_samples)
        return dict(zip(self.feature_names, shap_values.values[0]))
    
    def explain_instance_lime(self, 
                            instance: np.ndarray,
                            num_features: int = 5) -> Dict[str, float]:
        """
        Explain a single instance using LIME
        
        Args:
            instance: Single data instance to explain
            num_features: Number of features to show in explanation
            
        Returns:
            Dictionary of feature importance values
        """
        exp = self.lime_explainer.explain_instance(
            instance,
            self.model.predict_proba,
            num_features=num_features
        )
        return dict(exp.as_list())
    
    def explain_global_shap(self, 
                          data: np.ndarray,
                          num_samples: int = 1000) -> Dict[str, float]:
        """
        Get global feature importance using SHAP
        
        Args:
            data: Dataset to explain
            num_samples: Number of samples for SHAP approximation
            
        Returns:
            Dictionary of global feature importance values
        """
        shap_values = self.shap_explainer(data,
                                        max_evals=num_samples)
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        return dict(zip(self.feature_names, mean_abs_shap))
    
    def explain_global_ebm(self) -> Dict[str, float]:
        """
        Get global feature importance using Explainable Boosting Machine
        
        Returns:
            Dictionary of global feature importance values
        """
        ebm_global = self.ebm_explainer.explain_global()
        show(ebm_global)
        return dict(zip(self.feature_names, ebm_global.data()['scores']))
    
    def get_all_explanations(self, 
                           instance: np.ndarray,
                           data: np.ndarray = None) -> Dict[str, Dict]:
        """
        Get all available explanations for a given instance
        
        Args:
            instance: Single data instance to explain
            data: Optional dataset for global explanations
            
        Returns:
            Dictionary containing all explanation types
        """
        explanations = {
            'shap_local': self.explain_instance_shap(instance),
            'lime_local': self.explain_instance_lime(instance),
            'shap_global': self.explain_global_shap(data) if data is not None else None,
            'ebm_global': self.explain_global_ebm()
        }
        return explanations
