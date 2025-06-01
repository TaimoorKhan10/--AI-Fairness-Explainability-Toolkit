"""
Core explainability tools for AFET
"""

from typing import Dict, List, Optional, Union, Any, Callable
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show
import eli5
from eli5.sklearn import PermutationImportance
import logging

logger = logging.getLogger(__name__)


class ModelExplainer:
    """
    Main class for model explainability
    """
    
    def __init__(self, 
                 model: Any,
                 feature_names: List[str],
                 class_names: List[str],
                 training_data: np.ndarray,
                 y_train: Optional[np.ndarray] = None,
                 predict_function: Optional[Callable] = None):
        """
        Initialize the model explainer
        
        Args:
            model: Trained machine learning model
            feature_names: List of feature names
            class_names: List of class names
            training_data: Training data used for explanations
            y_train: Training labels for permutation importance
            predict_function: Optional custom predict function for SHAP
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.training_data = training_data
        self.y_train = y_train
        self.predict_function = predict_function or getattr(model, 'predict_proba', None)
        
        if self.predict_function is None:
            logger.warning("No predict_proba method found on model and no custom function provided")
            self.predict_function = getattr(model, 'predict', lambda x: x)
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        self.ebm_explainer = None
        self.perm_explainer = None
        
        # Create explainers
        self._initialize_explainers()
    
    def _initialize_explainers(self):
        """Initialize all explainers"""
        try:
            # SHAP Explainer
            self.shap_explainer = shap.Explainer(self.predict_function)
            logger.info("SHAP explainer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {str(e)}")
            self.shap_explainer = None
        
        try:
            # LIME Explainer
            self.lime_explainer = LimeTabularExplainer(
                self.training_data,
                feature_names=self.feature_names,
                class_names=self.class_names,
                discretize_continuous=True,
                mode='classification' if len(self.class_names) > 0 else 'regression'
            )
            logger.info("LIME explainer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LIME explainer: {str(e)}")
            self.lime_explainer = None
        
        try:
            # EBM Explainer
            self.ebm_explainer = ExplainableBoostingClassifier()
            pred_values = self.model.predict(self.training_data)
            self.ebm_explainer.fit(self.training_data, pred_values)
            logger.info("EBM explainer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EBM explainer: {str(e)}")
            self.ebm_explainer = None
            
        try:
            # Permutation Importance Explainer (only if y_train is provided)
            if self.y_train is not None:
                self.perm_explainer = PermutationImportance(self.model)
                self.perm_explainer.fit(self.training_data, self.y_train)
                logger.info("Permutation importance explainer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize permutation importance explainer: {str(e)}")
            self.perm_explainer = None
    
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
    
    def explain_permutation_importance(self) -> Dict[str, float]:
        """
        Get feature importance using permutation importance
        
        Returns:
            Dictionary of feature importance values
        """
        if self.perm_explainer is None or self.y_train is None:
            logger.warning("Permutation importance explainer not available")
            return {}
            
        try:
            perm_importance = eli5.explain_weights_df(self.perm_explainer, feature_names=self.feature_names)
            feature_importance = dict(zip(perm_importance['feature'], perm_importance['weight']))
            return feature_importance
        except Exception as e:
            logger.error(f"Error calculating permutation importance: {str(e)}")
            return {}
    
    def explain_partial_dependence(self, 
                                 features: List[int],
                                 data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate partial dependence for specified features
        
        Args:
            features: List of feature indices to calculate PD for
            data: Dataset to use for PD calculation
            
        Returns:
            Dictionary with PD values for each feature
        """
        try:
            from sklearn.inspection import partial_dependence
            
            results = {}
            for feature_idx in features:
                feature_name = self.feature_names[feature_idx]
                pd_result = partial_dependence(self.model, data, [feature_idx])
                results[feature_name] = {
                    'values': pd_result['values'][0],
                    'pd': pd_result['average'][0],
                }
            return results
        except Exception as e:
            logger.error(f"Error calculating partial dependence: {str(e)}")
            return {}
    
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
        explanations = {}
        
        # Add explanations if their respective explainers are available
        if self.shap_explainer is not None:
            explanations['shap_local'] = self.explain_instance_shap(instance)
            if data is not None:
                explanations['shap_global'] = self.explain_global_shap(data)
        
        if self.lime_explainer is not None:
            explanations['lime_local'] = self.explain_instance_lime(instance)
        
        if self.ebm_explainer is not None:
            explanations['ebm_global'] = self.explain_global_ebm()
        
        if self.perm_explainer is not None and self.y_train is not None:
            explanations['permutation_importance'] = self.explain_permutation_importance()
        
        if data is not None and len(self.feature_names) > 0:
            # Only include PD for the top 3 most important features
            important_features = self.get_top_important_features(3)
            if important_features:
                explanations['partial_dependence'] = self.explain_partial_dependence(important_features, data)
        
        return explanations
    
    def get_top_important_features(self, n: int = 3) -> List[int]:
        """
        Get indices of top important features based on available explainers
        
        Args:
            n: Number of top features to return
            
        Returns:
            List of feature indices
        """
        # Try to get feature importance from any available explainer
        importance = None
        
        if self.perm_explainer is not None and self.y_train is not None:
            importance_dict = self.explain_permutation_importance()
            if importance_dict:
                importance = np.array([importance_dict.get(f, 0) for f in self.feature_names])
        
        if importance is None and hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        
        if importance is None:
            # If no feature importance available, return first n features
            return list(range(min(n, len(self.feature_names))))
        
        # Return indices of top n features
        return list(np.argsort(importance)[-n:])
