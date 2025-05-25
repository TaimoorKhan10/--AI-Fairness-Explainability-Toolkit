"""
Advanced explainability tools for AFET
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show
import plotly.express as px
import plotly.graph_objects as go


class AdvancedExplainability:
    """
    Advanced explainability methods and visualizations
    """
    
    def __init__(self, 
                 model,
                 feature_names: List[str],
                 class_names: List[str],
                 training_data: np.ndarray):
        """
        Initialize advanced explainability tools
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.training_data = training_data
        
        # Initialize explainers
        self.shap_explainer = shap.Explainer(self.model.predict_proba)
        self.lime_explainer = LimeTabularExplainer(
            self.training_data,
            feature_names=self.feature_names,
            class_names=self.class_names,
            discretize_continuous=True
        )
        self.ebm_explainer = ExplainableBoostingClassifier()
        
    def create_shap_summary_plot(self, 
                               shap_values: np.ndarray,
                               max_display: int = 20) -> go.Figure:
        """
        Create an interactive SHAP summary plot
        """
        # Calculate mean absolute SHAP values
        mean_abs = np.abs(shap_values).mean(0)
        
        # Sort features by importance
        feature_order = np.argsort(-mean_abs)
        
        # Create plot
        fig = go.Figure()
        
        # Add SHAP values as points
        for i in feature_order[:max_display]:
            fig.add_trace(go.Box(
                y=shap_values[:, i],
                name=self.feature_names[i],
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8,
                marker=dict(color='rgba(99, 110, 250, 0.6)')
            ))
        
        fig.update_layout(
            title='SHAP Summary Plot',
            yaxis_title='Features',
            xaxis_title='SHAP value (impact on model output)',
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_shap_dependence_plot(self,
                                  shap_values: np.ndarray,
                                  feature_index: int,
                                  X: np.ndarray) -> go.Figure:
        """
        Create SHAP dependence plot
        """
        # Get the feature values and SHAP values
        feature_values = X[:, feature_index]
        shap_feature = shap_values[:, feature_index]
        
        # Create scatter plot
        fig = px.scatter(
            x=feature_values,
            y=shap_feature,
            title=f'SHAP Dependence Plot for {self.feature_names[feature_index]}',
            labels={'x': self.feature_names[feature_index],
                   'y': 'SHAP value'}
        )
        
        return fig
    
    def create_partial_dependence_plot(self,
                                     X: np.ndarray,
                                     feature_index: int,
                                     num_points: int = 100) -> go.Figure:
        """
        Create partial dependence plot
        """
        # Get feature values
        feature_values = X[:, feature_index]
        
        # Create grid of feature values
        feature_grid = np.linspace(
            feature_values.min(),
            feature_values.max(),
            num_points
        )
        
        # Create partial dependence values
        partial_dependence = []
        for value in feature_grid:
            X_temp = X.copy()
            X_temp[:, feature_index] = value
            partial_dependence.append(self.model.predict_proba(X_temp)[:, 1].mean())
        
        # Create plot
        fig = px.line(
            x=feature_grid,
            y=partial_dependence,
            title=f'Partial Dependence Plot for {self.feature_names[feature_index]}',
            labels={'x': self.feature_names[feature_index],
                   'y': 'Average prediction'}
        )
        
        return fig
    
    def create_feature_interaction_plot(self,
                                      shap_values: np.ndarray,
                                      feature1: int,
                                      feature2: int,
                                      X: np.ndarray) -> go.Figure:
        """
        Create SHAP interaction plot
        """
        # Create figure
        fig = px.scatter_3d(
            x=X[:, feature1],
            y=X[:, feature2],
            z=shap_values[:, feature1] * shap_values[:, feature2],
            title=f'SHAP Interaction Plot: {self.feature_names[feature1]} vs {self.feature_names[feature2]}',
            labels={
                'x': self.feature_names[feature1],
                'y': self.feature_names[feature2],
                'z': 'SHAP interaction value'
            }
        )
        
        return fig
    
    def create_decision_tree_visualization(self,
                                          max_depth: int = 3) -> go.Figure:
        """
        Create decision tree visualization (if model is tree-based)
        """
        try:
            from sklearn.tree import export_graphviz
            import graphviz
            
            # Export tree as graphviz
            dot_data = export_graphviz(
                self.model,
                out_file=None,
                feature_names=self.feature_names,
                class_names=self.class_names,
                filled=True,
                rounded=True,
                special_characters=True,
                max_depth=max_depth
            )
            
            # Create graph
            graph = graphviz.Source(dot_data)
            
            return graph
        except:
            return None
    
    def create_all_visualizations(self,
                                X: np.ndarray,
                                shap_values: np.ndarray = None) -> Dict[str, go.Figure]:
        """
        Create all available visualizations
        """
        if shap_values is None:
            shap_values = self.shap_explainer(X)
        
        visualizations = {
            'shap_summary': self.create_shap_summary_plot(shap_values),
            'partial_dependence': self.create_partial_dependence_plot(X, 0),
            'decision_tree': self.create_decision_tree_visualization()
        }
        
        # Add feature interaction plots
        for i in range(len(self.feature_names)):
            for j in range(i + 1, len(self.feature_names)):
                key = f'interaction_{self.feature_names[i]}_{self.feature_names[j]}'
                visualizations[key] = self.create_feature_interaction_plot(
                    shap_values,
                    feature1=i,
                    feature2=j,
                    X=X
                )
        
        return visualizations
