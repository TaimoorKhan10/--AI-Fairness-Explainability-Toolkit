"""
Advanced visualization tools for AFET
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from fairlearn.metrics import MetricFrame


class AdvancedVisualizations:
    """
    Class for creating advanced visualizations
    """
    
    def __init__(self):
        """
        Initialize visualization tools
        """
        pass
    
    def create_fairness_dashboard(self, 
                                 metrics: Dict[str, float],
                                 sensitive_features: pd.Series) -> go.Figure:
        """
        Create a dashboard showing fairness metrics across groups
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Demographic Parity', 'Equal Opportunity',
                          'Predictive Parity', 'Calibration Error')
        )
        
        # Get unique groups
        unique_groups = np.unique(sensitive_features)
        
        # Add demographic parity plot
        dp_values = [metrics[f'demographic_parity_{group}'] for group in unique_groups]
        fig.add_trace(
            go.Bar(
                x=unique_groups,
                y=dp_values,
                name='Demographic Parity'
            ),
            row=1, col=1
        )
        
        # Add equal opportunity plot
        eo_values = [metrics[f'equal_opportunity_{group}'] for group in unique_groups]
        fig.add_trace(
            go.Bar(
                x=unique_groups,
                y=eo_values,
                name='Equal Opportunity'
            ),
            row=1, col=2
        )
        
        # Add predictive parity plot
        pp_values = [metrics[f'positive_predictive_value_{group}'] for group in unique_groups]
        fig.add_trace(
            go.Bar(
                x=unique_groups,
                y=pp_values,
                name='Predictive Parity'
            ),
            row=2, col=1
        )
        
        # Add calibration error plot
        cal_values = [metrics[f'calibration_error_{group}'] for group in unique_groups]
        fig.add_trace(
            go.Bar(
                x=unique_groups,
                y=cal_values,
                name='Calibration Error'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Fairness Metrics Dashboard',
            showlegend=False,
            height=800
        )
        
        return fig
    
    def create_model_comparison_plot(self, 
                                   comparison_df: pd.DataFrame,
                                   metric: str) -> go.Figure:
        """
        Create model comparison plot for a specific metric
        """
        fig = px.bar(
            comparison_df,
            x='model',
            y=metric,
            color='model',
            title=f'Model Comparison: {metric}',
            labels={'model': 'Model', metric: metric}
        )
        
        return fig
    
    def create_fairness_tradeoff_plot(self,
                                     metrics: Dict[str, float],
                                     sensitive_features: pd.Series) -> go.Figure:
        """
        Create tradeoff plot between fairness and performance metrics
        """
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'model': list(metrics.keys()),
            'accuracy': [m['accuracy'] for m in metrics.values()],
            'demographic_parity': [m['demographic_parity'] for m in metrics.values()],
            'equalized_odds': [m['equalized_odds'] for m in metrics.values()]
        })
        
        # Create figure
        fig = px.scatter(
            df,
            x='accuracy',
            y='demographic_parity',
            color='model',
            title='Fairness vs Performance Tradeoff',
            labels={
                'accuracy': 'Accuracy',
                'demographic_parity': 'Demographic Parity Difference'
            }
        )
        
        return fig
    
    def create_feature_importance_heatmap(self,
                                         feature_importances: Dict[str, float],
                                         model_name: str) -> go.Figure:
        """
        Create heatmap of feature importances
        """
        # Sort features by importance
        sorted_importances = sorted(
            feature_importances.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Create DataFrame
        df = pd.DataFrame(
            sorted_importances,
            columns=['feature', 'importance']
        )
        
        # Create heatmap
        fig = px.imshow(
            df['importance'].values.reshape(1, -1),
            labels=dict(x="Features", y="Model"),
            x=df['feature'],
            y=[model_name],
            color_continuous_scale='RdBu_r'
        )
        
        return fig
    
    def create_fairness_distribution_plot(self,
                                         y_pred_proba: np.ndarray,
                                         sensitive_features: pd.Series,
                                         n_bins: int = 10) -> go.Figure:
        """
        Create distribution plot of predicted probabilities across groups
        """
        # Get unique groups
        unique_groups = np.unique(sensitive_features)
        
        # Create figure
        fig = go.Figure()
        
        # Add distribution for each group
        for group in unique_groups:
            group_mask = sensitive_features == group
            group_proba = y_pred_proba[group_mask]
            
            fig.add_trace(
                go.Histogram(
                    x=group_proba,
                    name=str(group),
                    opacity=0.7
                )
            )
        
        fig.update_layout(
            title='Probability Distribution Across Groups',
            xaxis_title='Predicted Probability',
            yaxis_title='Count',
            barmode='overlay'
        )
        
        return fig
