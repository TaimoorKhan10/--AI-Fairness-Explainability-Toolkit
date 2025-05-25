"""
Fairness Radar Plot

This module provides a radar plot visualization for comparing fairness metrics
across different demographic groups and models.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple

from afet.core.fairness_metrics import FairnessMetrics


class FairnessRadar:
    """
    A class for creating radar plots to visualize fairness metrics across different
    demographic groups and models.
    
    This visualization helps users understand how different fairness metrics
    compare across demographic groups and models, making it easier to identify
    where fairness issues might be occurring.
    """
    
    def __init__(self, title: str = "Fairness Metrics Radar Plot"):
        """
        Initialize the FairnessRadar visualization.
        
        Parameters
        ----------
        title : str, optional
            The title of the radar plot, by default "Fairness Metrics Radar Plot"
        """
        self.title = title
        self.metrics = [
            "Demographic Parity",
            "Equal Opportunity",
            "Equalized Odds",
            "Treatment Equality",
            "Predictive Parity"
        ]
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: Dict[str, np.ndarray],
        sensitive_features: np.ndarray,
        unique_groups: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate fairness metrics for each model and demographic group.
        
        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels
        y_pred : Dict[str, np.ndarray]
            Dictionary mapping model names to their predictions
        sensitive_features : np.ndarray
            Array of sensitive feature values for each instance
        unique_groups : Optional[List[str]], optional
            List of unique demographic groups, by default None
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing fairness metrics for each model and group
        """
        if unique_groups is None:
            unique_groups = np.unique(sensitive_features)
        
        results = []
        
        for model_name, predictions in y_pred.items():
            for group in unique_groups:
                # Create mask for the current group
                group_mask = sensitive_features == group
                
                # Skip if no samples in this group
                if not np.any(group_mask):
                    continue
                
                # Calculate metrics for this group
                metrics = FairnessMetrics(
                    y_true=y_true,
                    y_pred=predictions,
                    sensitive_features=sensitive_features
                )
                
                # Get metric values
                demographic_parity = 1 - abs(metrics.demographic_parity_difference(privileged_groups=[group]))
                equal_opportunity = 1 - abs(metrics.equal_opportunity_difference(privileged_groups=[group]))
                equalized_odds = 1 - abs(metrics.average_odds_difference(privileged_groups=[group]))
                
                # Calculate additional metrics
                # Treatment Equality: ratio of FN/FP between groups
                confusion = metrics.confusion_matrix_by_group()
                group_confusion = confusion.get(group, {})
                
                if group_confusion:
                    fp = group_confusion.get('false_positive', 0)
                    fn = group_confusion.get('false_negative', 0)
                    
                    # Avoid division by zero
                    treatment_equality = 1.0
                    if fp > 0:
                        treatment_equality = min(fn / fp if fn > fp else fp / fn, 1.0)
                else:
                    treatment_equality = 1.0
                
                # Predictive Parity: similar PPV across groups
                predictive_parity = 1 - abs(metrics.predictive_parity_difference(privileged_groups=[group]))
                
                results.append({
                    'Model': model_name,
                    'Group': group,
                    'Demographic Parity': demographic_parity,
                    'Equal Opportunity': equal_opportunity,
                    'Equalized Odds': equalized_odds,
                    'Treatment Equality': treatment_equality,
                    'Predictive Parity': predictive_parity
                })
        
        return pd.DataFrame(results)
    
    def create_plot(
        self,
        y_true: np.ndarray,
        y_pred: Dict[str, np.ndarray],
        sensitive_features: np.ndarray,
        model_names: Optional[List[str]] = None,
        group_names: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Create a radar plot of fairness metrics across models and demographic groups.
        
        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels
        y_pred : Dict[str, np.ndarray]
            Dictionary mapping model names to their predictions
        sensitive_features : np.ndarray
            Array of sensitive feature values for each instance
        model_names : Optional[List[str]], optional
            List of model names to include in the plot, by default None
        group_names : Optional[List[str]], optional
            List of demographic groups to include in the plot, by default None
            
        Returns
        -------
        go.Figure
            Plotly figure containing the radar plot
        """
        # Filter models if specified
        if model_names is not None:
            y_pred = {k: v for k, v in y_pred.items() if k in model_names}
        
        # Get unique groups
        if group_names is None:
            group_names = np.unique(sensitive_features)
        
        # Calculate metrics
        metrics_df = self._calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
            unique_groups=group_names
        )
        
        # Create subplots - one for each model
        fig = make_subplots(
            rows=1, 
            cols=len(y_pred),
            specs=[[{'type': 'polar'}] * len(y_pred)],
            subplot_titles=list(y_pred.keys())
        )
        
        # Add traces for each group within each model
        for i, model_name in enumerate(y_pred.keys()):
            model_data = metrics_df[metrics_df['Model'] == model_name]
            
            for group in group_names:
                group_data = model_data[model_data['Group'] == group]
                
                if not group_data.empty:
                    # Extract metric values and add the first value again to close the radar
                    values = [group_data[metric].values[0] for metric in self.metrics]
                    values.append(values[0])
                    
                    # Create radar trace
                    fig.add_trace(
                        go.Scatterpolar(
                            r=values,
                            theta=self.metrics + [self.metrics[0]],
                            name=f"{group}",
                            showlegend=i == 0,  # Only show legend for the first model
                        ),
                        row=1, col=i+1
                    )
        
        # Update layout
        fig.update_layout(
            title=self.title,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            height=600,
            width=250 * len(y_pred),
            margin=dict(l=80, r=80, t=100, b=80)
        )
        
        # Update each polar subplot
        for i in range(len(y_pred)):
            fig.update_polars(
                dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                subplot=i+1
            )
        
        return fig


def plot_fairness_radar(
    y_true: np.ndarray,
    y_pred: Dict[str, np.ndarray],
    sensitive_features: np.ndarray,
    model_names: Optional[List[str]] = None,
    group_names: Optional[List[str]] = None,
    title: str = "Fairness Metrics Radar Plot"
) -> go.Figure:
    """
    Create a radar plot of fairness metrics across models and demographic groups.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : Dict[str, np.ndarray]
        Dictionary mapping model names to their predictions
    sensitive_features : np.ndarray
        Array of sensitive feature values for each instance
    model_names : Optional[List[str]], optional
        List of model names to include in the plot, by default None
    group_names : Optional[List[str]], optional
        List of demographic groups to include in the plot, by default None
    title : str, optional
        The title of the radar plot, by default "Fairness Metrics Radar Plot"
        
    Returns
    -------
    go.Figure
        Plotly figure containing the radar plot
    """
    radar = FairnessRadar(title=title)
    return radar.create_plot(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        model_names=model_names,
        group_names=group_names
    )
