"""
Intersectional Fairness Heatmap Visualization

This module provides visualization tools for analyzing fairness across intersectional demographic groups,
helping users identify patterns and disparities across multiple protected attributes.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Union, Tuple

from afet.core.intersectional_fairness import IntersectionalFairnessMetrics


class IntersectionalHeatmap:
    """
    A class for creating heatmap visualizations to analyze fairness across intersectional demographic groups.
    
    This visualization helps users understand how multiple protected attributes interact to affect
    model outcomes and identify potential bias hotspots.
    """
    
    def __init__(self, title: str = "Intersectional Fairness Analysis"):
        """
        Initialize the IntersectionalHeatmap visualization.
        
        Parameters
        ----------
        title : str, optional
            The title of the heatmap, by default "Intersectional Fairness Analysis"
        """
        self.title = title
        self.metrics = [
            "Selection_Rate",
            "TPR",
            "TNR",
            "FPR",
            "FNR",
            "PPV",
            "NPV",
            "Accuracy"
        ]
    
    def _parse_group_names(self, group_metrics: pd.DataFrame, primary_attr: str, secondary_attr: str) -> pd.DataFrame:
        """
        Parse intersectional group names to extract individual attribute values.
        
        Parameters
        ----------
        group_metrics : pd.DataFrame
            DataFrame containing metrics for each intersectional group
        primary_attr : str
            Name of the primary attribute for the heatmap rows
        secondary_attr : str
            Name of the secondary attribute for the heatmap columns
            
        Returns
        -------
        pd.DataFrame
            DataFrame with parsed attribute values
        """
        # Create a copy of the metrics DataFrame
        metrics = group_metrics.copy()
        
        # Parse the group names to extract attribute values
        # Group names are in format "attr1_attr2_attr3"
        metrics[primary_attr] = metrics['Group'].apply(
            lambda x: x.split('_')[list(self.attr_indices.keys()).index(primary_attr)]
        )
        
        metrics[secondary_attr] = metrics['Group'].apply(
            lambda x: x.split('_')[list(self.attr_indices.keys()).index(secondary_attr)]
        )
        
        return metrics
    
    def create_heatmap(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_features: Dict[str, np.ndarray],
        primary_attr: str,
        secondary_attr: str,
        metric: str = "Selection_Rate"
    ) -> go.Figure:
        """
        Create a heatmap visualization for intersectional fairness analysis.
        
        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels
        y_pred : np.ndarray
            Predicted labels
        protected_features : Dict[str, np.ndarray]
            Dictionary mapping protected attribute names to their values for each instance
        primary_attr : str
            Name of the primary attribute for the heatmap rows
        secondary_attr : str
            Name of the secondary attribute for the heatmap columns
        metric : str, optional
            The metric to visualize, by default "Selection_Rate"
            
        Returns
        -------
        go.Figure
            Plotly figure containing the heatmap
        """
        # Store attribute indices for parsing group names
        self.attr_indices = {attr: i for i, attr in enumerate(protected_features.keys())}
        
        # Calculate intersectional fairness metrics
        ifm = IntersectionalFairnessMetrics(y_true, y_pred, protected_features)
        group_metrics = ifm.group_metrics()
        
        # Parse group names to extract individual attribute values
        metrics = self._parse_group_names(group_metrics, primary_attr, secondary_attr)
        
        # Create a pivot table for the heatmap
        pivot_table = metrics.pivot_table(
            index=primary_attr,
            columns=secondary_attr,
            values=metric,
            aggfunc='mean'
        )
        
        # Create the heatmap
        fig = px.imshow(
            pivot_table,
            labels=dict(
                x=secondary_attr,
                y=primary_attr,
                color=metric
            ),
            x=pivot_table.columns,
            y=pivot_table.index,
            color_continuous_scale="RdBu_r",
            title=f"{self.title}: {metric} by {primary_attr} and {secondary_attr}"
        )
        
        # Add text annotations
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                value = pivot_table.iloc[i, j]
                if not pd.isna(value):
                    fig.add_annotation(
                        x=j,
                        y=i,
                        text=f"{value:.3f}",
                        showarrow=False,
                        font=dict(color="black" if 0.3 < value < 0.7 else "white")
                    )
        
        # Update layout
        fig.update_layout(
            height=600,
            width=800,
            margin=dict(l=80, r=80, t=100, b=80)
        )
        
        return fig
    
    def create_multi_metric_heatmaps(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_features: Dict[str, np.ndarray],
        primary_attr: str,
        secondary_attr: str,
        metrics: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Create multiple heatmaps for different fairness metrics.
        
        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels
        y_pred : np.ndarray
            Predicted labels
        protected_features : Dict[str, np.ndarray]
            Dictionary mapping protected attribute names to their values for each instance
        primary_attr : str
            Name of the primary attribute for the heatmap rows
        secondary_attr : str
            Name of the secondary attribute for the heatmap columns
        metrics : Optional[List[str]], optional
            List of metrics to visualize, by default None
            
        Returns
        -------
        go.Figure
            Plotly figure containing multiple heatmaps
        """
        if metrics is None:
            metrics = ["Selection_Rate", "TPR", "FPR", "Accuracy"]
        
        # Store attribute indices for parsing group names
        self.attr_indices = {attr: i for i, attr in enumerate(protected_features.keys())}
        
        # Calculate intersectional fairness metrics
        ifm = IntersectionalFairnessMetrics(y_true, y_pred, protected_features)
        group_metrics = ifm.group_metrics()
        
        # Parse group names to extract individual attribute values
        parsed_metrics = self._parse_group_names(group_metrics, primary_attr, secondary_attr)
        
        # Create subplots
        n_metrics = len(metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[f"{metric}" for metric in metrics]
        )
        
        # Add heatmaps for each metric
        for i, metric in enumerate(metrics):
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            # Create a pivot table for the heatmap
            pivot_table = parsed_metrics.pivot_table(
                index=primary_attr,
                columns=secondary_attr,
                values=metric,
                aggfunc='mean'
            )
            
            # Add heatmap to subplot
            heatmap = go.Heatmap(
                z=pivot_table.values,
                x=pivot_table.columns,
                y=pivot_table.index,
                colorscale="RdBu_r",
                showscale=(i == 0),  # Only show colorbar for the first heatmap
                colorbar=dict(title=metric) if i == 0 else None
            )
            
            fig.add_trace(heatmap, row=row, col=col)
            
            # Add text annotations
            for i_row in range(len(pivot_table.index)):
                for j_col in range(len(pivot_table.columns)):
                    value = pivot_table.iloc[i_row, j_col]
                    if not pd.isna(value):
                        fig.add_annotation(
                            x=j_col,
                            y=i_row,
                            text=f"{value:.2f}",
                            showarrow=False,
                            font=dict(color="black" if 0.3 < value < 0.7 else "white"),
                            xref=f"x{i+1}" if i > 0 else "x",
                            yref=f"y{i+1}" if i > 0 else "y"
                        )
        
        # Update layout
        fig.update_layout(
            height=400 * n_rows,
            width=500 * n_cols,
            title=f"Intersectional Fairness Analysis: {primary_attr} × {secondary_attr}",
            margin=dict(l=80, r=80, t=100, b=80)
        )
        
        # Update x and y axis labels
        for i in range(n_metrics):
            fig.update_xaxes(title_text=secondary_attr, row=(i // n_cols) + 1, col=(i % n_cols) + 1)
            fig.update_yaxes(title_text=primary_attr, row=(i // n_cols) + 1, col=(i % n_cols) + 1)
        
        return fig


def plot_intersectional_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_features: Dict[str, np.ndarray],
    primary_attr: str,
    secondary_attr: str,
    metric: str = "Selection_Rate",
    title: str = "Intersectional Fairness Analysis"
) -> go.Figure:
    """
    Create a heatmap visualization for intersectional fairness analysis.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
    protected_features : Dict[str, np.ndarray]
        Dictionary mapping protected attribute names to their values for each instance
    primary_attr : str
        Name of the primary attribute for the heatmap rows
    secondary_attr : str
        Name of the secondary attribute for the heatmap columns
    metric : str, optional
        The metric to visualize, by default "Selection_Rate"
    title : str, optional
        The title of the heatmap, by default "Intersectional Fairness Analysis"
        
    Returns
    -------
    go.Figure
        Plotly figure containing the heatmap
    """
    heatmap = IntersectionalHeatmap(title=title)
    return heatmap.create_heatmap(
        y_true=y_true,
        y_pred=y_pred,
        protected_features=protected_features,
        primary_attr=primary_attr,
        secondary_attr=secondary_attr,
        metric=metric
    )


def plot_multi_metric_heatmaps(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_features: Dict[str, np.ndarray],
    primary_attr: str,
    secondary_attr: str,
    metrics: Optional[List[str]] = None,
    title: str = "Intersectional Fairness Analysis"
) -> go.Figure:
    """
    Create multiple heatmaps for different fairness metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
    protected_features : Dict[str, np.ndarray]
        Dictionary mapping protected attribute names to their values for each instance
    primary_attr : str
        Name of the primary attribute for the heatmap rows
    secondary_attr : str
        Name of the secondary attribute for the heatmap columns
    metrics : Optional[List[str]], optional
        List of metrics to visualize, by default None
    title : str, optional
        The title of the heatmap, by default "Intersectional Fairness Analysis"
        
    Returns
    -------
    go.Figure
        Plotly figure containing multiple heatmaps
    """
    heatmap = IntersectionalHeatmap(title=title)
    return heatmap.create_multi_metric_heatmaps(
        y_true=y_true,
        y_pred=y_pred,
        protected_features=protected_features,
        primary_attr=primary_attr,
        secondary_attr=secondary_attr,
        metrics=metrics
    )
