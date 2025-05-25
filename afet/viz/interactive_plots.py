"""
Interactive Visualization Tools for Fairness Analysis

This module provides interactive visualization tools for exploring model fairness
and explainability using Plotly.
"""

from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix
)


class FairnessDashboard:
    """
    Interactive dashboard for fairness analysis.
    
    Parameters:
    -----------
    model_names : list of str
        Names of the models to compare.
    sensitive_attr : str
        Name of the sensitive attribute column.
    """
    
    def __init__(self, model_names: List[str], sensitive_attr: str):
        self.model_names = model_names
        self.sensitive_attr = sensitive_attr
        self.fig = None
    
    def create_dashboard(self, 
                        y_true: np.ndarray,
                        y_pred: Dict[str, np.ndarray],
                        y_prob: Dict[str, np.ndarray],
                        sensitive: np.ndarray,
                        feature_names: Optional[List[str]] = None,
                        feature_importances: Optional[Dict[str, np.ndarray]] = None):
        """
        Create an interactive fairness dashboard.
        
        Parameters:
        -----------
        y_true : array-like of shape (n_samples,)
            Ground truth labels.
        y_pred : dict of array-like
            Dictionary of model predictions.
        y_prob : dict of array-like
            Dictionary of model predicted probabilities.
        sensitive : array-like of shape (n_samples,)
            Sensitive attribute values.
        feature_names : list of str, optional
            Names of the features.
        feature_importances : dict of array-like, optional
            Feature importances for each model.
        
        Returns:
        --------
        fig : plotly.graph_objects.Figure
            Interactive dashboard figure.
        """
        # Create subplots
        self.fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'ROC Curves by Sensitive Group',
                'Precision-Recall Curves',
                'Fairness Metrics',
                'Feature Importance (Top 10)'
            ),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                  [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Add ROC curves
        self._add_roc_curves(y_true, y_prob, sensitive)
        
        # Add PR curves
        self._add_pr_curves(y_true, y_prob)
        
        # Add fairness metrics
        self._add_fairness_metrics(y_true, y_pred, sensitive)
        
        # Add feature importance if available
        if feature_importances is not None and feature_names is not None:
            self._add_feature_importance(feature_importances, feature_names)
        
        # Update layout
        self.fig.update_layout(
            height=1000,
            width=1400,
            title_text="Fairness Analysis Dashboard",
            showlegend=True,
            hovermode='closest'
        )
        
        return self.fig
    
    def _add_roc_curves(self, y_true: np.ndarray, 
                       y_prob: Dict[str, np.ndarray],
                       sensitive: np.ndarray):
        """Add ROC curves for each model and sensitive group."""
        groups = np.unique(sensitive)
        
        for i, model_name in enumerate(self.model_names):
            for group in groups:
                mask = (sensitive == group)
                fpr, tpr, _ = roc_curve(y_true[mask], y_prob[model_name][mask])
                
                self.fig.add_trace(
                    go.Scatter(
                        x=fpr, y=tpr,
                        name=f"{model_name} - {group}",
                        line=dict(width=2, dash='solid' if i == 0 else 'dash'),
                        legendgroup=model_name,
                        showlegend=True
                    ),
                    row=1, col=1
                )
        
        # Add diagonal line
        self.fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                line=dict(color='black', dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        self.fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
        self.fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
    
    def _add_pr_curves(self, y_true: np.ndarray, 
                      y_prob: Dict[str, np.ndarray]):
        """Add precision-recall curves for each model."""
        for model_name in self.model_names:
            precision, recall, _ = precision_recall_curve(y_true, y_prob[model_name])
            
            self.fig.add_trace(
                go.Scatter(
                    x=recall, y=precision,
                    name=model_name,
                    line=dict(width=2),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Add baseline (prevalence)
        baseline = y_true.mean()
        self.fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[baseline, baseline],
                mode='lines',
                line=dict(color='black', dash='dash'),
                name='Baseline (Prevalence)'
            ),
            row=1, col=2
        )
        
        self.fig.update_xaxes(title_text="Recall", row=1, col=2)
        self.fig.update_yaxes(title_text="Precision", row=1, col=2)
    
    def _add_fairness_metrics(self, y_true: np.ndarray,
                            y_pred: Dict[str, np.ndarray],
                            sensitive: np.ndarray):
        """Add fairness metrics comparison."""
        from afet.core.fairness_metrics import FairnessMetrics
        
        metrics = []
        for model_name in self.model_names:
            fm = FairnessMetrics(
                y_true=y_true,
                y_pred=y_pred[model_name],
                sensitive_features=sensitive
            )
            
            metrics.append({
                'Model': model_name,
                'Demographic Parity': fm.demographic_parity_difference(),
                'Equal Opportunity': fm.equal_opportunity_difference(),
                'Average Odds': fm.average_odds_difference()
            })
        
        metrics_df = pd.DataFrame(metrics)
        
        for i, metric in enumerate(['Demographic Parity', 'Equal Opportunity', 'Average Odds'], 1):
            self.fig.add_trace(
                go.Bar(
                    x=self.model_names,
                    y=metrics_df[metric],
                    name=metric,
                    text=metrics_df[metric].round(3),
                    textposition='auto',
                    texttemplate='%{text:.3f}'
                ),
                row=2, col=1
            )
        
        self.fig.update_xaxes(title_text="Model", row=2, col=1)
        self.fig.update_yaxes(title_text="Fairness Metric Value", row=2, col=1)
    
    def _add_feature_importance(self, 
                              feature_importances: Dict[str, np.ndarray],
                              feature_names: List[str],
                              top_n: int = 10):
        """Add feature importance visualization."""
        for i, model_name in enumerate(self.model_names):
            # Get top N features
            idx = np.argsort(feature_importances[model_name])[-top_n:][::-1]
            top_features = [feature_names[i] for i in idx]
            top_importances = feature_importances[model_name][idx]
            
            self.fig.add_trace(
                go.Bar(
                    x=top_importances,
                    y=top_features,
                    name=model_name,
                    orientation='h',
                    showlegend=False
                ),
                row=2, col=2
            )
        
        self.fig.update_xaxes(title_text="Importance", row=2, col=2)
        self.fig.update_yaxes(title_text="Feature", row=2, col=2)


class ThresholdAnalysis:
    """
    Interactive threshold analysis for fairness-accuracy trade-offs.
    """
    
    @staticmethod
    def plot_threshold_analysis(y_true: np.ndarray,
                              y_prob: np.ndarray,
                              sensitive: np.ndarray,
                              model_name: str = 'Model') -> go.Figure:
        """
        Create an interactive threshold analysis plot.
        
        Parameters:
        -----------
        y_true : array-like of shape (n_samples,)
            Ground truth labels.
        y_prob : array-like of shape (n_samples,)
            Predicted probabilities.
        sensitive : array-like of shape (n_samples,)
            Sensitive attribute values.
        model_name : str, default='Model'
            Name of the model for the plot title.
            
        Returns:
        --------
        fig : plotly.graph_objects.Figure
            Interactive threshold analysis figure.
        """
        from afet.core.fairness_metrics import FairnessMetrics
        
        thresholds = np.linspace(0, 1, 101)
        groups = np.unique(sensitive)
        
        # Calculate metrics for each threshold
        metrics = []
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            fm = FairnessMetrics(y_true, y_pred, sensitive_features=sensitive)
            
            metrics.append({
                'Threshold': t,
                'Accuracy': fm.accuracy(),
                'Precision': fm.precision(),
                'Recall': fm.recall(),
                'F1': fm.f1_score(),
                'Demographic Parity': abs(fm.demographic_parity_difference()),
                'Equal Opportunity': abs(fm.equal_opportunity_difference()),
                'Group 0 TPR': fm.true_positive_rate(group=groups[0]),
                'Group 1 TPR': fm.true_positive_rate(group=groups[1]),
                'Group 0 FPR': fm.false_positive_rate(group=groups[0]),
                'Group 1 FPR': fm.false_positive_rate(group=groups[1])
            })
        
        metrics_df = pd.DataFrame(metrics)
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add traces for performance metrics
        for metric in ['Accuracy', 'F1']:
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['Threshold'],
                    y=metrics_df[metric],
                    name=metric,
                    line=dict(width=2)
                ),
                secondary_y=False
            )
        
        # Add traces for fairness metrics
        for metric in ['Demographic Parity', 'Equal Opportunity']:
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['Threshold'],
                    y=metrics_df[metric],
                    name=metric,
                    line=dict(dash='dash')
                ),
                secondary_y=True
            )
        
        # Add vertical line at default threshold (0.5)
        fig.add_vline(
            x=0.5,
            line=dict(color='red', dash='dash'),
            annotation_text="Default Threshold",
            annotation_position="top right"
        )
        
        # Update layout
        fig.update_layout(
            title=f"{model_name} - Threshold Analysis",
            xaxis_title="Classification Threshold",
            yaxis_title="Performance Metric",
            yaxis2_title="Fairness Metric",
            hovermode='x unified',
            height=600,
            width=1000
        )
        
        return fig


def plot_confusion_matrices(y_true: np.ndarray,
                           y_pred_dict: Dict[str, np.ndarray],
                           model_names: List[str],
                           class_names: Optional[List[str]] = None) -> go.Figure:
    """
    Plot confusion matrices for multiple models side by side.
    
    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.
    y_pred_dict : dict of array-like
        Dictionary of model predictions.
    model_names : list of str
        Names of the models.
    class_names : list of str, optional
        Names of the classes.
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Figure containing confusion matrices.
    """
    if class_names is None:
        class_names = ['Negative', 'Positive']
    
    fig = make_subplots(
        rows=1, 
        cols=len(model_names),
        subplot_titles=model_names,
        horizontal_spacing=0.1
    )
    
    for i, model_name in enumerate(model_names, 1):
        cm = confusion_matrix(y_true, y_pred_dict[model_name])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig.add_trace(
            go.Heatmap(
                z=cm_norm,
                x=class_names,
                y=class_names,
                colorscale='Blues',
                zmin=0,
                zmax=1,
                colorbar=dict(title='Normalized Count', x=1.15) if i == len(model_names) else None,
                text=cm,
                texttemplate="%{text}<br>%{z:.2f}",
                hoverinfo='text',
                hovertext=[[f"True: {true}<br>Pred: {pred}<br>Count: {count}<br>Pct: {pct:.1%}" 
                          for pred, (count, pct) in enumerate(zip(row, row_norm))] 
                         for true, (row, row_norm) in enumerate(zip(cm, cm_norm))]
            ),
            row=1, col=i
        )
        
        fig.update_xaxes(title_text="Predicted", row=1, col=i)
        fig.update_yaxes(title_text="Actual", row=1, col=i)
    
    fig.update_layout(
        title_text="Confusion Matrices",
        height=500,
        width=300 * len(model_names),
        showlegend=False
    )
    
    return fig
