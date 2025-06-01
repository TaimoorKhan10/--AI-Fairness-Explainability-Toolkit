"""
Visualization module for the AI Fairness and Explainability Toolkit (AFET).

This module provides various visualization tools for analyzing and explaining
machine learning models with a focus on fairness and interpretability.
"""

from .interactive_plots import FairnessDashboard, ThresholdAnalysis, plot_confusion_matrices
from .fairness_radar import FairnessRadar, plot_fairness_radar
from .fairness_radar_enhanced import FairnessRadarEnhanced
from .intersectional_heatmap import IntersectionalHeatmap, plot_intersectional_heatmap, plot_multi_metric_heatmaps

__all__ = [
    'FairnessDashboard',
    'ThresholdAnalysis',
    'plot_confusion_matrices',
    'FairnessRadar',
    'plot_fairness_radar',
    'FairnessRadarEnhanced',
    'IntersectionalHeatmap',
    'plot_intersectional_heatmap',
    'plot_multi_metric_heatmaps'
]
