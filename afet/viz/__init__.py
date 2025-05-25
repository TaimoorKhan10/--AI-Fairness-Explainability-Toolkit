"""
Visualization module for the AI Fairness and Explainability Toolkit (AFET).

This module provides various visualization tools for analyzing and explaining
machine learning models with a focus on fairness and interpretability.
"""

from .interactive_plots import FairnessDashboard, ThresholdAnalysis, plot_confusion_matrices

__all__ = [
    'FairnessDashboard',
    'ThresholdAnalysis',
    'plot_confusion_matrices'
]
