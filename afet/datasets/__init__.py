"""
Datasets module for the AI Fairness and Explainability Toolkit (AFET).

This module provides dataset utilities for loading, generating, and preprocessing
data for fairness analysis and evaluation.
"""

from .synthetic_generator import (
    BiasedDatasetGenerator,
    generate_credit_dataset,
    generate_hiring_dataset,
    generate_healthcare_dataset
)

__all__ = [
    'BiasedDatasetGenerator',
    'generate_credit_dataset',
    'generate_hiring_dataset',
    'generate_healthcare_dataset'
]
