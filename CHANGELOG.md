# Changelog

All notable changes to the AI Fairness and Explainability Toolkit (AFET) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- **Fairness Metrics (`core/fairness_metrics.py`):**
  - `equalized_odds()`: Implements TPR Parity and FPR Parity metrics using confusion matrix analysis
  - `treatment_equality()`: Calculates ratio of false negatives to false positives with robust division handling
  - Comprehensive metric aggregation via enhanced `get_all_fairness_metrics()` with exception handling
  - Improved edge case detection with dedicated error logging

- **Explainability Module (`core/explainability.py`):**
  - Integration with `eli5.permutation_importance` for model-agnostic feature importance
  - `partial_dependence()` calculation for visualizing feature impact on predictions
  - Error handling with `try/except` blocks during explainer initialization
  - Support for custom prediction functions via optional function parameters
  - Enhanced `get_all_explanations()` method with support for new explainers

- **Visualization Enhancements (`viz/fairness_radar_enhanced.py`):**
  - New `FairnessRadarEnhanced` class with advanced functionality:
    - Multi-model comparison via dictionary-based metrics input
    - Min-max normalization of metrics with `normalize=True` parameter
    - Theme customization with predefined color schemes
    - Threshold visualization with configurable values
    - Dual rendering support: interactive (Plotly) and static (Matplotlib)
    - `generate_metrics_table()` for pandas DataFrame comparison output

- **PyTorch Integration (`models/pytorch/`):**
  - `PyTorchModelWrapper` class implementing scikit-learn compatible interface
  - Methods: `fit()`, `predict()`, `predict_proba()`, `feature_importance()`
  - Inference optimization with configurable batch processing
  - Serialization via `save()` and `load()` class methods
  - Automatic device management (CPU/GPU) with CUDA detection
  - Support for both binary and multi-class classification

- **Model Registry Enhancement (`models/model_registry.py`):**
  - Multi-framework support with format-specific serialization:
    - `.joblib` for scikit-learn models
    - `.pt` for PyTorch models
    - `.h5` for TensorFlow models
    - `.pkl` for custom models
  - Framework-specific loading with dynamic import checking
  - Model type auto-detection based on instance properties
  - Improved metadata persistence with model type tracking
  - Filtering capability via `model_type` parameter in `list_models()`

- **Test Coverage:**
  - Unit tests for PyTorch wrapper functionality
  - Integration tests for fairness metrics with PyTorch models
  - Example script demonstrating end-to-end PyTorch workflow

### Changed
- Improved error handling with specific exception types and informative messages
- Enhanced logging with dedicated logger instances and appropriate log levels
- Optimized data processing with vectorized operations where possible
- Expanded docstrings with type annotations and detailed parameter descriptions

### Fixed
- Division by zero errors in fairness metrics with proper fallback values
- Type conversion issues in model prediction pipelines
- Initialization errors in explainability tools with better exception handling
- Inconsistent behavior between different model frameworks

## [0.1.1] - 2025-05-26
### Added
- PyPI package configuration for better README rendering
- Security policy and reporting guidelines
- Community contribution guidelines
- Pull request and issue templates
- CODEOWNERS file for automated code review assignment
- Documentation issue template

### Changed
- Updated project metadata for better PyPI integration
- Improved README with badges and better documentation
- Enhanced contribution guidelines

## [0.1.0] - 2025-05-25
### Added
- Initial release of AFET
- Core functionality for fairness assessment
- Basic visualization tools
- Documentation and examples

### Changed
- Initial project setup
- CI/CD pipeline configuration
- Repository structure
