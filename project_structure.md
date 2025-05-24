# Project Structure

```
afet/
├── core/                      # Core functionality
│   ├── metrics/               # Fairness and explainability metrics
│   │   ├── fairness/          # Fairness metrics implementation
│   │   └── explainability/    # Explainability metrics implementation
│   ├── visualization/         # Visualization components
│   └── mitigation/            # Bias mitigation strategies
├── datasets/                  # Sample datasets and data utilities
│   ├── synthetic/             # Synthetic datasets for testing
│   └── loaders/               # Dataset loading utilities
├── models/                    # Model wrappers and utilities
│   ├── tensorflow/            # TensorFlow model integrations
│   ├── pytorch/               # PyTorch model integrations
│   └── sklearn/               # Scikit-learn model integrations
├── dashboard/                 # Interactive dashboard components
│   ├── components/            # Reusable UI components
│   └── pages/                 # Dashboard pages
├── examples/                  # Example notebooks and scripts
│   ├── finance/               # Financial use cases
│   ├── healthcare/            # Healthcare use cases
│   ├── hiring/                # Hiring use cases
│   └── tutorials/             # Step-by-step tutorials
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
├── docs/                      # Documentation
│   ├── api/                   # API reference
│   ├── guides/                # User guides
│   └── contributing/          # Contribution guidelines
├── scripts/                   # Utility scripts
├── setup.py                   # Package installation
├── requirements.txt           # Dependencies
├── LICENSE                    # MIT License
└── README.md                  # Project overview
```

## Key Components

### Core

The `core` module contains the fundamental functionality of AFET:

- **Metrics**: Implementations of fairness metrics (e.g., demographic parity, equal opportunity) and explainability metrics (e.g., feature importance, SHAP values)
- **Visualization**: Components for visualizing fairness and explainability results
- **Mitigation**: Strategies for mitigating bias in models

### Datasets

The `datasets` module provides utilities for working with datasets:

- **Synthetic**: Synthetic datasets with controlled bias for testing
- **Loaders**: Utilities for loading and preprocessing datasets

### Models

The `models` module provides wrappers and utilities for working with different ML frameworks:

- **TensorFlow**: Integration with TensorFlow models
- **PyTorch**: Integration with PyTorch models
- **Scikit-learn**: Integration with Scikit-learn models

### Dashboard

The `dashboard` module contains the interactive dashboard components:

- **Components**: Reusable UI components
- **Pages**: Dashboard pages for different use cases

### Examples

The `examples` module contains example notebooks and scripts:

- **Finance**: Examples for financial use cases
- **Healthcare**: Examples for healthcare use cases
- **Hiring**: Examples for hiring use cases
- **Tutorials**: Step-by-step tutorials

### Tests

The `tests` module contains the test suite:

- **Unit**: Unit tests for individual components
- **Integration**: Integration tests for end-to-end workflows

### Documentation

The `docs` module contains the project documentation:

- **API**: API reference documentation
- **Guides**: User guides and tutorials
- **Contributing**: Guidelines for contributing to the project