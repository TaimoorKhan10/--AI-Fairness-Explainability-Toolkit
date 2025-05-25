Installation
===========

Prerequisites
------------
- Python 3.8 or higher
- pip (Python package manager)

Installation Methods
-------------------

### From PyPI (Recommended)

```bash
pip install ai-fairness-toolkit
```

### From Source

1. Clone the repository:
   ```bash
   git clone https://github.com/TaimoorKhan10/AI-Fairness-Explainability-Toolkit.git
   cd AI-Fairness-Explainability-Toolkit
   ```

2. Install with pip:
   ```bash
   pip install -e .
   ```

### Development Installation

For development, install with additional dependencies:

```bash
pip install -e ".[dev]"
```

Verify Installation
------------------

```python
import afet
print(f"AI Fairness Toolkit version: {afet.__version__}")
```
