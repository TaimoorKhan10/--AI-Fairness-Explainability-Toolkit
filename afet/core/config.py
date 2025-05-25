"""
Configuration and constants for AFET
"""

import os
from pathlib import Path
from typing import Dict, Any
import yaml


class Config:
    """
    Configuration manager for AFET
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration
        """
        self.config_path = config_path or os.path.join(
            Path.home(), '.afet', 'config.yaml'
        )
        self._config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file
        """
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._create_default_config()
        
    def _create_default_config(self) -> Dict[str, Any]:
        """
        Create default configuration
        """
        default_config = {
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'models': {
                'cache_dir': os.path.join(Path.home(), '.afet', 'models'),
                'max_cache_size': 1024 * 1024 * 1024  # 1GB
            },
            'dashboard': {
                'port': 8501,
                'host': '0.0.0.0',
                'theme': 'light'
            },
            'fairness': {
                'default_threshold': 0.1,
                'metrics': [
                    'demographic_parity',
                    'equalized_odds',
                    'predictive_parity',
                    'calibration'
                ]
            }
        }
        
        # Create config directory
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Save default config
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f)
        
        return default_config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        """
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value
        """
        self._config[key] = value
        self._save()
    
    def _save(self) -> None:
        """
        Save configuration to file
        """
        with open(self.config_path, 'w') as f:
            yaml.dump(self._config, f)

# Constants
class Constants:
    """
    Common constants used throughout AFET
    """
    
    # Model types
    MODEL_TYPES = {
        'classification': 'Classification',
        'regression': 'Regression',
        'clustering': 'Clustering'
    }
    
    # Metric types
    METRIC_TYPES = {
        'performance': ['accuracy', 'roc_auc', 'f1_score'],
        'fairness': ['demographic_parity', 'equalized_odds', 'predictive_parity'],
        'explainability': ['shap', 'lime', 'ebm']
    }
    
    # Protected attributes
    PROTECTED_ATTRIBUTES = [
        'gender', 'race', 'ethnicity', 'age', 'disability', 'national_origin',
        'religion', 'sexual_orientation'
    ]
    
    # Default thresholds
    DEFAULT_THRESHOLDS = {
        'demographic_parity': 0.1,
        'equalized_odds': 0.1,
        'predictive_parity': 0.1,
        'calibration': 0.05
    }

# Initialize global config
config = Config()
