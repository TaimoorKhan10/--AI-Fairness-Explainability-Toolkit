"""
Model registry for AFET
"""

from typing import Dict, Type, Any
import json
import os
from pathlib import Path
import joblib
from sklearn.base import BaseEstimator


class ModelRegistry:
    """
    Registry for managing machine learning models
    """
    
    def __init__(self, model_dir: str = None):
        """
        Initialize model registry
        """
        self.model_dir = model_dir or str(Path.home() / '.afet' / 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        self._models = {}
        self._load_models()
    
    def _load_models(self) -> None:
        """
        Load existing models from disk
        """
        for model_file in os.listdir(self.model_dir):
            if model_file.endswith('.joblib'):
                model_path = os.path.join(self.model_dir, model_file)
                try:
                    model = joblib.load(model_path)
                    self._models[model_file[:-7]] = model
                except Exception as e:
                    print(f"Warning: Could not load model {model_file}: {str(e)}")
    
    def register_model(self, 
                      model: BaseEstimator,
                      model_id: str,
                      metadata: Dict[str, Any]) -> None:
        """
        Register a new model
        """
        if model_id in self._models:
            raise ValueError(f"Model with ID {model_id} already exists")
            
        # Save model
        model_path = os.path.join(self.model_dir, f"{model_id}.joblib")
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata_path = os.path.join(self.model_dir, f"{model_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        # Add to registry
        self._models[model_id] = model
    
    def get_model(self, model_id: str) -> BaseEstimator:
        """
        Get a registered model
        """
        if model_id not in self._models:
            raise KeyError(f"Model {model_id} not found")
            
        return self._models[model_id]
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """
        List all registered models with metadata
        """
        models = {}
        for model_id in self._models:
            metadata_path = os.path.join(self.model_dir, f"{model_id}_metadata.json")
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                models[model_id] = metadata
            except Exception as e:
                print(f"Warning: Could not load metadata for {model_id}: {str(e)}")
        
        return models
    
    def delete_model(self, model_id: str) -> None:
        """
        Delete a registered model
        """
        if model_id not in self._models:
            raise KeyError(f"Model {model_id} not found")
            
        # Delete model file
        model_path = os.path.join(self.model_dir, f"{model_id}.joblib")
        metadata_path = os.path.join(self.model_dir, f"{model_id}_metadata.json")
        
        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        
        # Remove from registry
        del self._models[model_id]
    
    def update_metadata(self, 
                       model_id: str,
                       metadata: Dict[str, Any]) -> None:
        """
        Update model metadata
        """
        if model_id not in self._models:
            raise KeyError(f"Model {model_id} not found")
            
        metadata_path = os.path.join(self.model_dir, f"{model_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

# Initialize global registry
registry = ModelRegistry()
