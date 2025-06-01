"""
Model registry for AFET
"""

from typing import Dict, Type, Any, Union, Optional, List
import json
import os
import importlib
from pathlib import Path
import joblib
import pickle
import logging
from sklearn.base import BaseEstimator
from .pytorch.model_wrapper import PyTorchModelWrapper

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for managing machine learning models from different frameworks
    """
    
    # Supported model types
    MODEL_TYPES = {
        'sklearn': '.joblib',
        'pytorch': '.pt',
        'tensorflow': '.h5',
        'custom': '.pkl'
    }
    
    def __init__(self, model_dir: str = None):
        """
        Initialize model registry
        
        Args:
            model_dir: Directory to store models, defaults to ~/.afet/models
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
            # Check each supported file extension
            for model_type, ext in self.MODEL_TYPES.items():
                if model_file.endswith(ext):
                    model_path = os.path.join(self.model_dir, model_file)
                    model_id = model_file[:-len(ext)]
                    try:
                        # Load metadata first to determine how to load the model
                        metadata = self._load_metadata(model_id)
                        if metadata and 'model_type' in metadata:
                            # Use the model type from metadata
                            model_type = metadata['model_type']
                        
                        # Load model based on type
                        model = self._load_model_by_type(model_path, model_type)
                        if model is not None:
                            self._models[model_id] = model
                            logger.info(f"Loaded {model_type} model: {model_id}")
                    except Exception as e:
                        logger.warning(f"Could not load model {model_file}: {str(e)}")
    
    def _load_model_by_type(self, model_path: str, model_type: str) -> Any:
        """
        Load a model based on its type
        
        Args:
            model_path: Path to the model file
            model_type: Type of model ('sklearn', 'pytorch', 'tensorflow', 'custom')
            
        Returns:
            Loaded model instance
        """
        try:
            if model_type == 'sklearn':
                return joblib.load(model_path)
            elif model_type == 'pytorch':
                import torch
                return torch.load(model_path)
            elif model_type == 'tensorflow':
                # Load TensorFlow model if TF is available
                tf_spec = importlib.util.find_spec('tensorflow')
                if tf_spec is not None:
                    import tensorflow as tf
                    return tf.keras.models.load_model(model_path)
                else:
                    logger.error("TensorFlow not installed, cannot load model")
                    return None
            else:  # custom or unknown type
                return pickle.load(open(model_path, 'rb'))
        except Exception as e:
            logger.error(f"Error loading {model_type} model: {str(e)}")
            return None
    
    def _load_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Load metadata for a model
        
        Args:
            model_id: ID of the model
            
        Returns:
            Metadata dictionary or None if not found
        """
        metadata_path = os.path.join(self.model_dir, f"{model_id}_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata for {model_id}: {str(e)}")
        return None
    
    def register_model(self, 
                      model: Any,
                      model_id: str,
                      metadata: Dict[str, Any],
                      model_type: str = None) -> None:
        """
        Register a new model
        
        Args:
            model: Model instance to register
            model_id: Unique identifier for the model
            metadata: Dictionary of metadata about the model
            model_type: Type of model ('sklearn', 'pytorch', 'tensorflow', 'custom')
                       If not specified, will try to detect from model instance
        """
        if model_id in self._models:
            raise ValueError(f"Model with ID {model_id} already exists")
        
        # Determine model type if not provided
        if model_type is None:
            if isinstance(model, BaseEstimator):
                model_type = 'sklearn'
            elif isinstance(model, PyTorchModelWrapper):
                model_type = 'pytorch'
            elif 'tensorflow' in str(type(model)):
                model_type = 'tensorflow'
            else:
                model_type = 'custom'
        
        # Add model type to metadata
        metadata['model_type'] = model_type
        
        # Get file extension for this model type
        ext = self.MODEL_TYPES.get(model_type, '.pkl')
        model_path = os.path.join(self.model_dir, f"{model_id}{ext}")
        
        # Save model based on type
        try:
            if model_type == 'sklearn':
                joblib.dump(model, model_path)
            elif model_type == 'pytorch':
                if hasattr(model, 'save'):
                    model.save(model_path)
                else:
                    import torch
                    torch.save(model, model_path)
            elif model_type == 'tensorflow':
                if hasattr(model, 'save'):
                    model.save(model_path)
                else:
                    pickle.dump(model, open(model_path, 'wb'))
            else:  # custom
                pickle.dump(model, open(model_path, 'wb'))
                
            # Save metadata
            metadata_path = os.path.join(self.model_dir, f"{model_id}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            # Add to registry
            self._models[model_id] = model
            logger.info(f"Registered {model_type} model: {model_id}")
            
        except Exception as e:
            logger.error(f"Error registering model {model_id}: {str(e)}")
            raise
    
    def get_model(self, model_id: str) -> Any:
        """
        Get a registered model
        
        Args:
            model_id: ID of the model to retrieve
            
        Returns:
            Model instance
            
        Raises:
            KeyError: If model not found
        """
        if model_id not in self._models:
            # Try to load it first in case it was added after initialization
            metadata = self._load_metadata(model_id)
            if metadata and 'model_type' in metadata:
                model_type = metadata['model_type']
                ext = self.MODEL_TYPES.get(model_type, '.pkl')
                model_path = os.path.join(self.model_dir, f"{model_id}{ext}")
                if os.path.exists(model_path):
                    model = self._load_model_by_type(model_path, model_type)
                    if model is not None:
                        self._models[model_id] = model
                        return model
            
            raise KeyError(f"Model {model_id} not found")
            
        return self._models[model_id]
    
    def list_models(self, model_type: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        List all registered models with metadata
        
        Args:
            model_type: Optional filter by model type
            
        Returns:
            Dictionary of model IDs and their metadata
        """
        models = {}
        
        # Check all files in the model directory to find all models
        for filename in os.listdir(self.model_dir):
            if filename.endswith('_metadata.json'):
                model_id = filename[:-14]  # Remove '_metadata.json'
                metadata = self._load_metadata(model_id)
                
                if metadata is None:
                    continue
                    
                # Filter by model type if specified
                if model_type is not None and metadata.get('model_type') != model_type:
                    continue
                    
                models[model_id] = metadata
        
        return models
    
    def delete_model(self, model_id: str) -> None:
        """
        Delete a registered model
        
        Args:
            model_id: ID of the model to delete
            
        Raises:
            KeyError: If model not found
        """
        # Get metadata to determine file extension
        metadata = self._load_metadata(model_id)
        model_type = metadata.get('model_type', 'sklearn') if metadata else 'sklearn'
        ext = self.MODEL_TYPES.get(model_type, '.pkl')
        
        # Check if model exists
        model_path = os.path.join(self.model_dir, f"{model_id}{ext}")
        metadata_path = os.path.join(self.model_dir, f"{model_id}_metadata.json")
        
        if not os.path.exists(model_path) and model_id not in self._models:
            # Try other extensions before giving up
            found = False
            for ext_type in self.MODEL_TYPES.values():
                alt_path = os.path.join(self.model_dir, f"{model_id}{ext_type}")
                if os.path.exists(alt_path):
                    model_path = alt_path
                    found = True
                    break
            
            if not found:
                raise KeyError(f"Model {model_id} not found")
        
        # Delete model file
        if os.path.exists(model_path):
            os.remove(model_path)
            logger.info(f"Deleted model file: {model_path}")
            
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            logger.info(f"Deleted metadata file: {metadata_path}")
        
        # Remove from registry
        if model_id in self._models:
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
