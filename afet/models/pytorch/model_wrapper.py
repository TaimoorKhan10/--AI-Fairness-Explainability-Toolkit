"""
PyTorch model wrapper for AFET
"""

from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import numpy as np
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class PyTorchModelWrapper:
    """
    Wrapper for PyTorch models to make them compatible with AFET
    
    This wrapper adapts PyTorch models to provide a scikit-learn-like interface
    for prediction and probability estimation, which is required by AFET's
    fairness and explainability tools.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: Optional[str] = None,
                 preprocessing_fn: Optional[Callable] = None,
                 class_names: Optional[List[str]] = None,
                 feature_names: Optional[List[str]] = None,
                 batch_size: int = 32):
        """
        Initialize PyTorch model wrapper
        
        Args:
            model: PyTorch model (nn.Module)
            device: Device to run the model on ('cpu', 'cuda', or None for auto-detection)
            preprocessing_fn: Optional function to preprocess input data
            class_names: Optional list of class names
            feature_names: Optional list of feature names
            batch_size: Batch size for inference
        """
        self.model = model
        self.preprocessing_fn = preprocessing_fn
        self.class_names = class_names
        self.feature_names = feature_names
        self.batch_size = batch_size
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        logger.info(f"PyTorch model wrapper initialized on device: {self.device}")
    
    def _prepare_input(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Prepare input data for the model
        
        Args:
            X: Input data as numpy array or torch tensor
            
        Returns:
            Torch tensor ready for model input
        """
        # Apply preprocessing if provided
        if self.preprocessing_fn is not None:
            X = self.preprocessing_fn(X)
        
        # Convert to torch tensor if needed
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        
        # Move to device
        X = X.to(self.device)
        
        return X
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Make predictions with the model
        
        Args:
            X: Input data
            
        Returns:
            Class predictions as numpy array
        """
        # Ensure model is in evaluation mode
        self.model.eval()
        
        try:
            # Get predictions in batches to avoid memory issues
            all_preds = []
            
            with torch.no_grad():
                # Process in batches
                for i in range(0, len(X), self.batch_size):
                    batch = self._prepare_input(X[i:i+self.batch_size])
                    logits = self.model(batch)
                    
                    # Get predicted class (argmax for multi-class, threshold for binary)
                    if logits.shape[1] > 1:  # Multi-class
                        preds = torch.argmax(logits, dim=1)
                    else:  # Binary
                        preds = (logits > 0.5).long()
                    
                    all_preds.append(preds.cpu().numpy())
            
            return np.concatenate(all_preds)
        
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Get probability estimates for each class
        
        Args:
            X: Input data
            
        Returns:
            Probability estimates as numpy array
        """
        # Ensure model is in evaluation mode
        self.model.eval()
        
        try:
            # Get probabilities in batches to avoid memory issues
            all_probs = []
            
            with torch.no_grad():
                # Process in batches
                for i in range(0, len(X), self.batch_size):
                    batch = self._prepare_input(X[i:i+self.batch_size])
                    logits = self.model(batch)
                    
                    # Apply softmax for multi-class, sigmoid for binary
                    if logits.shape[1] > 1:  # Multi-class
                        probs = torch.softmax(logits, dim=1)
                    else:  # Binary
                        probs = torch.sigmoid(logits)
                        # For binary, return both 0 and 1 class probabilities
                        if probs.shape[1] == 1:
                            probs = torch.cat([1 - probs, probs], dim=1)
                    
                    all_probs.append(probs.cpu().numpy())
            
            return np.concatenate(all_probs)
        
        except Exception as e:
            logger.error(f"Error during probability estimation: {str(e)}")
            raise
    
    def feature_importances(self) -> Optional[np.ndarray]:
        """
        Get feature importances if available
        
        Returns:
            Feature importance scores or None if not available
        """
        # Check if model has feature importances
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        
        # Check if it's a linear model with accessible weights
        try:
            # Try to get weights from the first layer
            first_layer = None
            for module in self.model.modules():
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    first_layer = module
                    break
            
            if first_layer is not None:
                # For linear layer, use absolute weights as importance
                weights = first_layer.weight.data.abs().mean(dim=0).cpu().numpy()
                return weights
        except Exception as e:
            logger.warning(f"Could not extract feature importances: {str(e)}")
        
        return None
    
    def save(self, path: str):
        """
        Save the PyTorch model
        
        Args:
            path: Path to save the model
        """
        try:
            # Save model state dict and metadata
            state = {
                'model_state_dict': self.model.state_dict(),
                'class_names': self.class_names,
                'feature_names': self.feature_names
            }
            torch.save(state, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load(cls, 
             path: str, 
             model_class: nn.Module,
             model_args: Optional[Dict] = None,
             device: Optional[str] = None,
             preprocessing_fn: Optional[Callable] = None,
             batch_size: int = 32) -> 'PyTorchModelWrapper':
        """
        Load a saved PyTorch model
        
        Args:
            path: Path to load the model from
            model_class: PyTorch model class
            model_args: Arguments to initialize the model class
            device: Device to run the model on
            preprocessing_fn: Preprocessing function
            batch_size: Batch size for inference
            
        Returns:
            Loaded PyTorchModelWrapper instance
        """
        try:
            # Load state dict and metadata
            state = torch.load(path, map_location='cpu')
            
            # Initialize model
            model_args = model_args or {}
            model = model_class(**model_args)
            model.load_state_dict(state['model_state_dict'])
            
            # Create wrapper
            wrapper = cls(
                model=model,
                device=device,
                preprocessing_fn=preprocessing_fn,
                class_names=state.get('class_names'),
                feature_names=state.get('feature_names'),
                batch_size=batch_size
            )
            
            logger.info(f"Model loaded from {path}")
            return wrapper
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
