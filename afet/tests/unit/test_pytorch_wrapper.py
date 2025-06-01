"""
Unit tests for PyTorch model wrapper
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from afet.models.pytorch.model_wrapper import PyTorchModelWrapper


class SimpleNN(nn.Module):
    """
    Simple neural network for testing purposes
    """
    def __init__(self, input_dim, hidden_dim=32, output_dim=1):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.output_activation = nn.Sigmoid() if output_dim == 1 else nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        if hasattr(self, 'output_activation'):
            x = self.output_activation(x)
        return x


class TestPyTorchModelWrapper(unittest.TestCase):
    """
    Test case for PyTorchModelWrapper
    """
    
    def setUp(self):
        """
        Set up test data and models
        """
        # Create a binary classification dataset
        X, y = make_classification(
            n_samples=1000, 
            n_features=10, 
            n_informative=5, 
            n_redundant=2,
            random_state=42
        )
        
        # Split dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
        # Create a simple PyTorch model for binary classification
        input_dim = self.X_train.shape[1]
        self.binary_model = SimpleNN(input_dim=input_dim, output_dim=1)
        
        # Create a simple PyTorch model for multi-class classification
        self.X_multi, self.y_multi = make_classification(
            n_samples=1000, 
            n_features=10, 
            n_informative=5, 
            n_redundant=2,
            n_classes=3,
            random_state=42
        )
        self.X_multi_train, self.X_multi_test, self.y_multi_train, self.y_multi_test = train_test_split(
            self.X_multi, self.y_multi, test_size=0.2, random_state=42
        )
        self.X_multi_train = scaler.fit_transform(self.X_multi_train)
        self.X_multi_test = scaler.transform(self.X_multi_test)
        
        self.multi_model = SimpleNN(input_dim=input_dim, output_dim=3)
    
    def train_binary_model(self):
        """
        Helper to train the binary classification model
        """
        X_tensor = torch.FloatTensor(self.X_train)
        y_tensor = torch.FloatTensor(self.y_train.reshape(-1, 1))
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.binary_model.parameters(), lr=0.01)
        
        for _ in range(50):  # Just a few epochs for testing
            optimizer.zero_grad()
            outputs = self.binary_model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
        # Create the wrapper with the trained model
        return PyTorchModelWrapper(self.binary_model)
    
    def train_multi_model(self):
        """
        Helper to train the multi-class classification model
        """
        X_tensor = torch.FloatTensor(self.X_multi_train)
        y_tensor = torch.LongTensor(self.y_multi_train)
        
        # Modify the model for multi-class classification
        self.multi_model.output_activation = nn.Softmax(dim=1)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.multi_model.parameters(), lr=0.01)
        
        for _ in range(50):  # Just a few epochs for testing
            optimizer.zero_grad()
            outputs = self.multi_model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        # Create the wrapper with the trained model
        return PyTorchModelWrapper(
            self.multi_model, 
            loss_fn=nn.CrossEntropyLoss(), 
            multi_class=True
        )
    
    def test_initialization(self):
        """
        Test that the wrapper initializes correctly
        """
        wrapper = PyTorchModelWrapper(self.binary_model)
        self.assertIsInstance(wrapper, PyTorchModelWrapper)
        self.assertEqual(wrapper.model, self.binary_model)
        
        # Test with custom parameters
        wrapper = PyTorchModelWrapper(
            self.binary_model,
            optimizer=optim.SGD(self.binary_model.parameters(), lr=0.1),
            loss_fn=nn.MSELoss(),
            device='cpu',
            batch_size=64
        )
        self.assertIsInstance(wrapper.optimizer, optim.SGD)
        self.assertIsInstance(wrapper.loss_fn, nn.MSELoss)
        self.assertEqual(wrapper.batch_size, 64)
    
    def test_predict_binary(self):
        """
        Test prediction for binary classification
        """
        wrapper = self.train_binary_model()
        
        # Test predict method
        y_pred = wrapper.predict(self.X_test)
        self.assertEqual(y_pred.shape, self.y_test.shape)
        self.assertTrue(np.all((y_pred == 0) | (y_pred == 1)))
        
        # Check accuracy is better than random
        accuracy = accuracy_score(self.y_test, y_pred)
        self.assertGreater(accuracy, 0.6)
    
    def test_predict_proba_binary(self):
        """
        Test probability prediction for binary classification
        """
        wrapper = self.train_binary_model()
        
        # Test predict_proba method
        y_proba = wrapper.predict_proba(self.X_test)
        self.assertEqual(y_proba.shape, (len(self.X_test), 2))
        
        # Check probabilities sum to 1
        self.assertTrue(np.allclose(np.sum(y_proba, axis=1), 1.0))
    
    def test_multi_class(self):
        """
        Test multi-class classification
        """
        wrapper = self.train_multi_model()
        
        # Test predict method
        y_pred = wrapper.predict(self.X_multi_test)
        self.assertEqual(y_pred.shape, self.y_multi_test.shape)
        
        # Check classes are in the expected range
        self.assertTrue(np.all((y_pred >= 0) & (y_pred <= 2)))
        
        # Test predict_proba method
        y_proba = wrapper.predict_proba(self.X_multi_test)
        self.assertEqual(y_proba.shape, (len(self.X_multi_test), 3))
        
        # Check probabilities sum to 1
        self.assertTrue(np.allclose(np.sum(y_proba, axis=1), 1.0))
    
    def test_save_load(self):
        """
        Test model persistence
        """
        import tempfile
        import os
        
        wrapper = self.train_binary_model()
        
        # Save the model
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            model_path = tmp.name
        
        wrapper.save(model_path)
        self.assertTrue(os.path.exists(model_path))
        
        # Load the model
        new_wrapper = PyTorchModelWrapper.load(model_path)
        self.assertIsInstance(new_wrapper, PyTorchModelWrapper)
        
        # Check predictions match
        X_sample = self.X_test[:10]
        original_preds = wrapper.predict(X_sample)
        loaded_preds = new_wrapper.predict(X_sample)
        self.assertTrue(np.array_equal(original_preds, loaded_preds))
        
        # Clean up
        os.remove(model_path)
    
    def test_feature_importance(self):
        """
        Test feature importance calculation
        """
        wrapper = self.train_binary_model()
        
        # Calculate feature importance
        importances = wrapper.feature_importance(self.X_test, self.y_test)
        self.assertEqual(len(importances), self.X_test.shape[1])
        
        # Check importance values are reasonable
        self.assertTrue(np.all(importances >= 0))


if __name__ == '__main__':
    unittest.main()
