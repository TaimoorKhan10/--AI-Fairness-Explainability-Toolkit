"""
Integration test for PyTorch model with fairness metrics and visualizations
"""

import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from afet.models.pytorch.model_wrapper import PyTorchModelWrapper
from afet.core.fairness_metrics import (
    demographic_parity,
    equal_opportunity,
    disparate_impact,
    statistical_parity_difference,
    equalized_odds,
    treatment_equality,
    get_all_fairness_metrics
)
from afet.viz.fairness_radar_enhanced import FairnessRadarEnhanced


class SimpleNN(nn.Module):
    """Simple neural network for binary classification"""
    def __init__(self, input_size, hidden_size=64):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


class TestPytorchFairnessIntegration(unittest.TestCase):
    """Integration test for PyTorch with fairness metrics"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data and models once for all tests"""
        try:
            # Load Adult dataset (if not available, test will be skipped)
            adult = fetch_openml(data_id=1590, as_frame=True)
            cls.dataset_available = True
            
            df = adult.data
            df['target'] = adult.target
            
            # Select sensitive attribute (sex)
            df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
            
            # Prepare features and target
            X = df.drop(['target', 'sex', 'race', 'native-country', 'relationship', 'education'], axis=1)
            y = (df['target'] == '>50K').astype(int)
            sensitive_attr = df['sex']
            
            # Encode categorical features
            cat_cols = X.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            
            # Convert to numeric
            X = X.astype(float)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
                X_scaled, y, sensitive_attr, test_size=0.2, random_state=42
            )
            
            cls.X_train, cls.X_test = X_train, X_test
            cls.y_train, cls.y_test = y_train.values, y_test.values
            cls.sens_train, cls.sens_test = sens_train.values, sens_test.values
            
            # Create and train a PyTorch model
            input_size = X_train.shape[1]
            model = SimpleNN(input_size)
            
            # Convert data to PyTorch tensors
            X_tensor = torch.FloatTensor(X_train)
            y_tensor = torch.FloatTensor(y_train.values.reshape(-1, 1))
            
            # Train the model
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Training loop (minimal training for test purposes)
            for epoch in range(5):
                optimizer.zero_grad()
                outputs = model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
            
            # Create model wrapper
            cls.model_wrapper = PyTorchModelWrapper(model)
            
            # For comparison, create a slightly different model
            model2 = SimpleNN(input_size)
            # Train with slightly different parameters
            optimizer2 = optim.SGD(model2.parameters(), lr=0.01)
            for epoch in range(5):
                optimizer2.zero_grad()
                outputs = model2(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer2.step()
                
            cls.model_wrapper2 = PyTorchModelWrapper(model2)
            
        except Exception as e:
            print(f"Error setting up test data: {e}")
            cls.dataset_available = False
    
    def setUp(self):
        """Skip tests if dataset is not available"""
        if not self.dataset_available:
            self.skipTest("Adult dataset not available")
    
    def test_fairness_metrics_with_pytorch(self):
        """Test fairness metrics with PyTorch model"""
        # Get predictions
        y_pred = self.model_wrapper.predict(self.X_test)
        
        # Calculate individual fairness metrics
        dp = demographic_parity(self.y_test, y_pred, self.sens_test)
        eo = equal_opportunity(self.y_test, y_pred, self.sens_test)
        di = disparate_impact(self.y_test, y_pred, self.sens_test)
        spd = statistical_parity_difference(self.y_test, y_pred, self.sens_test)
        eq_odds = equalized_odds(self.y_test, y_pred, self.sens_test)
        te = treatment_equality(self.y_test, y_pred, self.sens_test)
        
        # Check metric values
        self.assertIsInstance(dp, float)
        self.assertIsInstance(eo, float)
        self.assertIsInstance(di, float)
        self.assertIsInstance(spd, float)
        self.assertIsInstance(eq_odds, dict)
        self.assertIsInstance(te, float)
        
        # Test comprehensive metrics
        all_metrics = get_all_fairness_metrics(self.y_test, y_pred, self.sens_test)
        self.assertIsInstance(all_metrics, dict)
        self.assertIn('demographic_parity', all_metrics)
        self.assertIn('equal_opportunity', all_metrics)
        self.assertIn('disparate_impact', all_metrics)
        self.assertIn('statistical_parity_difference', all_metrics)
        self.assertIn('equalized_odds', all_metrics)
        self.assertIn('treatment_equality', all_metrics)
    
    def test_fairness_visualization_with_pytorch(self):
        """Test fairness visualization with PyTorch models"""
        # Get predictions from both models
        y_pred1 = self.model_wrapper.predict(self.X_test)
        y_pred2 = self.model_wrapper2.predict(self.X_test)
        
        # Calculate metrics for both models
        metrics1 = get_all_fairness_metrics(self.y_test, y_pred1, self.sens_test)
        metrics2 = get_all_fairness_metrics(self.y_test, y_pred2, self.sens_test)
        
        # Prepare metrics for visualization
        model_metrics = {
            'PyTorch Model 1': metrics1,
            'PyTorch Model 2': metrics2
        }
        
        # Create fairness radar visualization
        radar = FairnessRadarEnhanced()
        fig = radar.plot(
            model_metrics, 
            normalize=True,
            plot_type='both',
            save_path=None,  # Don't save for test
            show_plot=False  # Don't show for test
        )
        
        # Check that the figure was created
        self.assertIsNotNone(fig)
        
        # Create comparison table
        table = radar.generate_metrics_table(model_metrics)
        self.assertIsInstance(table, pd.DataFrame)
        self.assertEqual(len(table), len(metrics1))
        
        # Check model comparison data
        comparison = radar.compare_models(model_metrics)
        self.assertIsInstance(comparison, dict)
        self.assertEqual(len(comparison), 2)  # Two models


if __name__ == '__main__':
    unittest.main()
