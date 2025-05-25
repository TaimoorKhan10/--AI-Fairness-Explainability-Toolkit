"""
Synthetic medical diagnosis dataset generator
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class MedicalDataset:
    """
    Generate synthetic medical diagnosis dataset
    """
    
    def __init__(self, 
                 n_samples: int = 1000,
                 random_state: int = 42):
        """
        Initialize dataset generator
        """
        self.n_samples = n_samples
        self.random_state = random_state
        
    def generate_features(self) -> pd.DataFrame:
        """
        Generate synthetic medical features
        """
        np.random.seed(self.random_state)
        
        # Medical features
        age = np.random.normal(50, 15, self.n_samples)
        blood_pressure = np.random.normal(120, 20, self.n_samples)
        cholesterol = np.random.normal(200, 30, self.n_samples)
        glucose = np.random.normal(90, 15, self.n_samples)
        
        # Demographic features
        gender = np.random.choice(['Male', 'Female'], self.n_samples)
        race = np.random.choice(['White', 'Black', 'Asian', 'Other'], self.n_samples)
        insurance_type = np.random.choice(['Private', 'Public', 'None'], self.n_samples)
        
        # Lifestyle factors
        smoking = np.random.choice([0, 1], self.n_samples, p=[0.8, 0.2])
        alcohol_consumption = np.random.normal(2, 1, self.n_samples)
        physical_activity = np.random.normal(3, 1, self.n_samples)
        
        # Create DataFrame
        df = pd.DataFrame({
            'age': age,
            'blood_pressure': blood_pressure,
            'cholesterol': cholesterol,
            'glucose': glucose,
            'gender': gender,
            'race': race,
            'insurance_type': insurance_type,
            'smoking': smoking,
            'alcohol_consumption': alcohol_consumption,
            'physical_activity': physical_activity
        })
        
        return df
    
    def generate_target(self, 
                       X: pd.DataFrame,
                       bias_factor: float = 0.1) -> pd.Series:
        """
        Generate synthetic diagnosis outcome with potential bias
        """
        np.random.seed(self.random_state)
        
        # Base probability based on medical features
        base_prob = (
            0.01 * X['age'] +
            0.005 * X['blood_pressure'] +
            0.01 * X['cholesterol'] +
            0.01 * X['glucose'] -
            0.5 * X['smoking'] +
            0.1 * X['physical_activity']
        )
        
        # Add bias based on protected attributes
        bias = np.zeros(self.n_samples)
        
        # Insurance bias
        bias[X['insurance_type'] == 'None'] += bias_factor * 2
        
        # Race bias
        bias[X['race'] == 'Black'] += bias_factor
        
        # Final probability
        final_prob = np.clip(base_prob + bias, 0, 1)
        
        # Generate binary target
        y = np.random.binomial(1, final_prob)
        
        return pd.Series(y, name='diagnosis_positive')
    
    def generate_dataset(self,
                        test_size: float = 0.2,
                        bias_factor: float = 0.1) -> Dict[str, pd.DataFrame]:
        """
        Generate complete dataset with train/test split
        """
        # Generate features
        X = self.generate_features()
        
        # Generate target with bias
        y = self.generate_target(X, bias_factor)
        
        # One-hot encode categorical variables
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        # Scale continuous features
        scaler = StandardScaler()
        X_scaled = X_encoded.copy()
        X_scaled[['age', 'blood_pressure', 'cholesterol', 'glucose',
                 'alcohol_consumption', 'physical_activity']] = scaler.fit_transform(
            X_scaled[['age', 'blood_pressure', 'cholesterol', 'glucose',
                     'alcohol_consumption', 'physical_activity']]
        )
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'original_features': X,
            'target': y
        }

# Example usage
if __name__ == "__main__":
    generator = MedicalDataset(n_samples=10000)
    generator.generate_dataset()
