"""
Synthetic loan dataset generator for AFET examples
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class SyntheticLoanDataset:
    """
    Generate synthetic loan dataset with potential fairness issues
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
        Generate synthetic features
        """
        np.random.seed(self.random_state)
        
        # Continuous features
        income = np.random.normal(50000, 15000, self.n_samples)
        credit_score = np.random.normal(700, 100, self.n_samples)
        debt_ratio = np.random.normal(0.3, 0.1, self.n_samples)
        
        # Categorical features
        gender = np.random.choice(['Male', 'Female'], self.n_samples)
        race = np.random.choice(['White', 'Black', 'Asian', 'Other'], self.n_samples)
        education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], self.n_samples)
        
        # Create DataFrame
        df = pd.DataFrame({
            'income': income,
            'credit_score': credit_score,
            'debt_ratio': debt_ratio,
            'gender': gender,
            'race': race,
            'education': education
        })
        
        return df
    
    def generate_target(self, 
                       X: pd.DataFrame,
                       bias_factor: float = 0.1) -> pd.Series:
        """
        Generate synthetic target variable with potential bias
        """
        np.random.seed(self.random_state)
        
        # Base probability based on features
        base_prob = (
            0.0001 * X['income'] +
            0.005 * X['credit_score'] -
            10 * X['debt_ratio']
        )
        
        # Add bias based on protected attributes
        bias = np.zeros(self.n_samples)
        
        # Gender bias
        bias[X['gender'] == 'Female'] += bias_factor
        
        # Race bias
        bias[X['race'] == 'Black'] += bias_factor * 2
        
        # Final probability
        final_prob = np.clip(base_prob + bias, 0, 1)
        
        # Generate binary target
        y = np.random.binomial(1, final_prob)
        
        return pd.Series(y, name='loan_approved')
    
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
        X_scaled[['income', 'credit_score', 'debt_ratio']] = scaler.fit_transform(
            X_scaled[['income', 'credit_score', 'debt_ratio']]
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
    
    def save_dataset(self, 
                    filepath: str,
                    test_size: float = 0.2,
                    bias_factor: float = 0.1):
        """
        Save dataset to CSV files
        """
        dataset = self.generate_dataset(test_size, bias_factor)
        
        # Save train data
        train_data = pd.concat([
            dataset['X_train'],
            dataset['y_train'].rename('loan_approved')
        ], axis=1)
        train_data.to_csv(f"{filepath}_train.csv", index=False)
        
        # Save test data
        test_data = pd.concat([
            dataset['X_test'],
            dataset['y_test'].rename('loan_approved')
        ], axis=1)
        test_data.to_csv(f"{filepath}_test.csv", index=False)
        
        # Save original features
        original_data = pd.concat([
            dataset['original_features'],
            dataset['target'].rename('loan_approved')
        ], axis=1)
        original_data.to_csv(f"{filepath}_original.csv", index=False)

# Example usage
if __name__ == "__main__":
    generator = SyntheticLoanDataset(n_samples=10000)
    generator.save_dataset("loan_dataset")
