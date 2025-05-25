"""
Synthetic hiring dataset generator
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class HiringDataset:
    """
    Generate synthetic hiring dataset
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
        Generate synthetic hiring features
        """
        np.random.seed(self.random_state)
        
        # Education and experience
        years_experience = np.random.normal(5, 3, self.n_samples)
        education_level = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                         self.n_samples,
                                         p=[0.2, 0.4, 0.3, 0.1])
        
        # Skills and certifications
        technical_skills = np.random.normal(70, 15, self.n_samples)
        leadership_skills = np.random.normal(60, 15, self.n_samples)
        certifications = np.random.binomial(1, 0.3, self.n_samples)
        
        # Demographic features
        gender = np.random.choice(['Male', 'Female'], self.n_samples)
        race = np.random.choice(['White', 'Black', 'Asian', 'Other'], self.n_samples)
        age = np.random.normal(35, 10, self.n_samples)
        
        # Previous job performance
        previous_performance = np.random.normal(75, 15, self.n_samples)
        previous_company_size = np.random.choice(['Small', 'Medium', 'Large'], 
                                               self.n_samples,
                                               p=[0.4, 0.3, 0.3])
        
        # Create DataFrame
        df = pd.DataFrame({
            'years_experience': years_experience,
            'education_level': education_level,
            'technical_skills': technical_skills,
            'leadership_skills': leadership_skills,
            'certifications': certifications,
            'gender': gender,
            'race': race,
            'age': age,
            'previous_performance': previous_performance,
            'previous_company_size': previous_company_size
        })
        
        return df
    
    def generate_target(self, 
                       X: pd.DataFrame,
                       bias_factor: float = 0.1) -> pd.Series:
        """
        Generate synthetic hiring decision with potential bias
        """
        np.random.seed(self.random_state)
        
        # Base probability based on qualifications
        base_prob = (
            0.1 * X['years_experience'] +
            0.01 * X['technical_skills'] +
            0.01 * X['leadership_skills'] +
            0.2 * X['previous_performance']
        )
        
        # Add bias based on protected attributes
        bias = np.zeros(self.n_samples)
        
        # Gender bias
        bias[X['gender'] == 'Female'] -= bias_factor
        
        # Race bias
        bias[X['race'] == 'Black'] -= bias_factor * 1.5
        
        # Age bias
        bias[X['age'] > 45] -= bias_factor * 2
        
        # Final probability
        final_prob = np.clip(base_prob + bias, 0, 1)
        
        # Generate binary target
        y = np.random.binomial(1, final_prob)
        
        return pd.Series(y, name='hired')
    
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
        X_scaled[['years_experience', 'technical_skills', 'leadership_skills',
                 'age', 'previous_performance']] = scaler.fit_transform(
            X_scaled[['years_experience', 'technical_skills', 'leadership_skills',
                     'age', 'previous_performance']]
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
    generator = HiringDataset(n_samples=10000)
    generator.generate_dataset()
