"""
Synthetic Dataset Generator

This module provides utilities for generating synthetic datasets with controlled bias
for testing and demonstrating fairness metrics and mitigation techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from sklearn.preprocessing import StandardScaler


class BiasedDatasetGenerator:
    """
    A class for generating synthetic datasets with controlled bias for fairness testing.
    
    This generator allows users to create datasets with specific types and levels of bias
    across different demographic groups, which is useful for testing fairness metrics
    and mitigation techniques.
    """
    
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize the BiasedDatasetGenerator.
        
        Parameters
        ----------
        random_state : Optional[int], optional
            Random seed for reproducibility, by default None
        """
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def generate_credit_data(
        self,
        n_samples: int = 5000,
        bias_level: float = 0.2,
        include_gender: bool = True,
        include_age: bool = True,
        include_race: bool = True
    ) -> pd.DataFrame:
        """
        Generate a synthetic credit approval dataset with controlled bias.
        
        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate, by default 5000
        bias_level : float, optional
            Level of bias to introduce (0.0 to 1.0), by default 0.2
        include_gender : bool, optional
            Whether to include gender as a protected attribute, by default True
        include_age : bool, optional
            Whether to include age as a protected attribute, by default True
        include_race : bool, optional
            Whether to include race as a protected attribute, by default True
            
        Returns
        -------
        pd.DataFrame
            Synthetic credit approval dataset with controlled bias
        """
        # Generate demographic features
        data = {}
        
        # Generate gender with equal distribution if included
        if include_gender:
            data['gender'] = self.rng.choice(
                ['Male', 'Female'],
                size=n_samples,
                p=[0.5, 0.5]
            )
        
        # Generate age groups if included
        if include_age:
            # Generate continuous age first
            age_continuous = self.rng.normal(45, 15, n_samples).astype(int)
            age_continuous = np.clip(age_continuous, 18, 90)
            
            # Then convert to age groups
            age_groups = pd.cut(
                age_continuous,
                bins=[18, 30, 45, 60, 100],
                labels=['18-30', '31-45', '46-60', '60+']
            )
            data['age_group'] = age_groups
            data['age'] = age_continuous
        
        # Generate race with realistic distribution if included
        if include_race:
            data['race'] = self.rng.choice(
                ['White', 'Black', 'Hispanic', 'Asian', 'Other'],
                size=n_samples,
                p=[0.6, 0.13, 0.18, 0.06, 0.03]
            )
        
        # Generate income with demographic-based differences
        income = self.rng.normal(60000, 20000, n_samples)
        
        # Apply bias based on demographics
        if include_gender:
            # Gender pay gap
            income[data['gender'] == 'Female'] *= (1 - bias_level * 0.3)
        
        if include_race:
            # Racial income disparities
            income[data['race'] == 'White'] *= (1 + bias_level * 0.1)
            income[data['race'] == 'Black'] *= (1 - bias_level * 0.2)
            income[data['race'] == 'Hispanic'] *= (1 - bias_level * 0.15)
            income[data['race'] == 'Asian'] *= (1 + bias_level * 0.05)
        
        if include_age:
            # Age-based income differences (mid-career peak)
            income[data['age_group'] == '18-30'] *= (1 - bias_level * 0.2)
            income[data['age_group'] == '31-45'] *= (1 + bias_level * 0.1)
            income[data['age_group'] == '46-60'] *= (1 + bias_level * 0.15)
            income[data['age_group'] == '60+'] *= (1 - bias_level * 0.1)
        
        income = np.clip(income, 20000, 200000).astype(int)
        data['income'] = income
        
        # Generate credit score (300-850)
        credit_score = self.rng.normal(650, 100, n_samples)
        
        # Apply bias to credit scores
        if include_gender:
            credit_score[data['gender'] == 'Female'] -= bias_level * 20
        
        if include_race:
            credit_score[data['race'] == 'White'] += bias_level * 15
            credit_score[data['race'] == 'Black'] -= bias_level * 30
            credit_score[data['race'] == 'Hispanic'] -= bias_level * 20
        
        if include_age:
            credit_score[data['age_group'] == '18-30'] -= bias_level * 25
            credit_score[data['age_group'] == '60+'] += bias_level * 10
        
        credit_score = np.clip(credit_score, 300, 850).astype(int)
        data['credit_score'] = credit_score
        
        # Generate debt-to-income ratio (0.1 to 0.8)
        dti = self.rng.beta(2, 5, n_samples) * 0.7 + 0.1
        
        # Apply bias to DTI
        if include_gender:
            dti[data['gender'] == 'Female'] += bias_level * 0.05
        
        if include_race:
            dti[data['race'] == 'Black'] += bias_level * 0.08
            dti[data['race'] == 'Hispanic'] += bias_level * 0.06
        
        if include_age:
            dti[data['age_group'] == '18-30'] += bias_level * 0.07
        
        dti = np.clip(dti, 0.1, 0.8)
        data['debt_to_income'] = dti
        
        # Generate loan amount (5k to 1M)
        loan_amount = self.rng.lognormal(10, 0.8, n_samples).astype(int)
        loan_amount = np.clip(loan_amount, 5000, 1000000)
        data['loan_amount'] = loan_amount
        
        # Generate employment length (0-30 years)
        employment_length = self.rng.exponential(5, n_samples).astype(int)
        employment_length = np.clip(employment_length, 0, 30)
        data['employment_length'] = employment_length
        
        # Generate target (loan approval) with bias
        # Base probability from legitimate features
        prob_approval = (
            0.3 +
            0.4 * (credit_score - 300) / 550 +  # Normalize credit score
            0.2 * (1 - dti) +
            0.1 * (income / 200000)
        )
        
        # Add bias based on protected attributes
        if include_gender:
            prob_approval[data['gender'] == 'Female'] -= bias_level * 0.1
        
        if include_race:
            prob_approval[data['race'] == 'Black'] -= bias_level * 0.15
            prob_approval[data['race'] == 'Hispanic'] -= bias_level * 0.1
        
        if include_age:
            prob_approval[data['age_group'] == '18-30'] -= bias_level * 0.05
            prob_approval[data['age_group'] == '60+'] -= bias_level * 0.05
        
        # Add some random noise
        prob_approval += self.rng.normal(0, 0.1, n_samples)
        prob_approval = np.clip(prob_approval, 0, 1)
        
        # Generate binary outcome
        data['approved'] = (self.rng.random(n_samples) < prob_approval).astype(int)
        
        return pd.DataFrame(data)
    
    def generate_hiring_data(
        self,
        n_samples: int = 5000,
        bias_level: float = 0.2,
        include_gender: bool = True,
        include_age: bool = True,
        include_race: bool = True
    ) -> pd.DataFrame:
        """
        Generate a synthetic hiring dataset with controlled bias.
        
        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate, by default 5000
        bias_level : float, optional
            Level of bias to introduce (0.0 to 1.0), by default 0.2
        include_gender : bool, optional
            Whether to include gender as a protected attribute, by default True
        include_age : bool, optional
            Whether to include age as a protected attribute, by default True
        include_race : bool, optional
            Whether to include race as a protected attribute, by default True
            
        Returns
        -------
        pd.DataFrame
            Synthetic hiring dataset with controlled bias
        """
        # Generate demographic features
        data = {}
        
        # Generate gender with equal distribution if included
        if include_gender:
            data['gender'] = self.rng.choice(
                ['Male', 'Female'],
                size=n_samples,
                p=[0.5, 0.5]
            )
        
        # Generate age groups if included
        if include_age:
            # Generate continuous age first
            age_continuous = self.rng.normal(35, 10, n_samples).astype(int)
            age_continuous = np.clip(age_continuous, 18, 65)
            
            # Then convert to age groups
            age_groups = pd.cut(
                age_continuous,
                bins=[18, 25, 35, 45, 65],
                labels=['18-25', '26-35', '36-45', '46+']
            )
            data['age_group'] = age_groups
            data['age'] = age_continuous
        
        # Generate race with realistic distribution if included
        if include_race:
            data['race'] = self.rng.choice(
                ['White', 'Black', 'Hispanic', 'Asian', 'Other'],
                size=n_samples,
                p=[0.6, 0.13, 0.18, 0.06, 0.03]
            )
        
        # Generate education level
        data['education'] = self.rng.choice(
            ['High School', 'Bachelor', 'Master', 'PhD'],
            size=n_samples,
            p=[0.2, 0.5, 0.25, 0.05]
        )
        
        # Generate years of experience with demographic-based differences
        experience = self.rng.normal(8, 5, n_samples)
        
        # Apply bias based on demographics
        if include_gender:
            # Gender experience gap (e.g., due to career breaks)
            experience[data['gender'] == 'Female'] *= (1 - bias_level * 0.2)
        
        if include_age:
            # Age naturally correlates with experience
            experience = 0.7 * experience + 0.3 * (data['age'] - 18) / 2
        
        experience = np.clip(experience, 0, 30).astype(int)
        data['experience'] = experience
        
        # Generate skills score (1-10)
        skills = self.rng.normal(7, 2, n_samples)
        
        # No bias in inherent skills, but there might be bias in how skills are evaluated
        skills = np.clip(skills, 1, 10)
        data['skills'] = skills
        
        # Generate interview score (1-10) with bias
        interview = self.rng.normal(7, 2, n_samples)
        
        # Apply bias to interview scores
        if include_gender:
            interview[data['gender'] == 'Female'] -= bias_level * 1
        
        if include_race:
            interview[data['race'] == 'Black'] -= bias_level * 1.5
            interview[data['race'] == 'Hispanic'] -= bias_level * 1
            interview[data['race'] == 'Asian'] += bias_level * 0.5
        
        if include_age:
            interview[data['age_group'] == '46+'] -= bias_level * 1
        
        interview = np.clip(interview, 1, 10)
        data['interview_score'] = interview
        
        # Generate target (hiring decision) with bias
        # Base probability from legitimate features
        prob_hire = (
            0.1 +
            0.3 * (skills - 1) / 9 +  # Normalize skills
            0.3 * (interview - 1) / 9 +  # Normalize interview
            0.2 * (experience / 30) +  # Normalize experience
            0.1 * (data['education'] == 'PhD') * 1.0 +
            0.05 * (data['education'] == 'Master') * 1.0
        )
        
        # Add bias based on protected attributes
        if include_gender:
            prob_hire[data['gender'] == 'Female'] -= bias_level * 0.15
        
        if include_race:
            prob_hire[data['race'] == 'Black'] -= bias_level * 0.2
            prob_hire[data['race'] == 'Hispanic'] -= bias_level * 0.15
        
        if include_age:
            prob_hire[data['age_group'] == '46+'] -= bias_level * 0.2
        
        # Add some random noise
        prob_hire += self.rng.normal(0, 0.1, n_samples)
        prob_hire = np.clip(prob_hire, 0, 1)
        
        # Generate binary outcome
        data['hired'] = (self.rng.random(n_samples) < prob_hire).astype(int)
        
        return pd.DataFrame(data)
    
    def generate_healthcare_data(
        self,
        n_samples: int = 5000,
        bias_level: float = 0.2,
        include_gender: bool = True,
        include_age: bool = True,
        include_race: bool = True,
        include_ses: bool = True  # Socioeconomic status
    ) -> pd.DataFrame:
        """
        Generate a synthetic healthcare dataset with controlled bias.
        
        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate, by default 5000
        bias_level : float, optional
            Level of bias to introduce (0.0 to 1.0), by default 0.2
        include_gender : bool, optional
            Whether to include gender as a protected attribute, by default True
        include_age : bool, optional
            Whether to include age as a protected attribute, by default True
        include_race : bool, optional
            Whether to include race as a protected attribute, by default True
        include_ses : bool, optional
            Whether to include socioeconomic status as a protected attribute, by default True
            
        Returns
        -------
        pd.DataFrame
            Synthetic healthcare dataset with controlled bias
        """
        # Generate demographic features
        data = {}
        
        # Generate gender with equal distribution if included
        if include_gender:
            data['gender'] = self.rng.choice(
                ['Male', 'Female'],
                size=n_samples,
                p=[0.5, 0.5]
            )
        
        # Generate age groups if included
        if include_age:
            # Generate continuous age first
            age_continuous = self.rng.normal(50, 20, n_samples).astype(int)
            age_continuous = np.clip(age_continuous, 18, 90)
            
            # Then convert to age groups
            age_groups = pd.cut(
                age_continuous,
                bins=[18, 35, 50, 65, 90],
                labels=['18-35', '36-50', '51-65', '65+']
            )
            data['age_group'] = age_groups
            data['age'] = age_continuous
        
        # Generate race with realistic distribution if included
        if include_race:
            data['race'] = self.rng.choice(
                ['White', 'Black', 'Hispanic', 'Asian', 'Other'],
                size=n_samples,
                p=[0.6, 0.13, 0.18, 0.06, 0.03]
            )
        
        # Generate socioeconomic status if included
        if include_ses:
            data['ses'] = self.rng.choice(
                ['Low', 'Medium', 'High'],
                size=n_samples,
                p=[0.3, 0.5, 0.2]
            )
        
        # Generate health metrics
        
        # Blood pressure (systolic)
        bp = self.rng.normal(120, 15, n_samples)
        
        # Apply demographic effects on blood pressure
        if include_age:
            bp += 0.5 * (data['age'] - 50)
        
        if include_race:
            bp[data['race'] == 'Black'] += 5
        
        bp = np.clip(bp, 80, 200).astype(int)
        data['blood_pressure'] = bp
        
        # Cholesterol
        chol = self.rng.normal(190, 35, n_samples)
        
        # Apply demographic effects on cholesterol
        if include_age:
            chol += 0.5 * (data['age'] - 50)
        
        if include_gender:
            chol[data['gender'] == 'Male'] += 5
        
        chol = np.clip(chol, 100, 300).astype(int)
        data['cholesterol'] = chol
        
        # Body Mass Index (BMI)
        bmi = self.rng.normal(26, 5, n_samples)
        
        # Apply demographic effects on BMI
        if include_ses and include_race:
            # Intersectional effect of race and SES on BMI
            bmi[(data['race'] == 'Black') & (data['ses'] == 'Low')] += bias_level * 3
            bmi[(data['race'] == 'Hispanic') & (data['ses'] == 'Low')] += bias_level * 2.5
        
        bmi = np.clip(bmi, 15, 45)
        data['bmi'] = bmi
        
        # Generate symptoms severity (1-10)
        symptoms = self.rng.normal(5, 2, n_samples)
        symptoms = np.clip(symptoms, 1, 10)
        data['symptoms_severity'] = symptoms
        
        # Generate pain level reported (1-10) with bias
        pain = self.rng.normal(5, 2, n_samples)
        
        # Apply bias to pain assessment (e.g., women's pain being taken less seriously)
        if include_gender:
            # The bias is in how the pain is recorded/believed, not in the actual pain
            pain[data['gender'] == 'Female'] -= bias_level * 1.5
        
        if include_race:
            pain[data['race'] == 'Black'] -= bias_level * 2
            pain[data['race'] == 'Hispanic'] -= bias_level * 1.5
        
        pain = np.clip(pain, 1, 10)
        data['pain_level'] = pain
        
        # Generate target (referral to specialist) with bias
        # Base probability from legitimate features
        prob_referral = (
            0.1 +
            0.2 * (bp - 80) / 120 +  # Normalize blood pressure
            0.2 * (chol - 100) / 200 +  # Normalize cholesterol
            0.2 * (bmi - 15) / 30 +  # Normalize BMI
            0.2 * (symptoms - 1) / 9 +  # Normalize symptoms
            0.1 * (pain - 1) / 9  # Normalize pain
        )
        
        # Add bias based on protected attributes
        if include_gender:
            prob_referral[data['gender'] == 'Female'] -= bias_level * 0.15
        
        if include_race:
            prob_referral[data['race'] == 'Black'] -= bias_level * 0.2
            prob_referral[data['race'] == 'Hispanic'] -= bias_level * 0.15
        
        if include_ses:
            prob_referral[data['ses'] == 'Low'] -= bias_level * 0.25
        
        # Add some random noise
        prob_referral += self.rng.normal(0, 0.1, n_samples)
        prob_referral = np.clip(prob_referral, 0, 1)
        
        # Generate binary outcome
        data['referred'] = (self.rng.random(n_samples) < prob_referral).astype(int)
        
        return pd.DataFrame(data)


def generate_credit_dataset(
    n_samples: int = 5000,
    bias_level: float = 0.2,
    include_gender: bool = True,
    include_age: bool = True,
    include_race: bool = True,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate a synthetic credit approval dataset with controlled bias.
    
    Parameters
    ----------
    n_samples : int, optional
        Number of samples to generate, by default 5000
    bias_level : float, optional
        Level of bias to introduce (0.0 to 1.0), by default 0.2
    include_gender : bool, optional
        Whether to include gender as a protected attribute, by default True
    include_age : bool, optional
        Whether to include age as a protected attribute, by default True
    include_race : bool, optional
        Whether to include race as a protected attribute, by default True
    random_state : Optional[int], optional
        Random seed for reproducibility, by default None
        
    Returns
    -------
    pd.DataFrame
        Synthetic credit approval dataset with controlled bias
    """
    generator = BiasedDatasetGenerator(random_state=random_state)
    return generator.generate_credit_data(
        n_samples=n_samples,
        bias_level=bias_level,
        include_gender=include_gender,
        include_age=include_age,
        include_race=include_race
    )


def generate_hiring_dataset(
    n_samples: int = 5000,
    bias_level: float = 0.2,
    include_gender: bool = True,
    include_age: bool = True,
    include_race: bool = True,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate a synthetic hiring dataset with controlled bias.
    
    Parameters
    ----------
    n_samples : int, optional
        Number of samples to generate, by default 5000
    bias_level : float, optional
        Level of bias to introduce (0.0 to 1.0), by default 0.2
    include_gender : bool, optional
        Whether to include gender as a protected attribute, by default True
    include_age : bool, optional
        Whether to include age as a protected attribute, by default True
    include_race : bool, optional
        Whether to include race as a protected attribute, by default True
    random_state : Optional[int], optional
        Random seed for reproducibility, by default None
        
    Returns
    -------
    pd.DataFrame
        Synthetic hiring dataset with controlled bias
    """
    generator = BiasedDatasetGenerator(random_state=random_state)
    return generator.generate_hiring_data(
        n_samples=n_samples,
        bias_level=bias_level,
        include_gender=include_gender,
        include_age=include_age,
        include_race=include_race
    )


def generate_healthcare_dataset(
    n_samples: int = 5000,
    bias_level: float = 0.2,
    include_gender: bool = True,
    include_age: bool = True,
    include_race: bool = True,
    include_ses: bool = True,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate a synthetic healthcare dataset with controlled bias.
    
    Parameters
    ----------
    n_samples : int, optional
        Number of samples to generate, by default 5000
    bias_level : float, optional
        Level of bias to introduce (0.0 to 1.0), by default 0.2
    include_gender : bool, optional
        Whether to include gender as a protected attribute, by default True
    include_age : bool, optional
        Whether to include age as a protected attribute, by default True
    include_race : bool, optional
        Whether to include race as a protected attribute, by default True
    include_ses : bool, optional
        Whether to include socioeconomic status as a protected attribute, by default True
    random_state : Optional[int], optional
        Random seed for reproducibility, by default None
        
    Returns
    -------
    pd.DataFrame
        Synthetic healthcare dataset with controlled bias
    """
    generator = BiasedDatasetGenerator(random_state=random_state)
    return generator.generate_healthcare_data(
        n_samples=n_samples,
        bias_level=bias_level,
        include_gender=include_gender,
        include_age=include_age,
        include_race=include_race,
        include_ses=include_ses
    )
