"""
Bias Mitigation Techniques for Fair Machine Learning

This module implements various techniques to mitigate bias in machine learning models,
including pre-processing, in-processing, and post-processing methods.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
import scipy.optimize as optim


class PrejudiceRemover(BaseEstimator, ClassifierMixin):
    """
    Prejudice Remover is an in-processing technique that adds a discrimination-aware
    regularization term to the learning objective.
    
    Parameters:
    -----------
    eta : float, default=1.0
        Fairness penalty parameter. Higher values enforce stronger fairness.
    sensitive_cols : list of str or list of int, default=None
        Names or indices of sensitive attribute columns.
    max_iter : int, default=100
        Maximum number of iterations for optimization.
    tol : float, default=1e-4
        Tolerance for stopping criteria.
    """
    
    def __init__(self, eta: float = 1.0, sensitive_cols: Optional[Union[List[str], List[int]]] = None,
                 max_iter: int = 100, tol: float = 1e-4):
        self.eta = eta
        self.sensitive_cols = sensitive_cols
        self.max_iter = max_iter
        self.tol = tol
        self.le_ = None
        self.classes_ = None
        
    def _sigmoid(self, z):
        """Sigmoid function."""
        return 1 / (1 + np.exp(-z))
    
    def _objective(self, w, X, y, sensitive):
        """Objective function with fairness regularization."""
        z = np.dot(X, w[:-1]) + w[-1]  # w[-1] is bias
        y_pred = self._sigmoid(z)
        
        # Cross-entropy loss
        loss = -np.mean(y * np.log(y_pred + 1e-10) + (1 - y) * np.log(1 - y_pred + 1e-10))
        
        # Fairness regularization
        if self.eta > 0 and len(np.unique(sensitive)) > 1:
            # Calculate covariance between sensitive attribute and predictions
            cov = np.cov(sensitive, y_pred)[0, 1]
            loss += self.eta * cov**2
            
        return loss
    
    def fit(self, X, y, sensitive):
        """
        Fit the model to the training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sensitive : array-like of shape (n_samples,)
            Sensitive attribute values.
            
        Returns:
        --------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        
        # Encode sensitive attribute if not numeric
        if not np.issubdtype(sensitive.dtype, np.number):
            self.le_ = LabelEncoder()
            sensitive = self.le_.fit_transform(sensitive)
        
        # Initialize weights
        n_features = X.shape[1]
        w0 = np.zeros(n_features + 1)  # +1 for bias
        
        # Optimize
        result = optim.minimize(
            fun=self._objective,
            x0=w0,
            args=(X, y, sensitive),
            method='L-BFGS-B',
            options={'maxiter': self.max_iter, 'gtol': self.tol}
        )
        
        self.coef_ = result.x[:-1]
        self.intercept_ = result.x[-1]
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        check_is_fitted(self)
        X = check_array(X)
        
        z = np.dot(X, self.coef_) + self.intercept_
        proba = self._sigmoid(z)
        return np.column_stack((1 - proba, proba))
    
    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


class Reweighing:
    """
    Reweighing is a pre-processing technique that assigns weights to the training
    examples to ensure fairness before model training.
    
    Parameters:
    -----------
    sensitive_attr : str or int
        Name or index of the sensitive attribute column.
    target_attr : str or int
        Name or index of the target attribute column.
    """
    
    def __init__(self, sensitive_attr: Union[str, int], target_attr: Union[str, int]):
        self.sensitive_attr = sensitive_attr
        self.target_attr = target_attr
        self.weights_ = None
        
    def fit(self, X, y=None):
        """
        Calculate the weights for each training example.
        
        Parameters:
        -----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Target values. If None, y is taken from X using target_attr.
            
        Returns:
        --------
        self : object
            Fitted transformer.
        """
        if y is None:
            if isinstance(X, pd.DataFrame):
                y = X[self.target_attr]
                X = X.drop(columns=[self.target_attr])
            else:
                raise ValueError("y must be provided if X is not a DataFrame")
        
        if isinstance(X, pd.DataFrame):
            sensitive = X[self.sensitive_attr]
            X = X.drop(columns=[self.sensitive_attr])
        else:
            sensitive = X[:, self.sensitive_attr]
            X = np.delete(X, self.sensitive_attr, axis=1)
        
        # Calculate weights
        df = pd.DataFrame({
            'sensitive': sensitive,
            'target': y
        })
        
        # Calculate group sizes
        group_sizes = df.groupby(['sensitive', 'target']).size().unstack(fill_value=0)
        
        # Calculate weights
        weights = pd.Series(1.0, index=df.index)
        for s in group_sizes.index:
            for t in group_sizes.columns:
                mask = (df['sensitive'] == s) & (df['target'] == t)
                if group_sizes.loc[s, t] > 0:
                    weights[mask] = 1.0 / group_sizes.loc[s, t]
        
        # Normalize weights
        self.weights_ = weights / weights.mean()
        
        return self
    
    def transform(self, X, y=None):
        """
        Return the weights for the input data.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Input data.
        y : array-like, optional
            Target values. Ignored.
            
        Returns:
        --------
        weights : ndarray of shape (n_samples,)
            Weights for each sample.
        """
        if self.weights_ is None:
            raise ValueError("fit must be called before transform")
        return self.weights_.values
    
    def fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        return self.fit(X, y).transform(X, y)


class CalibratedEqualizedOddsPostprocessing:
    """
    Post-processing technique that adjusts classifier outputs to satisfy
    equalized odds constraints.
    
    Parameters:
    -----------
    cost_constraint : str, default='weighted'
        Constraints to optimize:
        - 'weighted': Weighted sum of error rate and fairness violation
        - 'fpr': False positive rate equality
        - 'fnr': False negative rate equality
        - 'weighted': Weighted sum of FPR and FNR differences
    """
    
    def __init__(self, cost_constraint: str = 'weighted'):
        self.cost_constraint = cost_constraint
        self.p0 = None  # Probability thresholds for group 0
        self.p1 = None  # Probability thresholds for group 1
        
    def _find_thresholds(self, y_true, y_prob, sensitive):
        """Find optimal probability thresholds for each group."""
        from sklearn.metrics import roc_curve
        
        groups = np.unique(sensitive)
        if len(groups) != 2:
            raise ValueError("Only binary sensitive attributes are supported")
            
        # Find ROC curves for each group
        fpr0, tpr0, thresholds0 = roc_curve(
            y_true[sensitive == groups[0]],
            y_prob[sensitive == groups[0]]
        )
        
        fpr1, tpr1, thresholds1 = roc_curve(
            y_true[sensitive == groups[1]],
            y_prob[sensitive == groups[1]]
        )
        
        # Find thresholds that achieve similar FPR or FNR
        if self.cost_constraint == 'fpr':
            # Find thresholds where FPRs are closest
            target_fpr = (fpr0[1:] + fpr1[1:]) / 2
            idx0 = np.argmin(np.abs(fpr0[1:, None] - target_fpr), axis=0)
            idx1 = np.argmin(np.abs(fpr1[1:, None] - target_fpr), axis=0)
        elif self.cost_constraint == 'fnr':
            # Find thresholds where FNRs are closest
            target_fnr = ((1 - tpr0[1:]) + (1 - tpr1[1:])) / 2
            idx0 = np.argmin(np.abs((1 - tpr0[1:, None]) - target_fnr), axis=0)
            idx1 = np.argmin(np.abs((1 - tpr1[1:, None]) - target_fnr), axis=0)
        else:  # 'weighted'
            # Weighted combination of FPR and FNR differences
            combined_diff = np.abs(fpr0[1:, None] - fpr1[1:]) + np.abs((1 - tpr0[1:, None]) - (1 - tpr1[1:]))
            idx0, idx1 = np.unravel_index(np.argmin(combined_diff), combined_diff.shape)
            
        return thresholds0[idx0], thresholds1[idx1]
    
    def fit(self, y_true, y_prob, sensitive):
        """
        Fit the post-processor.
        
        Parameters:
        -----------
        y_true : array-like of shape (n_samples,)
            Ground truth labels.
        y_prob : array-like of shape (n_samples,)
            Predicted probabilities of the positive class.
        sensitive : array-like of shape (n_samples,)
            Sensitive attribute values.
            
        Returns:
        --------
        self : object
            Fitted post-processor.
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        sensitive = np.asarray(sensitive)
        
        groups = np.unique(sensitive)
        if len(groups) != 2:
            raise ValueError("Only binary sensitive attributes are supported")
            
        self.p0, self.p1 = self._find_thresholds(y_true, y_prob, sensitive)
        
        return self
    
    def predict(self, y_prob, sensitive):
        """
        Predict class labels using the post-processed probabilities.
        
        Parameters:
        -----------
        y_prob : array-like of shape (n_samples,)
            Predicted probabilities of the positive class.
        sensitive : array-like of shape (n_samples,)
            Sensitive attribute values.
            
        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        y_prob = np.asarray(y_prob)
        sensitive = np.asarray(sensitive)
        
        if self.p0 is None or self.p1 is None:
            raise ValueError("fit must be called before predict")
            
        groups = np.unique(sensitive)
        if len(groups) != 2:
            raise ValueError("Only binary sensitive attributes are supported")
            
        y_pred = np.zeros_like(y_prob, dtype=int)
        y_pred[sensitive == groups[0]] = (y_prob[sensitive == groups[0]] >= self.p0).astype(int)
        y_pred[sensitive == groups[1]] = (y_prob[sensitive == groups[1]] >= self.p1).astype(int)
        
        return y_pred
