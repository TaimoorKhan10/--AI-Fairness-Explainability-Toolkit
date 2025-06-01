"""
Example demonstrating how to use the AFET toolkit with PyTorch models

This example shows how to:
1. Create and train a PyTorch model
2. Wrap it with PyTorchModelWrapper
3. Evaluate fairness metrics
4. Generate fairness visualizations
5. Use explainability tools
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from afet.models.pytorch.model_wrapper import PyTorchModelWrapper
from afet.core.fairness_metrics import get_all_fairness_metrics
from afet.viz.fairness_radar_enhanced import FairnessRadarEnhanced
from afet.core.explainability import Explainer


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


def load_data():
    """Load and prepare Adult dataset"""
    print("Loading and preparing the Adult dataset...")
    adult = fetch_openml(data_id=1590, as_frame=True)
    
    df = adult.data
    df['target'] = adult.target
    
    # Select sensitive attribute (sex)
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
    
    # Prepare features and target
    X = df.drop(['target', 'sex', 'race', 'native-country', 'relationship'], axis=1)
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
    
    # Convert targets to numpy arrays
    y_train, y_test = y_train.values, y_test.values
    sens_train, sens_test = sens_train.values, sens_test.values
    
    # Create feature names for explainability
    feature_names = X.columns.tolist()
    
    return X_train, X_test, y_train, y_test, sens_train, sens_test, feature_names


def train_pytorch_model(X_train, y_train, input_size, epochs=10, learning_rate=0.001):
    """Create and train a PyTorch model"""
    print(f"Training PyTorch model with {epochs} epochs...")
    
    # Create model
    model = SimpleNN(input_size)
    
    # Convert data to PyTorch tensors
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
    
    # Set up training
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    # Create wrapper
    wrapper = PyTorchModelWrapper(model)
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('pytorch_training_loss.png')
    
    return wrapper


def evaluate_fairness(model, X_test, y_test, sens_test):
    """Evaluate fairness metrics for the model"""
    print("Evaluating fairness metrics...")
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Calculate comprehensive fairness metrics
    metrics = get_all_fairness_metrics(y_test, y_pred, sens_test)
    
    # Print key metrics
    print("\nFairness Metrics:")
    print(f"Demographic Parity: {metrics['demographic_parity']:.4f}")
    print(f"Equal Opportunity: {metrics['equal_opportunity']:.4f}")
    print(f"Disparate Impact: {metrics['disparate_impact']:.4f}")
    print(f"Statistical Parity Difference: {metrics['statistical_parity_difference']:.4f}")
    print(f"Treatment Equality: {metrics['treatment_equality']:.4f}")
    print(f"TPR Parity (Equalized Odds): {metrics['equalized_odds']['tpr_parity']:.4f}")
    print(f"FPR Parity (Equalized Odds): {metrics['equalized_odds']['fpr_parity']:.4f}")
    
    return metrics


def visualize_fairness(metrics1, metrics2=None):
    """Generate fairness visualization"""
    print("Generating fairness visualization...")
    
    # Create model metrics dictionary
    if metrics2 is not None:
        model_metrics = {
            'PyTorch Model': metrics1,
            'PyTorch Model (Alt)': metrics2
        }
    else:
        model_metrics = {'PyTorch Model': metrics1}
    
    # Create and configure fairness radar
    radar = FairnessRadarEnhanced()
    
    # Generate visualization
    fig = radar.plot(
        model_metrics,
        normalize=True,
        plot_type='both',
        theme='blue',
        show_threshold=True,
        threshold_value=0.8,
        save_path='pytorch_fairness_radar.png',
        show_plot=True
    )
    
    # Generate metrics comparison table if multiple models
    if metrics2 is not None:
        table = radar.generate_metrics_table(model_metrics)
        print("\nMetrics Comparison Table:")
        print(table)
        
        # Save table to CSV
        table.to_csv('pytorch_metrics_comparison.csv')
    
    return radar


def explain_model(model, X_train, X_test, y_train, y_test, feature_names):
    """Generate explanations for the model"""
    print("Generating model explanations...")
    
    # Create explainer
    explainer = Explainer(model, X_train, feature_names=feature_names)
    
    # Get sample for explanation
    sample_idx = 0
    sample = X_test[sample_idx:sample_idx+1]
    
    # Generate SHAP explanation
    shap_values = explainer.explain_instance_shap(sample)
    explainer.plot_shap_values(shap_values, feature_names, save_path='pytorch_shap_explanation.png')
    
    # Calculate feature importance
    print("\nFeature Importance:")
    importances = model.feature_importance(X_test, y_test)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print(importance_df.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    bars = plt.barh(importance_df['Feature'][:10][::-1], importance_df['Importance'][:10][::-1])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('pytorch_feature_importance.png')
    
    # Generate partial dependence for top feature
    top_feature = importance_df['Feature'].iloc[0]
    top_feature_idx = feature_names.index(top_feature)
    pdp = explainer.partial_dependence(X_test, [top_feature_idx])
    
    return explainer


def train_alternative_model(X_train, y_train, input_size):
    """Train an alternative model for comparison"""
    print("Training alternative PyTorch model...")
    
    # Create model with different architecture
    model = SimpleNN(input_size, hidden_size=32)
    
    # Convert data to PyTorch tensors
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
    
    # Set up training with different optimizer
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Training loop (fewer epochs)
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
    
    # Create wrapper
    wrapper = PyTorchModelWrapper(model)
    return wrapper


def save_and_load_model(model, X_sample):
    """Demonstrate model persistence"""
    print("Demonstrating model persistence...")
    
    # Save model
    model_path = 'pytorch_model.pt'
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Load model
    loaded_model = PyTorchModelWrapper.load(model_path)
    print("Model loaded successfully")
    
    # Verify predictions match
    original_preds = model.predict(X_sample)
    loaded_preds = loaded_model.predict(X_sample)
    match = np.array_equal(original_preds, loaded_preds)
    print(f"Predictions match: {match}")
    
    return loaded_model


def main():
    """Main function to run the example"""
    print("PyTorch Fairness Example\n" + "="*50)
    
    # Load data
    X_train, X_test, y_train, y_test, sens_train, sens_test, feature_names = load_data()
    input_size = X_train.shape[1]
    
    # Train primary model
    model = train_pytorch_model(X_train, y_train, input_size)
    
    # Train alternative model for comparison
    alt_model = train_alternative_model(X_train, y_train, input_size)
    
    # Evaluate fairness for both models
    metrics = evaluate_fairness(model, X_test, y_test, sens_test)
    alt_metrics = evaluate_fairness(alt_model, X_test, y_test, sens_test)
    
    # Visualize fairness comparison
    radar = visualize_fairness(metrics, alt_metrics)
    
    # Generate explanations
    explainer = explain_model(model, X_train, X_test, y_train, y_test, feature_names)
    
    # Demonstrate model persistence
    loaded_model = save_and_load_model(model, X_test[:5])
    
    print("\nExample completed successfully!")
    print("Generated files:")
    print("- pytorch_training_loss.png")
    print("- pytorch_fairness_radar.png")
    print("- pytorch_metrics_comparison.csv")
    print("- pytorch_shap_explanation.png")
    print("- pytorch_feature_importance.png")
    print("- pytorch_model.pt")


if __name__ == "__main__":
    main()
