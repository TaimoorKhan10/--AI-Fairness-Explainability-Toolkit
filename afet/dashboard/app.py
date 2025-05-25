import streamlit as st
import numpy as np
import pandas as pd
from afet.core.fairness_metrics import FairnessMetrics
from afet.core.explainability import ModelExplainer
import plotly.express as px
import shap

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None

# Page configuration
st.set_page_config(
    page_title="AFET - AI Fairness and Explainability Toolkit",
    page_icon="📊",
    layout="wide"
)

# Sidebar
st.sidebar.title("AFET Dashboard")

# Main content
st.title("AI Fairness and Explainability Toolkit")

# Upload data section
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload your dataset", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.session_state.data = data
        st.write("Successfully loaded dataset!")

with col2:
    if st.session_state.data is not None:
        st.write("Dataset preview:")
        st.dataframe(st.session_state.data.head())

# Model selection
if st.session_state.data is not None:
    st.header("Model Selection")
    
    # Feature selection
    features = st.multiselect(
        "Select features",
        st.session_state.data.columns,
        default=list(st.session_state.data.columns[:-1])
    )
    
    # Target selection
    target = st.selectbox(
        "Select target variable",
        st.session_state.data.columns
    )
    
    # Protected attribute selection
    protected_attribute = st.selectbox(
        "Select protected attribute",
        st.session_state.data.columns
    )
    
    # Train model button
    if st.button("Train Model"):
        # Prepare data
        X = st.session_state.data[features]
        y = st.session_state.data[target]
        sensitive_features = st.session_state.data[protected_attribute]
        
        # Train a simple model (replace with your model)
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        
        st.session_state.model = {
            'model': model,
            'X': X,
            'y': y,
            'sensitive_features': sensitive_features
        }
        st.success("Model trained successfully!")

# Analysis section
if st.session_state.model is not None:
    st.header("Analysis Results")
    
    # Initialize metrics and explainer
    fairness_metrics = FairnessMetrics(
        protected_attribute=protected_attribute,
        favorable_label=1,
        unfavorable_label=0
    )
    
    model_explainer = ModelExplainer(
        model=st.session_state.model['model'],
        feature_names=features,
        class_names=['0', '1'],
        training_data=st.session_state.model['X'].values
    )
    
    # Calculate predictions
    y_pred = st.session_state.model['model'].predict(st.session_state.model['X'])
    
    # Fairness metrics
    st.subheader("Fairness Metrics")
    metrics = fairness_metrics.get_comprehensive_metrics(
        y_pred=y_pred,
        y_true=st.session_state.model['y'],
        sensitive_features=st.session_state.model['sensitive_features']
    )
    
    # Display metrics in a table
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    st.dataframe(metrics_df)
    
    # Explainability
    st.subheader("Model Explainability")
    
    # SHAP summary plot
    shap_values = model_explainer.shap_explainer(st.session_state.model['X'].values)
    fig = px.bar(
        x=shap_values.values.mean(axis=0),
        y=features,
        orientation='h',
        title='Global Feature Importance (SHAP)'
    )
    st.plotly_chart(fig)
    
    # Interactive instance explanation
    instance_idx = st.number_input(
        "Select instance to explain",
        min_value=0,
        max_value=len(st.session_state.model['X'])-1,
        value=0
    )
    
    if st.button("Explain Instance"):
        instance = st.session_state.model['X'].iloc[instance_idx].values
        explanations = model_explainer.get_all_explanations(
            instance=instance,
            data=st.session_state.model['X'].values
        )
        
        # Display explanations
        st.subheader("Instance Explanation")
        st.json(explanations)
