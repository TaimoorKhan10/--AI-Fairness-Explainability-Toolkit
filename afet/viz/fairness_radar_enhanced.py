"""
Enhanced fairness radar visualization for AFET
"""

from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)


class FairnessRadarEnhanced:
    """
    Enhanced radar plot visualization for fairness metrics comparison
    """
    
    def __init__(self, 
                 metrics: Dict[str, Dict[str, Union[float, Tuple[float, float]]]],
                 model_names: Optional[List[str]] = None,
                 display_metrics: Optional[List[str]] = None,
                 normalize: bool = True,
                 custom_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
                 threshold_lines: Optional[Dict[str, float]] = None,
                 theme: str = 'default'):
        """
        Initialize fairness radar plot
        
        Args:
            metrics: Dictionary of metrics per model, where each model's metrics are another dictionary
            model_names: Optional list of model names (if not provided, will use keys from metrics)
            display_metrics: Optional list of metrics to display (if not provided, will use all common metrics)
            normalize: Whether to normalize metrics to 0-1 range
            custom_ranges: Optional dictionary of custom ranges for specific metrics
            threshold_lines: Optional dictionary of threshold values to display as reference lines
            theme: Visual theme to use ('default', 'dark', 'light', 'colorblind')
        """
        self.metrics = metrics
        self.model_names = model_names or list(metrics.keys())
        self.normalize = normalize
        self.custom_ranges = custom_ranges or {}
        self.threshold_lines = threshold_lines or {}
        self.theme = theme
        
        # Extract common metrics across all models
        common_metrics = set()
        for model_name in self.model_names:
            if model_name not in metrics:
                logger.warning(f"Model {model_name} not found in provided metrics")
                continue
            model_metrics = set(metrics[model_name].keys())
            if not common_metrics:
                common_metrics = model_metrics
            else:
                common_metrics = common_metrics.intersection(model_metrics)
        
        self.display_metrics = display_metrics or list(common_metrics)
        
        # Filter to include only scalar metrics (not tuples like equalized odds)
        self.display_metrics = [m for m in self.display_metrics if self._is_scalar_metric(m)]
        
        # Apply normalization if requested
        if self.normalize:
            self._normalize_metrics()
    
    def _is_scalar_metric(self, metric: str) -> bool:
        """Check if a metric is scalar (not a tuple or other complex type)"""
        for model_name in self.model_names:
            if model_name in self.metrics and metric in self.metrics[model_name]:
                value = self.metrics[model_name][metric]
                if isinstance(value, (tuple, list, np.ndarray, dict)) and not isinstance(value, (int, float, bool)):
                    return False
        return True
    
    def _normalize_metrics(self):
        """Normalize metrics to 0-1 range for better visualization"""
        self.normalized_metrics = {model: {} for model in self.model_names}
        
        for metric in self.display_metrics:
            # Get all values for this metric across models
            values = []
            for model in self.model_names:
                if model in self.metrics and metric in self.metrics[model]:
                    value = self.metrics[model][metric]
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        values.append(value)
            
            if not values:
                continue
                
            # Use custom range or determine min/max
            if metric in self.custom_ranges:
                min_val, max_val = self.custom_ranges[metric]
            else:
                min_val, max_val = min(values), max(values)
                
            # Avoid division by zero
            if min_val == max_val:
                min_val = min_val - 0.5 if min_val != 0 else 0
                max_val = max_val + 0.5 if max_val != 0 else 1
            
            # Normalize values
            for model in self.model_names:
                if model in self.metrics and metric in self.metrics[model]:
                    value = self.metrics[model][metric]
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        normalized = (value - min_val) / (max_val - min_val)
                        self.normalized_metrics[model][metric] = max(0, min(1, normalized))
    
    def plot_interactive(self, 
                        title: str = "Fairness Metrics Comparison",
                        height: int = 600,
                        width: int = 800) -> go.Figure:
        """
        Create an interactive radar plot using Plotly
        
        Args:
            title: Plot title
            height: Plot height
            width: Plot width
            
        Returns:
            Plotly figure object
        """
        # Set up theme colors
        if self.theme == 'dark':
            bg_color = '#111111'
            text_color = '#ffffff'
            grid_color = '#333333'
            colorscale = 'Plasma'
        elif self.theme == 'light':
            bg_color = '#ffffff'
            text_color = '#111111'
            grid_color = '#dddddd'
            colorscale = 'Viridis'
        elif self.theme == 'colorblind':
            bg_color = '#ffffff'
            text_color = '#111111'
            grid_color = '#dddddd'
            colorscale = 'Plotly'
        else:  # default
            bg_color = '#f8f9fa'
            text_color = '#343a40'
            grid_color = '#dee2e6'
            colorscale = 'Bluered'

        # Create figure
        fig = go.Figure()
        
        # Add threshold lines if specified
        if self.threshold_lines:
            theta = [metric for metric in self.display_metrics]
            theta.append(theta[0])  # Close the loop
            
            for threshold_name, threshold_value in self.threshold_lines.items():
                threshold_r = [threshold_value] * len(theta)
                
                fig.add_trace(go.Scatterpolar(
                    r=threshold_r,
                    theta=theta,
                    fill=None,
                    mode='lines',
                    line=dict(color='rgba(100, 100, 100, 0.7)', dash='dot'),
                    name=f"{threshold_name} ({threshold_value})",
                    hoverinfo='text',
                    text=[f"{threshold_name}: {threshold_value}" for _ in theta]
                ))

        # Add traces for each model
        for i, model in enumerate(self.model_names):
            if model not in self.metrics:
                continue
                
            # Prepare radar chart data
            theta = []
            r = []
            
            for metric in self.display_metrics:
                if metric in self.metrics[model]:
                    theta.append(metric)
                    if self.normalize:
                        r.append(self.normalized_metrics[model].get(metric, 0))
                    else:
                        r.append(self.metrics[model].get(metric, 0))
            
            # Close the loop
            theta.append(theta[0])
            r.append(r[0])
            
            # Add trace
            fig.add_trace(go.Scatterpolar(
                r=r,
                theta=theta,
                fill='toself',
                name=model,
                hoverinfo='text',
                text=[f"{model}<br>{t}: {v:.4f}" for t, v in zip(theta, r)]
            ))

        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1] if self.normalize else None,
                    tickfont=dict(color=text_color),
                    gridcolor=grid_color
                ),
                angularaxis=dict(
                    tickfont=dict(color=text_color),
                    gridcolor=grid_color
                ),
                bgcolor=bg_color
            ),
            title=dict(
                text=title,
                font=dict(color=text_color, size=16)
            ),
            legend=dict(
                font=dict(color=text_color),
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            height=height,
            width=width,
            margin=dict(l=80, r=80, t=100, b=100)
        )
        
        return fig
    
    def plot_static(self, 
                   title: str = "Fairness Metrics Comparison",
                   figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Create a static radar plot using Matplotlib
        
        Args:
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        # Set style
        if self.theme == 'dark':
            plt.style.use('dark_background')
            cmap = plt.cm.plasma
        elif self.theme == 'colorblind':
            plt.style.use('default')
            cmap = plt.cm.tab10
        else:
            plt.style.use('default')
            cmap = plt.cm.viridis
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        
        # Number of metrics
        N = len(self.display_metrics)
        if N < 3:
            logger.warning("At least 3 metrics needed for radar plot. Adding dummy metrics.")
            # Add dummy metrics for visualization if less than 3
            self.display_metrics.extend([f"dummy_{i}" for i in range(3 - N)])
            for model in self.model_names:
                for i in range(3 - N):
                    self.metrics[model][f"dummy_{i}"] = 0
                    if self.normalize:
                        self.normalized_metrics[model][f"dummy_{i}"] = 0
            N = len(self.display_metrics)
        
        # Compute angle for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Set up axis
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw labels
        plt.xticks(angles[:-1], self.display_metrics)
        
        # Add threshold lines if specified
        if self.threshold_lines:
            for threshold_name, threshold_value in self.threshold_lines.items():
                values = [threshold_value] * (N + 1)
                ax.plot(angles, values, 'k--', alpha=0.75, linewidth=1, label=f"{threshold_name} ({threshold_value})")
        
        # Draw each model
        for i, model in enumerate(self.model_names):
            if model not in self.metrics:
                continue
                
            values = []
            for metric in self.display_metrics:
                if self.normalize:
                    values.append(self.normalized_metrics[model].get(metric, 0))
                else:
                    values.append(self.metrics[model].get(metric, 0))
            
            values += values[:1]  # Close the loop
            
            # Plot values
            ax.plot(angles, values, linewidth=2, label=model, color=cmap(i / len(self.model_names)))
            ax.fill(angles, values, alpha=0.1, color=cmap(i / len(self.model_names)))
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Add title
        plt.title(title, size=16, y=1.1)
        
        return fig
    
    def generate_metrics_table(self) -> pd.DataFrame:
        """
        Generate a DataFrame table of all metrics for comparison
        
        Returns:
            Pandas DataFrame with metrics
        """
        # Create DataFrame
        df = pd.DataFrame(index=self.model_names, columns=self.display_metrics)
        
        # Fill values
        for model in self.model_names:
            if model not in self.metrics:
                continue
                
            for metric in self.display_metrics:
                if metric in self.metrics[model]:
                    df.loc[model, metric] = self.metrics[model][metric]
                else:
                    df.loc[model, metric] = np.nan
        
        return df
