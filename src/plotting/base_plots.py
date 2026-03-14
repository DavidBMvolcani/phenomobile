"""
Base plotting functions for phenomobile project.

This module contains generic plotting functions that can be used
across different project types and datasets.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import root_mean_squared_error

class BasePlot:
    def __init__(self):
        pass

    def plot_prediction_vs_actual(self, features, target, df, condition=None, show=True):
        """
        Plot predicted vs actual values with metrics for multi-feature models.
        
        PARAMETERS:
        - features: list of feature column names
        - target: target column name
        - df: dataframe containing the data
        - condition: optional condition for filtering
        - show: whether to display the plot
        
        RETURNS:
        - matplotlib plot object
        """
        # Filter data if condition provided
        if condition:
            # This would need to be implemented based on specific filtering logic
            pass
        
        # Get X, y data
        if target in features:
            features = [f for f in features if f != target]
        X = df[features].values
        y = df[target].values
        
        # Train model and predict
        lr = LinearRegression()
        lr.fit(X, y)
        y_pred = lr.predict(X)
        
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        rmse = root_mean_squared_error(y, y_pred)
        
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(y, y_pred, alpha=0.6, label='Predictions')
        
        # Add perfect prediction line
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', 
                label='Perfect Prediction', linewidth=2)
        
        # Formatting
        plt.xlabel(f'True {target}')
        plt.ylabel(f'Predicted {target}')
        plt.title(f'Linear Regression: Predicted vs Actual (R²={r2:.4f}, RMSE={rmse:.4f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if show:
            plt.show()
        
        return plt


    def plot_heatmap_of_r2_score_of_ndi(self, r2_ndi_df, model, target, show=True):
        """
        Plot heatmap of R² scores for NDI combinations.
        
        PARAMETERS:
        - r2_ndi_df: dataframe with R² scores for NDI combinations
        - model: model name for title
        - target: target variable name for title
        - show: whether to display the plot
        
        RETURNS:
        - matplotlib plot object
        """
        if r2_ndi_df is None:
            raise ValueError("r2_ndi_df cannot be None")
        
        ndi_df = r2_ndi_df.copy()
        
        # Clean up dataframe if needed
        if ndi_df.columns.str.contains("^Unnamed").any():
            # This would need to be implemented based on specific cleanup logic
            pass
        
        # Plot the heatmap using Seaborn
        plt.figure(figsize=(10, 6))
        sns.heatmap(ndi_df)
        
        # Add titles and labels
        msg = f'''R2 scores of {model}
            for NDI(band_1,band_2) to {target} for all the records in the dataset'''
        plt.title(msg)
        plt.xlabel('Columns: band 1')
        plt.ylabel('Index: band2')
        
        if show:
            plt.show()
        
        return plt


    def plot_linear_regression_generic(self, trainable_features, target, df, 
                                    indicator='', color_map={}, condition=None,
                                    plot_separate=False, show=True):
        """
        Generic linear regression plotting function.
        
        PARAMETERS:
        - trainable_features: list of feature column names
        - target: target column name
        - df: dataframe containing the data
        - indicator: column name for point indicators
        - color_map: color mapping for points
        - condition: condition for filtering
        - plot_separate: whether to create separate plots
        - show: whether to display the plot
        
        RETURNS:
        - matplotlib plot object
        """
        # Filter data if condition provided
        if condition:
            # This would need to be implemented based on specific filtering logic
            pass
        
        # Helper function to plot one dataframe
        def plot_single(ax, sub_df, title_suffix=''):
            X = sub_df[trainable_features[0]].values.reshape(-1, 1)
            y = sub_df[target].values
            
            lr = LinearRegression()
            lr.fit(X, y)
            y_pred = lr.predict(X)
            
            r2 = r2_score(y, y_pred)
            rmse = root_mean_squared_error(y, y_pred)
            
            # Plot regression line
            ax.plot(X, y_pred, color='red', label='Regression Line')
            
            # Plot points
            ax.scatter(X, y, label='Data Points')
            
            # Formatting
            ax.set_xlabel(trainable_features[0])
            ax.set_ylabel(target)
            ax.set_title(f'Linear Regression {title_suffix} (R²={r2:.3f}, RMSE={rmse:.3f})')
            ax.grid(True)
            ax.legend()
        
        if plot_separate:
            # If plotting separate subplots, define categories
            categories = ['White and Blue Led', 'White Led', 'Shade', 'Control']
            n_rows, n_cols = 2, 2
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10), sharey=True)
            axes = axes.flatten()
            
            for ax, cat in zip(axes, categories):
                # Filter by category (would need specific implementation)
                sub_df = df  # Placeholder
                plot_single(ax, sub_df, title_suffix=f'{cat}')
            
            # Hide any unused axes
            for i in range(len(categories), len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
        else:
            plt.figure(figsize=(10, 5))
            plot_single(plt.gca(), df, title_suffix='Linear Regression')
        
        if show:
            plt.show()
        
        return plt

    def save_plot(self, plt, plot_path):
        """Save plot to file."""
        plt.savefig(plot_path)
        print(f"Saved plot to {plot_path}")
