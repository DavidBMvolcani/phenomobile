"""
Anthocyanin-specific plotting functions for phenomobile project.

This module contains plotting functions specifically designed for anthocyanin
datasets and experiments.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List


def get_anthocyanin_color_map(categories: Dict = None) -> Dict[str, str]:
    """
    Get color mapping for anthocyanin samples.
    
    PARAMETERS:
    - categories: optional categories dict from config
    
    RETURNS:
    - Dictionary mapping sample IDs to colors
    """
    color_map = {f"R{i}": "red" for i in range(1, 16)}
    color_map.update({f"C{i}": "red" for i in range(1, 6)})
    color_map.update({f"G{i}": "green" for i in range(1, 16)})
    color_map.update({f"C{i}": "green" for i in range(6, 11)})
    
    # If categories provided, use them to build color map
    if categories:
        color_map = {}
        for category_name, sample_ids in categories.items():
            color = "red" if "RED" in category_name else "green"
            for sample_id in sample_ids:
                color_map[sample_id] = color
    
    return color_map


def set_anthocyanin_markers(categories: Dict = None) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Set marker shapes for anthocyanin samples.
    
    PARAMETERS:
    - categories: optional categories dict from config
    
    RETURNS:
    - Tuple of (markers_dict, labels_dict)
    """
    if categories:
        # Use categories from config
        markers = {}
        for category_name, sample_ids in categories.items():
            if "white_blue_led" in category_name.lower():
                marker = 'o'  # circle
            elif "white_led" in category_name.lower():
                marker = 's'  # square
            elif "shade" in category_name.lower():
                marker = '^'  # triangle up
            elif "control" in category_name.lower():
                marker = 'D'  # diamond
            else:
                marker = 'v'  # triangle down (default)
            
            for sample_id in sample_ids:
                markers[sample_id] = marker
        
        labels = {
            'o': 'White and Blue Led',
            's': 'White Led',
            '^': 'Shade',
            'D': 'control'
        }
    else:
        # Use default anthocyanin categories
        letters = ["R", "G"]
        count = 5
        grp_A = [f"{letter}{i}" for letter in letters for i in range(1, count + 1)]
        grp_B = [f"{letter}{i}" for letter in letters for i in range(6, 2*count + 1)]
        grp_C = [f"{letter}{i}" for letter in letters for i in range(11, 3*count + 1)]
        grp_D = [f"C{i}" for i in range(1, 2*count + 1)]

        # mark the groups. Common marker options:
        # 'o' → circle, 's' → square, '^' → triangle up, 'D' → diamond
        markers = {a: 'o' for a in grp_A}
        markers.update({b: 's' for b in grp_B})
        markers.update({c: '^' for c in grp_C})
        markers.update({d: 'D' for d in grp_D})

        labels = {
            'o': 'White and Blue Led',
            's': 'White Led',
            '^': 'Shade',
            'D': 'control'
        }
    
    return markers, labels


def set_anthocyanin_categories_shapes_in_plot(ax, X, y, df, target, indicator, 
                                           categories: Dict = None):
    """
    Set anthocyanin category shapes in plot.
    
    PARAMETERS:
    - ax: matplotlib axis object
    - X: feature values
    - y: target values
    - df: dataframe containing the data
    - target: target column name
    - indicator: indicator column name
    - categories: optional categories dict from config
    """
    markers, labels = set_anthocyanin_markers(categories)
    
    y_to_mark = {}
    for index, row in df.iterrows():
        point_shape = markers.get(row[indicator], 'v')  # default shape: triangle down
        y_to_mark[row[target]] = point_shape
    
    # assign the shapes to the points
    for mark in set(markers.values()):
        y_points = [y_val for y_val in y if y_to_mark.get(y_val) == mark]
        x_points = [x_val for x_val, y_val in zip(X, y) if y_to_mark.get(y_val) == mark]
        ax.scatter(x_points, y_points, marker=mark, label=labels[mark])


def plot_anthocyanin_values_in_one_line_plot(plt, y_arr, df, target, indicator,
                                           trainable_features=None, hp_order='vsh',
                                           categories: Dict = None):
    """
    Plot anthocyanin values in one line plot.
    
    PARAMETERS:
    - plt: matplotlib pyplot object
    - y_arr: array of y values
    - df: dataframe containing the data
    - target: target column name
    - indicator: indicator column name
    - trainable_features: list of trainable features
    - hp_order: hyperparameter order
    - categories: optional categories dict from config
    
    RETURNS:
    - matplotlib plot object
    """
    markers, labels = set_anthocyanin_markers(categories)
    
    y_to_mark = {}
    for index, row in df.iterrows():
        point_shape = markers.get(row[indicator], 'v')  # default shape: triangle down
        y_to_mark[row[target]] = point_shape
    
    # assign the shapes to the points
    for mark in set(markers.values()):
        y_points = [y for y in y_arr if y_to_mark.get(y) == mark]
        dummy_values = np.zeros_like(y_points)
        plt.scatter(dummy_values, y_points, marker=mark, label=labels[mark])
    
    # annotate the indicator values and hyper-parameter values (as string)
    ax = plt.gca()
    color_map = get_anthocyanin_color_map(categories)
    
    for idx, row in df.iterrows():
        point_color = color_map.get(row[indicator], 'black')
        ax.annotate(
            str(row[indicator]),
            (0, row[target]),
            textcoords='offset points',
            xytext=(15, 0),
            ha='left',
            color=point_color
        )
        
        if trainable_features and hp_order == 'vsh':
            hyperParameter_lst = row[trainable_features[:-1]].values.tolist()
            ax.annotate(
                str(hyperParameter_lst),
                (row[trainable_features[0]], row[target]),
                textcoords='offset points',
                xytext=(0, 10),
                ha='center',
                color=point_color
            )
    
    return plt


def plot_anthocyanin_linear_regression(trainable_features, target, df,
                                     indicator='', color_map={}, condition=None,
                                     plot_separate=False, show=True,
                                     categories: Dict = None):
    """
    Anthocyanin-specific linear regression plotting.
    
    PARAMETERS:
    - trainable_features: list of feature column names
    - target: target column name
    - df: dataframe containing the data
    - indicator: column name for point indicators
    - color_map: color mapping for points
    - condition: condition for filtering
    - plot_separate: whether to create separate plots
    - show: whether to display the plot
    - categories: optional categories dict from config
    
    RETURNS:
    - matplotlib plot object
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.metrics import root_mean_squared_error
    
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
        
        # Plot points with anthocyanin-specific styling
        if target == 'Anthocyanin':
            if not color_map:
                color_map = get_anthocyanin_color_map(categories)
            set_anthocyanin_categories_shapes_in_plot(
                ax, X, y, sub_df, target, indicator, categories
            )
        else:
            ax.scatter(X, y, label='Data Points')
        
        # Formatting
        ax.set_xlabel(trainable_features[0])
        ax.set_ylabel(target)
        ax.set_title(f'Linear Regression {title_suffix} (R²={r2:.3f}, RMSE={rmse:.3f})')
        ax.grid(True)
        ax.legend()
    
    if plot_separate:
        # If plotting separate subplots, define categories
        categories_list = ['White and Blue Led', 'White Led', 'Shade', 'Control']
        n_rows, n_cols = 2, 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10), sharey=True)
        axes = axes.flatten()
        
        for ax, cat in zip(axes, categories_list):
            # Filter by category (would need specific implementation)
            sub_df = df  # Placeholder - would need filter_df_by_category
            plot_single(ax, sub_df, title_suffix=f'{cat}')
        
        # Hide any unused axes
        for i in range(len(categories_list), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
    else:
        plt.figure(figsize=(10, 5))
        plot_single(plt.gca(), df, title_suffix='Linear Regression')
    
    if show:
        plt.show()
    
    return plt
