"""Visualization module for Ford Ka analysis.

This module implements visualization best practices including:
- Consistent styling and theming
- Clear and informative plots
- Error handling and input validation
- Comprehensive documentation
- Accessibility considerations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Visualizer:
    """Handles creation and management of visualizations."""
    
    def __init__(self, output_dir: Union[str, Path] = 'plots'):
        """Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.setup_style()
        
    def setup_style(self) -> None:
        """Set up consistent plot styling."""
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # Set up consistent styling
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
        
    def save_plot(self, name: str, fig: Optional[plt.Figure] = None, 
                 tight_layout: bool = True) -> None:
        """Save plot to file.
        
        Args:
            name: Name of the plot file
            fig: Figure to save (if None, uses current figure)
            tight_layout: Whether to apply tight_layout before saving
        """
        try:
            if tight_layout:
                plt.tight_layout()
            plt.savefig(self.output_dir / f"{name}.png", dpi=300, bbox_inches='tight')
            plt.close(fig if fig is not None else plt.gcf())
            logger.info(f"Saved plot: {name}.png")
        except Exception as e:
            logger.error(f"Error saving plot {name}: {str(e)}")
            raise
    
    def create_distribution_plot(self, data: pd.DataFrame, variable: str,
                               title: Optional[str] = None) -> None:
        """Create distribution plot with statistical annotations.
        
        Args:
            data: DataFrame containing the data
            variable: Variable to plot
            title: Optional plot title
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram with KDE
        sns.histplot(data=data, x=variable, kde=True, ax=ax1)
        ax1.set_title(f'{variable} Distribution')
        
        # Box plot
        sns.boxplot(data=data, y=variable, ax=ax2)
        ax2.set_title(f'{variable} Box Plot')
        
        # Add statistical annotations
        stats_text = (
            f"Mean: {data[variable].mean():.2f}\n"
            f"Median: {data[variable].median():.2f}\n"
            f"Std: {data[variable].std():.2f}\n"
            f"Skew: {data[variable].skew():.2f}"
        )
        ax1.text(0.95, 0.95, stats_text,
                transform=ax1.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if title:
            fig.suptitle(title)
            
        self.save_plot(f"{variable.lower()}_distribution")
    
    def create_correlation_heatmap(self, data: pd.DataFrame, 
                                 variables: List[str],
                                 title: Optional[str] = None) -> None:
        """Create correlation heatmap with significance levels.
        
        Args:
            data: DataFrame containing the data
            variables: List of variables to include
            title: Optional plot title
        """
        # Calculate correlations
        corr = data[variables].corr()
        
        # Calculate p-values
        p_values = pd.DataFrame(np.zeros_like(corr), 
                              index=corr.index, 
                              columns=corr.columns)
        
        for i in range(len(variables)):
            for j in range(len(variables)):
                if i != j:
                    stat, p = stats.pearsonr(data[variables[i]].dropna(),
                                           data[variables[j]].dropna())
                    p_values.iloc[i, j] = p
        
        # Create plot
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr))
        
        # Plot correlations
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', square=True)
        
        # Add significance markers
        for i in range(len(variables)):
            for j in range(len(variables)):
                if i < j:  # Upper triangle only
                    if p_values.iloc[i, j] < 0.001:
                        plt.text(j+0.5, i+0.5, '***', ha='center', va='center')
                    elif p_values.iloc[i, j] < 0.01:
                        plt.text(j+0.5, i+0.5, '**', ha='center', va='center')
                    elif p_values.iloc[i, j] < 0.05:
                        plt.text(j+0.5, i+0.5, '*', ha='center', va='center')
        
        if title:
            plt.title(title)
            
        self.save_plot('correlation_heatmap')
    
    def create_group_comparison(self, data: pd.DataFrame, variable: str,
                              group_col: str = 'PreferenceGroup',
                              title: Optional[str] = None) -> None:
        """Create group comparison plot with statistical annotations.
        
        Args:
            data: DataFrame containing the data
            variable: Variable to compare
            group_col: Column containing group labels
            title: Optional plot title
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot
        sns.boxplot(data=data, x=group_col, y=variable, ax=ax1)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        
        # Violin plot
        sns.violinplot(data=data, x=group_col, y=variable, ax=ax2)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        
        # Perform ANOVA
        groups = [group for _, group in data.groupby(group_col)[variable]]
        f_stat, p_val = stats.f_oneway(*groups)
        
        # Add statistical annotation
        stats_text = (
            f"ANOVA:\nF-stat: {f_stat:.2f}\n"
            f"p-value: {p_val:.4f}"
        )
        ax1.text(0.95, 0.95, stats_text,
                transform=ax1.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if title:
            fig.suptitle(title)
            
        self.save_plot(f"{variable.lower()}_group_comparison")
    
    def create_feature_importance_plot(self, importance_df: pd.DataFrame,
                                     title: Optional[str] = None,
                                     n_features: int = 10) -> None:
        """Create feature importance plot.
        
        Args:
            importance_df: DataFrame with feature importance scores
            title: Optional plot title
            n_features: Number of top features to show
        """
        plt.figure(figsize=(10, 6))
        
        # Sort and select top features
        plot_data = importance_df.sort_values('importance', ascending=True).tail(n_features)
        
        # Create horizontal bar plot
        sns.barplot(data=plot_data, y='feature', x='importance')
        
        if title:
            plt.title(title)
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        
        self.save_plot('feature_importance')
    
    def create_cluster_visualization(self, X: np.ndarray, labels: np.ndarray,
                                   feature_names: List[str],
                                   title: Optional[str] = None) -> None:
        """Create comprehensive cluster visualization.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            feature_names: Names of features
            title: Optional plot title
        """
        # Create PCA projection
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 5))
        
        # Plot 1: PCA visualization
        ax1 = fig.add_subplot(131)
        scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
        ax1.set_title('PCA Visualization')
        plt.colorbar(scatter, ax=ax1)
        
        # Plot 2: Cluster sizes
        ax2 = fig.add_subplot(132)
        sns.countplot(x=labels, ax=ax2)
        ax2.set_title('Cluster Sizes')
        
        # Plot 3: Feature distributions by cluster
        ax3 = fig.add_subplot(133)
        cluster_means = pd.DataFrame(X, columns=feature_names).groupby(labels).mean()
        sns.heatmap(cluster_means, cmap='coolwarm', center=0, ax=ax3)
        ax3.set_title('Feature Means by Cluster')
        
        if title:
            fig.suptitle(title)
            
        self.save_plot('cluster_visualization')
    
    def create_demographic_profile(self, data: pd.DataFrame,
                                 demographic_vars: List[str],
                                 group_col: Optional[str] = None) -> None:
        """Create demographic profile visualization.
        
        Args:
            data: DataFrame containing demographic data
            demographic_vars: List of demographic variables
            group_col: Optional column for grouping
        """
        n_vars = len(demographic_vars)
        fig = plt.figure(figsize=(15, 5 * ((n_vars + 1) // 2)))
        
        for i, var in enumerate(demographic_vars, 1):
            ax = fig.add_subplot((n_vars + 1) // 2, 2, i)
            
            if data[var].dtype in ['int64', 'float64']:
                if group_col:
                    sns.boxplot(data=data, x=group_col, y=var, ax=ax)
                else:
                    sns.histplot(data=data, x=var, kde=True, ax=ax)
            else:
                if group_col:
                    sns.countplot(data=data, x=var, hue=group_col, ax=ax)
                else:
                    sns.countplot(data=data, x=var, ax=ax)
                    
            ax.set_title(f'{var} Distribution')
            if data[var].dtype not in ['int64', 'float64']:
                plt.xticks(rotation=45)
        
        plt.tight_layout()
        self.save_plot('demographic_profile')

def plot_group_demographics(group_data: pd.DataFrame, group_name: str):
    """Create comprehensive demographic plots for a group."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Demographic Profile of {group_name}\n', fontsize=16, y=1.02)

    # Age Distribution
    sns.histplot(data=group_data, x='Age', bins=20, ax=ax1)
    mean_age = group_data['Age'].mean()
    ax1.axvline(mean_age, color='r', linestyle='--', label=f'Mean Age: {mean_age:.1f}')
    ax1.set_title('Age Distribution')
    ax1.legend()

    # Marital Status and Trendiness
    sns.boxplot(data=group_data, x='MaritalStatus', y='Q1', ax=ax2)
    ax2.set_title('Trendiness by Marital Status')
    marital_means = group_data.groupby('MaritalStatus')['Q1'].mean()
    for i, mean in enumerate(marital_means):
        ax2.text(i, mean, f'{mean:.2f}', ha='center', va='bottom')

    # Income Distribution
    income_counts = group_data['IncomeCategory'].value_counts().sort_index()
    ax3.pie(income_counts, labels=[f'Income {i}' for i in income_counts.index],
            autopct='%1.1f%%', startangle=90)
    ax3.set_title('Income Distribution')

    # Trendiness by Income and Children
    sns.boxplot(data=group_data, x='IncomeCategory', y='Q1', 
                hue='NumberChildren', ax=ax4)
    ax4.set_title('Trendiness by Income and Number of Children')

    save_plot(f"{group_name.lower().replace(' ', '_')}_demographics")

def plot_psychographic_profile(data: pd.DataFrame, questions: list, 
                             title: Optional[str] = None):
    """Create radar chart of psychographic profile."""
    # Prepare data
    mean_profiles = data.groupby('PreferenceGroup')[questions].mean()
    
    # Set up the angles
    angles = np.linspace(0, 2*np.pi, len(questions), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Plot data for each group
    for group in mean_profiles.index:
        values = mean_profiles.loc[group].values
        values = np.concatenate((values, [values[0]]))
        ax.plot(angles, values, 'o-', linewidth=2, label=group)
        ax.fill(angles, values, alpha=0.25)
    
    # Customize plot
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    
    # Create labels with descriptions
    labels = [f"{q}\n{QUESTION_DESCRIPTIONS.get(q, '')}" for q in questions]
    ax.set_xticklabels(labels)
    
    plt.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))
    if title:
        plt.title(title)
        
    save_plot('psychographic_profile')

def plot_group_comparisons(data: pd.DataFrame):
    """Create plots comparing all groups."""
    # Trendiness Distribution
    plt.figure(figsize=FIGURE_SIZE)
    sns.boxplot(data=data, x='PreferenceGroup', y='Q1')
    plt.title('Trendiness Scores Distribution by Group')
    plt.xticks(rotation=45)
    save_plot('trendiness_distribution')
    
    # Income Distribution
    plt.figure(figsize=FIGURE_SIZE)
    income_dist = pd.crosstab(data['PreferenceGroup'], 
                             data['IncomeCategory'], 
                             normalize='index') * 100
    income_dist.plot(kind='bar', stacked=True)
    plt.title('Income Distribution by Group')
    plt.legend(title='Income Category', bbox_to_anchor=(1.05, 1))
    save_plot('income_distribution_by_group')
    
    # Age Distribution
    plt.figure(figsize=FIGURE_SIZE)
    age_dist = pd.crosstab(data['PreferenceGroup'], 
                          data['AgeGroup'], 
                          normalize='index') * 100
    age_dist.plot(kind='bar', stacked=True)
    plt.title('Age Distribution by Group')
    plt.legend(title='Age Group', bbox_to_anchor=(1.05, 1))
    save_plot('age_distribution_by_group') 