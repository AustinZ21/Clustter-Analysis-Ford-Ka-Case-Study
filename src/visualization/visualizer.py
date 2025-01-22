import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from scipy.stats import chi2_contingency

class Visualizer:
    """Handles visualization of Ford Ka analysis results"""
    
    def __init__(self):
        """Initialize visualizer with default style settings"""
        # Use a default matplotlib style instead of seaborn
        plt.style.use('default')
        sns.set_theme()  # This will set up seaborn's default styling
        self.figsize = (15, 10)
    
    def plot_optimal_k(self, results: Dict, analysis_type: str) -> None:
        """
        Plot elbow curve and silhouette scores for optimal k analysis
        Args:
            results: Dictionary containing clustering results
            analysis_type: Type of analysis ('demographic' or 'psychographic')
        """
        plt.figure(figsize=(15, 5))
        
        # Elbow curve
        plt.subplot(1, 2, 1)
        plt.plot(results['k_values'], results['inertias'], marker='o')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Inertia')
        plt.title(f'Elbow Method - {analysis_type.title()} Data')
        
        # Silhouette scores
        plt.subplot(1, 2, 2)
        plt.plot(results['k_values'], results['silhouette_scores'], marker='o')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Silhouette Score')
        plt.title(f'Silhouette Analysis - {analysis_type.title()} Data')
        
        plt.tight_layout()
        plt.savefig(f'optimal_k_{analysis_type.lower()}.png')
        plt.close()
    
    def plot_cluster_analysis(self, data: pd.DataFrame, 
                            results: Dict, 
                            analysis_type: str) -> None:
        """
        Create detailed cluster analysis plots
        Args:
            data: Original data
            results: Dictionary containing clustering results
            analysis_type: Type of analysis ('demographic' or 'psychographic')
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        if analysis_type == 'demographic':
            self._plot_demographic_clusters(data, results, axes)
        else:
            self._plot_psychographic_clusters(data, results, axes)
        
        plt.tight_layout()
        plt.savefig(f'{analysis_type}_cluster_analysis.png')
        plt.close()
    
    def _plot_demographic_clusters(self, data: pd.DataFrame, 
                                 results: Dict, 
                                 axes: np.ndarray) -> None:
        """Create demographic cluster analysis plots"""
        # Scatter plot of Age vs NumberChildren
        sns.scatterplot(
            data=data,
            x='Age',
            y='NumberChildren',
            hue=results['labels'],
            ax=axes[0, 0]
        )
        axes[0, 0].set_title('Age vs NumberChildren by Cluster')
        
        # Distribution of clusters across PreferenceGroup
        if 'PreferenceGroup' in data.columns:
            pd.crosstab(
                results['labels'],
                data['PreferenceGroup']
            ).plot(kind='bar', ax=axes[0, 1])
            axes[0, 1].set_title('Clusters vs PreferenceGroup')
        
        # Parallel coordinates plot of cluster centers
        # Get only the columns that exist in both data and centers
        common_cols = [col for col in data.columns if col in self.demo_vars]
        centers = pd.DataFrame(
            results['centers'],
            columns=common_cols
        )
        if not centers.empty:
            pd.plotting.parallel_coordinates(
                centers.reset_index().rename(columns={'index': 'Cluster'}),
                'Cluster',
                ax=axes[1, 0]
            )
            axes[1, 0].set_title('Cluster Centers')
        
        # Cluster sizes
        pd.Series(results['labels']).value_counts().plot(
            kind='bar',
            ax=axes[1, 1]
        )
        axes[1, 1].set_title('Cluster Sizes')
    
    def _plot_psychographic_clusters(self, data: pd.DataFrame, 
                                   results: Dict, 
                                   axes: np.ndarray) -> None:
        """Create psychographic cluster analysis plots"""
        # Scatter plot of first two questions
        if 'Q1' in data.columns and 'Q2' in data.columns:
            sns.scatterplot(
                data=data,
                x='Q1',
                y='Q2',
                hue=results['labels'],
                ax=axes[0, 0]
            )
            axes[0, 0].set_title('Q1 vs Q2 by Cluster')
        
        # Distribution of clusters across PreferenceGroup
        if 'PreferenceGroup' in data.columns:
            pd.crosstab(
                results['labels'],
                data['PreferenceGroup']
            ).plot(kind='bar', ax=axes[0, 1])
            axes[0, 1].set_title('Clusters vs PreferenceGroup')
        
        # Parallel coordinates plot of selected questions
        selected_qs = [q for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'] 
                      if q in data.columns]
        if selected_qs:
            centers = pd.DataFrame(
                results['centers'],
                columns=data.columns
            )
            pd.plotting.parallel_coordinates(
                centers[selected_qs].reset_index().rename(
                    columns={'index': 'Cluster'}),
                'Cluster',
                ax=axes[1, 0]
            )
            axes[1, 0].set_title('Cluster Centers (Selected Questions)')
        
        # Cluster sizes
        pd.Series(results['labels']).value_counts().plot(
            kind='bar',
            ax=axes[1, 1]
        )
        axes[1, 1].set_title('Cluster Sizes')
    
    def plot_group_analysis(self, data: pd.DataFrame, 
                          group: int, 
                          analysis_results: Dict) -> None:
        """
        Create visualizations for group analysis
        Args:
            data: Original data
            group: Group number to analyze
            analysis_results: Dictionary containing group analysis results
        """
        plt.figure(figsize=(15, 10))
        
        # Age distribution
        plt.subplot(2, 2, 1)
        sns.histplot(
            data=data[data['PreferenceGroup'] == group],
            x='Age',
            bins=20
        )
        plt.title(f'Age Distribution of Group {group}')
        
        # Income vs Number of Children
        plt.subplot(2, 2, 2)
        sns.scatterplot(
            data=data[data['PreferenceGroup'] == group],
            x='IncomeCategory',
            y='NumberChildren'
        )
        plt.title('Income vs Number of Children')
        
        # Marital Status by Gender
        plt.subplot(2, 2, 3)
        group_data = data[data['PreferenceGroup'] == group]
        pd.crosstab(
            group_data['MaritalStatus'],
            group_data['Gender']
        ).plot(kind='bar', stacked=True)
        plt.title('Marital Status by Gender')
        
        # Top question scores
        plt.subplot(2, 2, 4)
        if 'top_questions' in analysis_results:
            pd.Series(analysis_results['top_questions']).plot(kind='bar')
            plt.title('Top Scoring Questions')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'group_{group}_analysis.png')
        plt.close()
    
    def plot_q1_preference_analysis(self, data: pd.DataFrame) -> None:
        """
        Analyze and visualize relationship between Q1 and PreferenceGroup
        Args:
            data: DataFrame containing Q1 and PreferenceGroup columns
        """
        plt.figure(figsize=(15, 10))
        
        # 1. Cross-tabulation heatmap
        plt.subplot(2, 2, 1)
        crosstab = pd.crosstab(data['Q1'], data['PreferenceGroup'])
        sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('Q1 vs PreferenceGroup Distribution')
        plt.xlabel('PreferenceGroup')
        plt.ylabel('Q1 Response')
        
        # 2. Percentage distribution
        plt.subplot(2, 2, 2)
        pct_crosstab = pd.crosstab(
            data['Q1'], 
            data['PreferenceGroup'], 
            normalize='columns'
        ) * 100
        sns.heatmap(pct_crosstab, annot=True, fmt='.1f', cmap='YlOrRd')
        plt.title('Q1 vs PreferenceGroup (% within Group)')
        plt.xlabel('PreferenceGroup')
        plt.ylabel('Q1 Response')
        
        # 3. Box plot
        plt.subplot(2, 2, 3)
        sns.boxplot(data=data, x='PreferenceGroup', y='Q1')
        plt.title('Q1 Distribution by PreferenceGroup')
        
        # 4. Violin plot
        plt.subplot(2, 2, 4)
        sns.violinplot(data=data, x='PreferenceGroup', y='Q1')
        plt.title('Q1 Distribution Density by PreferenceGroup')
        
        plt.tight_layout()
        plt.savefig('q1_preference_analysis.png')
        plt.close()
        
        # Print statistical analysis
        print("\nQ1 vs PreferenceGroup Analysis:")
        
        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(crosstab)
        print(f"\nChi-square test:")
        print(f"Chi-square statistic: {chi2:.2f}")
        print(f"p-value: {p_value:.4f}")
        print(f"Degrees of freedom: {dof}")
        
        # Basic statistics by group
        print("\nQ1 Statistics by PreferenceGroup:")
        stats = data.groupby('PreferenceGroup')['Q1'].agg([
            'count', 'mean', 'std', 'min', 'median', 'max'
        ])
        print(stats.round(2))
        
        # Response distribution
        print("\nQ1 Response Distribution (%) by PreferenceGroup:")
        print(pct_crosstab.round(1))
    
    @property
    def demo_vars(self) -> List[str]:
        """Get list of demographic variables"""
        return ['Age', 'MaritalStatus', 'Gender', 'NumberChildren', 
                'IncomeCategory', 'FirstTimePurchase']
    
    @property
    def psyc_vars(self) -> List[str]:
        """Get list of psychographic variables"""
        return [f'Q{i}' for i in range(1, 63)] 