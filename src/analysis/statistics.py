"""Statistical analysis module for Ford Ka customer data.

This module implements statistical analysis best practices including:
- Hypothesis testing
- Effect size calculations
- Correlation analysis
- Distribution testing
- Statistical validation
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class StatisticalAnalyzer:
    """Advanced statistical analysis for customer data."""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize with customer data.
        
        Args:
            data: DataFrame containing customer data
        """
        self.data = data
        self.scaler = StandardScaler()
        
    def test_normality(self, variable: str) -> Dict[str, float]:
        """Test for normality using multiple methods.
        
        Args:
            variable: Name of the variable to test
            
        Returns:
            Dictionary containing test results
        """
        data = self.data[variable].dropna()
        
        # Shapiro-Wilk test
        shapiro_stat, shapiro_p = stats.shapiro(data)
        
        # D'Agostino's K^2 test
        k2_stat, k2_p = stats.normaltest(data)
        
        results = {
            'shapiro_statistic': shapiro_stat,
            'shapiro_pvalue': shapiro_p,
            'k2_statistic': k2_stat,
            'k2_pvalue': k2_p,
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data)
        }
        
        logger.info(f"Normality test results for {variable}:")
        logger.info(f"Shapiro-Wilk p-value: {shapiro_p:.4f}")
        logger.info(f"D'Agostino's K^2 p-value: {k2_p:.4f}")
        
        return results
    
    def compare_groups(self, variable: str, grouping: str) -> Dict[str, float]:
        """Compare variable across groups using appropriate statistical tests.
        
        Args:
            variable: Variable to compare
            grouping: Grouping variable
            
        Returns:
            Dictionary containing test results
        """
        groups = [group for _, group in self.data.groupby(grouping)[variable]]
        
        # Test for normality first
        normal_distribution = all(
            stats.shapiro(group)[1] > 0.05 for group in groups
        )
        
        if normal_distribution:
            # Use one-way ANOVA for normal distributions
            f_stat, p_val = stats.f_oneway(*groups)
            test_name = "ANOVA"
        else:
            # Use Kruskal-Wallis for non-normal distributions
            h_stat, p_val = stats.kruskal(*groups)
            test_name = "Kruskal-Wallis"
        
        # Calculate effect size (Eta-squared)
        effect_size = self._calculate_effect_size(variable, grouping)
        
        results = {
            'test_name': test_name,
            'statistic': f_stat if normal_distribution else h_stat,
            'p_value': p_val,
            'effect_size': effect_size
        }
        
        logger.info(f"Group comparison results for {variable} by {grouping}:")
        logger.info(f"Test used: {test_name}")
        logger.info(f"p-value: {p_val:.4f}")
        
        return results
    
    def correlation_analysis(self, variables: List[str], 
                           method: str = 'pearson') -> pd.DataFrame:
        """Perform correlation analysis with significance testing.
        
        Args:
            variables: List of variables to analyze
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            
        Returns:
            DataFrame containing correlation coefficients and p-values
        """
        if method not in ['pearson', 'spearman', 'kendall']:
            raise ValueError("Method must be 'pearson', 'spearman', or 'kendall'")
        
        corr_matrix = pd.DataFrame(index=variables, columns=variables)
        p_values = pd.DataFrame(index=variables, columns=variables)
        
        for var1 in variables:
            for var2 in variables:
                if method == 'pearson':
                    coef, p_val = stats.pearsonr(
                        self.data[var1].dropna(),
                        self.data[var2].dropna()
                    )
                elif method == 'spearman':
                    coef, p_val = stats.spearmanr(
                        self.data[var1].dropna(),
                        self.data[var2].dropna()
                    )
                else:  # kendall
                    coef, p_val = stats.kendalltau(
                        self.data[var1].dropna(),
                        self.data[var2].dropna()
                    )
                    
                corr_matrix.loc[var1, var2] = coef
                p_values.loc[var1, var2] = p_val
        
        logger.info(f"Completed {method} correlation analysis")
        return {'correlations': corr_matrix, 'p_values': p_values}
    
    def _calculate_effect_size(self, variable: str, grouping: str) -> float:
        """Calculate eta-squared effect size.
        
        Args:
            variable: Variable to analyze
            grouping: Grouping variable
            
        Returns:
            Eta-squared effect size
        """
        groups = self.data.groupby(grouping)[variable]
        ss_between = sum(len(group) * ((group.mean() - self.data[variable].mean()) ** 2) 
                        for name, group in groups)
        ss_total = sum((self.data[variable] - self.data[variable].mean()) ** 2)
        
        return ss_between / ss_total
    
    def visualize_distributions(self, variables: List[str]) -> None:
        """Create distribution plots with statistical annotations.
        
        Args:
            variables: List of variables to visualize
        """
        n_vars = len(variables)
        fig, axes = plt.subplots(n_vars, 2, figsize=(12, 4*n_vars))
        
        for i, var in enumerate(variables):
            # Histogram with KDE
            sns.histplot(data=self.data, x=var, kde=True, ax=axes[i, 0])
            axes[i, 0].set_title(f'{var} Distribution')
            
            # Q-Q plot
            stats.probplot(self.data[var].dropna(), dist="norm", plot=axes[i, 1])
            axes[i, 1].set_title(f'{var} Q-Q Plot')
            
            # Add statistical annotations
            normality_results = self.test_normality(var)
            axes[i, 0].text(0.05, 0.95, 
                          f"Shapiro-Wilk p: {normality_results['shapiro_pvalue']:.4f}\n"
                          f"Skewness: {normality_results['skewness']:.2f}\n"
                          f"Kurtosis: {normality_results['kurtosis']:.2f}",
                          transform=axes[i, 0].transAxes,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('plots/distribution_analysis.png')
        plt.close()
        
    def analyze_relationships(self, target: str, features: List[str]) -> Dict:
        """Analyze relationships between target and feature variables.
        
        Args:
            target: Target variable
            features: List of feature variables
            
        Returns:
            Dictionary containing analysis results
        """
        results = {}
        
        for feature in features:
            # Calculate correlation
            corr_result = self.correlation_analysis([target, feature])
            correlation = corr_result['correlations'].iloc[0, 1]
            p_value = corr_result['p_values'].iloc[0, 1]
            
            # Calculate mutual information if categorical
            if self.data[feature].dtype == 'object' or self.data[target].dtype == 'object':
                from sklearn.metrics import mutual_info_score
                mi_score = mutual_info_score(
                    self.data[feature].fillna('missing'),
                    self.data[target].fillna('missing')
                )
            else:
                mi_score = None
            
            results[feature] = {
                'correlation': correlation,
                'p_value': p_value,
                'mutual_info': mi_score
            }
            
        return results

def calculate_group_statistics(data: pd.DataFrame, group_col: str = 'PreferenceGroup') -> Dict:
    """Calculate basic statistics for each group."""
    stats_dict = {}
    
    for group in data[group_col].unique():
        group_data = data[data[group_col] == group]
        
        stats_dict[group] = {
            'size': len(group_data),
            'age_mean': group_data['Age'].mean(),
            'age_std': group_data['Age'].std(),
            'income_mode': group_data['IncomeCategory'].mode().iloc[0],
            'marital_dist': group_data['MaritalStatus'].value_counts(normalize=True),
            'trendiness_mean': group_data['Q1'].mean(),
            'trendiness_std': group_data['Q1'].std()
        }
    
    return stats_dict

def find_differentiating_questions(data: pd.DataFrame, 
                                 n_top: int = 10) -> pd.Series:
    """Find questions that best differentiate between groups."""
    q_columns = [col for col in data.columns if col.startswith('Q')]
    variances = data.groupby('PreferenceGroup')[q_columns].mean().var()
    return variances.sort_values(ascending=False).head(n_top)

def compare_groups(data: pd.DataFrame, 
                  var: str, 
                  group1: str, 
                  group2: str) -> Dict:
    """Statistically compare two groups on a variable."""
    group1_data = data[data['PreferenceGroup'] == group1][var]
    group2_data = data[data['PreferenceGroup'] == group2][var]
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
    
    # Calculate effect size (Cohen's d)
    n1, n2 = len(group1_data), len(group2_data)
    var1, var2 = group1_data.var(), group2_data.var()
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_se
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'mean_difference': group1_data.mean() - group2_data.mean()
    }

def analyze_demographic_patterns(data: pd.DataFrame, 
                               group: str) -> Dict:
    """Analyze demographic patterns for a specific group."""
    group_data = data[data['PreferenceGroup'] == group]
    
    # Age analysis
    age_stats = group_data['Age'].describe()
    age_groups = pd.qcut(group_data['Age'], q=4)
    age_trendiness = group_data.groupby(age_groups)['Q1'].mean()
    
    # Income analysis
    income_dist = group_data['IncomeCategory'].value_counts(normalize=True)
    income_trendiness = group_data.groupby('IncomeCategory')['Q1'].mean()
    
    # Marital status analysis
    marital_dist = group_data['MaritalStatus'].value_counts(normalize=True)
    marital_trendiness = group_data.groupby('MaritalStatus')['Q1'].mean()
    
    return {
        'age_stats': age_stats,
        'age_trendiness': age_trendiness,
        'income_distribution': income_dist,
        'income_trendiness': income_trendiness,
        'marital_distribution': marital_dist,
        'marital_trendiness': marital_trendiness
    }

def calculate_correlations(data: pd.DataFrame, 
                         questions: List[str]) -> pd.DataFrame:
    """Calculate correlations between specified questions."""
    return data[questions].corr() 