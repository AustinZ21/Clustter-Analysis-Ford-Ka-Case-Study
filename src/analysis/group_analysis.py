"""Group-specific analysis functions."""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from ..visualization.plots import (plot_group_demographics, 
                                 plot_psychographic_profile)
from .statistics import (analyze_demographic_patterns,
                       find_differentiating_questions)

class GroupAnalyzer:
    """Class for analyzing specific Ford Ka customer groups."""
    
    def __init__(self, data_dict: Dict[str, pd.DataFrame]):
        """Initialize with data dictionary containing all groups."""
        self.full_data = data_dict['full_data']
        self.group1_data = data_dict['group1']
        self.group2_data = data_dict['group2']
        self.group3_data = data_dict['group3']
    
    def analyze_group1(self, output_plots: bool = True) -> Dict:
        """Comprehensive analysis of Group 1 (Ka Chooser - Top 3)."""
        results = {}
        
        # Basic demographics
        results['demographics'] = {
            'size': len(self.group1_data),
            'age_stats': self.group1_data['Age'].describe(),
            'marital_status': self.group1_data['MaritalStatus'].value_counts(normalize=True),
            'income': self.group1_data['IncomeCategory'].value_counts(normalize=True),
            'children': self.group1_data['NumberChildren'].value_counts(normalize=True)
        }
        
        # Trendiness analysis
        results['trendiness'] = {
            'mean': self.group1_data['Q1'].mean(),
            'median': self.group1_data['Q1'].median(),
            'std': self.group1_data['Q1'].std(),
            'by_marital': self.group1_data.groupby('MaritalStatus')['Q1'].mean(),
            'by_income': self.group1_data.groupby('IncomeCategory')['Q1'].mean(),
            'by_children': self.group1_data.groupby('NumberChildren')['Q1'].mean()
        }
        
        # Find differentiating questions
        results['differentiating_questions'] = find_differentiating_questions(
            self.full_data, n_top=10
        )
        
        # Demographic patterns
        results['patterns'] = analyze_demographic_patterns(
            self.full_data, 
            'Group 1 (Ka Chooser - Top 3)'
        )
        
        if output_plots:
            # Generate visualizations
            plot_group_demographics(
                self.group1_data, 
                "Group 1 (Ka Chooser - Top 3)"
            )
            
            plot_psychographic_profile(
                self.full_data,
                results['differentiating_questions'].index.tolist(),
                "Psychographic Profile Comparison"
            )
        
        return results
    
    def get_key_insights(self) -> str:
        """Generate key insights about Group 1."""
        insights = []
        
        # Age insights
        mean_age = self.group1_data['Age'].mean()
        age_std = self.group1_data['Age'].std()
        insights.append(f"Average age: {mean_age:.1f} years (±{age_std:.1f})")
        
        # Marital status
        married_pct = (self.group1_data['MaritalStatus'] == 1).mean() * 100
        insights.append(f"Married: {married_pct:.1f}%")
        
        # Income
        high_income = (self.group1_data['IncomeCategory'] >= 5).mean() * 100
        insights.append(f"High income (categories 5-6): {high_income:.1f}%")
        
        # Trendiness
        mean_trend = self.group1_data['Q1'].mean()
        insights.append(f"Average trendiness score: {mean_trend:.2f}")
        
        # Children
        with_children = (self.group1_data['NumberChildren'] > 0).mean() * 100
        insights.append(f"With children: {with_children:.1f}%")
        
        return "\n".join(insights)
    
    def compare_with_other_groups(self, variable: str) -> Dict:
        """Compare Group 1 with other groups on a specific variable."""
        group1_mean = self.group1_data[variable].mean()
        group2_mean = self.group2_data[variable].mean()
        group3_mean = self.group3_data[variable].mean()
        
        return {
            'group1_mean': group1_mean,
            'difference_vs_group2': group1_mean - group2_mean,
            'difference_vs_group3': group1_mean - group3_mean,
            'percent_difference_vs_group2': (group1_mean - group2_mean) / group2_mean * 100,
            'percent_difference_vs_group3': (group1_mean - group3_mean) / group3_mean * 100
        } 