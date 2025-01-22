"""Main script for Ford Ka customer analysis."""

import pandas as pd
from pathlib import Path
from data.data_loader import load_and_prepare_data
from visualization.plots import setup_plot_style
from analysis.group_analysis import GroupAnalyzer

def main():
    """Run the main analysis pipeline."""
    # Set up
    setup_plot_style()
    
    # Load and prepare data
    print("Loading and preparing data...")
    data_dict = load_and_prepare_data()
    
    # Initialize group analyzer
    analyzer = GroupAnalyzer(data_dict)
    
    # Analyze Group 1
    print("\nAnalyzing Group 1 (Ka Chooser - Top 3)...")
    group1_results = analyzer.analyze_group1()
    
    # Print key insights
    print("\nKey Insights for Group 1:")
    print(analyzer.get_key_insights())
    
    # Compare trendiness across groups
    print("\nTrendiness Comparison:")
    trendiness_comparison = analyzer.compare_with_other_groups('Q1')
    print(f"Difference vs Group 2: {trendiness_comparison['difference_vs_group2']:.2f}")
    print(f"Difference vs Group 3: {trendiness_comparison['difference_vs_group3']:.2f}")
    
    print("\nAnalysis complete! Check the outputs directory for visualizations.")

if __name__ == "__main__":
    main() 