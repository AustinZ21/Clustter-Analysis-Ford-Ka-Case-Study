from typing import Dict, Optional
import pandas as pd
from ..data.data_loader import DataLoader
from ..data.preprocessor import DataPreprocessor
from .cluster_analyzer import ClusterAnalyzer
from ..visualization.visualizer import Visualizer

class AnalysisCoordinator:
    """Coordinates the Ford Ka analysis process"""
    
    def __init__(self, data_path: str):
        """
        Initialize coordinator with components
        Args:
            data_path: Path to data directory
        """
        self.data_loader = DataLoader(data_path)
        self.preprocessor = DataPreprocessor()
        self.cluster_analyzer = ClusterAnalyzer()
        self.visualizer = Visualizer()
        
        # Data containers
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.processed_data: Dict[str, pd.DataFrame] = {}
        self.clustering_results: Dict[str, Dict] = {}
    
    def run_analysis(self) -> None:
        """Run the complete analysis pipeline"""
        try:
            # Load and preprocess data
            self._load_and_preprocess_data()
            
            # Analyze Q1 and PreferenceGroup relationship first
            self._analyze_q1_relationship()
            
            # Perform clustering analysis
            self._perform_clustering_analysis()
            
            # Analyze groups
            self._analyze_groups()
            
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            raise
    
    def _load_and_preprocess_data(self) -> None:
        """Load and preprocess all data"""
        print("\nLoading and preprocessing data...")
        
        # Load raw data
        self.raw_data = self.data_loader.load_all_data()
        
        # Preprocess data
        self.processed_data = self.preprocessor.preprocess_all_data(
            self.raw_data
        )
    
    def _analyze_q1_relationship(self) -> None:
        """Analyze relationship between Q1 and PreferenceGroup"""
        print("\n" + "="*50)
        print("Q1 and PreferenceGroup Relationship Analysis")
        print("="*50)
        
        # Basic Q1 statistics
        q1_data = self.raw_data['combined']['Q1']
        print("\nOverall Q1 Statistics:")
        print(f"Mean: {q1_data.mean():.2f}")
        print(f"Median: {q1_data.median():.2f}")
        print(f"Std Dev: {q1_data.std():.2f}")
        print(f"Range: {q1_data.min():.0f} - {q1_data.max():.0f}")
        
        # Create cross-tabulation
        print("\nCross-tabulation of Q1 and PreferenceGroup:")
        crosstab = pd.crosstab(
            self.raw_data['combined']['Q1'],
            self.raw_data['combined']['PreferenceGroup'],
            margins=True
        )
        print(crosstab)
        
        # Calculate percentages
        print("\nPercentage Distribution within each PreferenceGroup:")
        pct_crosstab = pd.crosstab(
            self.raw_data['combined']['Q1'],
            self.raw_data['combined']['PreferenceGroup'],
            normalize='columns'
        ) * 100
        print(pct_crosstab.round(1))
        
        # Create visualizations
        print("\nGenerating Q1 analysis visualizations...")
        self.visualizer.plot_q1_preference_analysis(
            self.raw_data['combined']
        )
    
    def _perform_clustering_analysis(self) -> None:
        """Perform clustering analysis on both demographic and psychographic data"""
        print("\nPerforming clustering analysis...")
        
        # Demographic clustering
        self.clustering_results['demographic'] = self.cluster_analyzer.find_optimal_k(
            self.processed_data['demo_std'],
            'demographic'
        )
        self.visualizer.plot_optimal_k(
            self.clustering_results['demographic'],
            'demographic'
        )
        self.visualizer.plot_cluster_analysis(
            self.raw_data['demographic'],
            self.clustering_results['demographic'],
            'demographic'
        )
        
        # Psychographic clustering
        self.clustering_results['psychographic'] = self.cluster_analyzer.find_optimal_k(
            self.processed_data['psyc_std'],
            'psychographic'
        )
        self.visualizer.plot_optimal_k(
            self.clustering_results['psychographic'],
            'psychographic'
        )
        self.visualizer.plot_cluster_analysis(
            self.raw_data['psychographic'],
            self.clustering_results['psychographic'],
            'psychographic'
        )
        
        # Compare clusters with preference groups
        if 'PreferenceGroup' in self.raw_data['combined'].columns:
            self._compare_clusters_with_preferences()
    
    def _compare_clusters_with_preferences(self) -> None:
        """Compare clustering results with actual preference groups"""
        print("\nComparing clusters with preference groups...")
        
        # Compare demographic clusters
        demo_comparison = self.cluster_analyzer.compare_clusters_with_preference_groups(
            self.clustering_results['demographic']['labels'],
            self.raw_data['combined']['PreferenceGroup']
        )
        print("\nDemographic Clusters vs PreferenceGroup:")
        print(demo_comparison)
        
        # Compare psychographic clusters
        psyc_comparison = self.cluster_analyzer.compare_clusters_with_preference_groups(
            self.clustering_results['psychographic']['labels'],
            self.raw_data['combined']['PreferenceGroup']
        )
        print("\nPsychographic Clusters vs PreferenceGroup:")
        print(psyc_comparison)
    
    def _analyze_groups(self) -> None:
        """Analyze characteristics of each preference group"""
        print("\nAnalyzing preference groups...")
        
        for group in [1, 2, 3]:  # Assuming these are the preference groups
            group_analysis = self.cluster_analyzer.analyze_clusters(
                self.raw_data['combined'],
                self.raw_data['combined']['PreferenceGroup'] == group,
                self.preprocessor.demographic_variables + 
                self.preprocessor.psychographic_variables
            )
            
            # Visualize group analysis
            self.visualizer.plot_group_analysis(
                self.raw_data['combined'],
                group,
                group_analysis
            )
            
            print(f"\nGroup {group} Analysis:")
            print(f"Size: {group_analysis[f'Cluster_0']['size']} respondents")
            print(f"Proportion: {group_analysis[f'Cluster_0']['proportion']:.2%}")
            
            # Print top characteristics
            print("\nTop Characteristics:")
            for var, stats in sorted(
                group_analysis[f'Cluster_0']['variables'].items(),
                key=lambda x: abs(x[1]['mean']),
                reverse=True
            )[:5]:
                print(f"{var}: {stats['mean']:.2f} (±{stats['std']:.2f})")
                
    def get_results(self) -> Dict:
        """
        Get all analysis results
        Returns:
            Dictionary containing all analysis results
        """
        return {
            'raw_data': self.raw_data,
            'processed_data': self.processed_data,
            'clustering_results': self.clustering_results
        } 