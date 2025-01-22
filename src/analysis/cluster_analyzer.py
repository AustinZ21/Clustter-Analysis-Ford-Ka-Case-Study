"""Cluster analysis module for Ford Ka customer segmentation.

This module implements best practices for real-world cluster analysis including:
- Data preprocessing and scaling
- Optimal cluster number selection using multiple metrics
- Cluster validation and stability analysis
- Feature importance analysis
- Cluster interpretation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ClusterAnalyzer:
    """Advanced cluster analysis for customer segmentation."""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize with customer data.
        
        Args:
            data: DataFrame containing customer data
        """
        self.data = data
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.pca = PCA(random_state=42)
        self.best_model = None
        self.cluster_labels = None
        self.feature_importance = None
        
    def preprocess_data(self, features: List[str], handle_missing: bool = True) -> np.ndarray:
        """Preprocess data for clustering.
        
        Args:
            features: List of feature columns to use
            handle_missing: Whether to handle missing values
            
        Returns:
            Preprocessed data array
        """
        X = self.data[features].copy()
        
        if handle_missing:
            # Handle missing values
            X = X.fillna(X.median())
            
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Log preprocessing results
        logger.info(f"Preprocessed {len(features)} features")
        logger.info(f"Data shape after preprocessing: {X_scaled.shape}")
        
        return X_scaled
        
    def find_optimal_clusters(self, X: np.ndarray, max_clusters: int = 10) -> Dict:
        """Find optimal number of clusters using multiple metrics.
        
        Args:
            X: Preprocessed data array
            max_clusters: Maximum number of clusters to try
            
        Returns:
            Dictionary with metrics for each number of clusters
        """
        metrics = {
            'n_clusters': [],
            'inertia': [],
            'silhouette': [],
            'calinski': []
        }
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X)
            
            metrics['n_clusters'].append(k)
            metrics['inertia'].append(kmeans.inertia_)
            metrics['silhouette'].append(silhouette_score(X, labels))
            metrics['calinski'].append(calinski_harabasz_score(X, labels))
            
        return metrics
        
    def assess_cluster_stability(self, X: np.ndarray, n_clusters: int, 
                               n_iterations: int = 100) -> float:
        """Assess cluster stability through multiple iterations.
        
        Args:
            X: Preprocessed data array
            n_clusters: Number of clusters to assess
            n_iterations: Number of iterations for stability check
            
        Returns:
            Average similarity score between iterations
        """
        base_kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
        similarities = []
        
        for _ in range(n_iterations):
            # Fit new model with different random state
            iter_kmeans = KMeans(n_clusters=n_clusters, random_state=np.random.randint(1000))
            iter_labels = iter_kmeans.fit_predict(X)
            
            # Compare with base model using adjusted mutual information score
            from sklearn.metrics.cluster import adjusted_mutual_info_score
            similarity = adjusted_mutual_info_score(base_kmeans.labels_, iter_labels)
            similarities.append(similarity)
            
        return np.mean(similarities)
        
    def analyze_feature_importance(self, X: np.ndarray, features: List[str]) -> pd.DataFrame:
        """Analyze feature importance for clustering results.
        
        Args:
            X: Preprocessed data array
            features: List of feature names
            
        Returns:
            DataFrame with feature importance scores
        """
        if self.cluster_labels is None:
            raise ValueError("Must perform clustering before analyzing feature importance")
            
        # Use Random Forest to determine feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, self.cluster_labels)
        
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
        
    def perform_clustering(self, features: List[str], n_clusters: Optional[int] = None) -> Dict:
        """Perform complete clustering analysis.
        
        Args:
            features: List of features to use
            n_clusters: Number of clusters (if None, will be determined automatically)
            
        Returns:
            Dictionary containing clustering results and analysis
        """
        # Preprocess data
        X = self.preprocess_data(features)
        
        # Find optimal number of clusters if not specified
        if n_clusters is None:
            metrics = self.find_optimal_clusters(X)
            n_clusters = metrics['n_clusters'][np.argmax(metrics['silhouette'])]
            logger.info(f"Optimal number of clusters determined: {n_clusters}")
        
        # Perform clustering
        self.best_model = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_labels = self.best_model.fit_predict(X)
        
        # Analyze feature importance
        feature_importance = self.analyze_feature_importance(X, features)
        
        # Assess stability
        stability_score = self.assess_cluster_stability(X, n_clusters)
        
        # Prepare results
        results = {
            'n_clusters': n_clusters,
            'labels': self.cluster_labels,
            'cluster_centers': self.best_model.cluster_centers_,
            'feature_importance': feature_importance,
            'stability_score': stability_score,
            'silhouette_score': silhouette_score(X, self.cluster_labels)
        }
        
        logger.info(f"Clustering completed with {n_clusters} clusters")
        logger.info(f"Stability score: {stability_score:.3f}")
        logger.info(f"Silhouette score: {results['silhouette_score']:.3f}")
        
        return results
    
    def visualize_clusters(self, X: np.ndarray, features: List[str]) -> None:
        """Create comprehensive cluster visualizations.
        
        Args:
            X: Preprocessed data array
            features: List of feature names
        """
        if self.cluster_labels is None:
            raise ValueError("Must perform clustering before visualization")
            
        # PCA for dimensionality reduction
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        
        # Create visualization
        plt.figure(figsize=(15, 5))
        
        # Plot 1: PCA visualization
        plt.subplot(131)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.cluster_labels, cmap='viridis')
        plt.title('Cluster Visualization (PCA)')
        
        # Plot 2: Feature importance
        importance_df = self.analyze_feature_importance(X, features)
        plt.subplot(132)
        sns.barplot(data=importance_df.head(10), x='importance', y='feature')
        plt.title('Top 10 Important Features')
        
        # Plot 3: Cluster sizes
        plt.subplot(133)
        sns.countplot(x=self.cluster_labels)
        plt.title('Cluster Sizes')
        
        plt.tight_layout()
        plt.savefig('plots/cluster_analysis_summary.png')
        plt.close()
        
    def interpret_clusters(self, features: List[str]) -> pd.DataFrame:
        """Generate interpretable cluster profiles.
        
        Args:
            features: List of feature names
            
        Returns:
            DataFrame with cluster profiles
        """
        if self.cluster_labels is None:
            raise ValueError("Must perform clustering before interpretation")
            
        # Add cluster labels to original data
        data_with_clusters = self.data.copy()
        data_with_clusters['Cluster'] = self.cluster_labels
        
        # Calculate cluster profiles
        profiles = []
        for cluster in range(len(np.unique(self.cluster_labels))):
            cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster]
            
            # Calculate summary statistics for each feature
            profile = {
                'Cluster': cluster,
                'Size': len(cluster_data),
                'Percentage': len(cluster_data) / len(self.data) * 100
            }
            
            # Add feature statistics
            for feature in features:
                profile[f"{feature}_mean"] = cluster_data[feature].mean()
                profile[f"{feature}_std"] = cluster_data[feature].std()
                
            profiles.append(profile)
            
        return pd.DataFrame(profiles)

    def analyze_clusters(self, data: pd.DataFrame, 
                        labels: np.ndarray, 
                        variables: List[str]) -> Dict:
        """
        Analyze characteristics of each cluster
        Args:
            data: Original data
            labels: Cluster labels
            variables: Variables to analyze
        Returns:
            Dictionary containing cluster analysis results
        """
        analysis = {}
        
        # Add cluster labels to data
        data_with_clusters = data.copy()
        data_with_clusters['Cluster'] = labels
        
        # Analyze each cluster
        for cluster in range(len(np.unique(labels))):
            cluster_mask = data_with_clusters['Cluster'] == cluster
            cluster_data = data_with_clusters[cluster_mask]
            
            # Calculate basic statistics for each variable
            analysis[f'Cluster_{cluster}'] = {
                'size': len(cluster_data),
                'proportion': len(cluster_data) / len(data),
                'variables': {}
            }
            
            for var in variables:
                analysis[f'Cluster_{cluster}']['variables'][var] = {
                    'mean': cluster_data[var].mean(),
                    'std': cluster_data[var].std(),
                    'median': cluster_data[var].median()
                }
        
        return analysis
    
    def compare_clusters_with_preference_groups(self, 
                                              labels: np.ndarray,
                                              preference_groups: pd.Series) -> pd.DataFrame:
        """
        Create contingency table comparing clusters with preference groups
        Args:
            labels: Cluster labels
            preference_groups: Series containing preference group assignments
        Returns:
            Contingency table DataFrame
        """
        return pd.crosstab(
            labels,
            preference_groups,
            margins=True
        ) 