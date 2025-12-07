"""
clustering_utils.py
-----------------------------------
Helper functions for clustering analysis, evaluation, and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


# -------------------------
# PATH MANAGEMENT
# -------------------------
def get_base_dir() -> Path:
    """Get base directory (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def get_output_dir() -> Path:
    """Get clustering output directory."""
    output_dir = get_base_dir() / "output" / "clustering"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# -------------------------
# CLUSTER SELECTION
# -------------------------
def find_optimal_clusters(X, k_range=(2, 9), random_state=42):
    """
    Find optimal number of clusters using silhouette analysis.
    
    Args:
        X: Feature matrix
        k_range: Range of k values to test (min, max)
        random_state: Random seed
    
    Returns:
        Dictionary with k values, silhouette scores, and optimal k
    """
    print(f"\n🔄 Performing silhouette analysis for k in range {k_range[0]}-{k_range[1]-1}...")
    
    silhouette_scores = []
    k_values = range(k_range[0], k_range[1])
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"   k={k}: Silhouette score = {silhouette_avg:.4f}")
    
    # Find optimal k (highest silhouette score)
    optimal_k = k_values[np.argmax(silhouette_scores)]
    optimal_score = max(silhouette_scores)
    
    print(f"\n✅ Optimal number of clusters: k={optimal_k} (silhouette score = {optimal_score:.4f})")
    
    return {
        'k_values': list(k_values),
        'silhouette_scores': silhouette_scores,
        'optimal_k': optimal_k,
        'optimal_score': optimal_score
    }


# -------------------------
# CLUSTER EVALUATION
# -------------------------
def evaluate_clusters(X, cluster_labels, kmeans_model):
    """
    Evaluate cluster quality using multiple metrics.
    
    Args:
        X: Feature matrix
        cluster_labels: Cluster assignments
        kmeans_model: Fitted KMeans model
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Overall silhouette score
    silhouette_avg = silhouette_score(X, cluster_labels)
    
    # Per-cluster silhouette scores
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
    # Other metrics
    calinski_harabasz = calinski_harabasz_score(X, cluster_labels)
    davies_bouldin = davies_bouldin_score(X, cluster_labels)
    
    # Within-cluster sum of squares (cohesion)
    inertia = kmeans_model.inertia_
    
    # Cluster sizes
    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
    
    # Per-cluster silhouette scores
    cluster_silhouettes = {}
    for i in range(len(np.unique(cluster_labels))):
        cluster_silhouettes[i] = sample_silhouette_values[cluster_labels == i].mean()
    
    return {
        'silhouette_avg': silhouette_avg,
        'silhouette_by_cluster': cluster_silhouettes,
        'calinski_harabasz': calinski_harabasz,
        'davies_bouldin': davies_bouldin,
        'inertia': inertia,
        'cluster_sizes': cluster_sizes.to_dict()
    }


# -------------------------
# VISUALIZATIONS
# -------------------------
def plot_silhouette_analysis(silhouette_results, save_path=None):
    """
    Plot silhouette analysis results.
    
    Args:
        silhouette_results: Dictionary from find_optimal_clusters()
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(silhouette_results['k_values'], silhouette_results['silhouette_scores'], 
             marker='o', linewidth=2, markersize=8)
    plt.axvline(x=silhouette_results['optimal_k'], color='r', linestyle='--', 
                label=f"Optimal k={silhouette_results['optimal_k']}")
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Average Silhouette Score', fontsize=12)
    plt.title('Silhouette Analysis for Optimal Cluster Selection', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✅ Saved silhouette analysis plot to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_cluster_centroids(centroids_df, save_path=None):
    """
    Plot cluster centroids as heatmap.
    
    Args:
        centroids_df: DataFrame with clusters as rows and features as columns
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=(max(12, len(centroids_df.columns) * 0.5), max(6, len(centroids_df) * 0.8)))
    sns.heatmap(centroids_df.T, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                center=0, cbar_kws={'label': 'Centroid Value'})
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Cluster Centroids (Prototypes)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✅ Saved cluster centroids plot to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_cluster_distributions(df, cluster_col, features, save_path=None):
    """
    Plot feature distributions by cluster.
    
    Args:
        df: DataFrame with cluster assignments
        cluster_col: Name of cluster column
        features: List of feature names to plot
        save_path: Path to save plot (optional)
    """
    n_features = len(features)
    n_clusters = df[cluster_col].nunique()
    
    fig, axes = plt.subplots(n_features, 1, figsize=(10, n_features * 3))
    if n_features == 1:
        axes = [axes]
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        # Box plot for numeric features
        if df[feature].dtype in ['int64', 'float64']:
            df.boxplot(column=feature, by=cluster_col, ax=ax)
            ax.set_title(f'{feature} Distribution by Cluster', fontsize=11)
            ax.set_xlabel('Cluster', fontsize=10)
            ax.set_ylabel(feature, fontsize=10)
        else:
            # Bar plot for categorical features
            crosstab = pd.crosstab(df[cluster_col], df[feature], normalize='index') * 100
            crosstab.plot(kind='bar', ax=ax, width=0.8)
            ax.set_title(f'{feature} Distribution by Cluster', fontsize=11)
            ax.set_xlabel('Cluster', fontsize=10)
            ax.set_ylabel('Percentage', fontsize=10)
            ax.legend(title=feature, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax.grid(alpha=0.3)
    
    plt.suptitle('Feature Distributions by Cluster', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✅ Saved feature distributions plot to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_response_by_cluster(df, cluster_col, response_col, campaign_features=None, save_path=None):
    """
    Plot subscription rates and campaign features by cluster.
    
    Args:
        df: DataFrame with cluster assignments and response data
        cluster_col: Name of cluster column
        response_col: Name of response/target column
        campaign_features: List of campaign feature names (optional)
        save_path: Path to save plot (optional)
    """
    n_plots = 1 + (len(campaign_features) if campaign_features else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]
    
    # Plot 1: Subscription rates by cluster
    # Convert response to numeric (yes=1, no=0) before computing mean
    # Note: 1 = "yes" (subscribed), 0 = "no" (did not subscribe)
    response_rates = pd.DataFrame({
        'mean': df.groupby(cluster_col)[response_col].apply(lambda x: (x == 'yes').mean() * 100),
        'count': df.groupby(cluster_col)[response_col].count()
    })
    
    axes[0].bar(response_rates.index, response_rates['mean'], color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Cluster', fontsize=12)
    axes[0].set_ylabel('Subscription Rate (%)', fontsize=12)
    axes[0].set_title('Subscription Rate by Cluster\n(Percentage with y="yes" who subscribed)', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim([0, max(response_rates['mean']) * 1.15])  # Add some space for labels
    
    # Add count labels
    for idx, row in response_rates.iterrows():
        axes[0].text(idx, row['mean'] + max(response_rates['mean']) * 0.02, 
                    f"n={int(row['count'])}", 
                    ha='center', fontsize=9)
    
    # Plot campaign features if provided
    if campaign_features:
        for idx, feature in enumerate(campaign_features, start=1):
            if idx < len(axes):
                if df[feature].dtype in ['int64', 'float64']:
                    # Box plot for numeric campaign features
                    df.boxplot(column=feature, by=cluster_col, ax=axes[idx])
                    axes[idx].set_title(f'{feature} by Cluster', fontsize=11)
                    axes[idx].set_xlabel('Cluster', fontsize=10)
                    axes[idx].set_ylabel(feature, fontsize=10)
                else:
                    # Bar plot for categorical campaign features
                    crosstab = pd.crosstab(df[cluster_col], df[feature], normalize='index') * 100
                    crosstab.plot(kind='bar', ax=axes[idx], width=0.8)
                    axes[idx].set_title(f'{feature} by Cluster', fontsize=11)
                    axes[idx].set_xlabel('Cluster', fontsize=10)
                    axes[idx].set_ylabel('Percentage', fontsize=10)
                    axes[idx].legend(title=feature, bbox_to_anchor=(1.05, 1), loc='upper left')
                axes[idx].grid(alpha=0.3)
    
    plt.suptitle('Telemarketing Response by Cluster', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✅ Saved response analysis plot to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_cluster_comparison(cluster_summary_df, save_path=None):
    """
    Plot cluster comparison heatmap.
    
    Args:
        cluster_summary_df: DataFrame with clusters as rows and features as columns
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=(max(12, len(cluster_summary_df.columns) * 0.5), 
                         max(6, len(cluster_summary_df) * 0.8)))
    sns.heatmap(cluster_summary_df.T, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                center=0, cbar_kws={'label': 'Normalized Value'})
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Cluster Comparison Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✅ Saved cluster comparison heatmap to: {save_path}")
    else:
        plt.show()
    plt.close()

