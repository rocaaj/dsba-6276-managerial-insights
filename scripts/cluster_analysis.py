"""
cluster_analysis.py
-----------------------------------
Main analysis script for Cluster Analysis research question:
"What distinct customer archetypes exist, and how do they respond to 
different telemarketing strategies?"

Uses K-means clustering on client features to discover customer segments,
then analyzes how each segment responds to telemarketing campaigns.
"""

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
from sklearn.cluster import KMeans

# Add scripts directory to path for imports
scripts_dir = Path(__file__).resolve().parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

# Import utilities
from clustering_utils import (
    get_output_dir, find_optimal_clusters, evaluate_clusters,
    plot_silhouette_analysis, plot_cluster_centroids, 
    plot_cluster_distributions, plot_response_by_cluster,
    plot_cluster_comparison
)

# -------------------------
# CONFIG
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CLUSTERING_DATA_PATH = BASE_DIR / "data" / "bank-full-clustering.csv"
FULL_DATA_PATH = BASE_DIR / "data" / "bank-full.csv"  # For response analysis
OUTPUT_DIR = get_output_dir()
RANDOM_STATE = 42
K_RANGE = (2, 9)  # Range of k values to test for silhouette analysis

# -------------------------
# FUNCTIONS
# -------------------------
def load_clustering_data(path: Path) -> pd.DataFrame:
    """Load preprocessed client features for clustering."""
    print(f"\n🔄 Loading preprocessed clustering data from: {path}")
    df = pd.read_csv(path)
    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def load_full_data(path: Path) -> pd.DataFrame:
    """Load full dataset for response analysis."""
    print(f"\n🔄 Loading full dataset from: {path}")
    df = pd.read_csv(path, sep=';')
    print(f"✅ Loaded full dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def perform_clustering(X, optimal_k, random_state=42):
    """
    Perform K-means clustering with optimal k.
    
    Returns:
        Fitted KMeans model and cluster labels
    """
    print(f"\n🔄 Performing K-means clustering with k={optimal_k}...")
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    print(f"✅ Clustering complete")
    print(f"   Cluster sizes: {pd.Series(cluster_labels).value_counts().sort_index().to_dict()}")
    
    return kmeans, cluster_labels


def analyze_cluster_characteristics(X, cluster_labels, feature_names):
    """
    Analyze cluster characteristics using centroids and distributions.
    
    Returns:
        DataFrame with cluster centroids and summary statistics
    """
    print("\n" + "="*60)
    print("CLUSTER CHARACTERISTICS ANALYSIS")
    print("="*60)
    
    # Create DataFrame with features and cluster assignments
    df_clustered = pd.DataFrame(X, columns=feature_names)
    df_clustered['cluster'] = cluster_labels
    
    # Calculate cluster centroids (means)
    centroids = df_clustered.groupby('cluster').mean()
    
    # Calculate summary statistics
    cluster_summary = df_clustered.groupby('cluster').agg(['mean', 'std', 'count'])
    
    print("\n📊 Cluster Centroids (Prototypes):")
    print(centroids.round(3))
    
    print("\n📊 Cluster Sizes:")
    cluster_sizes = df_clustered['cluster'].value_counts().sort_index()
    for cluster, size in cluster_sizes.items():
        pct = (size / len(df_clustered)) * 100
        print(f"   Cluster {cluster}: {size} observations ({pct:.1f}%)")
    
    return centroids, cluster_summary, df_clustered


def analyze_telemarketing_response(df_full, cluster_labels):
    """
    Analyze how each cluster responds to telemarketing strategies.
    
    Args:
        df_full: Full dataset with campaign and response features
        cluster_labels: Cluster assignments for each observation
    
    Returns:
        DataFrames with response analysis by cluster
    """
    print("\n" + "="*60)
    print("TELEMARKETING RESPONSE ANALYSIS")
    print("="*60)
    
    # Add cluster labels to full dataset
    df_analysis = df_full.copy()
    df_analysis['cluster'] = cluster_labels
    
    # Analyze subscription rates by cluster
    print("\n📊 Subscription Rates by Cluster:")
    response_rates = df_analysis.groupby('cluster')['y'].agg([
        ('subscription_rate', lambda x: (x == 'yes').mean() * 100),
        ('total_contacts', 'count'),
        ('subscriptions', lambda x: (x == 'yes').sum())
    ]).round(2)
    print(response_rates)
    
    # Analyze campaign features by cluster
    campaign_features = ['campaign', 'contact', 'month']
    available_campaign = [f for f in campaign_features if f in df_analysis.columns]
    
    if available_campaign:
        print("\n📊 Campaign Features by Cluster:")
        campaign_summary = {}
        for feature in available_campaign:
            if df_analysis[feature].dtype in ['int64', 'float64']:
                summary = df_analysis.groupby('cluster')[feature].agg(['mean', 'std']).round(2)
                campaign_summary[feature] = summary
                print(f"\n{feature}:")
                print(summary)
            else:
                crosstab = pd.crosstab(df_analysis['cluster'], df_analysis[feature], 
                                      normalize='index') * 100
                campaign_summary[feature] = crosstab
                print(f"\n{feature} distribution:")
                print(crosstab.round(2))
    
    return response_rates, campaign_summary, df_analysis


def save_results(kmeans_model, cluster_labels, centroids, evaluation_metrics,
                 silhouette_results, response_rates, campaign_summary, 
                 df_clustered, feature_names):
    """Save all clustering results to files."""
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Save K-means model
    model_path = OUTPUT_DIR / "kmeans_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(kmeans_model, f)
    print(f"✅ Saved K-means model to: {model_path}")
    
    # Save cluster assignments
    assignments_df = pd.DataFrame({
        'cluster': cluster_labels
    })
    assignments_path = OUTPUT_DIR / "cluster_assignments.csv"
    assignments_df.to_csv(assignments_path, index=False)
    print(f"✅ Saved cluster assignments to: {assignments_path}")
    
    # Save cluster centroids
    centroids_path = OUTPUT_DIR / "cluster_centroids.csv"
    centroids.to_csv(centroids_path)
    print(f"✅ Saved cluster centroids to: {centroids_path}")
    
    # Save silhouette analysis results
    silhouette_df = pd.DataFrame({
        'k': silhouette_results['k_values'],
        'silhouette_score': silhouette_results['silhouette_scores']
    })
    silhouette_path = OUTPUT_DIR / "silhouette_scores.csv"
    silhouette_df.to_csv(silhouette_path, index=False)
    print(f"✅ Saved silhouette scores to: {silhouette_path}")
    
    # Save cluster evaluation metrics
    eval_df = pd.DataFrame([{
        'silhouette_avg': evaluation_metrics['silhouette_avg'],
        'calinski_harabasz': evaluation_metrics['calinski_harabasz'],
        'davies_bouldin': evaluation_metrics['davies_bouldin'],
        'inertia': evaluation_metrics['inertia']
    }])
    eval_path = OUTPUT_DIR / "cluster_evaluation.csv"
    eval_df.to_csv(eval_path, index=False)
    print(f"✅ Saved cluster evaluation metrics to: {eval_path}")
    
    # Save per-cluster silhouette scores
    cluster_silhouette_df = pd.DataFrame([evaluation_metrics['silhouette_by_cluster']]).T
    cluster_silhouette_df.columns = ['silhouette_score']
    cluster_silhouette_path = OUTPUT_DIR / "silhouette_scores_by_cluster.csv"
    cluster_silhouette_df.to_csv(cluster_silhouette_path)
    print(f"✅ Saved per-cluster silhouette scores to: {cluster_silhouette_path}")
    
    # Save cluster summary
    cluster_summary_data = []
    for cluster in range(len(np.unique(cluster_labels))):
        cluster_data = {
            'cluster': cluster,
            'size': evaluation_metrics['cluster_sizes'][cluster],
            'silhouette_score': evaluation_metrics['silhouette_by_cluster'][cluster]
        }
        # Add centroid values
        for feature in feature_names:
            cluster_data[f'centroid_{feature}'] = centroids.loc[cluster, feature]
        cluster_summary_data.append(cluster_data)
    
    summary_df = pd.DataFrame(cluster_summary_data)
    summary_path = OUTPUT_DIR / "cluster_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Saved cluster summary to: {summary_path}")
    
    # Save response analysis
    response_path = OUTPUT_DIR / "response_by_cluster.csv"
    response_rates.to_csv(response_path)
    print(f"✅ Saved response rates by cluster to: {response_path}")
    
    # Save campaign features analysis
    if campaign_summary:
        for feature, summary_data in campaign_summary.items():
            campaign_path = OUTPUT_DIR / f"campaign_{feature}_by_cluster.csv"
            if isinstance(summary_data, pd.DataFrame):
                summary_data.to_csv(campaign_path)
                print(f"✅ Saved {feature} analysis to: {campaign_path}")


def main():
    """Main clustering analysis pipeline."""
    print("="*60)
    print("CLUSTER ANALYSIS: CUSTOMER ARCHETYPES")
    print("="*60)
    print("\nResearch Question:")
    print("What distinct customer archetypes exist, and how do they")
    print("respond to different telemarketing strategies?")
    print("="*60)
    
    # Load preprocessed client features
    X = load_clustering_data(CLUSTERING_DATA_PATH)
    feature_names = X.columns.tolist()
    X_array = X.values
    
    # Perform silhouette analysis to find optimal k
    print("\n" + "="*60)
    print("SILHOUETTE ANALYSIS")
    print("="*60)
    silhouette_results = find_optimal_clusters(X_array, k_range=K_RANGE, random_state=RANDOM_STATE)
    
    # Plot silhouette analysis
    plot_silhouette_analysis(
        silhouette_results,
        save_path=OUTPUT_DIR / "optimal_clusters.png"
    )
    
    # Perform clustering with optimal k
    print("\n" + "="*60)
    print("K-MEANS CLUSTERING")
    print("="*60)
    kmeans_model, cluster_labels = perform_clustering(
        X_array, 
        silhouette_results['optimal_k'],
        random_state=RANDOM_STATE
    )
    
    # Evaluate clusters
    print("\n" + "="*60)
    print("CLUSTER EVALUATION")
    print("="*60)
    evaluation_metrics = evaluate_clusters(X_array, cluster_labels, kmeans_model)
    
    print(f"\n📊 Cluster Quality Metrics:")
    print(f"   Average Silhouette Score: {evaluation_metrics['silhouette_avg']:.4f}")
    print(f"   Calinski-Harabasz Score: {evaluation_metrics['calinski_harabasz']:.2f}")
    print(f"   Davies-Bouldin Score: {evaluation_metrics['davies_bouldin']:.4f}")
    print(f"   Inertia (WCSS): {evaluation_metrics['inertia']:.2f}")
    
    print(f"\n📊 Per-Cluster Silhouette Scores:")
    for cluster, score in evaluation_metrics['silhouette_by_cluster'].items():
        print(f"   Cluster {cluster}: {score:.4f}")
    
    # Analyze cluster characteristics
    centroids, cluster_summary, df_clustered = analyze_cluster_characteristics(
        X_array, cluster_labels, feature_names
    )
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Cluster centroids plot
    plot_cluster_centroids(
        centroids,
        save_path=OUTPUT_DIR / "cluster_centroids_plot.png"
    )
    
    # Feature distributions by cluster (select key features)
    key_features = feature_names[:min(5, len(feature_names))]  # Top 5 features
    plot_cluster_distributions(
        df_clustered,
        cluster_col='cluster',
        features=key_features,
        save_path=OUTPUT_DIR / "feature_distributions_by_cluster.png"
    )
    
    # Load full data for response analysis
    df_full = load_full_data(FULL_DATA_PATH)
    
    # Analyze telemarketing response
    response_rates, campaign_summary, df_analysis = analyze_telemarketing_response(
        df_full, cluster_labels
    )
    
    # Response analysis visualization
    campaign_features = ['campaign', 'contact', 'month']
    available_campaign = [f for f in campaign_features if f in df_analysis.columns]
    
    plot_response_by_cluster(
        df_analysis,
        cluster_col='cluster',
        response_col='y',
        campaign_features=available_campaign[:2] if available_campaign else None,  # Limit to 2 for visualization
        save_path=OUTPUT_DIR / "response_analysis_by_cluster.png"
    )
    
    # Cluster comparison heatmap
    plot_cluster_comparison(
        centroids,
        save_path=OUTPUT_DIR / "cluster_comparison_heatmap.png"
    )
    
    # Save all results
    save_results(
        kmeans_model, cluster_labels, centroids, evaluation_metrics,
        silhouette_results, response_rates, campaign_summary,
        df_clustered, feature_names
    )
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - kmeans_model.pkl")
    print("  - cluster_assignments.csv")
    print("  - cluster_centroids.csv")
    print("  - cluster_summary.csv")
    print("  - silhouette_scores.csv")
    print("  - silhouette_scores_by_cluster.csv")
    print("  - cluster_evaluation.csv")
    print("  - response_by_cluster.csv")
    print("  - optimal_clusters.png")
    print("  - cluster_centroids_plot.png")
    print("  - feature_distributions_by_cluster.png")
    print("  - response_analysis_by_cluster.png")
    print("  - cluster_comparison_heatmap.png")


if __name__ == "__main__":
    main()

