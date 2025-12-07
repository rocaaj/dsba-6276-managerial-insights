"""
utils.py
-----------------------------------
Helper functions for evaluation metrics, visualizations, and path management.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    confusion_matrix, classification_report
)

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
    """Get output directory."""
    output_dir = get_base_dir() / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# -------------------------
# EVALUATION METRICS
# -------------------------
def evaluate_binary_classification(y_true, y_pred, y_proba=None):
    """
    Compute comprehensive binary classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for ROC-AUC/PR-AUC)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    # ROC-AUC and PR-AUC require probabilities
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['roc_auc'] = np.nan
        
        try:
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
            metrics['pr_auc'] = auc(recall_vals, precision_vals)
        except ValueError:
            metrics['pr_auc'] = np.nan
    else:
        metrics['roc_auc'] = np.nan
        metrics['pr_auc'] = np.nan
    
    return metrics


def print_classification_report(y_true, y_pred):
    """Print detailed classification report."""
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred))
    print("="*60)


# -------------------------
# VISUALIZATIONS
# -------------------------
def plot_roc_curve(y_true, y_proba, model_name="Model", save_path=None):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        model_name: Name for legend
        save_path: Path to save plot (optional)
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✅ Saved ROC curve to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name for title
        save_path: Path to save plot (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['No Subscription', 'Subscription'],
                yticklabels=['No Subscription', 'Subscription'])
    plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✅ Saved confusion matrix to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_coefficients(coef_df, top_n=20, save_path=None):
    """
    Plot coefficient bar plot sorted by absolute value.
    
    Args:
        coef_df: DataFrame with columns ['Feature', 'Coefficient']
        top_n: Number of top features to plot
        save_path: Path to save plot (optional)
    """
    # Sort by absolute coefficient
    coef_df_sorted = coef_df.copy()
    coef_df_sorted['AbsCoefficient'] = coef_df_sorted['Coefficient'].abs()
    coef_df_sorted = coef_df_sorted.sort_values('AbsCoefficient', ascending=False)
    
    # Exclude intercept if present
    if 'Intercept' in coef_df_sorted['Feature'].values:
        coef_df_sorted = coef_df_sorted[coef_df_sorted['Feature'] != 'Intercept']
    
    # Top N features
    top_features = coef_df_sorted.head(top_n)
    
    plt.figure(figsize=(10, max(6, top_n * 0.3)))
    colors = ['steelblue' if x > 0 else 'coral' for x in top_features['Coefficient']]
    plt.barh(range(len(top_features)), top_features['Coefficient'], color=colors)
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Coefficient (Logit)', fontsize=12)
    plt.title(f'Top {top_n} Logistic Regression Coefficients', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✅ Saved coefficient plot to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_odds_ratios(coef_df, top_n=20, save_path=None):
    """
    Plot odds ratio bar plot sorted by value.
    
    Args:
        coef_df: DataFrame with columns ['Feature', 'OddsRatio']
        top_n: Number of top features to plot
        save_path: Path to save plot (optional)
    """
    # Sort by odds ratio (farthest from 1)
    coef_df_sorted = coef_df.copy()
    coef_df_sorted['OddsRatioDistance'] = (coef_df_sorted['OddsRatio'] - 1).abs()
    coef_df_sorted = coef_df_sorted.sort_values('OddsRatioDistance', ascending=False)
    
    # Exclude intercept if present
    if 'Intercept' in coef_df_sorted['Feature'].values:
        coef_df_sorted = coef_df_sorted[coef_df_sorted['Feature'] != 'Intercept']
    
    # Top N features
    top_features = coef_df_sorted.head(top_n)
    
    plt.figure(figsize=(10, max(6, top_n * 0.3)))
    colors = ['steelblue' if x > 1 else 'coral' for x in top_features['OddsRatio']]
    plt.barh(range(len(top_features)), top_features['OddsRatio'], color=colors)
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Odds Ratio', fontsize=12)
    plt.title(f'Top {top_n} Odds Ratios', fontsize=14, fontweight='bold')
    plt.axvline(x=1, color='black', linestyle='-', linewidth=1, label='No Effect (OR=1)')
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✅ Saved odds ratio plot to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_feature_significance(coef_df, save_path=None):
    """
    Plot p-value significance visualization.
    
    Args:
        coef_df: DataFrame with columns ['Feature', 'p_value', 'Coefficient']
        save_path: Path to save plot (optional)
    """
    # Exclude intercept
    plot_df = coef_df[coef_df['Feature'] != 'Intercept'].copy()
    
    # Create significance categories
    plot_df['Significance'] = 'Not Significant'
    plot_df.loc[plot_df['p_value'] < 0.001, 'Significance'] = 'p < 0.001'
    plot_df.loc[(plot_df['p_value'] >= 0.001) & (plot_df['p_value'] < 0.01), 'Significance'] = 'p < 0.01'
    plot_df.loc[(plot_df['p_value'] >= 0.01) & (plot_df['p_value'] < 0.05), 'Significance'] = 'p < 0.05'
    
    # Sort by p-value
    plot_df = plot_df.sort_values('p_value')
    
    # Color mapping
    color_map = {
        'p < 0.001': 'darkgreen',
        'p < 0.01': 'green',
        'p < 0.05': 'lightgreen',
        'Not Significant': 'lightgray'
    }
    
    # Create scatter plot
    plt.figure(figsize=(12, max(8, len(plot_df) * 0.15)))
    
    for sig_level in ['p < 0.001', 'p < 0.01', 'p < 0.05', 'Not Significant']:
        subset = plot_df[plot_df['Significance'] == sig_level]
        if len(subset) > 0:
            plt.scatter(subset['p_value'], range(len(subset)), 
                       c=color_map[sig_level], label=sig_level, s=50, alpha=0.7)
    
    plt.yticks(range(len(plot_df)), plot_df['Feature'])
    plt.xlabel('P-value', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Statistical Significance of Features (P-values)', fontsize=14, fontweight='bold')
    plt.axvline(x=0.05, color='red', linestyle='--', linewidth=1, label='α = 0.05')
    plt.axvline(x=0.01, color='orange', linestyle='--', linewidth=1, label='α = 0.01')
    plt.axvline(x=0.001, color='darkred', linestyle='--', linewidth=1, label='α = 0.001')
    plt.legend(loc='upper right')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✅ Saved significance plot to: {save_path}")
    else:
        plt.show()
    plt.close()

