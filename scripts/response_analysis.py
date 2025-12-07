"""
response_analysis.py
-----------------------------------
Main analysis script for Response Analysis research question:
"Which customer or campaign factors are statistically significant predictors 
of term deposit purchase?"

Uses logistic regression with statistical inference (p-values, odds ratios)
to draw managerial insights.
"""

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Add scripts directory to path for imports
scripts_dir = Path(__file__).resolve().parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

# Import utilities
from utils import (
    get_output_dir, evaluate_binary_classification, print_classification_report,
    plot_roc_curve, plot_confusion_matrix, plot_coefficients, 
    plot_odds_ratios, plot_feature_significance
)

# -------------------------
# CONFIG
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "bank-full-processed.csv"
OUTPUT_DIR = get_output_dir()
TARGET_COL = "y"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# -------------------------
# FUNCTIONS
# -------------------------
def load_data(path: Path) -> pd.DataFrame:
    """Load preprocessed dataset."""
    print(f"\n🔄 Loading preprocessed data from: {path}")
    df = pd.read_csv(path)
    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def split_data(df: pd.DataFrame):
    """
    Split dataset into train/test sets (stratified).
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    print(f"\n🔄 Splitting data (stratified, test_size={TEST_SIZE}, random_state={RANDOM_STATE})...")
    
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"✅ Train set: {X_train.shape[0]} samples")
    print(f"✅ Test set: {X_test.shape[0]} samples")
    print(f"📊 Target distribution (train): {y_train.value_counts().to_dict()}")
    print(f"📊 Target distribution (test): {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Train logistic regression using both sklearn and statsmodels.
    
    sklearn: For predictions and evaluation
    statsmodels: For statistical inference (p-values, confidence intervals)
    
    Returns:
        sklearn_model, statsmodels_results, coefficient_table
    """
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION TRAINING")
    print("="*60)
    
    # Train sklearn model (for predictions)
    print("\n🔄 Training sklearn Logistic Regression...")
    sklearn_model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver='lbfgs'
    )
    sklearn_model.fit(X_train, y_train)
    print("✅ sklearn model trained")
    
    # Train statsmodels model (for statistical inference)
    print("\n🔄 Training statsmodels Logistic Regression for statistical inference...")
    
    # Add constant for intercept
    X_train_sm = sm.add_constant(X_train)
    X_test_sm = sm.add_constant(X_test)
    
    # Fit statsmodels logistic regression
    sm_model = sm.Logit(y_train, X_train_sm)
    sm_results = sm_model.fit(disp=False, method='lbfgs')
    print("✅ statsmodels model trained")
    
    # Extract coefficient table
    coef_table = pd.DataFrame({
        'Feature': ['Intercept'] + list(X_train.columns),
        'Coefficient': sm_results.params.values,
        'StdErr': sm_results.bse.values,
        'z_value': sm_results.tvalues.values,
        'p_value': sm_results.pvalues.values,
        'OddsRatio': np.exp(sm_results.params.values),
        'CI_lower_95': np.exp(sm_results.conf_int()[0].values),
        'CI_upper_95': np.exp(sm_results.conf_int()[1].values)
    })
    
    # Sort by absolute coefficient
    coef_table['AbsCoefficient'] = coef_table['Coefficient'].abs()
    coef_table = coef_table.sort_values('AbsCoefficient', ascending=False)
    
    print("\n📊 Top 10 Features by Absolute Coefficient:")
    print(coef_table[['Feature', 'Coefficient', 'p_value', 'OddsRatio']].head(10).to_string(index=False))
    
    # Statistical significance summary
    significant_05 = (coef_table['p_value'] < 0.05).sum()
    significant_01 = (coef_table['p_value'] < 0.01).sum()
    significant_001 = (coef_table['p_value'] < 0.001).sum()
    
    print(f"\n📊 Statistical Significance Summary:")
    print(f"   p < 0.05: {significant_05} features")
    print(f"   p < 0.01: {significant_01} features")
    print(f"   p < 0.001: {significant_001} features")
    
    return sklearn_model, sm_results, coef_table


def evaluate_model(model, X_test, y_test, coef_table):
    """
    Evaluate model performance and generate visualizations.
    
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    metrics = evaluate_binary_classification(y_test, y_pred, y_proba)
    
    print("\n📊 Performance Metrics:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1-Score:  {metrics['f1_score']:.4f}")
    print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"   PR-AUC:    {metrics['pr_auc']:.4f}")
    
    # Print classification report
    print_classification_report(y_test, y_pred)
    
    # Generate visualizations
    print("\n🔄 Generating visualizations...")
    
    # ROC curve
    plot_roc_curve(
        y_test, y_proba, 
        model_name="Logistic Regression",
        save_path=OUTPUT_DIR / "roc_curve.png"
    )
    
    # Confusion matrix
    plot_confusion_matrix(
        y_test, y_pred,
        model_name="Logistic Regression",
        save_path=OUTPUT_DIR / "confusion_matrix.png"
    )
    
    # Coefficient plot
    plot_coefficients(
        coef_table[['Feature', 'Coefficient']],
        top_n=20,
        save_path=OUTPUT_DIR / "coefficient_plot.png"
    )
    
    # Odds ratio plot
    plot_odds_ratios(
        coef_table[['Feature', 'OddsRatio']],
        top_n=20,
        save_path=OUTPUT_DIR / "odds_ratio_plot.png"
    )
    
    # Significance plot
    plot_feature_significance(
        coef_table[['Feature', 'p_value', 'Coefficient']],
        save_path=OUTPUT_DIR / "feature_significance_plot.png"
    )
    
    print("✅ All visualizations saved")
    
    return metrics


def save_results(model, coef_table, metrics):
    """Save model, coefficient table, and metrics to files."""
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Save model
    model_path = OUTPUT_DIR / "logistic_regression_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Saved model to: {model_path}")
    
    # Save full coefficient table
    coef_path = OUTPUT_DIR / "coefficient_summary.csv"
    coef_table.to_csv(coef_path, index=False)
    print(f"✅ Saved coefficient summary to: {coef_path}")
    
    # Save statistically significant features (p < 0.05)
    significant_features = coef_table[coef_table['p_value'] < 0.05].copy()
    significant_path = OUTPUT_DIR / "significant_features.csv"
    significant_features.to_csv(significant_path, index=False)
    print(f"✅ Saved significant features (p < 0.05) to: {significant_path}")
    print(f"   {len(significant_features)} statistically significant features")
    
    # Save evaluation metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = OUTPUT_DIR / "evaluation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✅ Saved evaluation metrics to: {metrics_path}")
    
    # Print summary of significant features
    print("\n📊 Statistically Significant Features (p < 0.05):")
    print("="*60)
    sig_features_display = significant_features[['Feature', 'Coefficient', 'p_value', 'OddsRatio']].copy()
    sig_features_display = sig_features_display.sort_values('p_value')
    print(sig_features_display.to_string(index=False))
    print("="*60)


def main():
    """Main analysis pipeline."""
    print("="*60)
    print("RESPONSE ANALYSIS: STATISTICAL SIGNIFICANCE OF PREDICTORS")
    print("="*60)
    print("\nResearch Question:")
    print("Which customer or campaign factors are statistically significant")
    print("predictors of term deposit purchase?")
    print("="*60)
    
    # Load data
    df = load_data(DATA_PATH)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Train model
    sklearn_model, sm_results, coef_table = train_logistic_regression(
        X_train, y_train, X_test, y_test
    )
    
    # Evaluate model
    metrics = evaluate_model(sklearn_model, X_test, y_test, coef_table)
    
    # Save results
    save_results(sklearn_model, coef_table, metrics)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - logistic_regression_model.pkl")
    print("  - coefficient_summary.csv")
    print("  - significant_features.csv")
    print("  - evaluation_metrics.csv")
    print("  - roc_curve.png")
    print("  - confusion_matrix.png")
    print("  - coefficient_plot.png")
    print("  - odds_ratio_plot.png")
    print("  - feature_significance_plot.png")


if __name__ == "__main__":
    main()
