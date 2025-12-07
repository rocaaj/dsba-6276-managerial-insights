"""
clustering_preprocessing.py
-----------------------------------
Preprocesses client features only for clustering analysis.
Focuses on customer characteristics: age, job, marital, education, 
default, balance, housing, loan.

Steps:
1. Load raw CSV (semicolon-separated)
2. Select client features only
3. Map job to income levels (low/medium/high)
4. Encode categorical variables (ordinal for ordered, one-hot for nominal)
5. Scale numeric variables (StandardScaler)
6. Save processed dataset to data/bank-full-clustering.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

# -------------------------
# CONFIG
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "bank-full.csv"
OUTPUT_PATH = BASE_DIR / "data" / "bank-full-clustering.csv"
RANDOM_STATE = 42

# Client features only (for clustering)
CLIENT_FEATURES = ["age", "job", "marital", "education", "default", "balance", "housing", "loan"]

# Categorical encoding strategy
ORDINAL_CATEGORIES = {
    "education": ["unknown", "primary", "secondary", "tertiary"],
    "job_income": ["low", "medium", "high"]  # Income level grouping for jobs
}

# Job to income level mapping (same as response analysis)
JOB_INCOME_MAPPING = {
    # High income
    "management": "high",
    "entrepreneur": "high",
    "self-employed": "high",
    # Medium income
    "technician": "medium",
    "admin.": "medium",
    "services": "medium",
    "retired": "medium",
    "unknown": "medium",  # Map unknown to medium
    # Low income
    "blue-collar": "low",
    "housemaid": "low",
    "student": "low",
    "unemployed": "low"
}

# -------------------------
# FUNCTIONS
# -------------------------
def load_data(path: Path) -> pd.DataFrame:
    """Load raw dataset from CSV."""
    print(f"\n🔄 Loading dataset from: {path}")
    df = pd.read_csv(path, sep=';')
    print(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    return df


def select_client_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only client features for clustering.
    
    Returns:
        DataFrame with client features only
    """
    print("\n" + "="*60)
    print("CLIENT FEATURE SELECTION")
    print("="*60)
    
    # Check which features are available
    available_features = [f for f in CLIENT_FEATURES if f in df.columns]
    missing_features = [f for f in CLIENT_FEATURES if f not in df.columns]
    
    if missing_features:
        print(f"⚠️  Warning: Missing features: {missing_features}")
    
    df_client = df[available_features].copy()
    
    print(f"\n✅ Selected {len(available_features)} client features:")
    print(f"   {available_features}")
    print(f"   Shape: {df_client.shape}")
    
    return df_client


def check_missing_values(df: pd.DataFrame) -> None:
    """Check and report missing values."""
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n⚠️  Missing values detected:")
        print(missing[missing > 0])
    else:
        print("\n✅ No missing values detected")


def map_job_to_income(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map job types to income levels (low/medium/high) for better interpretability.
    
    Returns:
        DataFrame with 'job' column replaced by 'job_income' column
    """
    if "job" not in df.columns:
        return df
    
    print("\n🔄 Mapping job types to income levels...")
    
    # Create mapping
    df_mapped = df.copy()
    df_mapped["job_income"] = df_mapped["job"].map(JOB_INCOME_MAPPING)
    
    # Check for any unmapped jobs
    unmapped = df_mapped[df_mapped["job_income"].isna()]["job"].unique()
    if len(unmapped) > 0:
        print(f"⚠️  Warning: Found unmapped job types: {unmapped}")
        print(f"   Mapping to 'medium' by default")
        df_mapped["job_income"] = df_mapped["job_income"].fillna("medium")
    
    # Show mapping distribution
    mapping_counts = df_mapped["job_income"].value_counts()
    print(f"✅ Job income mapping distribution:")
    for income_level, count in mapping_counts.items():
        print(f"   {income_level}: {count} ({count/len(df_mapped)*100:.1f}%)")
    
    # Drop original job column and keep job_income
    df_mapped = df_mapped.drop(columns=["job"])
    
    return df_mapped


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables using mixed strategy:
    - Ordinal encoding for ordered categories (education, job_income)
    - One-hot encoding for nominal categories (marital, default, etc.)
    
    Returns:
        DataFrame with encoded categoricals
    """
    print("\n" + "="*60)
    print("CATEGORICAL ENCODING")
    print("="*60)
    
    # Identify categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    print(f"\n📊 Categorical columns: {cat_cols}")
    print(f"📊 Numeric columns: {num_cols}")
    
    # Separate ordinal and one-hot columns
    ordinal_cols = [col for col in cat_cols if col in ORDINAL_CATEGORIES]
    onehot_cols = [col for col in cat_cols if col not in ORDINAL_CATEGORIES]
    
    print(f"\n🔢 Ordinal encoding: {ordinal_cols}")
    print(f"🔤 One-hot encoding: {onehot_cols}")
    
    X_encoded = df[num_cols].copy()  # Start with numeric columns
    
    # Ordinal encoding
    if ordinal_cols:
        ordinal_encoder = OrdinalEncoder(
            categories=[ORDINAL_CATEGORIES[col] for col in ordinal_cols],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        X_ordinal = pd.DataFrame(
            ordinal_encoder.fit_transform(df[ordinal_cols]),
            columns=ordinal_cols,
            index=df.index
        )
        X_encoded = pd.concat([X_encoded, X_ordinal], axis=1)
        print(f"✅ Ordinal encoded {len(ordinal_cols)} features")
    
    # One-hot encoding
    if onehot_cols:
        onehot_encoder = OneHotEncoder(
            drop='first',  # Drop first category to avoid multicollinearity
            handle_unknown='ignore',
            sparse_output=False
        )
        X_onehot = pd.DataFrame(
            onehot_encoder.fit_transform(df[onehot_cols]),
            columns=onehot_encoder.get_feature_names_out(onehot_cols),
            index=df.index
        )
        X_encoded = pd.concat([X_encoded, X_onehot], axis=1)
        print(f"✅ One-hot encoded {len(onehot_cols)} features into {X_onehot.shape[1]} columns")
    
    print(f"\n✅ Encoded dataset shape: {X_encoded.shape}")
    return X_encoded


def scale_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale numeric features using StandardScaler (z-score normalization).
    
    Returns:
        DataFrame with scaled numeric features
    """
    print("\n" + "="*60)
    print("NUMERIC FEATURE SCALING")
    print("="*60)
    
    # Identify numeric columns
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()
    
    print(f"\n📊 Numeric columns to scale: {num_cols}")
    print(f"📊 Categorical columns (no scaling): {cat_cols}")
    
    if num_cols:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(df[num_cols]),
            columns=num_cols,
            index=df.index
        )
        
        # Combine scaled numeric with categorical
        if cat_cols:
            X_processed = pd.concat([X_scaled, df[cat_cols]], axis=1)
        else:
            X_processed = X_scaled
        
        print(f"✅ Scaled {len(num_cols)} numeric features using StandardScaler")
    else:
        X_processed = df
        print("⚠️  No numeric features to scale")
    
    print(f"\n✅ Final processed dataset shape: {X_processed.shape}")
    return X_processed


def save_data(df: pd.DataFrame, output_path: Path) -> None:
    """Save processed dataset to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved processed dataset to: {output_path}")


def main():
    """Main preprocessing pipeline for clustering."""
    print("="*60)
    print("CLUSTERING DATA PREPROCESSING")
    print("="*60)
    print("\nFocus: Client features only for customer segmentation")
    print("="*60)
    
    # Load data
    df = load_data(INPUT_PATH)
    
    # Select client features only
    df_client = select_client_features(df)
    
    # Check missing values
    check_missing_values(df_client)
    
    # Map job to income levels
    df_mapped = map_job_to_income(df_client)
    
    # Encode categoricals
    df_encoded = encode_categoricals(df_mapped)
    
    # Scale numeric features
    df_processed = scale_numeric_features(df_encoded)
    
    # Save processed data
    save_data(df_processed, OUTPUT_PATH)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"\nProcessed dataset saved to: {OUTPUT_PATH}")
    print(f"Final shape: {df_processed.shape}")
    print(f"Features: {list(df_processed.columns)}")


if __name__ == "__main__":
    main()

