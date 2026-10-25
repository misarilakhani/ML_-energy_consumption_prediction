"""
Preprocessing Module - Smart Home Energy Consumption Prediction
Author: Machine Learning Engineer & Research Team
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib


def analyze_target_distribution(df: pd.DataFrame, target_col: str = 'Appliances') -> dict:
    """
    Analyzes target distribution, skewness, outliers, and quantiles (Stage 4 & Stage 5).
    Documents why regression cannot use classification balancing (e.g. SMOTE) and why
    high consumption spikes must be retained as valid physical phenomena.
    """
    s = df[target_col]
    q25 = float(s.quantile(0.25))
    q50 = float(s.median())
    q75 = float(s.quantile(0.75))
    q90 = float(s.quantile(0.90))
    q95 = float(s.quantile(0.95))
    q99 = float(s.quantile(0.99))
    iqr = q75 - q25
    upper_bound = q75 + 1.5 * iqr
    outlier_count = int((s > upper_bound).sum())
    skewness = float(s.skew())
    kurtosis = float(s.kurt())

    analysis = {
        'count': int(len(s)),
        'mean': float(s.mean()),
        'std': float(s.std()),
        'min': float(s.min()),
        'q25': q25,
        'median': q50,
        'q75': q75,
        'q90': q90,
        'q95': q95,
        'q99': q99,
        'max': float(s.max()),
        'iqr': iqr,
        'iqr_upper_bound': upper_bound,
        'outlier_count': outlier_count,
        'outlier_percentage': float((outlier_count / len(s)) * 100),
        'skewness': skewness,
        'kurtosis': kurtosis,
        'data_balancing_verdict': (
            "Regression problem: classification resampling (SMOTE / random undersampling) is mathematically "
            "inapplicable because target is continuous Wh. Synthetic interpolation distorts physical time-series "
            "dynamics. Legitimate consumption peaks (e.g., ovens, laundry dryers) represent genuine load spikes "
            "and must not be deleted or clipped, as detecting high-demand periods is vital for smart grid demand response."
        )
    }
    return analysis


def chronological_split(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15):
    """
    Splits time-series data chronologically into Train (70%), Validation (15%), and Test (15%).
    Strictly avoids random shuffling to prevent future-information leakage.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)

    print(f"Chronological split sizes:")
    print(f"  Train:      {len(train_df)} rows ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"  Validation: {len(val_df)} rows ({val_df['date'].min()} to {val_df['date'].max()})")
    print(f"  Test:       {len(test_df)} rows ({test_df['date'].min()} to {test_df['date'].max()})")

    return train_df, val_df, test_df


def fit_and_transform_scaler(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                             scaler_type: str = 'robust', scaler_save_path: str = None):
    """
    Fits scaler exclusively on training data and transforms validation and test sets.
    Prevents data leakage from future sets into training representations.
    """
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()

    feature_cols = X_train.columns.tolist()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=feature_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

    if scaler_save_path:
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
        joblib.dump(scaler, scaler_save_path)
        print(f"Scaler saved to: {scaler_save_path}")

    return scaler, X_train_scaled, X_val_scaled, X_test_scaled


if __name__ == '__main__':
    data_path = os.path.join('data', 'processed', 'clean_energydata.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        analysis = analyze_target_distribution(df)
        print("Target distribution analysis:")
        print(json.dumps(analysis, indent=2))
