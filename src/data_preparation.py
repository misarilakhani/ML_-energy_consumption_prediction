"""
Data Preparation Module - Smart Home Energy Consumption Prediction
Author: Machine Learning Engineer & Research Team
Dataset: UCI Appliances Energy Prediction (energydata_complete.csv)
"""

import os
import pandas as pd
import numpy as np


def load_and_prepare_data(raw_data_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Loads raw CSV, validates chronological integrity, checks sampling frequency,
    detects duplicates and anomalies, and exports clean prepared dataset.
    """
    print(f"[STAGE 3] Loading raw dataset from: {raw_data_path}")
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data file not found at {raw_data_path}")

    df_raw = pd.read_csv(raw_data_path)
    initial_rows, initial_cols = df_raw.shape
    print(f"Raw dataset shape: {initial_rows} rows x {initial_cols} columns")

    # 1. Parse datetime
    print("Parsing 'date' column into datetime format...")
    df = df_raw.copy()
    df['date'] = pd.to_datetime(df['date'])

    # 2. Chronological sorting
    print("Sorting chronologically by timestamp...")
    df = df.sort_values('date').reset_index(drop=True)

    # 3. Check timestamp uniqueness and duplicates
    duplicates_count = df.duplicated().sum()
    date_duplicates = df['date'].duplicated().sum()
    print(f"Full row duplicates: {duplicates_count}")
    print(f"Timestamp duplicates: {date_duplicates}")

    # 4. Check sampling interval consistency
    time_diffs = df['date'].diff().dropna()
    expected_interval = pd.Timedelta(minutes=10)
    intervals_equal = (time_diffs == expected_interval).all()
    print(f"Sampling interval is strictly 10 minutes: {intervals_equal}")
    if not intervals_equal:
        non_standard = (time_diffs != expected_interval).sum()
        print(f"Warning: {non_standard} intervals deviate from 10 minutes.")

    # 5. Missing values check
    missing_series = df.isnull().sum()
    total_missing = missing_series.sum()
    print(f"Total missing values across all columns: {total_missing}")

    # 6. Check data types and ranges
    target_col = 'Appliances'
    print(f"Target variable '{target_col}' range: min={df[target_col].min()} Wh, max={df[target_col].max()} Wh, mean={df[target_col].mean():.2f} Wh")

    # 7. Check random variables rv1 and rv2
    if 'rv1' in df.columns and 'rv2' in df.columns:
        identical_rv = np.allclose(df['rv1'], df['rv2'])
        print(f"Random variables rv1 and rv2 are identical: {identical_rv}")
        corr_rv1 = df['rv1'].corr(df[target_col])
        print(f"Correlation between rv1 and {target_col}: {corr_rv1:.5f}")

    # Final summary
    final_rows, final_cols = df.shape
    print(f"Prepared dataset shape: {final_rows} rows x {final_cols} columns (Retained 100% of observations)")

    # Save to processed directory if specified
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Successfully saved clean prepared dataset to: {output_path}")

    return df


if __name__ == '__main__':
    raw_path = os.path.join('data', 'raw', 'energydata_complete.csv')
    processed_path = os.path.join('data', 'processed', 'clean_energydata.csv')
    df_prepared = load_and_prepare_data(raw_path, processed_path)
