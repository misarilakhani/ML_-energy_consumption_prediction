"""
Feature Engineering Module - Smart Home Energy Consumption Prediction
Author: Machine Learning Engineer & Research Team
Task: Predict appliance energy consumption (Wh) for the next 10-minute interval (t+1)
      using strictly measurements available at or before prediction time (t).
"""

import os
import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame, drop_rv: bool = True) -> pd.DataFrame:
    """
    Creates temporal, cyclical, lagged, rolling, and physical domain features.
    
    Prediction Task:
      Input: Information available at current time t (sensors at t, consumption at t, t-1, t-2, etc.)
      Target: Appliance energy consumption at next 10-minute interval (t+1), denoted as 'target_next_appliances'.
      
    Guarantees:
      - Strictly NO future-information leakage.
      - Rolling features are computed strictly on past and current values (up to t).
      - Random noise variables (rv1, rv2) are removed when drop_rv=True.
    """
    print("[STAGE 7] Feature Engineering: Creating features for next-interval prediction...")
    df_feat = df.copy()

    # Ensure datetime format and sort chronologically
    df_feat['date'] = pd.to_datetime(df_feat['date'])
    df_feat = df_feat.sort_values('date').reset_index(drop=True)

    # 1. Target Variable: Next interval appliance consumption (t+1)
    df_feat['target_next_appliances'] = df_feat['Appliances'].shift(-1)
    # Next interval timestamp for calendar features
    next_date = df_feat['date'] + pd.Timedelta(minutes=10)

    # 2. Calendar and Time Features for the prediction interval (t+1)
    df_feat['hour'] = next_date.dt.hour
    df_feat['day_of_week'] = next_date.dt.dayofweek
    df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)
    df_feat['month'] = next_date.dt.month

    # 3. Cyclical Temporal Encodings (sin / cos)
    df_feat['sin_hour'] = np.sin(2 * np.pi * df_feat['hour'] / 24.0)
    df_feat['cos_hour'] = np.cos(2 * np.pi * df_feat['hour'] / 24.0)
    df_feat['sin_day'] = np.sin(2 * np.pi * df_feat['day_of_week'] / 7.0)
    df_feat['cos_day'] = np.cos(2 * np.pi * df_feat['day_of_week'] / 7.0)

    # 4. Historical Lagged Consumption Features (available at or before time t)
    # Appliances at current time t is lag 0 relative to t, or lag 1 relative to target t+1
    df_feat['appliances_current'] = df_feat['Appliances']            # consumption at t
    df_feat['appliances_lag_10m'] = df_feat['Appliances'].shift(1)   # consumption at t-1 (10 min ago)
    df_feat['appliances_lag_20m'] = df_feat['Appliances'].shift(2)   # consumption at t-2 (20 min ago)
    df_feat['appliances_lag_30m'] = df_feat['Appliances'].shift(3)   # consumption at t-3 (30 min ago)
    df_feat['appliances_lag_1h']  = df_feat['Appliances'].shift(6)   # consumption at t-6 (1 hour ago)
    df_feat['appliances_lag_24h'] = df_feat['Appliances'].shift(144) # consumption at t-144 (24 hours ago)

    # 5. Rolling Statistics (Strictly computed on historical values up to t)
    # Rolling 30 min (3 intervals up to t): mean, std
    df_feat['appliances_roll_mean_30m'] = df_feat['Appliances'].rolling(window=3).mean()
    df_feat['appliances_roll_std_30m']  = df_feat['Appliances'].rolling(window=3).std().fillna(0)
    # Rolling 1 hour (6 intervals up to t): mean, max
    df_feat['appliances_roll_mean_1h']  = df_feat['Appliances'].rolling(window=6).mean()
    df_feat['appliances_roll_max_1h']   = df_feat['Appliances'].rolling(window=6).max()

    # 6. Domain-Specific Physical and Thermal Features
    # Thermal gradient between inside living space and outdoor weather
    df_feat['T_diff_indoor_outdoor'] = df_feat['T1'] - df_feat['T_out']
    # Thermal gradient between kitchen and living room (activity / cooking signature)
    df_feat['T_kitchen_living_diff'] = df_feat['T1'] - df_feat['T2']
    # Atmospheric dew point depression (dryness indicator)
    df_feat['dewpoint_depression']   = df_feat['T_out'] - df_feat['Tdewpoint']
    # Aggregated indoor environment
    indoor_temp_cols = [f'T{i}' for i in range(1, 10)]
    indoor_hum_cols  = [f'RH_{i}' for i in range(1, 10)]
    df_feat['T_indoor_mean']  = df_feat[indoor_temp_cols].mean(axis=1)
    df_feat['RH_indoor_mean'] = df_feat[indoor_hum_cols].mean(axis=1)

    # 7. Remove uninformative random variables (rv1, rv2)
    if drop_rv:
        drop_cols = [col for col in ['rv1', 'rv2'] if col in df_feat.columns]
        if drop_cols:
            print(f"Dropping uninformative random noise columns: {drop_cols}")
            df_feat = df_feat.drop(columns=drop_cols)

    # 8. Drop warm-up rows (due to 24h lag = 144 steps) and final row (due to shift(-1) target)
    initial_len = len(df_feat)
    df_clean = df_feat.dropna().reset_index(drop=True)
    dropped_count = initial_len - len(df_clean)
    print(f"Dropped {dropped_count} boundary rows (due to 24-hr lag initialization and next-interval target shift).")
    print(f"Engineered dataset shape: {df_clean.shape[0]} rows x {df_clean.shape[1]} columns.")

    return df_clean


def get_feature_groups():
    """
    Returns dictionary of categorized feature names for interpretability.
    """
    return {
        'temporal': ['hour', 'day_of_week', 'is_weekend', 'month', 'sin_hour', 'cos_hour', 'sin_day', 'cos_day'],
        'historical_consumption': [
            'appliances_current', 'appliances_lag_10m', 'appliances_lag_20m', 
            'appliances_lag_30m', 'appliances_lag_1h', 'appliances_lag_24h',
            'appliances_roll_mean_30m', 'appliances_roll_std_30m', 
            'appliances_roll_mean_1h', 'appliances_roll_max_1h'
        ],
        'indoor_climate': [
            'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 
            'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9', 'RH_9',
            'T_indoor_mean', 'RH_indoor_mean', 'T_kitchen_living_diff'
        ],
        'outdoor_weather': [
            'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 
            'Tdewpoint', 'dewpoint_depression', 'T_diff_indoor_outdoor'
        ],
        'other': ['lights']
    }


if __name__ == '__main__':
    data_path = os.path.join('data', 'processed', 'clean_energydata.csv')
    output_path = os.path.join('data', 'processed', 'featured_energydata.csv')
    df = pd.read_csv(data_path)
    df_featured = create_features(df)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_featured.to_csv(output_path, index=False)
    print(f"Saved featured dataset to: {output_path}")
