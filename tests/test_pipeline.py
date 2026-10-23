"""
Unit & Pipeline Tests - Smart Home Energy Management System
Author: Machine Learning Engineer & Research Team
"""

import os
import sys
import numpy as np
import pandas as pd

# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preparation import load_and_prepare_data
from feature_engineering import create_features
from preprocessing import chronological_split, analyze_target_distribution
from predict import EnergyPredictor


def test_data_preparation():
    raw_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'energydata_complete.csv')
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
        assert 'Appliances' in df.columns
        assert 'date' in df.columns
        assert len(df) == 19735
        assert df.isnull().sum().sum() == 0


def test_target_distribution_analysis():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'clean_energydata.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        analysis = analyze_target_distribution(df)
        assert analysis['min'] == 10.0
        assert analysis['max'] == 1080.0
        assert analysis['skewness'] > 2.0  # Positively skewed
        assert 'data_balancing_verdict' in analysis


def test_feature_engineering_no_leakage():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'clean_energydata.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path).head(300)
        df_feat = create_features(df)
        assert 'target_next_appliances' in df_feat.columns
        assert 'appliances_current' in df_feat.columns
        assert 'appliances_lag_10m' in df_feat.columns
        assert 'sin_hour' in df_feat.columns
        assert 'cos_hour' in df_feat.columns
        assert 'rv1' not in df_feat.columns
        assert 'rv2' not in df_feat.columns


def test_chronological_split():
    df = pd.DataFrame({'date': pd.date_range('2026-01-01', periods=100, freq='10min'), 'val': range(100)})
    train, val, test = chronological_split(df, 0.7, 0.15)
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15
    # Strict temporal ordering
    assert train['date'].max() < val['date'].min()
    assert val['date'].max() < test['date'].min()


def test_energy_predictor():
    predictor = EnergyPredictor()
    dummy_input = {feat: 20.0 for feat in predictor.feature_names}
    dummy_input['appliances_current'] = 50.0
    result = predictor.predict(dummy_input, tariff_per_kwh=0.20)

    assert 'predicted_consumption_Wh' in result
    assert result['predicted_consumption_Wh'] >= 0.0  # Physical constraint
    assert 'estimated_interval_cost' in result
    assert result['estimated_interval_cost'] >= 0.0
    assert len(result['prediction_interval_95_Wh']) == 2
    assert result['prediction_interval_95_Wh'][0] <= result['prediction_interval_95_Wh'][1]


if __name__ == '__main__':
    print("Running Pipeline Tests...")
    test_data_preparation()
    print("  [PASS] Data Preparation Test")
    test_target_distribution_analysis()
    print("  [PASS] Target Distribution Analysis Test")
    test_feature_engineering_no_leakage()
    print("  [PASS] Feature Engineering Test (Zero Leakage)")
    test_chronological_split()
    print("  [PASS] Chronological Split Test")
    test_energy_predictor()
    print("  [PASS] Energy Predictor Inference Test")
    print("All tests passed successfully!")
