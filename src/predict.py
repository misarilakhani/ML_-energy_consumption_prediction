"""
Inference & Prediction Pipeline - Smart Home Energy Management
Author: Machine Learning Engineer & Research Team
Exposes consistent inference interface matching training pipeline, calculates costs,
and provides exploratory advisory scenarios.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib


class EnergyPredictor:
    def __init__(self, model_path: str = None, features_path: str = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if model_path is None:
            model_path = os.path.join(base_dir, 'models', 'best_model.joblib')
        if features_path is None:
            features_path = os.path.join(base_dir, 'models', 'feature_names.json')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Feature names file not found at: {features_path}")

        self.model = joblib.load(model_path)
        with open(features_path, 'r') as f:
            self.feature_names = json.load(f)

        # Standard error estimated from test residuals for prediction interval (+- 1.96 std for 95% interval)
        self.std_error_wh = 56.4

    def predict(self, input_data: dict or pd.DataFrame, tariff_per_kwh: float = 0.15) -> dict:
        """
        Predicts next 10-minute appliance energy consumption (Wh and kWh) and estimates cost.
        """
        if isinstance(input_data, dict):
            df_in = pd.DataFrame([input_data])
        else:
            df_in = input_data.copy()

        # Check for missing features and fill with 0 or mean if necessary
        for feat in self.feature_names:
            if feat not in df_in.columns:
                df_in[feat] = 0.0

        # Maintain exact column order
        X_infer = df_in[self.feature_names]

        # Predict
        pred_wh = float(self.model.predict(X_infer)[0])
        # Physical constraint: energy consumption cannot be negative
        pred_wh = max(0.0, pred_wh)
        pred_kwh = pred_wh / 1000.0
        cost_estimate = pred_kwh * tariff_per_kwh

        lower_bound_wh = max(0.0, pred_wh - 1.96 * self.std_error_wh)
        upper_bound_wh = pred_wh + 1.96 * self.std_error_wh

        # Load level classification
        if pred_wh < 60:
            regime = "Standby / Low Demand"
            recommendation = "Optimal time to run flexible background loads or charging cycles."
        elif pred_wh <= 200:
            regime = "Moderate Domestic Demand"
            recommendation = "Normal domestic activity. Standard tariff conditions apply."
        else:
            regime = "Peak / High Demand Surge"
            recommendation = "High-load appliances active. Consider postponing non-essential loads if under a time-of-use (TOU) tariff."

        return {
            'predicted_consumption_Wh': round(pred_wh, 2),
            'predicted_consumption_kWh': round(pred_kwh, 4),
            'prediction_interval_95_Wh': [round(lower_bound_wh, 2), round(upper_bound_wh, 2)],
            'estimated_interval_cost': round(cost_estimate, 4),
            'tariff_rate': tariff_per_kwh,
            'consumption_regime': regime,
            'exploratory_suggestion': recommendation,
            'disclaimer': "Exploratory advisory scenario only. Observational ML predictions do not guarantee causal savings."
        }


def get_sample_inference_scenarios():
    """
    Returns pre-configured realistic operational test scenarios from the holdout dataset.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_pred_path = os.path.join(base_dir, 'outputs', 'metrics', 'test_predictions.csv')
    if os.path.exists(test_pred_path):
        df_test = pd.read_csv(test_pred_path)
        # 1. Standby night example
        night_sample = df_test[df_test['hour'] == 3].iloc[0]
        # 2. Evening cooking/peak example
        peak_sample = df_test[(df_test['hour'] == 18) & (df_test['appliances_current'] > 150)].iloc[0]
        # 3. Afternoon moderate example
        afternoon_sample = df_test[df_test['hour'] == 14].iloc[0]

        return {
            'Night Standby (03:00)': night_sample.to_dict(),
            'Evening Peak Cooking (18:00)': peak_sample.to_dict(),
            'Afternoon Moderate (14:00)': afternoon_sample.to_dict()
        }
    return {}


if __name__ == '__main__':
    predictor = EnergyPredictor()
    scenarios = get_sample_inference_scenarios()
    if scenarios:
        for name, scen in scenarios.items():
            result = predictor.predict(scen, tariff_per_kwh=0.18)
            print(f"\n--- Scenario: {name} ---")
            print(f"Actual Next Wh: {scen.get('target_next_appliances', 'N/A')}")
            print(f"Predicted Wh:   {result['predicted_consumption_Wh']} Wh ({result['predicted_consumption_kWh']} kWh)")
            print(f"95% Interval:   {result['prediction_interval_95_Wh']} Wh")
            print(f"Estimated Cost: ${result['estimated_interval_cost']}")
            print(f"Advisory:       {result['exploratory_suggestion']}")
