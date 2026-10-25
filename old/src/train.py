"""
Model Training & Tuning Module - Smart Home Energy Management
Author: Machine Learning Engineer & Research Team
Stages: Stage 8 (Baselines), Stage 9 (ML Models), Stage 10 (Hyperparameter Tuning), Stage 11 (Evaluation Setup)
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from preprocessing import chronological_split


def calculate_metrics(y_true, y_pred, model_name: str, split_name: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        'model': model_name,
        'split': split_name,
        'MAE_Wh': float(round(mae, 3)),
        'MSE_Wh2': float(round(mse, 3)),
        'RMSE_Wh': float(round(rmse, 3)),
        'R2': float(round(r2, 4))
    }


def train_and_evaluate_all():
    print("=" * 70)
    print("SMART HOME ENERGY MANAGEMENT: MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 70)

    data_path = os.path.join('data', 'processed', 'featured_energydata.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Featured dataset missing at {data_path}. Run feature_engineering.py first.")

    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])

    # Chronological Split (70% Train, 15% Val, 15% Test)
    train_df, val_df, test_df = chronological_split(df, train_ratio=0.70, val_ratio=0.15)

    target_col = 'target_next_appliances'
    ignore_cols = ['date', target_col]
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    print(f"Total features used: {len(feature_cols)}")

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    all_metrics = []
    models_dict = {}

    # -------------------------------------------------------------
    # STAGE 8: Baseline Models
    # -------------------------------------------------------------
    print("\n--- [STAGE 8] Training Baseline Models ---")
    
    # 1. Mean Baseline
    train_mean = y_train.mean()
    y_pred_val_mean = np.full_like(y_val, fill_value=train_mean)
    y_pred_test_mean = np.full_like(y_test, fill_value=train_mean)
    
    m_val_mean = calculate_metrics(y_val, y_pred_val_mean, "Baseline: Mean", "Validation")
    m_test_mean = calculate_metrics(y_test, y_pred_test_mean, "Baseline: Mean", "Test")
    all_metrics.extend([m_val_mean, m_test_mean])
    print(f"Mean Baseline        -> Val RMSE: {m_val_mean['RMSE_Wh']:.2f} Wh, Val R2: {m_val_mean['R2']:.4f}")

    # 2. Naive Persistence Baseline (predict next = current consumption)
    # appliances_current is the latest consumption reading at time t
    y_pred_val_persist = X_val['appliances_current'].values
    y_pred_test_persist = X_test['appliances_current'].values
    
    m_val_persist = calculate_metrics(y_val, y_pred_val_persist, "Baseline: Naive Persistence", "Validation")
    m_test_persist = calculate_metrics(y_test, y_pred_test_persist, "Baseline: Naive Persistence", "Test")
    all_metrics.extend([m_val_persist, m_test_persist])
    print(f"Persistence Baseline -> Val RMSE: {m_val_persist['RMSE_Wh']:.2f} Wh, Val R2: {m_val_persist['R2']:.4f}")

    # -------------------------------------------------------------
    # STAGE 9: Machine Learning Models
    # -------------------------------------------------------------
    print("\n--- [STAGE 9] Training Machine Learning Models ---")

    # Model 1: Linear / Ridge Regression Pipeline
    print("Training Model 1: Ridge Linear Regression (with StandardScaler)...")
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=10.0, random_state=42))
    ])
    lr_pipeline.fit(X_train, y_train)
    models_dict['Ridge_Linear_Regression'] = lr_pipeline

    m_train_lr = calculate_metrics(y_train, lr_pipeline.predict(X_train), "Ridge Regression", "Train")
    m_val_lr = calculate_metrics(y_val, lr_pipeline.predict(X_val), "Ridge Regression", "Validation")
    m_test_lr = calculate_metrics(y_test, lr_pipeline.predict(X_test), "Ridge Regression", "Test")
    all_metrics.extend([m_train_lr, m_val_lr, m_test_lr])
    print(f"Ridge Regression     -> Val RMSE: {m_val_lr['RMSE_Wh']:.2f} Wh, Val R2: {m_val_lr['R2']:.4f}")

    # Model 2: Random Forest Regressor
    print("Training Model 2: Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    models_dict['Random_Forest'] = rf_model

    m_train_rf = calculate_metrics(y_train, rf_model.predict(X_train), "Random Forest (Default)", "Train")
    m_val_rf = calculate_metrics(y_val, rf_model.predict(X_val), "Random Forest (Default)", "Validation")
    m_test_rf = calculate_metrics(y_test, rf_model.predict(X_test), "Random Forest (Default)", "Test")
    all_metrics.extend([m_train_rf, m_val_rf, m_test_rf])
    print(f"Random Forest        -> Val RMSE: {m_val_rf['RMSE_Wh']:.2f} Wh, Val R2: {m_val_rf['R2']:.4f}")

    # Model 3: Histogram Gradient Boosting Regressor
    print("Training Model 3: Histogram Gradient Boosting Regressor...")
    hgb_model = HistGradientBoostingRegressor(
        max_iter=120,
        learning_rate=0.08,
        max_depth=10,
        min_samples_leaf=20,
        l2_regularization=1.5,
        random_state=42
    )
    hgb_model.fit(X_train, y_train)
    models_dict['Hist_Gradient_Boosting'] = hgb_model

    m_train_hgb = calculate_metrics(y_train, hgb_model.predict(X_train), "Hist Gradient Boosting (Default)", "Train")
    m_val_hgb = calculate_metrics(y_val, hgb_model.predict(X_val), "Hist Gradient Boosting (Default)", "Validation")
    m_test_hgb = calculate_metrics(y_test, hgb_model.predict(X_test), "Hist Gradient Boosting (Default)", "Test")
    all_metrics.extend([m_train_hgb, m_val_hgb, m_test_hgb])
    print(f"HistGradientBoosting -> Val RMSE: {m_val_hgb['RMSE_Wh']:.2f} Wh, Val R2: {m_val_hgb['R2']:.4f}")

    # -------------------------------------------------------------
    # STAGE 10: Hyperparameter Tuning
    # -------------------------------------------------------------
    print("\n--- [STAGE 10] Hyperparameter Tuning via TimeSeriesSplit ---")
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Tuning HistGradientBoostingRegressor
    hgb_param_grid = {
        'learning_rate': [0.05, 0.08, 0.12],
        'max_depth': [8, 12],
        'min_samples_leaf': [15, 25],
        'l2_regularization': [0.5, 2.0]
    }
    print(f"Running GridSearchCV over {len(hgb_param_grid['learning_rate']) * len(hgb_param_grid['max_depth']) * len(hgb_param_grid['min_samples_leaf']) * len(hgb_param_grid['l2_regularization'])} configurations using TimeSeriesSplit...")
    hgb_grid = GridSearchCV(
        estimator=HistGradientBoostingRegressor(max_iter=100, random_state=42),
        param_grid=hgb_param_grid,
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    hgb_grid.fit(X_train, y_train)
    tuned_hgb = hgb_grid.best_estimator_
    models_dict['Tuned_Hist_Gradient_Boosting'] = tuned_hgb
    print(f"Best HistGradientBoosting Params: {hgb_grid.best_params_}")
    print(f"Best Cross-Validation RMSE: {-hgb_grid.best_score_:.2f} Wh")

    m_train_tuned_hgb = calculate_metrics(y_train, tuned_hgb.predict(X_train), "Tuned HistGradientBoosting", "Train")
    m_val_tuned_hgb = calculate_metrics(y_val, tuned_hgb.predict(X_val), "Tuned HistGradientBoosting", "Validation")
    m_test_tuned_hgb = calculate_metrics(y_test, tuned_hgb.predict(X_test), "Tuned HistGradientBoosting", "Test")
    all_metrics.extend([m_train_tuned_hgb, m_val_tuned_hgb, m_test_tuned_hgb])
    print(f"Tuned HistGB         -> Val RMSE: {m_val_tuned_hgb['RMSE_Wh']:.2f} Wh, Val R2: {m_val_tuned_hgb['R2']:.4f}")

    # Also Tune Random Forest on a refined practical grid
    rf_param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [12, 18],
        'min_samples_leaf': [2, 4]
    }
    rf_grid = GridSearchCV(
        estimator=RandomForestRegressor(max_features='sqrt', random_state=42, n_jobs=-1),
        param_grid=rf_param_grid,
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    tuned_rf = rf_grid.best_estimator_
    models_dict['Tuned_Random_Forest'] = tuned_rf
    print(f"Best Random Forest Params: {rf_grid.best_params_}")

    m_train_tuned_rf = calculate_metrics(y_train, tuned_rf.predict(X_train), "Tuned Random Forest", "Train")
    m_val_tuned_rf = calculate_metrics(y_val, tuned_rf.predict(X_val), "Tuned Random Forest", "Validation")
    m_test_tuned_rf = calculate_metrics(y_test, tuned_rf.predict(X_test), "Tuned Random Forest", "Test")
    all_metrics.extend([m_train_tuned_rf, m_val_tuned_rf, m_test_tuned_rf])
    print(f"Tuned Random Forest  -> Val RMSE: {m_val_tuned_rf['RMSE_Wh']:.2f} Wh, Val R2: {m_val_tuned_rf['R2']:.4f}")

    # -------------------------------------------------------------
    # STAGE 11: Final Model Selection & Serialization
    # -------------------------------------------------------------
    print("\n--- [STAGE 11] Model Comparison and Selection ---")
    metrics_df = pd.DataFrame(all_metrics)
    
    # Save metrics tables
    os.makedirs(os.path.join('outputs', 'metrics'), exist_ok=True)
    metrics_df.to_csv(os.path.join('outputs', 'metrics', 'model_comparison.csv'), index=False)
    with open(os.path.join('outputs', 'metrics', 'model_comparison.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)

    val_comparison = metrics_df[metrics_df['split'] == 'Validation'].sort_values('RMSE_Wh')
    print("\nValidation Performance Summary (Sorted by RMSE):")
    print(val_comparison.to_string(index=False))

    test_comparison = metrics_df[metrics_df['split'] == 'Test'].sort_values('RMSE_Wh')
    print("\nFinal Test Performance Summary (Untouched Test Set):")
    print(test_comparison.to_string(index=False))

    # Identify Best Model based on Validation RMSE
    best_candidate_name = val_comparison.iloc[0]['model']
    print(f"\nBest Performing Model on Validation Set: {best_candidate_name}")
    
    if "HistGradientBoosting" in best_candidate_name:
        final_best_model = tuned_hgb
        best_key = 'tuned_hist_gradient_boosting'
    elif "Random Forest" in best_candidate_name:
        final_best_model = tuned_rf
        best_key = 'tuned_random_forest'
    else:
        final_best_model = lr_pipeline
        best_key = 'ridge_linear_regression'

    # Save Models
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(lr_pipeline, os.path.join(models_dir, 'ridge_linear_regression.joblib'))
    joblib.dump(rf_model, os.path.join(models_dir, 'random_forest_default.joblib'))
    joblib.dump(tuned_rf, os.path.join(models_dir, 'random_forest_tuned.joblib'))
    joblib.dump(hgb_model, os.path.join(models_dir, 'hist_gradient_boosting_default.joblib'))
    joblib.dump(tuned_hgb, os.path.join(models_dir, 'hist_gradient_boosting_tuned.joblib'))
    joblib.dump(final_best_model, os.path.join(models_dir, 'best_model.joblib'))

    # Save feature names list
    with open(os.path.join(models_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_cols, f, indent=2)

    # Save Test Set with predictions for detailed evaluation plotting
    test_eval_df = test_df.copy()
    test_eval_df['pred_mean'] = y_pred_test_mean
    test_eval_df['pred_persist'] = y_pred_test_persist
    test_eval_df['pred_ridge'] = lr_pipeline.predict(X_test)
    test_eval_df['pred_rf'] = tuned_rf.predict(X_test)
    test_eval_df['pred_hgb'] = tuned_hgb.predict(X_test)
    test_eval_df['pred_best'] = final_best_model.predict(X_test)
    test_eval_df.to_csv(os.path.join('outputs', 'metrics', 'test_predictions.csv'), index=False)

    # Save Hyperparameter tuning report
    tuning_report = {
        'hgb_best_params': hgb_grid.best_params_,
        'hgb_best_cv_rmse': float(-hgb_grid.best_score_),
        'rf_best_params': rf_grid.best_params_,
        'rf_best_cv_rmse': float(-rf_grid.best_score_),
        'best_overall_model': best_candidate_name
    }
    with open(os.path.join('outputs', 'metrics', 'hyperparameter_tuning_report.json'), 'w') as f:
        json.dump(tuning_report, f, indent=2)

    print("\nAll models trained, evaluated, tuned, and serialized successfully.")
    return metrics_df, final_best_model


if __name__ == '__main__':
    train_and_evaluate_all()
