# Smart Home Energy Consumption Prediction and Optimization
## Machine Learning Innovative Assignment — Smart Energy Management

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Tests: 100% Passed](https://img.shields.io/badge/Tests-100%25_Passed-brightgreen.svg)](tests/test_pipeline.py)
[![Problem: Regression](https://img.shields.io/badge/Problem-Time--Series_Regression-orange.svg)]()

This repository contains an end-to-end, leak-free, academically rigorous machine learning project developed for the university Machine Learning Innovative Assignment.

---

## 1. Project Overview & Prediction Task

- **Domain:** Smart Energy Management
- **Dataset:** UCI Appliances Energy Prediction (`energydata_complete.csv`)
- **Dataset URL:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)
- **Problem Type:** Supervised Machine Learning — Continuous Time-Series Regression
- **Core Prediction Objective:**
  > **Predict household appliance energy consumption (Wh) for the next 10-minute interval ($t+1$) based strictly on historical consumption and environmental sensor data available at or before the current interval ($t$).**

---

## 2. Project Directory Structure

```text
smart-energy-management/
├── index.html                          # Zero-dependency interactive web dashboard (Double-click to open)
├── requirements.txt                    # Python package dependencies
├── README.md                           # Master setup and execution instructions
├── VIVA_GUIDE.md                       # Comprehensive oral examination and viva guide
├── GUIDELINE_COMPLIANCE.md             # Complete audit mapping every guideline to code & evidence
├── .gitignore                          # Standard git exclusion rules
├── app/
│   ├── streamlit_app.py                # Official Streamlit interactive dashboard
│   └── standalone_dashboard.html       # Standalone interactive dashboard with live Chart.js
├── data/
│   ├── raw/
│   │   └── energydata_complete.csv     # Original raw UCI dataset (19,735 rows x 29 cols)
│   └── processed/
│       ├── clean_energydata.csv        # Validated chronological clean dataset
│       └── featured_energydata.csv     # Engineered leak-free dataset (19,590 rows x 51 cols)
├── notebooks/
│   └── exploratory_analysis.ipynb      # Step-by-step academic walkthrough notebook
├── src/
│   ├── data_preparation.py             # Stage 2-3: CSV loading, datetime parsing, sampling verification
│   ├── preprocessing.py                # Stage 4-5: Target analysis, outlier review, chronological split
│   ├── feature_engineering.py          # Stage 7: Cyclical encodings, lags, rolling stats, physical gradients
│   ├── eda.py                          # Stage 6: Generation of all 10 mandatory EDA figures
│   ├── train.py                        # Stage 8-10: Baselines, Ridge, RF, HistGB & TimeSeriesSplit tuning
│   ├── evaluation.py                   # Stage 11: 7 diagnostic evaluation figures & residual audit
│   └── predict.py                      # Stage 12: Real-time inference engine & tariff cost calculator
├── models/
│   ├── best_model.joblib               # Best selected model (HistGradientBoosting)
│   ├── hist_gradient_boosting_default.joblib
│   ├── hist_gradient_boosting_tuned.joblib
│   ├── random_forest_default.joblib
│   ├── random_forest_tuned.joblib
│   ├── ridge_linear_regression.joblib
│   └── feature_names.json              # Exact 49-feature schema required for inference
├── outputs/
│   ├── figures/                        # 17 high-resolution publication figures (10 EDA + 7 Evaluation)
│   │   ├── fig1_appliance_distribution.png
│   │   ├── fig2_consumption_timeline.png
│   │   ├── fig3_hourly_consumption_pattern.png
│   │   ├── fig4_day_of_week_consumption.png
│   │   ├── fig5_temperature_vs_appliances.png
│   │   ├── fig6_humidity_vs_appliances.png
│   │   ├── fig7_correlation_heatmap.png
│   │   ├── fig8_high_consumption_zoom.png
│   │   ├── fig9_weather_relationships.png
│   │   ├── fig10_target_distribution_log.png
│   │   ├── eval_fig1_actual_vs_predicted.png
│   │   ├── eval_fig2_residual_distribution.png
│   │   ├── eval_fig3_error_over_time.png
│   │   ├── eval_fig4_model_comparison_bars.png
│   │   ├── eval_fig5_train_vs_val_overfitting.png
│   │   ├── eval_fig6_high_consumption_error_analysis.png
│   │   └── eval_fig7_worst_failure_cases.png
│   └── metrics/                        # Empirical evaluation results & summaries
│       ├── model_comparison.csv
│       ├── model_comparison.json
│       ├── eda_summary.json
│       ├── evaluation_diagnostics_summary.json
│       ├── hyperparameter_tuning_report.json
│       └── test_predictions.csv
├── presentation/
│   ├── presentation.md                 # 11-slide presentation guide structured for 4 speakers
│   └── index.html                      # Interactive slide deck (F11 for presentation mode)
├── reports/
│   └── Smart_Home_Energy_Management_Report.md  # Formal 15-section academic written report
├── scripts/
│   └── generate_notebook.py            # Reproducibility script to generate Jupyter notebook
└── tests/
    └── test_pipeline.py                # Automated test suite (100% passing)
```

---

## 3. Installation & Environment Setup

### 3.1 Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14 (64-bit).
- Standard web browser (Chrome, Edge, Firefox).

### 3.2 Install Dependencies
Clone this repository and install required packages:
```bash
pip install -r requirements.txt
```

---

## 4. Step-by-Step Reproduction Pipeline

The entire machine learning lifecycle is fully automated and reproducible via dedicated scripts:

### Step 1: Data Preparation & Integrity Check (Stage 2 & 3)
Validates chronological ordering, 10-minute uniform sampling, duplicate absence, and data types:
```bash
python src/data_preparation.py
```
*Output:* `data/processed/clean_energydata.csv` (19,735 rows preserved).

### Step 2: Exploratory Data Analysis (Stage 6)
Generates all 10 mandatory publication-grade analytical figures:
```bash
python src/eda.py
```
*Output:* 10 figures in `outputs/figures/` and statistical summary in `outputs/metrics/eda_summary.json`.

### Step 3: Feature Engineering (Stage 7)
Constructs 49 predictors (cyclical time, autoregressive lags, rolling past stats, thermal gradients) with zero lookahead leakage, dropping artificial noise columns `rv1` and `rv2`:
```bash
python src/feature_engineering.py
```
*Output:* `data/processed/featured_energydata.csv` (19,590 rows x 51 columns).

### Step 4: Model Training, Tuning & Benchmark (Stage 8, 9, 10, 11)
Trains Baselines, Ridge Linear Regression, Random Forest, and Histogram Gradient Boosting with `TimeSeriesSplit` cross-validation:
```bash
python src/train.py
```
*Output:* Serialized models in `models/`, benchmark metrics in `outputs/metrics/model_comparison.csv`, and predictions in `outputs/metrics/test_predictions.csv`.

### Step 5: Evaluation Diagnostics & Error Dissection (Stage 11)
Generates 7 diagnostic figures (Residual Q-Q plots, time-slice tracking, regime error analysis, failure case audits):
```bash
python src/evaluation.py
```
*Output:* 7 diagnostic figures in `outputs/figures/` and `outputs/metrics/evaluation_diagnostics_summary.json`.

### Step 6: Automated Test Verification
Verifies data preparation, zero-leakage feature engineering, chronological splitting, and real-time predictor constraints:
```bash
python tests/test_pipeline.py
```
*Output:* `All tests passed successfully!`

---

## 5. Model Performance Comparison

Evaluated on strict chronological splits: **Train (70%, $N=13,713$)**, **Validation (15%, $N=2,938$)**, and untouched **Holdout Test (15%, $N=2,939$)**:

| Model Architecture | Split | MAE (Wh) | MSE ($\text{Wh}^2$) | RMSE (Wh) | $R^2$ Score | Key Characteristics |
|---|---|---|---|---|---|---|
| **HistGradientBoosting (Default)** | **Validation** | **25.36** | **3181.36** | **56.40** | **0.6090** | **Best Selected Model** |
| Tuned HistGradientBoosting | Validation | 25.66 | 3245.65 | 56.97 | 0.6011 | L2 Regularized (0.5), Depth 12 |
| Ridge Linear Regression | Validation | 27.23 | 3521.48 | 59.34 | 0.5672 | Robust to sensor collinearity |
| Tuned Random Forest | Validation | 30.68 | 3711.99 | 60.93 | 0.5438 | 150 bagged decision trees |
| Random Forest (Default) | Validation | 33.83 | 4125.46 | 64.23 | 0.4930 | Unconstrained tree depth |
| Naive Persistence Baseline ($y_t$) | Validation | 25.79 | 4291.97 | 65.51 | 0.4726 | Reference short-term baseline |
| Mean Target Baseline ($\bar{y}$) | Validation | 53.67 | 8183.56 | 90.46 | -0.0057 | Naive uninformative baseline |
| **Ridge Linear Regression** | **Test (Holdout)** | **26.69** | **3663.31** | **60.52** | **0.5574** | Strong out-of-time stability |
| Tuned HistGradientBoosting | Test (Holdout) | 28.18 | 3848.05 | 62.03 | 0.5351 | Solid generalizability |
| HistGradientBoosting (Default) | Test (Holdout) | 30.86 | 4086.88 | 63.93 | 0.5062 | Consistent out-of-time accuracy |

---

## 6. Interactive Application Launch

### Option A: Standalone Instant Web Dashboard (Recommended for Viva)
No installations or server startup required! Simply double-click `index.html` at the project root or open `app/standalone_dashboard.html` in any web browser.
- Live interactive 10-minute prediction calculator.
- Preset test scenarios (Night Standby, Evening Peak Cooking, Afternoon Moderate).
- Real-time electricity tariff cost estimator.
- Dynamic Chart.js interactive diurnal and distribution plots.

### Option B: Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```
Access the dashboard at `http://localhost:8501`.

---

## 7. Presentation & Viva Preparation

- **Interactive Presentation Deck:** Open `presentation/index.html` in any browser. Press `F11` for full screen and navigate using arrow keys (`←` / `→`).
- **Speaker Guide (4 Students):** Read `presentation/presentation.md` for role-by-role speaker notes.
- **Viva Q&A Preparation:** Read `VIVA_GUIDE.md` for answers to the top 11 theoretical and practical defense questions.
- **Guideline Compliance Checklist:** Review `GUIDELINE_COMPLIANCE.md` for the line-by-line verification table.

---

## 8. Academic Integrity & Citation
1. Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). *Data driven prediction models of energy use of appliances in a low-energy house.* Energy and Buildings, 140, 81–97.
2. UCI Machine Learning Repository. (2017). *Appliances Energy Prediction Dataset.* DOI: 10.24432/C57688.
