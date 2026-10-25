# Smart Home Energy Consumption Prediction and Optimization
## Group Presentation Slides — 4 Student Speaker Guide

**Domain:** Smart Energy Management  
**Project:** Supervised Machine Learning for 10-Minute Appliance Energy Prediction  
**Dataset:** UCI Appliances Energy Prediction (`energydata_complete.csv`)  

---

### Slide 1: Title & Team Overview
- **Title:** Smart Home Energy Consumption Prediction & Demand Optimization
- **Subtitle:** An End-to-End Leak-Free Machine Learning Framework for Residential Energy Management
- **Target Task:** Predict next 10-minute appliance energy consumption (Wh) using historical telemetry.
- **Team Roles:**
  - **Speaker 1 (Student 1):** Problem Definition, Use Case, and Dataset Provenance
  - **Speaker 2 (Student 2):** Preprocessing, Leakage Prevention, and Exploratory Data Analysis
  - **Speaker 3 (Student 3):** Feature Engineering, Baseline Benchmarks, and ML Model Training
  - **Speaker 4 (Student 4):** Model Comparison, Streamlit Demo, Limitations & Conclusion

---

### Slide 2: Problem Definition & Smart Grid Context (Speaker 1)
- **The Challenge:** Domestic electrical appliances (dryers, ovens, heaters) cause sharp, volatile demand surges that stress residential distribution grids and inflate electricity costs.
- **The Goal:** Accurate next 10-minute appliance load forecasting ($\hat{y}_{t+1}$) enables automated demand response and load shifting.
- **Key Decision Principle:** What is known at inference time?
  - Available: Information up to time $t$ (temperatures, humidity, historical energy draws).
  - Target: Energy consumed during $(t, t+10\text{ min}]$.
  - Zero Lookahead: Concurrent or future sensors are strictly isolated.

---

### Slide 3: Dataset Architecture & Sensor Topology (Speaker 1)
- **Source:** University of Mons, Belgium (Luis Candanedo et al., 2017).
- **Scale:** 19,735 observations at strict 10-minute intervals over 4.5 months (Jan–May 2016).
- **Multi-Sensor Suite:**
  - 9 Indoor Room Temperatures ($T_1 \dots T_9$) and Humidities ($RH_1 \dots RH_9$).
  - Macro Weather: Outside temperature, pressure, windspeed, humidity, visibility from Chievres Airport.
  - Sub-metered Appliance Load (`Appliances`, Wh) and Lighting (`lights`, Wh).
- **Random Noise Audit:** Identified and investigated `rv1` and `rv2` (uncorrelated artificial noise, $r = -0.011$). Decision: Safely dropped.

---

### Slide 4: Data Preprocessing & Integrity Verification (Speaker 2)
- **Zero Missing Values:** All 19,735 records verified complete; zero duplicate timestamps.
- **Outlier Reality Check:**
  - Max energy: $1080\text{ Wh}$, 75th percentile: $100\text{ Wh}$, Median: $60\text{ Wh}$.
  - Spikes $> 175\text{ Wh}$ account for $10.9\%$ of data.
  - *Engineering Decision:* Retained all peaks without clipping. Removing spikes blinds models to peak load events essential for demand management.
- **Data Balancing in Regression:**
  - SMOTE / synthetic resampling does not apply to continuous time series.
  - Resampling distorts physical transition probabilities.
- **Chronological Split (Zero Leakage):**
  - Train: First 70% (Jan 12 – Apr 16, 2016)
  - Validation: Next 15% (Apr 16 – May 07, 2016)
  - Test: Final 15% (May 07 – May 27, 2016)
  - Scalers fitted exclusively on Train!

---

### Slide 5: Exploratory Data Analysis Highlights (Speaker 2)
- **Diurnal Pattern (Fig 3):** Clear daily rhythm; low night standby ($48.2\text{ Wh}$ at 03:00), prominent dinner surge ($190.4\text{ Wh}$ at 18:00).
- **Day of Week (Fig 4):** Weekend loads are slightly elevated ($100.6\text{ Wh}$ vs $96.6\text{ Wh}$) due to occupancy.
- **Multicollinearity (Fig 7):** Internal room temperatures strongly correlate ($r_{T_1, T_2} = 0.84$), demonstrating spatial thermal coupling.
- **Spike Duration (Fig 8):** Active appliance loads persist for 20 to 60 minutes (2–6 consecutive intervals), proving the power of autoregressive lags.

---

### Slide 6: Feature Engineering Architecture (Speaker 3)
- **49 Total Predictors Constructed:**
  1. *Cyclical Calendar:* $\sin(\text{hour})$, $\cos(\text{hour})$, $\sin(\text{day})$, $\cos(\text{day})$.
  2. *Historical Lags:* $y_t$ (current draw), $y_{t-1}$ (10m ago), $y_{t-2}$ (20m ago), $y_{t-6}$ (1h ago), $y_{t-144}$ (24h ago).
  3. *Rolling Past Statistics:* 30-minute rolling mean & std; 1-hour rolling mean & max.
  4. *Physical Gradients:* $T_1 - T_{out}$ (indoor-outdoor gradient), $T_1 - T_2$ (cooking thermal signature), dew point depression.
- **Strict Leakage Prevention:** All rolling windows and lags computed strictly using past records ($\le t$).

---

### Slide 7: Model Development & Time-Series Tuning (Speaker 3)
- **Baselines:**
  - Mean Baseline: Predicts constant training mean ($\bar{y} = 97.7\text{ Wh}$).
  - Naive Persistence: Predicts $\hat{y}_{t+1} = y_t$ (strong short-term benchmark!).
- **Supervised ML Models:**
  - *Model 1 — Ridge Linear Regression:* Scaled L2-regularized linear model.
  - *Model 2 — Random Forest Regressor:* 150 bagged decision trees with sqrt sub-sampling.
  - *Model 3 — Histogram Gradient Boosting (HistGB):* Gradient boosted trees with histogram binning.
- **Hyperparameter Tuning:** Evaluated using `TimeSeriesSplit(n_splits=3)` on Training split only (never touching test set).

---

### Slide 8: Experimental Model Benchmark Results (Speaker 4)

| Model Architecture | Split | MAE (Wh) | RMSE (Wh) | $R^2$ Score | Assessment |
|---|---|---|---|---|---|
| **HistGradientBoosting (Default)** | **Val** | **25.36** | **56.40** | **0.6090** | **Best Selected Model** |
| Tuned HistGradientBoosting | Val | 25.66 | 56.97 | 0.6011 | Robust L2 regularization |
| Ridge Linear Regression | Val | 27.23 | 59.34 | 0.5672 | Strong linear benchmark |
| Tuned Random Forest | Val | 30.68 | 60.93 | 0.5438 | Reduces default RF variance |
| Naive Persistence Baseline | Val | 25.79 | 65.51 | 0.4726 | Competitive baseline |
| Mean Target Baseline | Val | 53.67 | 90.46 | -0.0057 | Zero predictive value |
| **Ridge Linear Regression** | **Test** | **26.69** | **60.52** | **0.5574** | **Generalizes solidly** |
| Tuned HistGradientBoosting | Test | 28.18 | 62.03 | 0.5351 | Strong out-of-time stability |

---

### Slide 9: Diagnostic Evaluation & Error Dissection (Speaker 4)
- **Zero-Centered Residuals:** Mean error is $-2.80\text{ Wh}$, showing minimal systematic bias.
- **Consumption Regime Breakdown:**
  - Standby ($\le 60\text{ Wh}$): MAE = $11.8\text{ Wh}$ (Highly accurate).
  - Moderate ($60\text{--}200\text{ Wh}$): MAE = $32.4\text{ Wh}$.
  - Peak Spikes ($> 200\text{ Wh}$): MAE = $114.6\text{ Wh}$ (Hardest to forecast).
- **Failure Cases:** Unannounced sudden switch-ons from $50\text{ Wh}$ standby to $800+\text{ Wh}$ within a single 10-minute window cannot be anticipated by observational models without pre-event user intent sensors.

---

### Slide 10: Interactive Application & Practical Use Case (Speaker 4)
- **Interactive Demonstrator:**
  - Streamlit dashboard (`app/streamlit_app.py`) & Standalone web GUI (`index.html`).
  - Next 10-minute prediction with $95\%$ prediction interval ($\pm 110\text{ Wh}$).
  - Live cost calculation via user-input electricity tariff ($\$/\text{kWh}$).
- **Demand Response Scenarios:**
  - Automated advice to shift discretionary laundry/dishwasher cycles out of 17:00–20:00 peak window to 22:00 off-peak.
- **Academic Limitations:**
  - Single-house dataset (Belgium); cannot generalize across building classes without retraining.
  - Observational predictions represent correlational patterns, not guaranteed physical savings.

---

### Slide 11: Summary & Conclusion (All Speakers)
1. End-to-end reproducible machine learning pipeline established across all 12 stages.
2. Verified zero data leakage via chronological time-series splitting.
3. HistGradientBoosting and Ridge regression beat naive persistence by $14\text{--}16\%$ in RMSE.
4. Working dashboard and full academic deliverables submitted.
