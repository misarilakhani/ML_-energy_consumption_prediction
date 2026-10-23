# Smart Home Energy Management System
## Comprehensive Viva & Oral Examination Preparation Guide

This guide provides direct, clear, academically rigorous answers to potential questions asked by university professors and external evaluators during project defense and viva voce.

---

### Q1: What is the core problem and practical objective of this project?
**Answer:**  
The objective is to predict residential appliance electrical consumption in Watt-hours (Wh) for the upcoming 10-minute interval using historical energy readings, multi-room indoor temperature and humidity sensors, and local outdoor weather telemetry. This short-term forecasting capability enables smart home controllers and utility aggregators to anticipate demand spikes, schedule flexible background loads (e.g., laundry, dishwashing, EV charging), and minimize electricity bills under dynamic Time-of-Use (TOU) tariffs.

---

### Q2: Why is this formulated as a Regression problem rather than Classification?
**Answer:**  
Electric power and energy are continuous physical quantities governed by thermodynamics and circuit laws. Energy consumed over a 10-minute window can take any non-negative real value (e.g., $48.2\text{ Wh}$, $190.4\text{ Wh}$, $630.0\text{ Wh}$). 
- Discretizing consumption into arbitrary classes (e.g., "High", "Low") discards critical physical magnitude information required for load flow calculations, cost computations, and grid transformer sizing.
- Continuous regression predicts exact numerical energy draws in physical engineering units (Wh and kWh), allowing direct financial cost calculations ($\text{Cost} = \text{kWh} \times \text{Tariff}$).

---

### Q3: Why is standard Data Balancing (like SMOTE or Random Undersampling) not applicable here?
**Answer:**  
1. **Mathematical Inapplicability:** SMOTE (Synthetic Minority Over-sampling Technique) operates by interpolating between nearest neighbors in feature space for discrete class labels. In continuous time series, synthesizing artificial observations between unrelated timestamps disrupts physical time continuity and thermodynamic relationships.
2. **Distributional Realism:** Household electricity consumption naturally follows a right-skewed distribution where the majority of time is spent at idle/standby levels ($40\text{--}60\text{ Wh}$) with infrequent peak spikes ($200\text{--}1000\text{ Wh}$). Artificially downsampling the low-energy intervals would bias the model toward chronic overestimation of base loads.
3. **Appropriate Machine Learning Solution:** Instead of artificial data manipulation, we use loss formulations and algorithms that are naturally resilient to skewed continuous targets—specifically Tree-based Ensembles (Histogram Gradient Boosting and Random Forest) that partition continuous feature spaces non-parametrically.

---

### Q4: How were Missing Values and Outliers handled?
**Answer:**  
- **Missing Values:** The UCI Appliances Energy Prediction dataset contains exactly zero missing values across all 19,735 rows and 29 columns. This was empirically verified by checking `df.isnull().sum()`.
- **Outliers:** The target variable `Appliances` has a median of $60\text{ Wh}$ and maximum of $1080\text{ Wh}$, with $10.9\%$ of values exceeding $Q_3 + 1.5 \times IQR$ ($175\text{ Wh}$).
  - *Treatment Decision:* We **retained all high readings without deletion or clipping**.
  - *Engineering Justification:* In smart energy systems, extreme energy values are not measurement errors or sensor defects; they are legitimate physical activations of high-wattage domestic appliances (such as induction stoves, ovens, electric clothes dryers). Removing them would destroy the system's ability to anticipate the very peak load events that grid operators need to forecast.

---

### Q5: What is Data Leakage, and how did you prevent it in this project?
**Answer:**  
Data leakage occurs when information from outside the training dataset (especially future information) is inadvertently used to create features, fit scalers, or train models.
In this project, leakage was systematically prevented through:
1. **Chronological Splitting:** Never using random train-test splitting (like `train_test_split(shuffle=True)`). Shuffling causes future observations from Tuesday to predict Monday, creating unrealistic temporal overlap. We used strict chronological splitting: First 70% Train, Next 15% Validation, Final 15% untouched Test.
2. **Strict Filtration for Features:** At prediction time $t$, the model must predict $y_{t+1}$. All engineered features (lags $y_t, y_{t-1}, y_{t-6}$, rolling 30-min and 1-hour statistics, sensor readings) were calculated strictly using data up to time $t$. No future sensor values or future consumption readings were accessed.
3. **Scaler Isolation:** Feature scalers (StandardScaler / RobustScaler) were fitted exclusively on the 70% training split, and only applied (`transform()`) to validation and test sets.

---

### Q6: What did Feature Engineering accomplish?
**Answer:**  
Raw sensor timestamps and ambient readings alone have weak linear correlation with instantaneous energy consumption. Feature engineering created 49 meaningful physical and temporal signals:
1. **Cyclical Temporal Signals:** Encoded time using sine and cosine functions ($\sin(2\pi h / 24)$ and $\cos(2\pi h / 24)$) so that 23:50 and 00:00 are recognized as adjacent rather than numerically distant.
2. **Autoregressive Lags ($y_t, y_{t-1}, y_{t-2}, y_{t-6}, y_{t-144}$):** Captured short-term appliance operational cycles ($20\text{--}60\text{ min}$) and daily 24-hour routine cycles.
3. **Past Rolling Statistics:** Rolling 30-minute mean/std and 1-hour mean/max to measure short-term consumption trajectory and volatility.
4. **Physical Thermal Gradients:**
   - $T_1 - T_{out}$ (indoor vs outdoor temperature difference governing building thermal losses).
   - $T_1 - T_2$ (kitchen vs living room temperature difference indicating active cooking).
   - $T_{out} - T_{dewpoint}$ (dewpoint depression indicating ambient humidity stress).
5. **Noise Elimination:** Removed artificial columns `rv1` and `rv2` which were uncorrelated noise ($r = -0.011$).

---

### Q7: Explain MAE, MSE, RMSE, and R² in the context of this project.
**Answer:**  
- **MAE (Mean Absolute Error):** $\frac{1}{n} \sum |y_i - \hat{y}_i|$. Measured in Watt-hours (Wh). It represents the expected average error magnitude on a typical 10-minute prediction. Our best model achieves an MAE of **$25.36\text{ Wh}$**, which is very small relative to active appliance draws.
- **MSE (Mean Squared Error):** $\frac{1}{n} \sum (y_i - \hat{y}_i)^2$. Measured in $\text{Wh}^2$. Penalizes large errors quadratically.
- **RMSE (Root Mean Squared Error):** $\sqrt{\text{MSE}}$. Measured in Watt-hours (Wh). Because it squares errors before averaging, it heavily penalizes large peak misses. Our model achieves an RMSE of **$56.40\text{ Wh}$**.
- **$R^2$ (Coefficient of Determination):** $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$. Proportion of variance explained compared to predicting the training mean. Our model achieves $R^2 = 0.6090$ on validation, whereas the naive mean baseline achieves $R^2 \approx 0.0$.

---

### Q8: Which model performed best, and why?
**Answer:**  
**Histogram Gradient Boosting Regressor (HistGB)** performed best on validation data, achieving an RMSE of **$56.40\text{ Wh}$**, MAE of **$25.36\text{ Wh}$**, and $R^2 = 0.6090$.
*Why it won:*
1. **Non-linear Partitioning:** Captures complex, threshold-based interactions between indoor humidity surges (cooking steam), time-of-day, and prior cycle duration.
2. **Collinearity Resilience:** Kitchen and living room temperatures have high multicollinearity ($r = 0.84$), which damages unregularized linear models but does not hinder tree-based splits.
3. **Histogram Binning:** Discretizes continuous feature values into 256 integer bins, drastically reducing memory overhead and preventing individual noisy sensor fluctuations from causing overfitting.

---

### Q9: What were the Baseline Models, and did the ML models actually beat them?
**Answer:**  
Yes, decisively:
1. **Mean Baseline ($\hat{y} = \bar{y}$):** Validation RMSE = **$90.46\text{ Wh}$**, $R^2 = -0.0057$.
2. **Naive Persistence Baseline ($\hat{y}_{t+1} = y_t$):** Validation RMSE = **$65.51\text{ Wh}$**, $R^2 = 0.4726$. Persistence is a strong benchmark in energy time series because appliance cycles (dishwashers, washers) often last multiple 10-minute intervals.
3. **ML Performance:** HistGradientBoosting achieved Validation RMSE = **$56.40\text{ Wh}$** ($13.9\%$ error reduction over Persistence, and $37.7\%$ error reduction over Mean baseline).

---

### Q10: Where does the model fail, and what are its key limitations?
**Answer:**  
1. **Failure Cases (Sudden Switch-ons):** The greatest errors occur when an occupant abruptly turns on a high-draw appliance (e.g. from standby $50\text{ Wh}$ directly to $820\text{ Wh}$) within a single 10-minute window. Because there are no physical precursors in temperature or humidity sensors prior to the switch being turned on, an observational model cannot anticipate the instantaneous human decision.
2. **Single Dwelling Generalization:** The dataset comes from a single low-energy home in Belgium over 4.5 months. Thermal characteristics, family size, appliance types, and weather profiles differ in other geographic regions, meaning localized calibration is required before transfer.
3. **Prediction vs Causal Optimization:** The machine learning model captures predictive correlations, not causal mechanics. Our dashboard recommendations (e.g., shifting loads to night hours) are exploratory simulation scenarios rather than experimentally verified physical guarantees.

---

### Q11: How does the interactive application compute predictions and costs?
**Answer:**  
The Streamlit application and standalone web dashboard load the serialized model (`best_model.joblib`) and feature schema (`feature_names.json`).
1. The user selects a representative historical scenario or modifies key inputs (current load, past lags, room temperatures, outdoor weather, hour of day).
2. The pipeline converts the input into the exact 49-feature schema matching training.
3. The model outputs predicted consumption in Watt-hours ($\hat{y}$ Wh) and kilowatt-hours ($\text{kWh} = \text{Wh} / 1000$).
4. The financial cost is dynamically calculated using the user's input tariff: $\text{Cost} = \text{kWh} \times \text{Tariff Rate}$.
5. A $95\%$ prediction interval ($\hat{y} \pm 1.96 \times 56.4\text{ Wh}$) and data-informed load-shifting advisory message are presented.
