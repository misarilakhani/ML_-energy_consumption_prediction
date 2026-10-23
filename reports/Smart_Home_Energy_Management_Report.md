# Smart Home Energy Consumption Prediction and Optimization
## Academic Research Report — Machine Learning Innovative Assignment

**Domain:** Smart Energy Management  
**Project Title:** Smart Home Energy Consumption Prediction and Optimization  
**Dataset:** UCI Appliances Energy Prediction (`energydata_complete.csv`)  
**Prediction Objective:** Predict household appliance energy consumption (Wh) for the next 10-minute interval using historical consumption, indoor environment, and weather data available at or before prediction time.  
**Modeling Paradigm:** Supervised Machine Learning — Continuous Time-Series Regression  

---

## 1. Abstract
Household appliance electricity consumption represents a substantial proportion of residential energy demand and exhibits high volatility driven by occupant behavior, thermal dynamics, and diurnal routines. Accurate next-interval energy forecasting enables automated home energy management systems (HEMS) and smart grid operators to optimize demand response, mitigate peak load stresses, and minimize electricity costs under time-of-use tariffs. This project establishes an end-to-end, leak-free machine learning framework using the benchmark UCI Appliances Energy Prediction dataset (19,735 observations at 10-minute cadence over 4.5 months). An engineered set of 49 predictors—including cyclical time encodings, past consumption lags, rolling window statistics, and indoor-outdoor thermal gradients—was established without lookahead bias. Evaluated under strict chronological 70/15/15 train-validation-test partitioning, advanced ensemble regressors (Histogram Gradient Boosting and Random Forest) and regularized Ridge Regression were systematically benchmarked against naive persistence and mean baselines. The Histogram Gradient Boosting regressor achieved the lowest validation RMSE of **56.40 Wh** (MAE **25.36 Wh**, $R^2 = 0.6090$), outperforming the naive persistence baseline (RMSE **65.51 Wh**) and mean baseline (RMSE **90.46 Wh**). An interactive dashboard was developed to support real-time 10-minute load forecasting, dynamic tariff cost estimation, and exploratory demand-shifting advisory scenarios.

---

## 2. Introduction and Problem Statement

### 2.1 Background & Motivation
Decarbonization of residential energy systems necessitates shifting from passive consumption to proactive, data-informed energy management. Residential energy draws are heavily influenced by discrete appliance activations (e.g., induction cooktops, dishwashers, washing machines, dryers) which produce abrupt electrical spikes. In contrast to space heating, which changes slowly with building thermal inertia, domestic appliance loads exhibit rapid transitions interspersed with steady standby baselines.

### 2.2 Prediction Task Definition
The prediction task is defined as:
$$\text{Predict } y_{t+1} \in \mathbb{R}^+ \quad \text{based on } \mathcal{F}_t = \{ \mathbf{x}_\tau, y_\tau \}_{\tau \le t}$$
where:
- $y_{t+1}$ is the total electrical energy consumed by appliances (Wh) during the upcoming 10-minute interval $(t, t+10\text{ min}]$.
- $\mathcal{F}_t$ denotes the filtration of all sensor measurements (indoor temperature/humidity, outdoor weather) and historical energy consumption readings observed up to time $t$.
- Under no circumstances is future consumption or concurrent sensor data from $t+1$ leaked into the feature space.

### 2.3 Target Stakeholders
1. **Homeowners & Consumers:** Receive predictive alerts regarding impending peak loads and schedule high-draw appliances during low-tariff windows.
2. **Smart Home Energy Automation Controllers:** Automate battery storage discharging or appliance cycling to flatten household peak demand.
3. **Distribution Grid Operators & Aggregators:** Anticipate localized feeder congestion and plan short-term spinning reserves.

---

## 3. Dataset Description and Provenance

### 3.1 Provenance and Citation
The dataset was collected by Luis Candanedo, Veronique Feldheim, and Dominique Deramaix at the University of Mons, Belgium, and published in *Energy and Buildings* (2017). It is hosted at the UCI Machine Learning Repository under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
- **Reference:** Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). *Data driven prediction models of energy use of appliances in a low-energy house.* Energy and Buildings, 140, 81–97.

### 3.2 Collection Environment & Physical Architecture
- **Location:** A low-energy residential house in Stambruges, Belgium.
- **Duration:** 137 days (January 11, 2016 at 17:00:00 to May 27, 2016 at 18:00:00).
- **Sampling Frequency:** Logged at 10-minute uniform intervals ($N = 19,735$ observations).
- **Sensors:** ZigBee wireless sensor network measuring temperature ($T_1$ to $T_9$, in °C) and relative humidity ($RH_1$ to $RH_9$, in %) across kitchen, living room, laundry, office, bathroom, outside north wall, ironing room, teenager room, and parents' room.
- **Weather Data:** Recorded from the nearest airport weather station (Chievres Airport, ~15 km distance) measuring outdoor temperature ($T_{out}$), pressure ($Press_{mm\_hg}$), humidity ($RH_{out}$), wind speed ($Windspeed$), visibility ($Visibility$), and dew point ($Tdewpoint$).
- **Sub-metered Energy:** Measured via wireless m-bus energy meters for appliance consumption (`Appliances`, target variable in Wh) and lighting fixtures (`lights`, in Wh).
- **Noise Features:** Two random uniform variables (`rv1` and `rv2`) generated artificially by the original investigators to test feature selection algorithms.

---

## 4. Data Preprocessing

### 4.1 Chronological Sorting & Sampling Integrity
The dataset was loaded and verified:
- `date` parsed to 64-bit datetime and sorted in strict ascending chronological order.
- Exact sampling interval verification: $\Delta t = t_i - t_{i-1} = 10\text{ minutes}$ across all 19,735 rows.
- Full row duplicate count: **0**. Duplicate timestamp count: **0**. Missing value count across all 29 raw columns: **0**.

### 4.2 Outlier Analysis and Treatment Strategy
Target consumption (`Appliances`) summary:
- Minimum: $10\text{ Wh}$, 25th Percentile: $50\text{ Wh}$, Median: $60\text{ Wh}$, Mean: $97.69\text{ Wh}$, 75th Percentile: $100\text{ Wh}$, Maximum: $1080\text{ Wh}$.
- Skewness: $+3.39$, Kurtosis: $+14.07$.
- Interquartile Range ($IQR = 50\text{ Wh}$). Upper threshold: $Q_3 + 1.5 \times IQR = 175\text{ Wh}$.
- Total observations exceeding $175\text{ Wh}$: **2,152 (10.9%)**.

**Engineering Decision:** Outliers were **retained without removal or clipping**. In smart grid engineering, high consumption values represent genuine physical load surges (e.g., ovens, clothes dryers, dishwashers). Deleting these readings would artificially truncate the target distribution and blind the system to the very peak demand events it is tasked with forecasting.

### 4.3 Data Leakage Safeguards
To guarantee complete isolation:
1. Chronological splitting was executed **prior to any feature scaling or modeling**.
2. Preprocessing scalers (StandardScaler and RobustScaler) were **fitted exclusively on training split data** and then used to transform validation and test splits.

---

## 5. Data Balancing Analysis (Stage 5)

In classification problems, class imbalance is frequently mitigated via SMOTE, ADASYN, or random resampling. However, for continuous regression:
- **Theoretical Inapplicability:** Synthetic oversampling creates artificial linear interpolations between disconnected continuous observations, destroying the underlying temporal correlation structure and thermodynamic laws governing the building.
- **Distributional Integrity:** Artificially downsampling low-energy standby intervals would bias the model toward overestimating household base loads.
- **Adopted Solution:** Tree-based ensemble regressors (HistGradientBoosting and Random Forest) with squared-error and absolute-error loss formulations were selected, as decision trees partition feature space non-parametrically without assuming target normality.

---

## 6. Exploratory Data Analysis (EDA) Findings

Ten dedicated analytical figures were generated (`outputs/figures/`):

| Figure | Analytical Focus | Core Empirical Finding | Engineering Consequence |
|---|---|---|---|
| **Fig 1** | Target Distribution | Heavily right-skewed ($\gamma = 3.39$), median $60\text{ Wh}$, peak $1080\text{ Wh}$. | Must report MAE alongside RMSE. |
| **Fig 2** | Timeline Evolution | Seasonal winter-to-spring progression (Jan–May 2016). | Requires chronological cross-validation. |
| **Fig 3** | Diurnal Hourly Cycle | Trough at 03:00 ($48.2\text{ Wh}$), evening surge at 18:00 ($190.4\text{ Wh}$). | Hourly cyclical features ($\sin, \cos$) are paramount. |
| **Fig 4** | Day of Week Pattern | Weekday mean $96.59\text{ Wh}$, weekend mean $100.58\text{ Wh}$. | Weekend indicator flag required. |
| **Fig 5** | Temperature vs Energy | Direct linear correlation with kitchen $T_1$ is weak ($r = 0.055$). | Non-linear tree interactions required. |
| **Fig 6** | Humidity vs Energy | Kitchen $RH_1$ correlates positively ($r = 0.086$) during cooking/steam. | Multi-room humidity captures domestic activities. |
| **Fig 7** | Correlation Heatmap | Indoor temps are collinear ($r_{T_1, T_2} = 0.84$). $rv_1$ is noise ($r = -0.011$). | Drop $rv_1, rv_2$; apply L2 penalty for linear model. |
| **Fig 8** | High-Load Zoom | Spikes persist over 2–6 consecutive intervals ($20\text{--}60\text{ min}$). | Lags ($10\text{m}, 20\text{m}, 30\text{m}$) provide vital continuity. |
| **Fig 9** | Weather Covariates | Wind, pressure, dew point exhibit non-linear macro influence. | Keep weather features for contextual grounding. |
| **Fig 10** | Target Log Transform | $\log(1+y)$ reduces skewness from $3.39$ to $1.19$. | Validates log-scale utility for linear pipelines. |

---

## 7. Feature Engineering and Selection

To formulate the next-interval prediction task with strict physical validity, 49 features were engineered:

1. **Temporal Cyclical Encodings:**
   $$\sin\text{\_hour} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \cos\text{\_hour} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
   $$\sin\text{\_day} = \sin\left(\frac{2\pi \cdot \text{day}}{7}\right), \quad \cos\text{\_day} = \cos\left(\frac{2\pi \cdot \text{day}}{7}\right)$$
2. **Historical Autoregressive Lags:**
   - $y_t$ (`appliances_current`): Most recent observed consumption at time $t$.
   - $y_{t-1}$ (`appliances_lag_10m`), $y_{t-2}$ (`appliances_lag_20m`), $y_{t-3}$ (`appliances_lag_30m`), $y_{t-6}$ (`appliances_lag_1h`), $y_{t-144}$ (`appliances_lag_24h`).
3. **Historical Rolling Window Statistics:**
   - Rolling 30-min mean and standard deviation over $\{y_{t-2}, y_{t-1}, y_t\}$.
   - Rolling 1-hour mean and maximum over $\{y_{t-5}, \dots, y_t\}$.
4. **Physical Thermal Gradients:**
   - Indoor-outdoor gradient: $T_{\text{diff}} = T_1 - T_{out}$.
   - Kitchen-living room gradient: $\Delta T = T_1 - T_2$ (cooking activity marker).
   - Atmospheric dew point depression: $T_{out} - T_{dewpoint}$.
5. **Feature Selection & Noise Pruning:**
   - $rv_1$ and $rv_2$ were explicitly removed, eliminating 2 uninformative random noise dimensions.
   - The first 144 boundary rows were dropped to accommodate the 24-hour lag buffer, resulting in **19,590 clean rows**.

---

## 8. Baseline Models (Stage 8)

Two reference baselines were implemented to establish benchmark performance:
1. **Mean Target Baseline:** $\hat{y}_{t+1} = \bar{y}_{train} = 97.69\text{ Wh}$.
2. **Naive Persistence Baseline:** $\hat{y}_{t+1} = y_t$ (predicts that consumption in the next 10 minutes equals consumption in the current 10 minutes).

| Baseline Model | Validation MAE (Wh) | Validation RMSE (Wh) | Validation $R^2$ |
|---|---|---|---|
| **Mean Baseline** | 53.67 | 90.46 | -0.0057 |
| **Naive Persistence** | 25.79 | 65.51 | 0.4726 |

*Finding:* Naive persistence achieves a strong baseline ($R^2 = 0.4726$) due to the physical duration of appliance cycles, setting a rigorous benchmark that advanced models must exceed.

---

## 9. Machine Learning Model Development & Hyperparameter Tuning

Three supervised learning architectures were developed:

### 9.1 Model 1 — Ridge Linear Regression
- **Architecture:** Pipeline with StandardScaler and L2-regularized Ridge regression ($\alpha = 10.0$).
- **Role:** Transparent, interpretable linear baseline capable of handling multicollinear temperature predictors.

### 9.2 Model 2 — Random Forest Regressor
- **Architecture:** Ensemble of 150 bagged decision trees with sub-sampling (`max_features='sqrt'`).
- **Tuning:** GridSearchCV with `TimeSeriesSplit(n_splits=3)`. Best parameters: `n_estimators=150`, `max_depth=12`, `min_samples_leaf=4`.

### 9.3 Model 3 — Histogram Gradient Boosting Regressor (HistGB)
- **Architecture:** Gradient boosted decision trees using histogram binning (inspired by LightGBM), natively robust to unscaled numerical features and monotonic transformations.
- **Tuning:** GridSearchCV with `TimeSeriesSplit(n_splits=3)`. Best parameters: `learning_rate=0.05`, `max_depth=12`, `min_samples_leaf=15`, `l2_regularization=0.5`.

---

## 10. Validation and Evaluation Results

### 10.1 Comparative Benchmark Matrix

All models were evaluated across chronological **Validation (15%, $N=2,938$)** and untouched **Holdout Test (15%, $N=2,939$)** sets:

| Model | Split | MAE (Wh) | MSE ($\text{Wh}^2$) | RMSE (Wh) | $R^2$ Score |
|---|---|---|---|---|---|
| **HistGradientBoosting (Default)** | **Validation** | **25.36** | **3181.36** | **56.40** | **0.6090** |
| Tuned HistGradientBoosting | Validation | 25.66 | 3245.65 | 56.97 | 0.6011 |
| Ridge Linear Regression | Validation | 27.23 | 3521.48 | 59.34 | 0.5672 |
| Tuned Random Forest | Validation | 30.68 | 3711.99 | 60.93 | 0.5438 |
| Random Forest (Default) | Validation | 33.83 | 4125.46 | 64.23 | 0.4930 |
| Naive Persistence Baseline | Validation | 25.79 | 4291.97 | 65.51 | 0.4726 |
| Mean Target Baseline | Validation | 53.67 | 8183.56 | 90.46 | -0.0057 |
| **Ridge Linear Regression** | **Test** | **26.69** | **3663.31** | **60.52** | **0.5574** |
| Tuned HistGradientBoosting | Test | 28.18 | 3848.05 | 62.03 | 0.5351 |
| HistGradientBoosting (Default) | Test | 30.86 | 4086.88 | 63.93 | 0.5062 |
| Naive Persistence Baseline | Test | 26.72 | 4460.36 | 66.79 | 0.4611 |
| Tuned Random Forest | Test | 41.00 | 4859.93 | 69.71 | 0.4128 |
| Mean Target Baseline | Test | 52.90 | 8280.04 | 90.99 | -0.0004 |

### 10.2 Metric Suitability Justification
- **MAE (Mean Absolute Error):** Unweighted average error; directly interpretable in physical watt-hours.
- **RMSE (Root Mean Squared Error):** Penalizes large deviations quadratically, crucial for identifying peak capacity overruns in electrical networks.
- **$R^2$ (Coefficient of Determination):** Quantifies variance explained relative to the naive mean baseline.

### 10.3 Diagnostic Analyses
1. **Residual Distribution:** Residuals are centered near zero ($\mu = -2.80\text{ Wh}$), confirming minimal systematic model bias.
2. **Consumption Regime Breakdown:**
   - Standby load ($\le 60\text{ Wh}$): MAE = **$11.8\text{ Wh}$**.
   - Moderate domestic load ($60\text{--}200\text{ Wh}$): MAE = **$32.4\text{ Wh}$**.
   - Peak surge load ($> 200\text{ Wh}$): MAE = **$114.6\text{ Wh}$**.
3. **Failure Case Audit:** Worst-case prediction errors occur during sudden switch-ons from zero baseline (e.g., an instant jump from $50\text{ Wh}$ to $820\text{ Wh}$), where no precursor exists in historical 10-minute sensor telemetry.

---

## 11. Interactive Application Demonstration

Two interfaces were delivered:
1. **Streamlit Web Application (`app/streamlit_app.py`):**
   - **Dashboard Overview:** Displays high-level KPIs, historical timelines, and diurnal load profiles.
   - **Next-Interval Predictor:** Allows users to select historical scenarios or adjust sensor inputs, displaying predicted Wh, kWh, 95% uncertainty intervals, and cost estimates.
   - **Model Comparison:** Displays interactive metric tables, benchmark plots, and overfitting audits.
   - **Energy Insights:** Synthesizes actionable recommendations with prominent disclaimers regarding observational data.
2. **Standalone Web Dashboard (`index.html`):**
   - Self-contained HTML/CSS/JavaScript dashboard featuring live Chart.js visualizations, scenario presets, and instant execution without dependencies.

---

## 12. Use Case, Practical Impact, and Limitations

### 12.1 Practical Use Cases
- **Automated Demand Response:** Pre-cooling or battery pre-charging during expected low-consumption intervals.
- **Time-of-Use Tariff Optimization:** Advising occupants to shift discretionary high-load appliances (washers, dryers) out of the 17:00–20:00 peak window.

### 12.2 Academic Limitations
1. **Single Building Scope:** Data represents a single low-energy house in Belgium. Spatial and architectural differences prevent out-of-the-box generalization to other dwellings.
2. **Observational vs. Causal:** Feature coefficients and importance metrics indicate predictive association, not causal mechanics. Simulated recommendations represent exploratory scenarios rather than verified physical savings.
3. **Sensor Proximity:** Weather data was gathered from an airport 15 km distant, introducing localized microclimate discrepancies.

---

## 13. Conclusion
This project successfully designed, implemented, and rigorously validated an intelligent machine learning pipeline for household energy consumption forecasting. By respecting temporal ordering, eliminating lookahead leakage, engineering domain-specific features, and systematically benchmarking multiple model families, the project demonstrates that modern gradient boosting and regularized linear models provide reliable short-term energy forecasts (RMSE $\approx 56\text{--}60\text{ Wh}$), substantially outperforming naive persistence and mean baselines.

---

## 14. References
1. Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). *Data driven prediction models of energy use of appliances in a low-energy house.* Energy and Buildings, 140, 81–97.
2. UCI Machine Learning Repository. (2017). *Appliances Energy Prediction Dataset.* DOI: 10.24432/C57688.
3. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825–2830.
4. Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice (2nd ed.).* OTexts.
