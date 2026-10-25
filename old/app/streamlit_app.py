"""
Smart Home Energy Management - Interactive Streamlit Dashboard
Domain: Smart Energy Management
Dataset: UCI Appliances Energy Prediction
Task: Predict next 10-minute appliance energy consumption (Wh) using historical data & environmental sensors.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))

from predict import EnergyPredictor, get_sample_inference_scenarios

# Page Configuration
st.set_page_config(
    page_title="Smart Home Energy Management",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        font-weight: 600;
        border-radius: 6px 6px 0px 0px;
        padding-left: 18px;
        padding-right: 18px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    return EnergyPredictor()


@st.cache_data
def load_metrics_and_summary():
    metrics_path = os.path.join(BASE_DIR, 'outputs', 'metrics', 'model_comparison.csv')
    eda_path = os.path.join(BASE_DIR, 'outputs', 'metrics', 'eda_summary.json')
    diag_path = os.path.join(BASE_DIR, 'outputs', 'metrics', 'evaluation_diagnostics_summary.json')
    test_pred_path = os.path.join(BASE_DIR, 'outputs', 'metrics', 'test_predictions.csv')

    df_metrics = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else None
    df_test_preds = pd.read_csv(test_pred_path) if os.path.exists(test_pred_path) else None
    
    with open(eda_path, 'r') as f:
        eda_summary = json.load(f) if os.path.exists(eda_path) else {}

    with open(diag_path, 'r') as f:
        diag_summary = json.load(f) if os.path.exists(diag_path) else {}

    return df_metrics, eda_summary, diag_summary, df_test_preds


def main():
    st.markdown('<div class="main-title">⚡ Smart Home Energy Management System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">10-Minute Appliance Energy Prediction, Demand Forecasting & Cost Advisory</div>', unsafe_allow_html=True)

    df_metrics, eda_summary, diag_summary, df_test_preds = load_metrics_and_summary()
    predictor = load_predictor()

    # Sidebar: Global Configurations
    with st.sidebar:
        st.header("⚙️ Simulation Settings")
        tariff_rate = st.number_input("Electricity Tariff ($ / kWh)", min_value=0.01, max_value=2.00, value=0.18, step=0.01)
        st.info("Electricity tariff is used to estimate momentary operating cost for upcoming 10-minute intervals.")
        
        st.markdown("---")
        st.markdown("### 📌 Academic Info")
        st.markdown("**Course:** Machine Learning Innovative Assignment  \n**Dataset:** UCI Appliances Energy Prediction  \n**Model:** HistGradientBoosting Regressor  \n**Sampling Cadence:** 10 Minutes")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard Overview", 
        "🔮 Real-Time Prediction", 
        "🏆 Model Comparison", 
        "💡 Energy Insights"
    ])

    # -------------------------------------------------------------
    # TAB 1: DASHBOARD OVERVIEW
    # -------------------------------------------------------------
    with tab1:
        st.subheader("Household Energy Profile & Key Statistics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Monitored Observations", "19,590", "100% Complete")
        with c2:
            st.metric("Mean Appliance Load", f"{eda_summary.get('fig1', {}).get('mean', 97.7):.1f} Wh", "Standby ~ 50 Wh")
        with c3:
            st.metric("Diurnal Peak Window", "18:00 (Evening)", f"{eda_summary.get('fig3', {}).get('peak_mean_wh', 190.4):.1f} Wh Avg")
        with c4:
            st.metric("High-Load Spike Frequency", f"{eda_summary.get('fig8', {}).get('high_load_ratio', 10.0):.1f}%", "> 200 Wh Threshold")

        st.markdown("---")
        st.subheader("Historical Timeline & Diurnal Patterns")
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.markdown("**Consumption Timeline (Jan - May 2016)**")
            fig2_path = os.path.join(BASE_DIR, 'outputs', 'figures', 'fig2_consumption_timeline.png')
            if os.path.exists(fig2_path):
                st.image(fig2_path, use_container_width=True)
            else:
                st.write("Figure 2 not found.")

        with col_chart2:
            st.markdown("**Diurnal Pattern by Hour of Day**")
            fig3_path = os.path.join(BASE_DIR, 'outputs', 'figures', 'fig3_hourly_consumption_pattern.png')
            if os.path.exists(fig3_path):
                st.image(fig3_path, use_container_width=True)
            else:
                st.write("Figure 3 not found.")

    # -------------------------------------------------------------
    # TAB 2: REAL-TIME PREDICTION
    # -------------------------------------------------------------
    with tab2:
        st.subheader("Predict Next 10-Minute Appliance Energy (Wh)")
        st.markdown(
            "Predict appliance energy consumption ($t+1$) based strictly on sensors and past consumption available at current interval ($t$). "
            "No future information is exposed to the model."
        )

        sample_scenarios = get_sample_inference_scenarios()
        selected_scenario_name = st.selectbox(
            "Select a Representative Test Scenario or Custom Defaults:",
            list(sample_scenarios.keys()) + ["Custom Manual Input"]
        )

        if selected_scenario_name in sample_scenarios:
            active_data = sample_scenarios[selected_scenario_name].copy()
            actual_wh = active_data.get('target_next_appliances', None)
        else:
            active_data = {feat: 20.0 for feat in predictor.feature_names}
            active_data['appliances_current'] = 60.0
            actual_wh = None

        col_input1, col_input2, col_input3 = st.columns(3)
        with col_input1:
            st.markdown("**1. Historical Consumption at Time $t$**")
            curr_wh = st.slider("Current Appliance Draw at $t$ (Wh)", 10, 1000, int(active_data.get('appliances_current', 60)))
            lag_10m = st.number_input("Draw 10m ago at $t-1$ (Wh)", value=float(active_data.get('appliances_lag_10m', curr_wh)))
            lag_1h = st.number_input("Draw 1h ago at $t-6$ (Wh)", value=float(active_data.get('appliances_lag_1h', curr_wh)))

        with col_input2:
            st.markdown("**2. Indoor Room Temperatures (°C)**")
            t_kitchen = st.slider("Kitchen Temperature T1 (°C)", 14.0, 30.0, float(active_data.get('T1', 21.5)), 0.1)
            t_living = st.slider("Living Room Temperature T2 (°C)", 14.0, 30.0, float(active_data.get('T2', 20.5)), 0.1)
            rh_kitchen = st.slider("Kitchen Relative Humidity RH_1 (%)", 20.0, 80.0, float(active_data.get('RH_1', 40.0)), 0.5)

        with col_input3:
            st.markdown("**3. Weather & Contextual Factors**")
            t_out = st.slider("Outdoor Temperature T_out (°C)", -5.0, 35.0, float(active_data.get('T_out', 12.0)), 0.1)
            windspeed = st.slider("Windspeed (m/s)", 0.0, 15.0, float(active_data.get('Windspeed', 4.0)), 0.1)
            hour_val = st.slider("Prediction Hour of Day", 0, 23, int(active_data.get('hour', 18)))

        # Update input payload with user inputs
        active_data['appliances_current'] = curr_wh
        active_data['appliances_lag_10m'] = lag_10m
        active_data['appliances_lag_1h'] = lag_1h
        active_data['T1'] = t_kitchen
        active_data['T2'] = t_living
        active_data['RH_1'] = rh_kitchen
        active_data['T_out'] = t_out
        active_data['Windspeed'] = windspeed
        active_data['hour'] = hour_val
        active_data['sin_hour'] = np.sin(2 * np.pi * hour_val / 24.0)
        active_data['cos_hour'] = np.cos(2 * np.pi * hour_val / 24.0)
        active_data['T_diff_indoor_outdoor'] = t_kitchen - t_out
        active_data['T_kitchen_living_diff'] = t_kitchen - t_living

        if st.button("🚀 Run Prediction for Next 10-Minute Interval", type="primary"):
            pred_result = predictor.predict(active_data, tariff_per_kwh=tariff_rate)
            
            st.markdown("---")
            st.subheader("🎯 Model Prediction Results")
            res1, res2, res3, res4 = st.columns(4)
            with res1:
                st.metric("Predicted Appliance Load", f"{pred_result['predicted_consumption_Wh']} Wh", 
                          f"{pred_result['predicted_consumption_kWh']} kWh")
            with res2:
                if actual_wh is not None:
                    error_val = abs(actual_wh - pred_result['predicted_consumption_Wh'])
                    st.metric("Actual Observed (Holdout)", f"{actual_wh:.1f} Wh", f"|Error|: {error_val:.1f} Wh")
                else:
                    st.metric("Actual Observed", "N/A (Custom)")
            with res3:
                st.metric("Estimated Cost (10-min)", f"${pred_result['estimated_interval_cost']:.4f}", f"@ ${tariff_rate}/kWh")
            with res4:
                st.metric("Consumption Regime", pred_result['consumption_regime'])

            st.markdown(f"**95% Confidence Interval:** `[{pred_result['prediction_interval_95_Wh'][0]} Wh, {pred_result['prediction_interval_95_Wh'][1]} Wh]`")
            st.info(f"💡 **Exploratory Advisory:** {pred_result['exploratory_suggestion']}")
            st.caption(f"⚠️ {pred_result['disclaimer']}")

    # -------------------------------------------------------------
    # TAB 3: MODEL COMPARISON
    # -------------------------------------------------------------
    with tab3:
        st.subheader("Experimental Model Evaluation & Benchmark Comparison")
        st.markdown(
            "All models evaluated across chronological Train (70%), Validation (15%), and untouched Test (15%) splits. "
            "Metrics reported in physical engineering units: MAE (Wh), RMSE (Wh), and R²."
        )

        if df_metrics is not None:
            c_test, c_val = st.columns(2)
            with c_test:
                st.markdown("**Holdout Test Set Results (Untouched Benchmark)**")
                st.dataframe(df_metrics[df_metrics['split'] == 'Test'].sort_values('RMSE_Wh'), use_container_width=True)

            with c_val:
                st.markdown("**Validation Set Results (Tuning & Selection)**")
                st.dataframe(df_metrics[df_metrics['split'] == 'Validation'].sort_values('RMSE_Wh'), use_container_width=True)

        st.markdown("---")
        st.subheader("Visual Performance Benchmark & Overfitting Audit")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_bar_path = os.path.join(BASE_DIR, 'outputs', 'figures', 'eval_fig4_model_comparison_bars.png')
            if os.path.exists(fig_bar_path):
                st.image(fig_bar_path, caption="RMSE, MAE, and R² Comparison across all candidate models", use_container_width=True)

        with col_c2:
            fig_gap_path = os.path.join(BASE_DIR, 'outputs', 'figures', 'eval_fig5_train_vs_val_overfitting.png')
            if os.path.exists(fig_gap_path):
                st.image(fig_gap_path, caption="Generalization Gap: Train vs Validation Performance", use_container_width=True)

    # -------------------------------------------------------------
    # TAB 4: ENERGY INSIGHTS
    # -------------------------------------------------------------
    with tab4:
        st.subheader("Data-Informed Energy Management Insights")
        st.markdown(
            "Actionable takeaways synthesized from exploratory analysis and model diagnostics. "
            "All recommendations are explicitly labeled exploratory scenarios based on observational data."
        )

        st.markdown("""
        ### 🔍 Core Findings:
        1. **Diurnal Concentration:** Appliance electricity consumption peaks sharply between **17:00 and 20:00 (190.4 Wh average)**, aligning with peak utility tariffs.
        2. **Short-Cycle Persistence:** Active appliance events (cooking, drying, washing) last between 20 to 60 minutes. Lagged measurements provide the strongest predictive signals.
        3. **Thermal Multicollinearity:** Internal room temperatures (T1, T2) correlate at **r = 0.88**, necessitating regularized linear models or tree ensembles.
        4. **Uninformative Features:** The artificial variables `rv1` and `rv2` correlate with energy consumption at **r = -0.01** and were removed to protect model generalizability.
        
        ### ⚙️ Exploratory Energy Management Scenarios:
        - **Off-Peak Load Shifting:** Delaying major washing/drying cycles from 18:00 to 22:00 or 03:00 (standby ~48 Wh) can reduce electricity expenditure under Time-of-Use (TOU) tariffs without affecting comfort.
        - **Standby Baselines:** Baseline standby consumption rarely drops below 40 Wh. Smart smart plugs that de-energize inactive entertainment devices can shave ~15-20% off idle load.
        
        ### ⚠️ Academic Limitations:
        - **Single Household:** The dataset records a single experimental low-energy dwelling in Stambruges, Belgium over 4.5 months. It does not generalize to different building envelopes or climates without retraining.
        - **Correlation vs Causation:** Machine learning models identify predictive correlations, not causal relationships. Changing an input in the simulator does not prove an identical real-world physical reduction.
        """)

        fig_regime_path = os.path.join(BASE_DIR, 'outputs', 'figures', 'eval_fig6_high_consumption_error_analysis.png')
        if os.path.exists(fig_regime_path):
            st.image(fig_regime_path, caption="Prediction Error Analysis Segmented by Consumption Regime", use_container_width=True)


if __name__ == '__main__':
    main()
