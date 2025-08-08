import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, chi2
from jinja2 import Template
import warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducible results
np.random.seed(42)

# Try import plotly — used for interactive CVaR plots & forecast
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# ===================== Page config =====================
st.set_page_config(
    page_title="Market Risk Analysis for Farmers", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E8B57;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        color: #4682B4;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4682B4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🌾 Market Risk Analysis for Farmers</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center; color: #666;">Predict Market Risks & Plan Better</h3>', unsafe_allow_html=True)

# Instructions
with st.expander("📋 How to Use This App (Click to Read)", expanded=False):
    st.markdown("""
    ### Simple Steps:
    1. **Upload your market data** (Excel or CSV file with dates and prices)
    2. **Choose your settings** on the left panel (confidence level, simulations)
    3. **Select date range** and price column to analyze
    4. **View results** - see risk predictions and policy recommendations
    
    ### What You Need:
    - File with dates and market prices
    - For advanced analysis: columns named 'Modal' (prices) and 'Arrivals' (quantity)
    
    ### What You Get:
    - Risk predictions for your market
    - Policy recommendations to reduce losses
    - Future risk forecasts to plan ahead
    """)

# ===================== Sidebar Controls =====================
st.sidebar.markdown('<h2 style="color: #2E8B57;">⚙️ Settings</h2>', unsafe_allow_html=True)

# Confidence level with drag slider
st.sidebar.markdown("**📊 Risk Level (How sure you want to be)**")
confidence_level = st.sidebar.slider(
    "Higher = More Conservative", 
    min_value=0.90, 
    max_value=0.999, 
    value=0.95, 
    step=0.01, 
    format="%.1f%%",
    help="95% means you're 95% sure losses won't exceed the predicted amount"
)
z_score = norm.ppf(1 - confidence_level)

# Monte Carlo simulations
st.sidebar.markdown("**🎲 Number of Simulations**")
num_simulations = st.sidebar.number_input(
    "More simulations = More accurate (but slower)", 
    min_value=1000, 
    max_value=100000, 
    value=10000, 
    step=1000,
    help="Default 10,000 is good for most cases"
)

# Rolling window for backtesting
st.sidebar.markdown("**📅 Analysis Window**")
rolling_window = st.sidebar.number_input(
    "Weeks of data to use for each prediction", 
    min_value=10, 
    max_value=260, 
    value=52, 
    step=1,
    help="52 weeks = 1 year of data for each prediction"
)

# Forecast horizon
st.sidebar.markdown("**🔮 Forecast Period**")
forecast_horizon = st.sidebar.number_input(
    "How many weeks ahead to predict", 
    min_value=4, 
    max_value=52, 
    value=26, 
    step=1,
    help="26 weeks = 6 months ahead prediction"
)

# Options
st.sidebar.markdown("---")
st.sidebar.markdown("**📈 Display Options**")
run_forecast = st.sidebar.checkbox("Show future risk forecast", value=True)
interactive_plots = st.sidebar.checkbox("Use interactive charts", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Tip:** Start with default settings, then adjust as needed")

# ===================== File upload =====================
st.markdown('<h2 class="sub-header">📁 Upload Your Market Data</h2>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose your file (CSV, Excel)", 
    type=["csv", "xls", "xlsx"],
    help="File should contain dates and market prices"
)

if not PLOTLY_AVAILABLE and interactive_plots:
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Note:</strong> Interactive charts not available. Install plotly for better visualizations.
    </div>
    """, unsafe_allow_html=True)

if uploaded_file:
    try:
        # Read file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Date handling
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.dropna(subset=['Date'], inplace=True)
            df.set_index('Date', inplace=True)
            df = df.sort_index()
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df.dropna(subset=['date'], inplace=True)
            df.set_index('date', inplace=True)
            df = df.sort_index()

        st.markdown(f"""
        <div class="success-box">
            <strong>✅ Data loaded successfully!</strong><br>
            📅 <strong>Date Range:</strong> {df.index.min().date()} to {df.index.max().date()}<br>
            📊 <strong>Total Records:</strong> {len(df)}
        </div>
        """, unsafe_allow_html=True)

        # Column selection
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if not numeric_cols:
            st.error("❌ No numeric price columns found in your data!")
            st.stop()

        col1, col2 = st.columns([1, 1])
        with col1:
            selected_col = st.selectbox(
                "🎯 Select Market Price Column", 
                options=['All Columns'] + numeric_cols,
                help="Choose which price to analyze for risk"
            )

        with col2:
            date_range = st.date_input(
                "📅 Select Date Range", 
                [df.index.min(), df.index.max()],
                help="Choose period to analyze"
            )

        # Filter data by date range
        if len(date_range) == 2:
            df = df.loc[(df.index >= pd.to_datetime(date_range[0])) & 
                       (df.index <= pd.to_datetime(date_range[1]))]

        analysis_cols = numeric_cols if selected_col == 'All Columns' else [selected_col]
        results, briefs = [], []

        # ===================== VaR Analysis =====================
        st.markdown('<h2 class="sub-header">📊 Risk Analysis Results</h2>', unsafe_allow_html=True)

        for col in analysis_cols:
            series = df[col].copy()
            series.replace(0, np.nan, inplace=True)
            series.dropna(inplace=True)

            if len(series) < 30:
                st.warning(f"⚠️ Not enough data for {col}. Need at least 30 data points.")
                continue

            # Calculate returns
            log_returns = np.log(series / series.shift(1)).dropna()
            mu, sigma = log_returns.mean(), log_returns.std()

            # Set seed for reproducibility
            np.random.seed(42)

            # Historical VaR
            hist_var = np.percentile(log_returns, (1 - confidence_level) * 100)
            
            # Parametric VaR
            param_var = mu + z_score * sigma
            
            # Monte Carlo VaR & CVaR
            sim_returns = np.random.normal(mu, sigma, size=int(num_simulations))
            mc_var = np.percentile(sim_returns, (1 - confidence_level) * 100)
            mc_cvar = sim_returns[sim_returns <= mc_var].mean()

            results.append({
                "Market": col,
                "Historical VaR (%)": round((np.exp(hist_var) - 1) * 100, 3),
                "Parametric VaR (%)": round((np.exp(param_var) - 1) * 100, 3),
                "Monte Carlo VaR (%)": round((np.exp(mc_var) - 1) * 100, 3),
                "Monte Carlo CVaR (%)": round((np.exp(mc_cvar) - 1) * 100, 3),
            })

            # Enhanced Policy Brief
            risk_level = "HIGH" if abs(mc_cvar) > 0.1 else "MODERATE" if abs(mc_cvar) > 0.05 else "LOW"
            brief_template = Template("""
            ### 🎯 Market: {{ market }}
            
            **📈 Risk Summary ({{ conf_level }}% Confidence):**
            - **Historical Risk:** {{ hist }}% weekly loss
            - **Statistical Model Risk:** {{ param }}% weekly loss  
            - **Simulation Risk (VaR):** {{ mc }}% weekly loss
            - **Worst Case Risk (CVaR):** {{ cvar }}% weekly loss
            
            **🚨 Risk Level: {{ risk_level }}**
            
            **💡 What This Means:**
            {% if risk_level == "HIGH" %}
            - **High Risk Market:** Expect significant price swings
            - **Action Needed:** Consider reducing inventory or hedging
            - **Weekly Loss:** Could lose {{ cvar }}% in worst weeks
            {% elif risk_level == "MODERATE" %}
            - **Moderate Risk:** Some price volatility expected
            - **Manageable Risk:** Standard precautions recommended  
            - **Weekly Loss:** Could lose {{ cvar }}% in bad weeks
            {% else %}
            - **Low Risk Market:** Relatively stable prices
            - **Good News:** Lower chance of major losses
            - **Weekly Loss:** Maximum {{ cvar }}% in worst weeks
            {% endif %}
            
            **🛡️ Recommended Actions:**
            1. **Buffer Stock:** Keep {{ buffer_days }} days extra inventory
            2. **Price Contracts:** Lock prices for {{ contract_pct }}% of your crop  
            3. **Emergency Fund:** Save {{ emergency_fund }}% of revenue for bad times
            4. **Market Timing:** {% if risk_level == "HIGH" %}Sell quickly when prices are good{% else %}Normal selling strategy is fine{% endif %}
            
            **📊 Model Reliability:**
            {% if abs(hist_var - mc_var) < 0.02 %}
            ✅ Models agree well - predictions are reliable
            {% else %}
            ⚠️ Models show different results - use multiple strategies  
            {% endif %}
            """)
            
            # Calculate recommendations based on risk level
            buffer_days = 15 if risk_level == "HIGH" else 10 if risk_level == "MODERATE" else 7
            contract_pct = 40 if risk_level == "HIGH" else 25 if risk_level == "MODERATE" else 15
            emergency_fund = 15 if risk_level == "HIGH" else 10 if risk_level == "MODERATE" else 5
            
            brief = brief_template.render(
                market=col,
                conf_level=int(confidence_level*100),
                hist=round((np.exp(hist_var) - 1) * 100, 2),
                param=round((np.exp(param_var) - 1) * 100, 2),
                mc=round((np.exp(mc_var) - 1) * 100, 2),
                cvar=round((np.exp(mc_cvar) - 1) * 100, 2),
                risk_level=risk_level,
                buffer_days=buffer_days,
                contract_pct=contract_pct,
                emergency_fund=emergency_fund
            )
            briefs.append((col, brief))

            # Interactive Monte Carlo Plot
            if interactive_plots and PLOTLY_AVAILABLE:
                fig = go.Figure()
                
                # Histogram
                fig.add_trace(go.Histogram(
                    x=sim_returns,
                    nbinsx=50,
                    name='Simulated Returns',
                    opacity=0.7,
                    marker_color='lightblue'
                ))
                
                # VaR line
                fig.add_vline(
                    x=mc_var, 
                    line=dict(color='red', dash='dash', width=2),
                    annotation_text=f"VaR: {((np.exp(mc_var) - 1) * 100):.2f}%",
                    annotation_position="top"
                )
                
                # CVaR line
                fig.add_vline(
                    x=mc_cvar,
                    line=dict(color='purple', dash='dash', width=2), 
                    annotation_text=f"CVaR: {((np.exp(mc_cvar) - 1) * 100):.2f}%",
                    annotation_position="bottom"
                )
                
                fig.update_layout(
                    title=f"Risk Distribution for {col} ({int(num_simulations)} simulations)",
                    xaxis_title="Weekly Returns",
                    yaxis_title="Frequency",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                # Fallback matplotlib plot
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.histplot(sim_returns, bins=50, kde=True, ax=ax, color='lightblue')
                ax.axvline(mc_var, color='red', linestyle='--', linewidth=2, label=f"VaR {int(confidence_level*100)}%")
                ax.axvline(mc_cvar, color='purple', linestyle='--', linewidth=2, label=f"CVaR {int(confidence_level*100)}%")
                ax.set_title(f"Risk Distribution: {col}")
                ax.set_xlabel("Weekly Returns")
                ax.legend()
                st.pyplot(fig)

        # Display results table
        if results:
            st.markdown("### 📋 Risk Comparison Table")
            results_df = pd.DataFrame(results).set_index("Market")
            st.dataframe(results_df, use_container_width=True)

        # Display policy briefs
        if briefs:
            st.markdown("### 📝 Detailed Risk Analysis & Recommendations")
            for market, brief in briefs:
                with st.expander(f"📊 Analysis for {market} - Click to View Details"):
                    st.markdown(brief)

        # ===================== Advanced Analysis ===================== 
        if 'Modal' in df.columns and 'Arrivals' in df.columns:
            st.markdown('<h2 class="sub-header">🔍 Advanced Risk Testing</h2>', unsafe_allow_html=True)

            ds = df[['Modal', 'Arrivals']].copy()
            ds['Log_Returns'] = np.log(ds['Modal'] / ds['Modal'].shift(1))
            ds['Log_Arrivals'] = np.log(ds['Arrivals'].replace(0, np.nan)).fillna(method='bfill')
            ds.dropna(inplace=True)

            if len(ds) > rolling_window:
                # Set seed for reproducible backtesting
                np.random.seed(42)
                
                backtest_results = []
                for i in range(rolling_window, len(ds)):
                    train = ds.iloc[i - rolling_window:i]
                    test = ds.iloc[i]
                    
                    X = sm.add_constant(train['Log_Arrivals'])
                    y = train['Log_Returns']
                    model = sm.OLS(y, X).fit()

                    sim_arr = np.random.normal(train['Log_Arrivals'].mean(), train['Log_Arrivals'].std(), int(num_simulations))
                    sim_mu = model.params[0] + model.params[1] * sim_arr
                    sim_ret = np.random.normal(sim_mu, model.resid.std(), size=int(num_simulations))

                    mc_var_bt = np.percentile(sim_ret, (1 - confidence_level) * 100)
                    mc_cvar_bt = sim_ret[sim_ret <= mc_var_bt].mean()

                    actual_ret = test['Log_Returns']
                    backtest_results.append({
                        'Date': ds.index[i],
                        'Actual_Return': actual_ret,
                        'MC_VaR': mc_var_bt,
                        'MC_CVaR': mc_cvar_bt,
                        'Breach_VaR': actual_ret < mc_var_bt,
                        'Breach_CVaR': actual_ret < mc_cvar_bt
                    })

                bt_df = pd.DataFrame(backtest_results).set_index('Date')

                # Interactive CVaR backtest plot
                if interactive_plots and PLOTLY_AVAILABLE:
                    st.markdown("### 📊 Interactive CVaR Model Testing")
                    
                    fig = go.Figure()
                    
                    # Actual returns
                    fig.add_trace(go.Scatter(
                        x=bt_df.index,
                        y=((np.exp(bt_df['Actual_Return']) - 1) * 100),
                        mode='lines',
                        name='Actual Weekly Returns (%)',
                        line=dict(color='blue', width=1),
                        hovertemplate='Date: %{x}<br>Actual Return: %{y:.2f}%<extra></extra>'
                    ))
                    
                    # CVaR predictions  
                    fig.add_trace(go.Scatter(
                        x=bt_df.index,
                        y=((np.exp(bt_df['MC_CVaR']) - 1) * 100),
                        mode='lines',
                        name='Predicted CVaR (%)',
                        line=dict(color='red', width=2, dash='dash'),
                        hovertemplate='Date: %{x}<br>CVaR Prediction: %{y:.2f}%<extra></extra>'
                    ))
                    
                    # Fill area for CVaR
                    fig.add_trace(go.Scatter(
                        x=bt_df.index,
                        y=((np.exp(bt_df['MC_CVaR']) - 1) * 100),
                        fill='tonexty',
                        fillcolor='rgba(255,0,0,0.1)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Risk Zone',
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    fig.update_layout(
                        title="Model Performance: Predicted vs Actual Risk",
                        xaxis_title="Date",
                        yaxis_title="Weekly Return (%)",
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    # Fallback matplotlib
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(bt_df.index, (np.exp(bt_df['Actual_Return']) - 1) * 100, label="Actual Returns", alpha=0.7)
                    ax.plot(bt_df.index, (np.exp(bt_df['MC_CVaR']) - 1) * 100, label="CVaR Predictions", color='red', linestyle='--')
                    ax.fill_between(bt_df.index, (np.exp(bt_df['MC_CVaR']) - 1) * 100, -50, alpha=0.1, color='red')
                    ax.set_title("Model Performance Testing")
                    ax.set_ylabel("Weekly Return (%)")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

                # Model accuracy metrics
                breach_rate = bt_df['Breach_CVaR'].mean() * 100
                expected_rate = (1 - confidence_level) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Model Accuracy", f"{100-abs(breach_rate-expected_rate):.1f}%")
                with col2:
                    st.metric("Actual Breach Rate", f"{breach_rate:.1f}%")
                with col3:
                    st.metric("Expected Breach Rate", f"{expected_rate:.1f}%")

                if abs(breach_rate - expected_rate) < 2:
                    st.markdown("""
                    <div class="success-box">
                        <strong>✅ Model is Working Well!</strong><br>
                        The predictions closely match actual results. You can trust these risk estimates.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="warning-box">
                        <strong>⚠️ Model Needs Attention</strong><br>
                        Predictions don't match reality well. Use results with caution and consider multiple strategies.
                    </div>
                    """, unsafe_allow_html=True)

        # ===================== Future Risk Forecast =====================
        if run_forecast and 'Modal' in df.columns and 'Arrivals' in df.columns:
            st.markdown('<h2 class="sub-header">🔮 Future Risk Forecast</h2>', unsafe_allow_html=True)
            
            # Use latest data for forecasting with fixed seed
            np.random.seed(42)
            train_latest = ds.iloc[-rolling_window:]
            model_latest = sm.OLS(train_latest["Log_Returns"], sm.add_constant(train_latest["Log_Arrivals"])).fit()
            
            mu_arr = train_latest["Log_Arrivals"].mean()
            sigma_arr = train_latest["Log_Arrivals"].std()
            resid_sigma = model_latest.resid.std()

            forecast_weeks = list(range(1, int(forecast_horizon) + 1))
            forecast_cvars = []
            forecast_vars = []

            for h in forecast_weeks:
                # Use deterministic approach for more stable forecasts
                np.random.seed(42 + h)  # Different seed for each week but reproducible
                
                sim_arrivals_future = np.random.normal(mu_arr, sigma_arr, size=int(num_simulations))
                sim_mu_future = model_latest.params[0] + model_latest.params[1] * sim_arrivals_future
                sim_returns_future = np.random.normal(sim_mu_future, resid_sigma, size=int(num_simulations))
                
                var_future = np.percentile(sim_returns_future, (1 - confidence_level) * 100)
                cvar_future = sim_returns_future[sim_returns_future <= var_future].mean()
                
                forecast_cvars.append((h, (np.exp(cvar_future) - 1) * 100))
                forecast_vars.append((h, (np.exp(var_future) - 1) * 100))

            fc_df = pd.DataFrame(forecast_cvars, columns=["Week_Ahead", "Predicted_CVaR (%)"]).set_index("Week_Ahead")
            var_df = pd.DataFrame(forecast_vars, columns=["Week_Ahead", "Predicted_VaR (%)"]).set_index("Week_Ahead")
            forecast_df = fc_df.join(var_df)

            # Find worst periods
            worst_idx = fc_df["Predicted_CVaR (%)"].idxmin()
            worst_val = fc_df["Predicted_CVaR (%)"].min()
            
            # Risk assessment
            avg_cvar = fc_df["Predicted_CVaR (%)"].mean()
            risk_trend = "INCREASING" if fc_df["Predicted_CVaR (%)"].iloc[-5:].mean() < fc_df["Predicted_CVaR (%)"].iloc[:5].mean() else "STABLE"

            st.markdown(f"""
            <div class="info-box">
                <h4>🎯 Forecast Summary</h4>
                <p><strong>Worst Risk Period:</strong> Week {worst_idx} ahead with {worst_val:.2f}% potential weekly loss</p>
                <p><strong>Average Risk:</strong> {avg_cvar:.2f}% weekly loss over next {forecast_horizon} weeks</p>
                <p><strong>Risk Trend:</strong> {risk_trend} risk levels expected</p>
            </div>
            """, unsafe_allow_html=True)

            # Interactive forecast plot
            if interactive_plots and PLOTLY_AVAILABLE:
                fig = go.Figure()
                
                # CVaR forecast
                fig.add_trace(go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df["Predicted_CVaR (%)"],
                    mode='lines+markers',
                    name='CVaR Forecast',
                    line=dict(color='red', width=3),
                    marker=dict(size=8),
                    hovertemplate='Week %{x}: %{y:.2f}% potential loss<extra></extra>'
                ))
                
                # VaR forecast
                fig.add_trace(go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df["Predicted_VaR (%)"],
                    mode='lines+markers',
                    name='VaR Forecast',
                    line=dict(color='orange', width=2),
                    marker=dict(size=6),
                    hovertemplate='Week %{x}: %{y:.2f}% normal risk<extra></extra>'
                ))
                
                # Highlight worst week
                fig.add_vline(
                    x=worst_idx,
                    line=dict(color='red', dash='dash', width=2),
                    annotation_text=f"Highest Risk<br>Week {worst_idx}",
                    annotation_position="top"
                )
                
                # Risk zones
                fig.add_hrect(
                    y0=-15, y1=-10, fillcolor="red", opacity=0.2, 
                    annotation_text="HIGH RISK", annotation_position="inside top left"
                )
                fig.add_hrect(
                    y0=-10, y1=-5, fillcolor="orange", opacity=0.2,
                    annotation_text="MODERATE RISK", annotation_position="inside top left"  
                )
                fig.add_hrect(
                    y0=-5, y1=0, fillcolor="green", opacity=0.2,
                    annotation_text="LOW RISK", annotation_position="inside top left"
                )
                
                fig.update_layout(
                    title=f"Risk Forecast for Next {forecast_horizon} Weeks",
                    xaxis_title="Weeks Ahead",
                    yaxis_title="Potential Weekly Loss (%)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # Fallback matplotlib
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(forecast_df.index, forecast_df["Predicted_CVaR (%)"], marker='o', linewidth=2, label='CVaR Forecast')
                ax.plot(forecast_df.index, forecast_df["Predicted_VaR (%)"], marker='s', linewidth=2, label='VaR Forecast')
                ax.axvline(worst_idx, color='red', linestyle='--', alpha=0.7, label=f'Worst Risk (Week {worst_idx})')
                ax.fill_between(forecast_df.index, forecast_df["Predicted_CVaR (%)"], -20, alpha=0.1, color='red')
                ax.set_xlabel("Weeks Ahead")
                ax.set_ylabel("Potential Weekly Loss (%)")
                ax.set_title(f"Risk Forecast for Next {forecast_horizon} Weeks")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

            # Recommendations based on forecast
            if worst_val < -10:
                recommendation_level = "HIGH ALERT"
                rec_color = "error"
            elif worst_val < -5:
                recommendation_level = "CAUTION"
                rec_color = "warning"  
            else:
                recommendation_level = "NORMAL"
                rec_color = "success"
                
                            if rec_color == "error":
                    st.markdown(f"""
                    <div style="background-color: #ffe6e6; padding: 15px; border-radius: 10px; border-left: 5px solid #dc3545;">
                        <h4>🚨 {recommendation_level}</h4>
                        <p><strong>High risk detected around week {worst_idx}!</strong></p>
                        <ul>
                            <li>Consider selling inventory before week {worst_idx-2 if worst_idx > 2 else 1}</li>
                            <li>Arrange emergency funding of at least 15% of typical revenue</li>
                            <li>Contact buyers early to secure better prices</li>
                            <li>Consider forward contracts to lock in current prices</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                elif rec_color == "warning":
                    st.markdown(f"""
                    <div style="background-color: #fff3cd; padding: 15px; border-radius: 10px; border-left: 5px solid #ffc107;">
                        <h4>⚠️ {recommendation_level}</h4>
                        <p><strong>Moderate risk around week {worst_idx}</strong></p>
                        <ul>
                            <li>Monitor market closely around week {worst_idx}</li>
                            <li>Keep 7-10 days buffer stock</li>
                            <li>Have backup buyers ready</li>
                            <li>Save 10% of revenue for emergencies</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.markdown(f"""
                    <div style="background-color: #d4edda; padding: 15px; border-radius: 10px; border-left: 5px solid #28a745;">
                        <h4>✅ {recommendation_level}</h4>
                        <p><strong>Low risk expected - normal operations recommended</strong></p>
                        <ul>
                            <li>Standard farming and selling practices should work well</li>
                            <li>Good time to plan expansion or investments</li>
                            <li>Maintain normal inventory levels</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="info-box">
                <h4>📊 For Advanced Analysis</h4>
                <p>To get backtesting and forecasting features, your data should include:</p>
                <ul>
                    <li><strong>'Modal'</strong> column (market prices)</li>
                    <li><strong>'Arrivals'</strong> column (quantities/volumes)</li>
                </ul>
                <p>Current analysis shows basic risk metrics which are still very useful for planning!</p>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("💡 Make sure your file has a 'Date' column and at least one numeric price column")

else:
    # Sample data section for users without data
    st.markdown("""
    <div class="info-box">
        <h4>🎯 Don't have data? Try our sample!</h4>
        <p>Click the button below to see how the app works with sample tomato market data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📊 Load Sample Data", type="primary"):
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='W')
        
        # Create realistic market data with seasonality
        base_price = 50
        seasonal_factor = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 52)
        trend = 0.1 * np.arange(len(dates))
        noise = np.random.normal(0, 5, len(dates))
        
        modal_prices = base_price + seasonal_factor + trend + noise
        modal_prices = np.maximum(modal_prices, 20)  # Floor price
        
        # Create arrivals data (correlated with prices - higher arrivals, lower prices)
        base_arrivals = 1000
        arrivals_seasonal = -200 * np.sin(2 * np.pi * np.arange(len(dates)) / 52 + np.pi/4)
        arrivals_noise = np.random.normal(0, 100, len(dates))
        arrivals = base_arrivals + arrivals_seasonal + arrivals_noise
        arrivals = np.maximum(arrivals, 100)  # Floor arrivals
        
        sample_df = pd.DataFrame({
            'Date': dates,
            'Modal': modal_prices,
            'Arrivals': arrivals,
            'Wholesale': modal_prices * 0.8,
            'Retail': modal_prices * 1.3
        })
        
        st.success("✅ Sample tomato market data loaded!")
        st.dataframe(sample_df.head(10))
        st.download_button(
            "📥 Download Sample Data",
            sample_df.to_csv(index=False),
            "sample_market_data.csv",
            "text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; margin-top: 30px;">
    <h4 style="color: #2E8B57; margin-bottom: 10px;">🌾 Built for Farmers, By Economists</h4>
    <p style="color: #666; margin-bottom: 15px;">
        This app helps you understand market risks and make better farming decisions.<br>
        For questions or support, contact us below.
    </p>
    <div style="display: flex; justify-content: center; gap: 30px; flex-wrap: wrap;">
        <div>
            <strong>📧 Email:</strong> sumanecon.uas@outlook.com
        </div>
        <div>
            <strong>👨‍💼 Developer:</strong> Suman L
        </div>
        <div>
            <strong>🎓 University:</strong> UAS Bengaluru
        </div>
    </div>
    <p style="font-size: 12px; color: #999; margin-top: 15px;">
        Version 2.0 | Enhanced for Professional Use | Updated {pd.Timestamp.now().strftime('%B %Y')}
    </p>
</div>
""", unsafe_allow_html=True)
