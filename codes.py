# app.py
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
np.random.seed(103)

# Try import plotly for interactive charts
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
st.markdown(
    """
<style>
    .main-header { text-align: center; color: #2E8B57; font-size: 2.5rem; font-weight: bold; margin-bottom: 1rem; }
    .sub-header { color: #4682B4; font-size: 1.5rem; font-weight: bold; margin: 1rem 0; }
    .info-box { background-color: #f0f8ff; padding: 1rem; border-radius: 10px; border-left: 5px solid #4682B4; margin: 1rem 0; }
    .warning-box { background-color: #fff3cd; padding: 1rem; border-radius: 10px; border-left: 5px solid #ffc107; margin: 1rem 0; }
    .success-box { background-color: #d4edda; padding: 1rem; border-radius: 10px; border-left: 5px solid #28a745; margin: 1rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown('<h1 class="main-header">🌾 Market Risk Analysis for Farmers</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center; color: #666;">Predict Market Risks & Plan Better</h3>', unsafe_allow_html=True)

# Instructions
with st.expander("📋 How to Use This App (Click to Read)", expanded=False):
    st.markdown(
        """
### Simple Steps:
1. **Upload your market data** (Excel or CSV file with dates and prices)
2. **Choose your settings** on the left panel (confidence level, simulations)
3. **Select date range** and price column to analyze
4. **View results** - see risk predictions and policy recommendations

### What You Need:
- File with dates and market prices
- For advanced analysis: columns named 'Modal' (prices) and 'Arrivals' (quantity)

### What You Get:
- Risk predictions for your market (Historical VaR, Parametric VaR, Monte Carlo VaR & CVaR)
- Policy recommendations to reduce losses
- Future risk forecasts to plan ahead
"""
    )

# ===================== Sidebar Controls =====================
st.sidebar.markdown('<h2 style="color: #2E8B57;">⚙️ Settings</h2>', unsafe_allow_html=True)

# Confidence level as integer percent (clear UX)
st.sidebar.markdown("**📊 Confidence Level (VaR)**")
conf_percent = st.sidebar.slider(
    "Choose confidence level (percent)",
    min_value=90,
    max_value=99,
    value=95,
    step=1,
    help="95 means you're 95% sure losses won't exceed the predicted amount",
)
confidence_level = conf_percent / 100.0
z_score = norm.ppf(1 - confidence_level)

# Monte Carlo simulations
st.sidebar.markdown("**🎲 Number of Simulations**")
num_simulations = st.sidebar.number_input(
    "More simulations = more accurate (but slower)",
    min_value=1000,
    max_value=100000,
    value=10000,
    step=1000,
    help="Default 10,000 is good for most cases",
)

# Rolling window for backtesting
st.sidebar.markdown("**📅 Analysis Window (weeks)**")
rolling_window = st.sidebar.number_input(
    "Weeks of data to use for each prediction",
    min_value=10,
    max_value=520,
    value=52,
    step=1,
    help="52 weeks = 1 year of data for each prediction",
)

# Forecast horizon
st.sidebar.markdown("**🔮 Forecast Horizon (weeks)**")
forecast_horizon = st.sidebar.number_input(
    "How many weeks ahead to predict",
    min_value=4,
    max_value=104,
    value=26,
    step=1,
    help="26 weeks = 6 months ahead prediction",
)

# Options
st.sidebar.markdown("---")
st.sidebar.markdown("**📈 Display Options**")
run_forecast = st.sidebar.checkbox("Show future risk forecast", value=True)
interactive_plots = st.sidebar.checkbox("Use interactive charts (plotly)", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Tip:** Start with default settings, then adjust as needed")

# ===================== File upload =====================
st.markdown('<h2 class="sub-header">📁 Upload Your Market Data</h2>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose your file (CSV, Excel)", type=["csv", "xls", "xlsx"], help="File should contain dates and market prices")

if not PLOTLY_AVAILABLE and interactive_plots:
    st.markdown(
        """
    <div class="warning-box">
        <strong>⚠️ Note:</strong> Interactive charts not available. Install plotly for better visualizations.
    </div>
    """,
        unsafe_allow_html=True,
    )

def read_file(uploaded):
    if uploaded.name.endswith(".csv"):
        return pd.read_csv(uploaded)
    else:
        return pd.read_excel(uploaded)

def prepare_index_dates(df):
    # try common date column names
    for c in ["Date", "date", "DATE"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            df = df.dropna(subset=[c]).set_index(c).sort_index()
            return df
    # if index looks like dates already
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    return df

if uploaded_file:
    try:
        df = read_file(uploaded_file)
        df = prepare_index_dates(df)

        st.markdown(
            f"""
        <div class="success-box">
            <strong>✅ Data loaded successfully!</strong><br>
            📅 <strong>Date Range:</strong> {df.index.min().date() if not df.empty else 'N/A'} to {df.index.max().date() if not df.empty else 'N/A'}<br>
            📊 <strong>Total Records:</strong> {len(df)}
        </div>
        """,
            unsafe_allow_html=True,
        )

        # numeric columns
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            st.error("❌ No numeric price columns found in your data!")
            st.stop()

        col1, col2 = st.columns([1, 1])
        with col1:
            selected_col = st.selectbox("🎯 Select Market Price Column", options=["All Columns"] + numeric_cols, help="Choose which price to analyze for risk")
        with col2:
            # date range picker
            default_start = df.index.min().date()
            default_end = df.index.max().date()
            date_range = st.date_input("📅 Select Date Range", [default_start, default_end])

        # filter by date range
        if len(date_range) == 2:
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            df = df.loc[(df.index >= start) & (df.index <= end)]

        analysis_cols = numeric_cols if selected_col == "All Columns" else [selected_col]
        results = []
        briefs = []

        st.markdown('<h2 class="sub-header">📊 Risk Analysis Results</h2>', unsafe_allow_html=True)

        for col in analysis_cols:
            series = df[col].copy()
            series.replace(0, np.nan, inplace=True)
            series = series.dropna()
            if len(series) < 30:
                st.warning(f"⚠️ Not enough data for {col}. Need at least 30 data points.")
                continue

            # log returns
            log_returns = np.log(series / series.shift(1)).dropna()
            mu, sigma = log_returns.mean(), log_returns.std()

            # Historical VaR: empirical quantile of returns
            hist_var = np.percentile(log_returns, (1 - confidence_level) * 100)

            # Parametric VaR: assume normal with mu, sigma
            param_var = mu + z_score * sigma

            # Monte Carlo simulations and CVaR
            np.random.seed(44)
            sim_returns = np.random.normal(mu, sigma, size=int(num_simulations))
            mc_var = np.percentile(sim_returns, (1 - confidence_level) * 100)
            mc_cvar = sim_returns[sim_returns <= mc_var].mean()

            # convert to percent (weekly loss % approx)
            def to_pct(x):
                return (np.exp(x) - 1) * 100

            results.append(
                {
                    "Market": col,
                    "Historical VaR (%)": round(to_pct(hist_var), 3),
                    "Parametric VaR (%)": round(to_pct(param_var), 3),
                    "Monte Carlo VaR (%)": round(to_pct(mc_var), 3),
                    "Monte Carlo CVaR (%)": round(to_pct(mc_cvar), 3),
                }
            )

            # Policy brief (simple)
            risk_level = "HIGH" if abs(mc_cvar) > 0.1 else "MODERATE" if abs(mc_cvar) > 0.05 else "LOW"
            brief_template = Template(
                """
### 🎯 Market: {{ market }}

**📈 Risk Summary ({{ conf_level }}% Confidence):**
- **Historical Risk:** {{ hist }}% weekly loss
- **Statistical Model Risk:** {{ param }}% weekly loss
- **Montecarlo Risk (VaR):** {{ mc }}% weekly loss
- **ConditionalVaR Risk (CVaR):** {{ cvar }}% weekly loss

**🚨 Risk Level: {{ risk_level }}**

**💡 What This Means:**
{% if risk_level == "HIGH" %}
- **High Risk Market:** Expect significant price swings
- **Action Needed:** Consider reducing inventory or hedging
{% elif risk_level == "MODERATE" %}
- **Moderate Risk:** Some price volatility expected
- **Manageable Risk:** Standard precautions recommended
{% else %}
- **Low Risk Market:** Relatively stable prices
- **Good News:** Lower chance of major losses
{% endif %}

**🛡️ Recommended Actions:**
1. Buffer stock: keep {{ buffer_days }} days extra inventory
2. Price contracts: lock prices for {{ contract_pct }}% of your crop
3. Emergency fund: save {{ emergency_fund }}% of revenue
"""
            )
            buffer_days = 15 if risk_level == "HIGH" else 10 if risk_level == "MODERATE" else 7
            contract_pct = 40 if risk_level == "HIGH" else 25 if risk_level == "MODERATE" else 15
            emergency_fund = 15 if risk_level == "HIGH" else 10 if risk_level == "MODERATE" else 5

            brief = brief_template.render(
                market=col,
                conf_level=int(confidence_level * 100),
                hist=round(to_pct(hist_var), 2),
                param=round(to_pct(param_var), 2),
                mc=round(to_pct(mc_var), 2),
                cvar=round(to_pct(mc_cvar), 2),
                risk_level=risk_level,
                buffer_days=buffer_days,
                contract_pct=contract_pct,
                emergency_fund=emergency_fund,
            )
            briefs.append((col, brief))

            # ====== Plot: combined view (Historical distribution + parametric + MC VaR + CVaR) ======
            if interactive_plots and PLOTLY_AVAILABLE:
                fig = go.Figure()

                # Histogram of historical log-returns
                fig.add_trace(
                    go.Histogram(
                        x=log_returns,
                        nbinsx=50,
                        name="Historical (Empirical)",
                        opacity=0.6,
                    )
                )

                # Parametric normal density (scaled) as a line with method legend label
                xs = np.linspace(log_returns.min(), log_returns.max(), 200)
                pdf_vals = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
                # scale pdf to histogram height approximately
                scale_factor = len(log_returns) * (log_returns.max() - log_returns.min()) / 50
                fig.add_trace(go.Scatter(x=xs, y=pdf_vals * scale_factor, mode="lines", name="Parametric (Normal)"))

                # Vertical lines for VaRs & CVaR with meaningful legend names
                fig.add_vline(x=hist_var, line=dict(color="black", dash="dash"))
                fig.add_vline(x=param_var, line=dict(color="green", dash="dash"))
                fig.add_vline(x=mc_var, line=dict(color="red", dash="dash"))
                fig.add_vline(x=mc_cvar, line=dict(color="purple", dash="dash"), annotation_text=f"Conditional CVaR {to_pct(mc_cvar):.2f}%", annotation_position="top left")
                              
                # To show methods in legend explicitly, add invisible scatter traces with method names (helps legend clarity)
                fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="black"), name="Historical VaR"))
                fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="green"), name="Parametric VaR"))
                fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="red"), name="Monte Carlo VaR"))
                fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="purple"), name="Conditional VaR"))

                fig.update_layout(
                    title=f"Risk Distribution & Methods - {col}",
                    xaxis_title="Log Returns",
                    yaxis_title="Frequency / scaled density",
                    bargap=0.1,
                    height=450,
                    showlegend=True,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Matplotlib fallback: label methods clearly in legend
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.histplot(log_returns, bins=50, kde=False, ax=ax)
                xs = np.linspace(log_returns.min(), log_returns.max(), 200)
                pdf_vals = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
                scale_factor = len(log_returns) * (log_returns.max() - log_returns.min()) / 50
                ax.plot(xs, pdf_vals * scale_factor, linestyle="--", label="Parametric (Normal)")
                # VaR lines
                ax.axvline(hist_var, color="black", linestyle="--", label=f"Historical VaR {to_pct(hist_var):.2f}%")
                ax.axvline(param_var, color="green", linestyle="--", label=f"Parametric VaR {to_pct(param_var):.2f}%")
                ax.axvline(mc_var, color="red", linestyle="--", label=f"Monte Carlo VaR {to_pct(mc_var):.2f}%")
                ax.axvline(mc_cvar, color="purple", linestyle="--", label=f"Monte Carlo CVaR {to_pct(mc_cvar):.2f}%")
                ax.set_title(f"Risk Distribution & Methods - {col}")
                ax.set_xlabel("Log Returns")
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

        # ===================== Advanced Analysis (requires Modal & Arrivals) =====================
        if "Modal" in df.columns and "Arrivals" in df.columns:
            st.markdown('<h2 class="sub-header">🔍 Advanced Risk Testing</h2>', unsafe_allow_html=True)

            ds = df[["Modal", "Arrivals"]].copy()
            ds["Log_Returns"] = np.log(ds["Modal"] / ds["Modal"].shift(1))
            ds["Log_Arrivals"] = np.log(ds["Arrivals"].replace(0, np.nan)).fillna(method="bfill")
            ds = ds.dropna()

            if len(ds) > rolling_window:
                np.random.seed(42)
                backtest_results = []
                for i in range(int(rolling_window), len(ds)):
                    train = ds.iloc[i - int(rolling_window) : i]
                    test = ds.iloc[i]

                    X = sm.add_constant(train["Log_Arrivals"])
                    y = train["Log_Returns"]
                    model = sm.OLS(y, X).fit()

                    # simulate arrivals and returns
                    sim_arr = np.random.normal(train["Log_Arrivals"].mean(), train["Log_Arrivals"].std(), int(num_simulations))
                    sim_mu = model.params[0] + model.params[1] * sim_arr
                    sim_ret = np.random.normal(sim_mu, model.resid.std(), size=int(num_simulations))

                    mc_var_bt = np.percentile(sim_ret, (1 - confidence_level) * 100)
                    mc_cvar_bt = sim_ret[sim_ret <= mc_var_bt].mean()

                    actual_ret = test["Log_Returns"]
                    backtest_results.append(
                        {
                            "Date": ds.index[i],
                            "Actual_Return": actual_ret,
                            "MC_VaR": mc_var_bt,
                            "MC_CVaR": mc_cvar_bt,
                            "Breach_VaR": actual_ret < mc_var_bt,
                            "Breach_CVaR": actual_ret < mc_cvar_bt,
                        }
                    )

                bt_df = pd.DataFrame(backtest_results).set_index("Date")

                # -------------------------
                # Kupiec (Unconditional Coverage) test for VaR breaches
                # x = observed failures, n = sample size, p = expected failure prob = 1 - confidence_level
                # LR_uc = -2 * ( log((1-p)^(n-x) p^x) - log((1-p_hat)^(n-x) p_hat^x) )
                # p-value = 1 - chi2.cdf(LR_uc, df=1)
                # -------------------------
                n = len(bt_df)
                x = bt_df["Breach_VaR"].sum()
                p = 1 - confidence_level
                p_hat = x / n if n > 0 else 0.0

                # Guard against zero/one probabilities
                def safe_log(x):
                    return np.log(x) if x > 0 else -1e10

                if n > 0:
                    # Log-likelihood under H0 (p)
                    ll0 = (n - x) * safe_log(1 - p) + x * safe_log(p)
                    # Log-likelihood under H1 (p_hat)
                    # If p_hat is 0 or 1, adjust slightly to avoid -inf
                    ph = min(max(p_hat, 1e-10), 1 - 1e-10)
                    ll1 = (n - x) * safe_log(1 - ph) + x * safe_log(ph)

                    LR_uc = -2 * (ll0 - ll1)
                    kupiec_pvalue = 1 - chi2.cdf(LR_uc, df=1)
                else:
                    LR_uc = np.nan
                    kupiec_pvalue = np.nan

                # Interactive CVaR / Monte Carlo backtest plot with breaches highlighted
                if interactive_plots and PLOTLY_AVAILABLE:
                    st.markdown("### 📊 Interactive CVaR / Monte Carlo Backtest (with Kupiec Test)")
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=bt_df.index,
                            y=((np.exp(bt_df["Actual_Return"]) - 1) * 100),
                            mode="lines+markers",
                            name="Actual Weekly Returns (%)",
                            hovertemplate="Date: %{x}<br>Actual Return: %{y:.2f}%<extra></extra>",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=bt_df.index,
                            y=((np.exp(bt_df["MC_VaR"]) - 1) * 100),
                            mode="lines",
                            name="Monte Carlo VaR (%)",
                            line=dict(color="red", dash="dash"),
                            hovertemplate="Date: %{x}<br>VaR: %{y:.2f}%<extra></extra>",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=bt_df.index,
                            y=((np.exp(bt_df["MC_CVaR"]) - 1) * 100),
                            mode="lines",
                            name="Monte Carlo CVaR (%)",
                            line=dict(color="purple", dash="dot"),
                            hovertemplate="Date: %{x}<br>CVaR: %{y:.2f}%<extra></extra>",
                        )
                    )
                    
                    fig.update_layout(
                        title=f"Backtest: Actual vs Monte Carlo VaR/CVaR ({col}) — Kupiec LR={LR_uc:.2f} p={kupiec_pvalue:.3f}",
                        xaxis_title="Date",
                        yaxis_title="Weekly Return (%)",
                        height=450,
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Matplotlib fallback plot
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(bt_df.index, (np.exp(bt_df["Actual_Return"]) - 1) * 100, label="Actual Returns", alpha=0.7)
                    ax.plot(bt_df.index, (np.exp(bt_df["MC_VaR"]) - 1) * 100, label="Monte Carlo VaR", color="red", linestyle="--")
                    ax.plot(bt_df.index, (np.exp(bt_df["MC_CVaR"]) - 1) * 100, label="Monte Carlo CVaR", color="purple", linestyle="-.")
                    breaches = bt_df[bt_df["Breach_VaR"]]
                    if not breaches.empty:
                        ax.scatter(breaches.index, (np.exp(breaches["Actual_Return"]) - 1) * 100, label="VaR Breach")
                    ax.set_title(f"Backtest: Actual vs Monte Carlo VaR/CVaR ({col}) — Kupiec LR={LR_uc:.2f} p={kupiec_pvalue:.3f}")
                    ax.set_ylabel("Weekly Return (%)")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

                # Show Kupiec test summary and other metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Observed Breach Rate", f"{(x / n * 100) if n>0 else np.nan:.2f}%")
                with col2:
                    st.metric("Expected Breach Rate", f"{(p * 100):.2f}%")
                with col3:
                    st.metric("Kupiec p-value", f"{kupiec_pvalue:.3f}" if not np.isnan(kupiec_pvalue) else "N/A")

                if not np.isnan(kupiec_pvalue):
                    if kupiec_pvalue < 0.05:
                        st.markdown(
                            """
                        <div class="warning-box">
                            <strong>⚠️ Kupiec Test:</strong> Reject the null: Actual breaches ~ Expected breaches==> "Model is Bad").
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            """
                        <div class="success-box">
                            <strong>✅ Kupiec Test:</strong> Accept the null: Actual breaches ~ Expected breaches==> "Model is best").
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                # Model accuracy metrics (calibration)
                breach_rate = bt_df["Breach_CVaR"].mean() * 100
                expected_rate = (1 - confidence_level) * 100

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Model Calibration (approx)", f"{100 - abs(breach_rate - expected_rate):.1f}%")
                with col2:
                    st.metric("Actual Breach Rate (CVaR)", f"{breach_rate:.1f}%")
                with col3:
                    st.metric("Expected Breach Rate (CVaR)", f"{expected_rate:.1f}%")

                if abs(breach_rate - expected_rate) < 2:
                    st.markdown(
                        """
                    <div class="success-box">
                        <strong>✅ Model is Working Well!</strong><br>
                        The breach rate closely matches actual expectation.
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
    
        # ===================== Future Risk Forecast =====================
        if run_forecast and "Modal" in df.columns and "Arrivals" in df.columns:
            st.markdown('<h2 class="sub-header">🔮 Future Risk Forecast</h2>', unsafe_allow_html=True)

            ds = df[["Modal", "Arrivals"]].copy()
            ds["Log_Returns"] = np.log(ds["Modal"] / ds["Modal"].shift(1))
            ds["Log_Arrivals"] = np.log(ds["Arrivals"].replace(0, np.nan)).fillna(method="bfill")
            ds = ds.dropna()

            train_latest = ds.iloc[-int(rolling_window) :]
            model_latest = sm.OLS(train_latest["Log_Returns"], sm.add_constant(train_latest["Log_Arrivals"])).fit()

            mu_arr = train_latest["Log_Arrivals"].mean()
            sigma_arr = train_latest["Log_Arrivals"].std()
            resid_sigma = model_latest.resid.std()

            forecast_weeks = list(range(1, int(forecast_horizon) + 1))
            forecast_cvars = []
            forecast_vars = []

            for h in forecast_weeks:
                np.random.seed(42 + h)
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

            worst_idx = fc_df["Predicted_CVaR (%)"].idxmin()
            worst_val = fc_df["Predicted_CVaR (%)"].min()
            avg_cvar = fc_df["Predicted_CVaR (%)"].mean()
            # trend: compare first 5 vs last 5 (guard for short horizons)
            first_slice = fc_df["Predicted_CVaR (%)"].iloc[: min(5, len(fc_df))]
            last_slice = fc_df["Predicted_CVaR (%)"].iloc[max(0, len(fc_df) - 5) :]
            risk_trend = "INCREASING" if last_slice.mean() < first_slice.mean() else "STABLE"

            st.markdown(
                f"""
            <div class="info-box">
                <h4>🎯 Forecast Summary</h4>
                <p><strong>Worst Risk Period:</strong> Week {worst_idx} ahead with {worst_val:.2f}% potential weekly loss</p>
                <p><strong>Average Risk:</strong> {avg_cvar:.2f}% weekly loss over next {forecast_horizon} weeks</p>
                <p><strong>Risk Trend:</strong> {risk_trend} risk levels expected</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Plot forecast
            if interactive_plots and PLOTLY_AVAILABLE:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=forecast_df.index,
                        y=forecast_df["Predicted_CVaR (%)"],
                        mode="lines+markers",
                        name="CVaR Forecast",
                        line=dict(color="red", width=3),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=forecast_df.index,
                        y=forecast_df["Predicted_VaR (%)"],
                        mode="lines+markers",
                        name="VaR Forecast",
                        line=dict(color="orange", width=2),
                    )
                )
                fig.add_vline(x=worst_idx, line=dict(color="red", dash="dash", width=2), annotation_text=f"Highest Risk Week {worst_idx}", annotation_position="top")
                fig.update_layout(title=f"Risk Forecast for Next {forecast_horizon} Weeks", xaxis_title="Weeks Ahead", yaxis_title="Potential Weekly Loss (%)", height=450)
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(forecast_df.index, forecast_df["Predicted_CVaR (%)"], marker="o", linewidth=2, label="CVaR Forecast")
                ax.plot(forecast_df.index, forecast_df["Predicted_VaR (%)"], marker="s", linewidth=2, label="VaR Forecast")
                ax.axvline(worst_idx, color="red", linestyle="--", alpha=0.7, label=f"Worst Risk (Week {worst_idx})")
                ax.fill_between(forecast_df.index, forecast_df["Predicted_CVaR (%)"], -20, alpha=0.1, color="red")
                ax.set_xlabel("Weeks Ahead")
                ax.set_ylabel("Potential Weekly Loss (%)")
                ax.set_title(f"Risk Forecast for Next {forecast_horizon} Weeks")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

            # Recommendations based on forecast worst_val
            if worst_val < -10:
                recommendation_level = "HIGH ALERT"
                rec_type = "error"
            elif worst_val < -5:
                recommendation_level = "CAUTION"
                rec_type = "warning"
            else:
                recommendation_level = "NORMAL"
                rec_type = "success"

            if rec_type == "error":
                st.markdown(
                    f"""
                    <div style="background-color: #ffe6e6; padding: 15px; border-radius: 10px; border-left: 5px solid #dc3545;">
                        <h4>🚨 {recommendation_level}</h4>
                        <p><strong>High risk detected around week {worst_idx}!</strong></p>
                        <ul>
                            <li>-------------Farmers-----------------</li>
                            <li>Stagger Planting – Adjust sowing windows to avoid synchronized harvests and market gluts</li>
                            <li>FPO-Led Direct Marketing – Leverage collective sales to institutional buyers for better price realization</li>
                            <li>Farm-Gate Value Addition – Process surplus into storable products to stabilize income during price crashes</li>
                            <li>Arrange emergency funding of at least 15% of typical revenue</li>
                            <li>Plan early for alternate crops to secure better prices and income</li>
                            <li>-------------Policy makers-----------------</li>
                            <li>Price Stabilization Fund – Deploy targeted procurement/interventions during glut periods</li>
                            <li>Processing & Value Addition Incentives – Support decentralized tomato processing units to absorb surplus</li>
                            <li>Market Intelligence Systems – Strengthen real-time price forecasting and advisory dissemination to farmers</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif rec_type == "warning":
                st.markdown(
                    f"""
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
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="background-color: #d4edda; padding: 15px; border-radius: 10px; border-left: 5px solid #28a745;">
                        <h4>✅ {recommendation_level}</h4>
                        <p><strong>Low risk expected - normal operations recommended</strong></p>
                        <ul>
                            <li>Standard farming and selling practices should work well</li>
                            <li>Good time to plan expansion or investments</li>
                            <li>Maintain normal inventory levels</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:
            st.markdown(
                """
            <div class="info-box">
                <h4>📊 For Advanced Analysis</h4>
                <p>To get backtesting and forecasting features, your data should include:</p>
                <ul>
                    <li><strong>'Modal'</strong> column (market prices)</li>
                    <li><strong>'Arrivals'</strong> column (quantities/volumes)</li>
                </ul>
                <p>Basic per-column risk metrics are still computed above.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("💡 Make sure your file has a 'Date' column (or datetime index) and at least one numeric price column")

else:
    # Sample data section
    st.markdown(
        """
    <div class="info-box">
        <h4>🎯 Don't have data? Try our sample!</h4>
        <p>Click the button below to see how the app works with sample tomato market data.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if st.button("📊 Load Sample Data", type="primary"):
        np.random.seed(42)
        dates = pd.date_range(start="2022-01-01", end="2024-12-31", freq="W")
        base_price = 50
        seasonal_factor = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 52)
        trend = 0.1 * np.arange(len(dates))
        noise = np.random.normal(0, 5, len(dates))
        modal_prices = base_price + seasonal_factor + trend + noise
        modal_prices = np.maximum(modal_prices, 20)
        base_arrivals = 1000
        arrivals_seasonal = -200 * np.sin(2 * np.pi * np.arange(len(dates)) / 52 + np.pi / 4)
        arrivals_noise = np.random.normal(0, 100, len(dates))
        arrivals = base_arrivals + arrivals_seasonal + arrivals_noise
        arrivals = np.maximum(arrivals, 100)
        sample_df = pd.DataFrame(
            {
                "Date": dates,
                "Modal": modal_prices,
                "Arrivals": arrivals,
                "Wholesale": modal_prices * 0.8,
                "Retail": modal_prices * 1.3,
            }
        )
        st.success("✅ Sample tomato market data loaded!")
        st.dataframe(sample_df.head(10))
        st.download_button("📥 Download Sample Data", sample_df.to_csv(index=False), "sample_market_data.csv", "text/csv")

# Footer
st.markdown("---")
st.markdown(
    f"""
<div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; margin-top: 30px;">
    <h4 style="color: #2E8B57; margin-bottom: 10px;">🌾 Built for Farmers, By Economists</h4>
    <p style="color: #666; margin-bottom: 15px;">
        This app helps you understand market risks and make better farming decisions.<br>
        For questions or support, contact us below.
    </p>
    <div style="display: flex; justify-content: center; gap: 30px; flex-wrap: wrap;">
        <div><strong>📧 Email:</strong> sumanecon.uas@outlook.com</div>
        <div><strong>👨‍💼 Developer:</strong> Suman L</div>
        <div><strong>🎓 University:</strong> UAS Bengaluru</div>
    </div>
    <p style="font-size: 12px; color: #999; margin-top: 15px;">
        Version 2.1 | Enhanced for Professional Use | Updated {pd.Timestamp.now().strftime("%B %Y")}
    </p>
</div>
""",
    unsafe_allow_html=True,
)
