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

# Try import plotly — used for interactive CVaR plots & forecast
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# ===================== Page config =====================
st.set_page_config(page_title="Value_at_risk Analysis App [Suman_econ_UAS(B)]", layout="wide")
st.title("📉 Value at Risk Analysis App — Suman_econ_UAS(B)")

st.markdown("""
### 📌 Instructions Before Upload:
- Date column will be automatically converted to datetime and set as index.
- File format: `.csv`, `.xls`, or `.xlsx`.
- Data should contain at least one price series (numeric).
- Missing values will be handled (imputed/dropped) automatically.
""")

# ===================== Sidebar Controls =====================
st.sidebar.header("Analysis Controls")

# Confidence level slider (user requested)
confidence_level = st.sidebar.slider("Confidence Level (VaR / CVaR)", min_value=0.90, max_value=0.999, value=0.95, step=0.01, format="%.3f")
z_score = norm.ppf(1 - confidence_level)

# Monte Carlo simulations input
num_simulations = st.sidebar.number_input("Monte Carlo simulations", min_value=1000, max_value=200000, value=10000, step=1000)

# Rolling window for backtesting
rolling_window = st.sidebar.number_input("Rolling window for backtesting (weeks)", min_value=10, max_value=260, value=52, step=1)

# Forecast horizon for CVaR forecast
forecast_horizon = st.sidebar.number_input("Forecast horizon (weeks) for CVaR forecast", min_value=4, max_value=52, value=26, step=1)

# Quick toggles
st.sidebar.markdown("---")
run_forecast = st.sidebar.checkbox("Run near-future CVaR forecast", value=True)
interactive_cvar = st.sidebar.checkbox("Show interactive CVaR backtest plot", value=True)
st.sidebar.markdown("---")
st.sidebar.write("Default sim count: 10000. Increase if you need smoother tails (slower).")

# ===================== File upload =====================
uploaded_file = st.file_uploader("Upload your crop market data (weekly suggested)", type=["csv", "xls", "xlsx"])

if not PLOTLY_AVAILABLE:
    st.warning("Plotly is not installed. Interactive CVaR plots will not be available. Install with `pip install plotly`.")

if uploaded_file:
    # Read
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

    st.markdown(f"**Date Range:** {df.index.min().date()} to {df.index.max().date()}")

    # Column selection
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    selected_col = st.selectbox("Select Market Price Column for VaR analysis", options=['All'] + numeric_cols)

    # Date range selection (main area)
    date_range = st.date_input("Select date range to analyze", [df.index.min(), df.index.max()])
    df = df.loc[(df.index >= pd.to_datetime(date_range[0])) & (df.index <= pd.to_datetime(date_range[1]))]

    analysis_cols = numeric_cols if selected_col == 'All' else [selected_col]

    # Keep earlier defaults/variables
    window = rolling_window
    results, briefs = [], []

    # === VaR Analysis (as before) ===
    for col in analysis_cols:
        series = df[col].copy()
        series.replace(0, np.nan, inplace=True)
        series.dropna(inplace=True)

        if len(series) < 30:
            st.warning(f"Not enough data points in {col} to compute risk models.")
            continue

        log_returns = np.log(series / series.shift(1)).dropna()
        mu, sigma = log_returns.mean(), log_returns.std()

        # Historical VaR
        hist_var = np.percentile(log_returns, (1 - confidence_level) * 100)
        # Parametric VaR
        param_var = mu + z_score * sigma
        # Monte Carlo VaR & CVaR (global simulation)
        np.random.seed(42) 
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

        # Automated policy brief with interpretation improvements
        brief_template = Template("""
        **Market:** {{ market }}

        **Summary (95% default unless changed):**
        - Historical VaR: {{ hist }}%
        - Parametric VaR: {{ param }}%
        - Monte Carlo VaR: {{ mc }}%
        - Monte Carlo CVaR (Expected Shortfall): {{ cvar }}%

        **Interpretation & Policy Implications**
        - The Monte Carlo CVaR indicates the expected extreme weekly loss of about **{{ cvar }}%** in the worst 5% of cases.
        - If CVaR is materially larger (more negative) than Historical VaR, this suggests **fat tails** or more frequent extreme losses — recommend stress testing inventories and contingency purchase plans.
        - If Parametric VaR is less conservative than Historical/MC measures, parametric Gaussian assumptions may understate risk — prefer MC-based or historical measures for policy framing.
        - Suggested policy actions: maintain buffer stocks, forward contracting for 10-20% of expected arrivals during peak season, and prepare rapid procurement funding to stabilize supply.
        """)
        brief = brief_template.render(
            market=col,
            hist=round((np.exp(hist_var) - 1) * 100, 3),
            param=round((np.exp(param_var) - 1) * 100, 3),
            mc=round((np.exp(mc_var) - 1) * 100, 3),
            cvar=round((np.exp(mc_cvar) - 1) * 100, 3)
        )
        briefs.append((col, brief))

        # Monte Carlo histogram (keeps existing Matplotlib/Seaborn display)
        fig, ax = plt.subplots()
        sns.histplot(sim_returns, bins=50, kde=True, ax=ax)
        ax.axvline(mc_var, color='red', linestyle='--', label=f"VaR {int(confidence_level*100)}%")
        ax.axvline(mc_cvar, color='purple', linestyle='--', label=f"CVaR {int(confidence_level*100)}%")
        ax.set_title(f"Monte Carlo Simulated Returns: {col}")
        ax.set_xlabel("Log Returns")
        ax.legend()
        st.pyplot(fig)

    # Display results and policy briefs
    if results:
        st.subheader("📋 Model Comparison Table")
        st.dataframe(pd.DataFrame(results).set_index("Market"))

    if briefs:
        st.subheader("📝 Automated Policy Briefs (Interpretations)")
        for market, text in briefs:
            with st.expander(f"Policy Brief for {market}"):
                st.markdown(text)

    # === Monte Carlo VaR Backtesting (existing flow) ===
    if 'Modal' in df.columns and 'Arrivals' in df.columns:
        st.subheader("🔎 Monte Carlo VaR Backtesting (Log Returns ~ Log Arrivals)")

        ds = df[['Modal', 'Arrivals']].copy()
        ds['Log_Returns'] = np.log(ds['Modal'] / ds['Modal'].shift(1))
        # For arrivals, avoid zeros and take log; fill small gaps sensibly
        ds['Log_Arrivals'] = np.log(ds['Arrivals'].replace(0, np.nan)).fillna(method='bfill')
        ds.dropna(inplace=True)

        backtest_results = []
        # run rolling backtest for VaR
        for i in range(window, len(ds)):
            train = ds.iloc[i - window:i]
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
        bt_df['Breach_VaR'] = bt_df['Breach_VaR'].astype(int)
        # Show line chart (matplotlib/st.line_chart kept)
        st.line_chart(bt_df[['Actual_Return', 'MC_VaR']])
        st.markdown(f"**Observed Breach Rate (VaR):** {bt_df['Breach_VaR'].mean() * 100:.2f}%")

        # Kupiec Test as before
        F = bt_df['Breach_VaR'].sum()
        T = len(bt_df)
        p = 1 - confidence_level
        observed_p = bt_df['Breach_VaR'].mean()
        # small numerical guards
        observed_p = np.clip(observed_p, 1e-8, 1 - 1e-8)
        p = np.clip(p, 1e-8, 1 - 1e-8)
        stat = -2 * np.log(((1 - p) ** (T - F) * p ** F) / ((1 - observed_p) ** (T - F) * observed_p ** F))
        p_val = 1 - chi2.cdf(stat, df=1)
        st.markdown(f"**Kupiec's POF Test Statistic:** {stat:.4f}")
        st.markdown(f"**P-value:** {p_val:.4f}")
        if p_val < 0.05:
            st.error("❌ VaR model does not fit well (reject null hypothesis).")
        else:
            st.success("✅ VaR model fits well (fail to reject null hypothesis).")

        # === Optimized CVaR Backtesting (rolling) ===
        st.subheader("📊 Optimized CVaR Backtesting (rolling)")

        actual_returns = []
        predicted_cvar_pct = []

        for i in range(rolling_window, len(ds)):
            train = ds.iloc[i - rolling_window:i]
            test = ds.iloc[i]

            model = sm.OLS(train["Log_Returns"], sm.add_constant(train["Log_Arrivals"])).fit()
            resid_sigma = model.resid.std()

            sim_arrivals = np.random.normal(train["Log_Arrivals"].mean(), train["Log_Arrivals"].std(), int(num_simulations))
            sim_mu = model.params[0] + model.params[1] * sim_arrivals
            sim_returns = np.random.normal(sim_mu, resid_sigma, int(num_simulations))

            var = np.percentile(sim_returns, (1 - confidence_level) * 100)
            cvar = sim_returns[sim_returns <= var].mean()
            cvar_pct = (np.exp(cvar) - 1) * 100

            actual_log_ret = test["Log_Returns"]
            actual_ret_pct = (np.exp(actual_log_ret) - 1) * 100
            actual_returns.append(actual_ret_pct)
            predicted_cvar_pct.append(cvar_pct)

        results_df = pd.DataFrame({
            "Date": ds.index[rolling_window:],
            "Actual_Return (%)": actual_returns,
            "Predicted_CVaR (%)": predicted_cvar_pct
        })
        results_df = results_df.set_index("Date")
        results_df["Breached_CVaR"] = results_df["Actual_Return (%)"] < results_df["Predicted_CVaR (%)"]

        breach_rate = results_df["Breached_CVaR"].mean() * 100
        st.markdown(f"**Observed Breach Rate (CVaR):** {breach_rate:.2f}%")
        st.markdown("**Expected (Nominal) Breach Rate:** {:.2f}%".format((1 - confidence_level) * 100))
        if breach_rate > (1 - confidence_level) * 100:
            st.error("🔴 CVaR underestimates tail risk.")
        else:
            st.success("🟢 CVaR appears conservative or well-calibrated.")

        # Matplotlib visualization (keeps previous styling)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(results_df.index, results_df["Actual_Return (%)"], label="Actual Return")
        ax.plot(results_df.index, results_df["Predicted_CVaR (%)"], label="Predicted CVaR")
        ax.fill_between(results_df.index, results_df["Predicted_CVaR (%)"], -50, alpha=0.1)
        ax.axhline(0, linestyle='--', color='black', linewidth=0.8)
        ax.set_title("Optimized Backtest: CVaR for Weekly Tomato Prices")
        ax.set_xlabel("Date")
        ax.set_ylabel("Weekly Return (%)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # === Interactive CVaR plot (Plotly) ===
        if interactive_cvar and PLOTLY_AVAILABLE:
            st.markdown("**Interactive CVaR Backtest (zoom & hover):**")
            fig_i = go.Figure()
            fig_i.add_trace(go.Scatter(
                x=results_df.index, y=results_df["Actual_Return (%)"],
                mode='lines+markers', name='Actual Return', hovertemplate='%{x|%Y-%m-%d}<br>Actual: %{y:.3f}%'
            ))
            fig_i.add_trace(go.Scatter(
                x=results_df.index, y=results_df["Predicted_CVaR (%)"],
                mode='lines+markers', name='Predicted CVaR', hovertemplate='%{x|%Y-%m-%d}<br>Pred CVaR: %{y:.3f}%'
            ))
            fig_i.update_layout(
                height=500,
                xaxis_title="Date",
                yaxis_title="Weekly Return (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_i, use_container_width=True)
        elif interactive_cvar and not PLOTLY_AVAILABLE:
            st.info("Install plotly (`pip install plotly`) to enable interactive CVaR plot.")

        # === Near-future CVaR Forecast (Monte Carlo per future week) ===
        if run_forecast:
            st.subheader("🔮 Near-future CVaR Forecast (Monte Carlo scenarios)")

            # Use latest rolling window to estimate model
            train_latest = ds.iloc[-rolling_window:]
            model_latest = sm.OLS(train_latest["Log_Returns"], sm.add_constant(train_latest["Log_Arrivals"])).fit()
            mu_arr = train_latest["Log_Arrivals"].mean()
            sigma_arr = train_latest["Log_Arrivals"].std()
            resid_sigma = model_latest.resid.std()

            # For each horizon week, compute CVaR by simulating arrivals and returns
            forecast_weeks = list(range(1, int(forecast_horizon) + 1))
            forecast_cvars = []

            for h in forecast_weeks:
                # naive assumption: arrivals each future week IID ~ Normal(mu_arr, sigma_arr)
                sim_arrivals_future = np.random.normal(mu_arr, sigma_arr, size=(int(num_simulations),))  # simulate arrivals for week h
                sim_mu_future = model_latest.params[0] + model_latest.params[1] * sim_arrivals_future
                # returns conditional on simulated mean, adding residual volatility
                sim_returns_future = np.random.normal(sim_mu_future, resid_sigma, size=(int(num_simulations),))
                var_future = np.percentile(sim_returns_future, (1 - confidence_level) * 100)
                cvar_future = sim_returns_future[sim_returns_future <= var_future].mean()
                forecast_cvars.append((h, (np.exp(cvar_future) - 1) * 100))  # in percent

            # build forecast DF
            fc_df = pd.DataFrame(forecast_cvars, columns=["Week_Ahead", "Predicted_CVaR (%)"]).set_index("Week_Ahead")

            # find worst week (min CVaR)
            worst_idx = fc_df["Predicted_CVaR (%)"].idxmin()
            worst_val = fc_df["Predicted_CVaR (%)"].min()

            st.markdown(f"**Forecast result:** Over the next **{forecast_horizon}** weeks the model predicts the *worst* CVaR at **week ahead = {worst_idx}** with predicted CVaR **{worst_val:.3f}%** (most negative expected tail weekly loss).")
            st.markdown("**Interpretation:** This indicates that, under the model's assumption and stationary arrivals, the tail risk is expected to be highest around that horizon. Use this as an early-warning signal; combine with domain knowledge (seasonality, harvest schedules) before taking policy action.")

            # Interactive forecast plot
            if PLOTLY_AVAILABLE:
                fig_fc = go.Figure()
                fig_fc.add_trace(go.Bar(
                    x=fc_df.index, y=fc_df["Predicted_CVaR (%)"],
                    name='Predicted CVaR (%)'
                ))
                fig_fc.add_trace(go.Scatter(
                    x=fc_df.index, y=fc_df["Predicted_CVaR (%)"], mode='lines+markers', name='CVaR trend'
                ))
                fig_fc.add_vline(x=worst_idx, line=dict(color='red', dash='dash'), annotation_text="Worst CVaR", annotation_position="top right")
                fig_fc.update_layout(title="Forecasted Predicted CVaR for Future Weeks", xaxis_title="Week Ahead", yaxis_title="Predicted CVaR (%)", height=450)
                st.plotly_chart(fig_fc, use_container_width=True)
            else:
                # fallback matplotlib
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                ax2.bar(fc_df.index, fc_df["Predicted_CVaR (%)"])
                ax2.plot(fc_df.index, fc_df["Predicted_CVaR (%)"], marker='o')
                ax2.axvline(worst_idx, color='red', linestyle='--', label='Worst CVaR')
                ax2.set_xlabel("Week Ahead")
                ax2.set_ylabel("Predicted CVaR (%)")
                ax2.set_title("Forecasted Predicted CVaR for Future Weeks")
                ax2.legend()
                st.pyplot(fig2)

    else:
        st.info("To run backtesting and CVaR forecast, your dataset must include columns named 'Modal' and 'Arrivals' (weekly).")
else:
    st.info("Upload your weekly crop market data to begin analysis.")

# Footer
st.markdown("""
<hr style="border:1px solid #ccc" />

<div style="text-align: center; font-size: 14px; color: gray;">
    🚀 This app was built by <b>Suman L</b> <br>
    📬 For support or collaboration, contact: <a href="mailto:sumanecon.uas@outlook.com">sumanecon.uas@outlook.com</a>
</div>
""", unsafe_allow_html=True)
