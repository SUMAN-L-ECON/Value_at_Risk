# app.py
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm, chi2
from jinja2 import Template
import warnings
warnings.filterwarnings("ignore")

# ================ Page Config ================
st.set_page_config(page_title="Value_at_Risk Analysis App [Suman_econ_UAS(B)]", layout="wide")
st.title("📉 Value at Risk Analysis App — Suman_econ_UAS(B)")

# ================ Sidebar (controls) ================
st.sidebar.header("Analysis Controls")

# File upload guidance
st.sidebar.write(
    """
    **Instructions**  
    - Upload CSV / XLS / XLSX with a `Date` column.  
    - At least one numeric price column required.  
    - Missing values: minor gaps imputed or dropped.
    """
)

uploaded_file = st.sidebar.file_uploader("Upload your crop market data", type=["csv", "xls", "xlsx"])

# User-controlled parameters
confidence_level = st.sidebar.slider("Confidence level (VaR/CVaR)", min_value=0.90, max_value=0.999, value=0.95, step=0.01, format="%.3f")
num_simulations = st.sidebar.number_input("Monte Carlo simulations", min_value=1000, max_value=200000, value=10000, step=1000)
rolling_window = st.sidebar.slider("Rolling window for backtest (weeks)", min_value=26, max_value=156, value=52, step=1)
run_var_backtest = st.sidebar.checkbox("Run Monte Carlo VaR Backtest", value=True)
run_cvar_backtest = st.sidebar.checkbox("Run Optimized CVaR Backtest", value=True)

# Quick UI toggle for which analyses to run
st.sidebar.markdown("---")
st.sidebar.info("Tip: Lower `num_simulations` during quick tests. Increase for production runs.")

# ================ Main UI ================
if not uploaded_file:
    st.info("Upload your weekly crop market data to begin analysis.")
    st.stop()

# ================ Read file ================
if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

# Ensure Date handling
if 'Date' not in df.columns:
    st.error("Uploaded file must contain a 'Date' column.")
    st.stop()

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.dropna(subset=['Date'], inplace=True)
df.set_index('Date', inplace=True)
df = df.sort_index()

st.markdown(f"**Date Range:** {df.index.min().date()} — {df.index.max().date()}")

# Column selection
numeric_cols = df.select_dtypes(include='number').columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found in uploaded data. At least one price series expected.")
    st.stop()

selected_col = st.selectbox("Select Market Price Column to analyze", options=['All'] + numeric_cols)

# Date range filter (main area)
with st.expander("Select analysis date range"):
    date_start, date_end = st.date_input("Date range", [df.index.min().date(), df.index.max().date()])
    df = df.loc[(df.index >= pd.to_datetime(date_start)) & (df.index <= pd.to_datetime(date_end))]

analysis_cols = numeric_cols if selected_col == 'All' else [selected_col]

# Constants derived from user input
z_score = norm.ppf(1 - confidence_level)
results = []
briefs = []

# ================ VaR / CVaR per column analysis ================
for col in analysis_cols:
    st.markdown(f"---\n## 🔎 Market: **{col}**")
    series = df[col].copy()
    series.replace(0, np.nan, inplace=True)
    series.dropna(inplace=True)

    if len(series) < 30:
        st.warning(f"Not enough data points in {col} to compute risk models (need >= 30).")
        continue

    # Log returns
    log_returns = np.log(series / series.shift(1)).dropna()
    mu, sigma = log_returns.mean(), log_returns.std()

    # Historical VaR (in log returns)
    hist_var_log = np.percentile(log_returns, (1 - confidence_level) * 100)
    hist_var_pct = (np.exp(hist_var_log) - 1) * 100

    # Parametric VaR (normal)
    param_var_log = mu + z_score * sigma
    param_var_pct = (np.exp(param_var_log) - 1) * 100

    # Monte Carlo VaR & CVaR (simulated on log returns)
    sim_returns = np.random.normal(mu, sigma, size=num_simulations)
    mc_var_log = np.percentile(sim_returns, (1 - confidence_level) * 100)
    mc_var_pct = (np.exp(mc_var_log) - 1) * 100
    mc_cvar_log = sim_returns[sim_returns <= mc_var_log].mean()
    mc_cvar_pct = (np.exp(mc_cvar_log) - 1) * 100

    results.append({
        "Market": col,
        "Historical VaR (%)": round(hist_var_pct, 3),
        "Parametric VaR (%)": round(param_var_pct, 3),
        "Monte Carlo VaR (%)": round(mc_var_pct, 3),
        "Monte Carlo CVaR (%)": round(mc_cvar_pct, 3),
    })

    # ---------- Interactive MC histogram (plotly) ----------
    st.subheader("Monte Carlo: Simulated Log Returns Distribution (interactive)")
    fig_hist = px.histogram(sim_returns, nbins=70, marginal="rug", histnorm='probability',
                            labels={"value": "Simulated log returns"},
                            title=f"Simulated Log Returns — {col}")
    # Add VaR/CVaR vertical lines
    fig_hist.add_vline(x=mc_var_log, line_dash="dash", line_color="red", annotation_text=f"VaR (log) {mc_var_log:.4f}", annotation_position="top left")
    fig_hist.add_vline(x=mc_cvar_log, line_dash="dash", line_color="purple", annotation_text=f"CVaR (log) {mc_cvar_log:.4f}", annotation_position="top right")
    st.plotly_chart(fig_hist, use_container_width=True)

    # ---------- Summary metrics ----------
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Historical VaR (95%)", f"{hist_var_pct:.3f}%")
        st.metric("Parametric VaR (95%)", f"{param_var_pct:.3f}%")
    with col2:
        st.metric("Monte Carlo VaR (95%)", f"{mc_var_pct:.3f}%")
        st.metric("Monte Carlo CVaR (95%)", f"{mc_cvar_pct:.3f}%")

    # ---------- Automated policy brief (improved with interpretation) ----------
    brief_template = Template("""
    **Executive summary — {{ market }}**

    Over the selected period ({{ start }} to {{ end }}), the weekly return dynamics show:
    - **Historical VaR ({{ cl }}):** {{ hist }}% — simple empirical tail threshold.
    - **Parametric VaR ({{ cl }}):** {{ param }}% — assumes normality; may understate tails if returns are fat-tailed.
    - **Monte Carlo VaR ({{ cl }}):** {{ mc }}% — simulates using sample mean & volatility.
    - **Monte Carlo CVaR ({{ cl }}):** {{ cvar }}% — expected loss given a breach; preferred for tail-risk planning.

    **Interpretation & policy suggestions**
    - The observed **CVaR of {{ cvar }}%** suggests that in the worst {{ tail_pct }}% of weeks, expected losses are around **{{ cvar }}%**.
    - If breach rates (backtests) are higher than nominal ({{ nominal }}%), consider:
      1. **Hedging** (forward contracts or futures) to protect farmers/traders from extreme downside.
      2. **Improved storage/processing** to reduce exposure to weekly price shocks.
      3. **Diversification of procurement and sales windows** to spread risk across weeks or markets.
      4. **Market intelligence & early-warning** using arrival forecasts and weather signals.
    - If parametric VaR << historical/MC VaR, stress tests or fat-tailed models (t-distribution) are recommended.

    _Note: Numbers above use log-return transforms; convert to amount-loss estimates by multiplying with current price._
    """)
    brief_text = brief_template.render(
        market=col,
        start=df.index.min().date(),
        end=df.index.max().date(),
        cl=f"{int(confidence_level*100)}%",
        hist=round(hist_var_pct, 3),
        param=round(param_var_pct, 3),
        mc=round(mc_var_pct, 3),
        cvar=round(mc_cvar_pct, 3),
        tail_pct=round((1-confidence_level)*100, 3),
        nominal=f"{round((1-confidence_level)*100,3)}%"
    )
    with st.expander(f"Policy Brief & Interpretation — {col}", expanded=False):
        st.markdown(brief_text)

# ================ Show Aggregate Comparison Table ================
if results:
    st.subheader("📋 Model Comparison Table")
    df_results = pd.DataFrame(results).set_index("Market")
    st.dataframe(df_results)

# ================ Monte Carlo VaR Backtesting (if requested) ================
if run_var_backtest and 'Modal' in df.columns and 'Arrivals' in df.columns:
    st.markdown("---")
    st.header("🔎 Monte Carlo VaR Backtesting (Log Returns ~ Log Arrivals)")

    ds = df[['Modal', 'Arrivals']].copy()
    ds['Log_Returns'] = np.log(ds['Modal'] / ds['Modal'].shift(1))
    ds['Log_Arrivals'] = np.log(ds['Arrivals'].replace(0, np.nan)).fillna(method='bfill')
    ds.dropna(inplace=True)

    backtest_results = []
    for i in range(rolling_window, len(ds)):
        train = ds.iloc[i - rolling_window:i]
        test = ds.iloc[i]
        X = sm.add_constant(train['Log_Arrivals'])
        y = train['Log_Returns']
        model = sm.OLS(y, X).fit()

        sim_arr = np.random.normal(train['Log_Arrivals'].mean(), train['Log_Arrivals'].std(), num_simulations)
        sim_mu = model.params[0] + model.params[1] * sim_arr
        sim_ret = np.random.normal(sim_mu, model.resid.std(), size=num_simulations)

        mc_var_bt = np.percentile(sim_ret, (1 - confidence_level) * 100)
        actual_ret = test['Log_Returns']
        backtest_results.append({
            'Date': ds.index[i],
            'Actual_Return': actual_ret,
            'MC_VaR': mc_var_bt,
            'Breach_VaR': actual_ret < mc_var_bt
        })

    bt_df = pd.DataFrame(backtest_results).set_index('Date')
    bt_df['Breach_VaR'] = bt_df['Breach_VaR'].astype(int)

    # Plot interactive
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Actual_Return'],
                                mode='lines+markers', name='Actual Log-Return'))
    fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['MC_VaR'],
                                mode='lines', name='MC VaR (log)'))
    fig_bt.update_layout(title="Monte Carlo VaR Backtest (log returns)", xaxis_title="Date", yaxis_title="Log returns")
    st.plotly_chart(fig_bt, use_container_width=True)

    st.markdown(f"**Observed Breach Rate (VaR):** {bt_df['Breach_VaR'].mean()*100:.2f}%")

    # Kupiec POF test
    F = bt_df['Breach_VaR'].sum()
    T = len(bt_df)
    p = 1 - confidence_level
    observed_p = bt_df['Breach_VaR'].mean()
    # Handle edge cases for observed_p=0 or 1:
    observed_p_safe = np.clip(observed_p, 1e-12, 1 - 1e-12)
    stat = -2 * np.log(((1 - p)**(T - F) * p**F) / ((1 - observed_p_safe)**(T - F) * observed_p_safe**F))
    p_val = 1 - chi2.cdf(stat, df=1)

    st.markdown(f"**Kupiec's POF Test Statistic:** {stat:.4f}")
    st.markdown(f"**P-value:** {p_val:.4f}")
    if p_val < 0.05:
        st.error("❌ VaR model does not fit well (reject null hypothesis).")
    else:
        st.success("✅ VaR model fits well (fail to reject null hypothesis).")

# ================ Optimized CVaR Backtesting (interactive) ================
if run_cvar_backtest and 'Modal' in df.columns and 'Arrivals' in df.columns:
    st.markdown("---")
    st.header("📊 Optimized CVaR Backtesting (interactive)")

    ds = df[['Modal', 'Arrivals']].copy()
    ds['Log_Returns'] = np.log(ds['Modal'] / ds['Modal'].shift(1))
    ds['Log_Arrivals'] = np.log(ds['Arrivals'].replace(0, np.nan)).fillna(method='bfill')
    ds.dropna(inplace=True)

    actual_returns = []
    predicted_cvar_pct = []
    dates = []

    for i in range(rolling_window, len(ds)):
        train = ds.iloc[i - rolling_window:i]
        test = ds.iloc[i]

        model = sm.OLS(train["Log_Returns"], sm.add_constant(train["Log_Arrivals"])).fit()
        resid_sigma = model.resid.std()

        sim_arrivals = np.random.normal(train["Log_Arrivals"].mean(), train["Log_Arrivals"].std(), num_simulations)
        sim_mu = model.params[0] + model.params[1] * sim_arrivals
        sim_returns = np.random.normal(sim_mu, resid_sigma, num_simulations)

        var = np.percentile(sim_returns, (1 - confidence_level) * 100)
        cvar = sim_returns[sim_returns <= var].mean()
        cvar_pct = (np.exp(cvar) - 1) * 100

        actual_log_ret = test["Log_Returns"]
        actual_ret_pct = (np.exp(actual_log_ret) - 1) * 100

        dates.append(ds.index[i])
        actual_returns.append(actual_ret_pct)
        predicted_cvar_pct.append(cvar_pct)

    results_df = pd.DataFrame({
        "Date": dates,
        "Actual_Return (%)": actual_returns,
        "Predicted_CVaR (%)": predicted_cvar_pct
    }).set_index("Date")
    results_df["Breached_CVaR"] = results_df["Actual_Return (%)"] < results_df["Predicted_CVaR (%)"]

    breach_rate = results_df["Breached_CVaR"].mean() * 100
    st.markdown(f"**Observed Breach Rate (CVaR):** {breach_rate:.2f}%")
    st.markdown("**Expected (Nominal) Breach Rate:** {:.2f}%".format((1 - confidence_level) * 100))

    if breach_rate > (1 - confidence_level) * 100:
        st.error("🔴 CVaR underestimates tail risk (observed breach > nominal).")
    else:
        st.success("🟢 CVaR appears conservative or well-calibrated (observed breach ≤ nominal).")

    # Interactive CVaR backtest plot (plotly)
    fig_cvar = go.Figure()
    fig_cvar.add_trace(go.Scatter(x=results_df.index, y=results_df["Actual_Return (%)"],
                                  mode='lines+markers', name='Actual Return (%)'))
    fig_cvar.add_trace(go.Scatter(x=results_df.index, y=results_df["Predicted_CVaR (%)"],
                                  mode='lines', name='Predicted CVaR (%)', line=dict(color='red')))
    fig_cvar.add_trace(go.Scatter(
        x=list(results_df.index) + list(results_df.index[::-1]),
        y=list(results_df["Predicted_CVaR (%)"]) + [-50]*len(results_df),
        fill='toself',
        fillcolor='rgba(255,0,0,0.1)',
        line=dict(color='rgba(255,0,0,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig_cvar.update_layout(title="Optimized CVaR Backtest: Actual vs Predicted CVaR (%)",
                           xaxis_title="Date", yaxis_title="Weekly Return (%)")
    st.plotly_chart(fig_cvar, use_container_width=True)

    # Show a small summary dataframe and download option
    st.dataframe(results_df.head(200))

    csv = results_df.to_csv(index=True)
    st.download_button(label="Download CVaR backtest CSV", data=csv, file_name="cvar_backtest_results.csv", mime="text/csv")

# ================ Footer ================
st.markdown("""
<hr style="border:1px solid #ccc" />

<div style="text-align: center; font-size: 14px; color: gray;">
    🚀 This app was built by <b>Suman L</b> <br>
    📬 For support or collaboration, contact: <a href="mailto:sumanecon.uas@outlook.com">sumanecon.uas@outlook.com</a>
</div>
""", unsafe_allow_html=True)
