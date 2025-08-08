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
import hashlib
import plotly.express as px

warnings.filterwarnings("ignore")

# -------------------- Page setup --------------------
st.set_page_config(
    page_title="Value_at_Risk Analysis App [Suman_econ_UAS(B)]",
    layout="wide"
)

# Large friendly title for farmers
st.title("📉 Value at Risk & CVaR Backtesting — Farmer Friendly")

st.markdown("""
This app computes VaR / CVaR for your market price series, performs backtests,
and forecasts when the worst CVaR (largest expected tail loss) is likely to happen
in the near future. Use the controls on the left to adjust confidence level, simulations and forecast horizon.
""")

# -------------------- Sidebar controls --------------------
st.sidebar.header("Analysis Controls")

confidence_level = st.sidebar.slider("Confidence level (%)", min_value=90, max_value=99, value=95, step=1) / 100.0
num_simulations = st.sidebar.number_input("Monte Carlo simulations", min_value=1000, max_value=200000, value=10000, step=1000)
rolling_window = st.sidebar.number_input("Rolling window (weeks)", min_value=26, max_value=156, value=52, step=1)
forecast_horizon = st.sidebar.number_input("Forecast horizon (weeks)", min_value=4, max_value=52, value=12, step=1)
display_interpretations = st.sidebar.checkbox("Show automated policy interpretations", value=True)

z_score = norm.ppf(1 - confidence_level)

# -------------------- File upload --------------------
uploaded_file = st.file_uploader("Upload your crop market data (CSV / XLS / XLSX)", type=["csv", "xls", "xlsx"])

def _make_deterministic_seed(series: pd.Series) -> int:
    """Create a deterministic 32-bit seed from series content so runs are repeatable for same dataset."""
    # Use last 500 values (if long) to make seed stable to full file noise
    arr = series.dropna().astype(str).tail(500).to_string().encode("utf-8")
    h = hashlib.sha256(arr).hexdigest()
    return int(h[:8], 16)  # 32-bit-ish integer

if uploaded_file:
    # Read file robustly
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Ensure Date column is parsed & set as index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df.set_index('Date', inplace=True)
        df = df.sort_index()

    st.markdown(f"**Data range:** {df.index.min().date()} to {df.index.max().date()}")

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        st.error("No numeric price columns found. Make sure your dataset includes at least one price series.")
        st.stop()

    selected_col = st.selectbox("Select Market Price Column", options=['All'] + numeric_cols)

    # date-range filter for convenience
    date_range = st.date_input("Select date range to analyze", [df.index.min(), df.index.max()])
    df = df.loc[(df.index >= pd.to_datetime(date_range[0])) & (df.index <= pd.to_datetime(date_range[1]))]

    analysis_cols = numeric_cols if selected_col == 'All' else [selected_col]

    # Prepare UI layout: left column for controls/brief, right for results
    left_col, right_col = st.columns([1, 2])

    results, briefs = [], []

    for col in analysis_cols:
        series = df[col].copy()
        series.replace(0, np.nan, inplace=True)
        series.dropna(inplace=True)

        if len(series) < 30:
            st.warning(f"Not enough data points for {col}. Need at least 30 non-null observations.")
            continue

        # Deterministic RNG seeded from data
        seed = _make_deterministic_seed(series)
        rng = np.random.default_rng(seed)

        # compute log returns
        log_returns = np.log(series / series.shift(1)).dropna()
        mu, sigma = log_returns.mean(), log_returns.std()

        # Historical VaR (in log returns)
        hist_var = np.percentile(log_returns, (1 - confidence_level) * 100)
        # Parametric VaR (Gaussian assumption)
        param_var = mu + z_score * sigma

        # Monte Carlo simulation using deterministic RNG
        sim_returns = rng.normal(mu, sigma, size=num_simulations)
        mc_var = np.percentile(sim_returns, (1 - confidence_level) * 100)
        mc_cvar = sim_returns[sim_returns <= mc_var].mean()

        # append numeric results (convert to percentage returns)
        results.append({
            "Market": col,
            "Historical VaR (%)": round((np.exp(hist_var) - 1) * 100, 2),
            "Parametric VaR (%)": round((np.exp(param_var) - 1) * 100, 2),
            "Monte Carlo VaR (%)": round((np.exp(mc_var) - 1) * 100, 2),
            "Monte Carlo CVaR (%)": round((np.exp(mc_cvar) - 1) * 100, 2),
        })

        # automated brief template - friendly language
        brief_template = Template("""
**Market:** {{ market }}

- Historical VaR at {{ conf*100|round(0) }}%: **{{ hist }}%**  
- Parametric VaR (Gaussian) at {{ conf*100|round(0) }}%: **{{ param }}%**  
- Monte Carlo VaR at {{ conf*100|round(0) }}%: **{{ mc }}%**  
- Monte Carlo CVaR (Expected Shortfall) at {{ conf*100|round(0) }}%: **{{ cvar }}%**

**Simple interpretation (for farmers):**  
There is a roughly *{{ conf*100|round(0) }}%* chance that weekly price change will be worse than **{{ cvar }}%** (a fall of this percent) when we look at the worst part of historical outcomes. Use storage, early sale, or insurance if falls of this size are risky for you.
        """)
        brief_text = brief_template.render(
            market=col,
            conf=confidence_level,
            hist=round((np.exp(hist_var) - 1) * 100, 2),
            param=round((np.exp(param_var) - 1) * 100, 2),
            mc=round((np.exp(mc_var) - 1) * 100, 2),
            cvar=round((np.exp(mc_cvar) - 1) * 100, 2)
        )
        briefs.append((col, brief_text))

        # Monte Carlo histogram (keeps original look)
        fig_mc, ax_mc = plt.subplots()
        sns.histplot(sim_returns, bins=50, kde=True, ax=ax_mc)
        ax_mc.axvline(mc_var, color='red', linestyle='--', label=f"VaR {(confidence_level*100):.0f}%")
        ax_mc.axvline(mc_cvar, color='purple', linestyle='--', label=f"CVaR {(confidence_level*100):.0f}%")
        ax_mc.set_title(f"Monte Carlo Simulated Returns: {col}")
        ax_mc.set_xlabel("Log Returns")
        ax_mc.legend()

        # Display Monte Carlo histogram in right column
        right_col.pyplot(fig_mc)

        # -------------------- Backtesting (Monte Carlo VaR) --------------------
        # Only run backtests if Modal & Arrivals exist (keep same logic)
        if 'Modal' in df.columns and 'Arrivals' in df.columns:
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

                # use deterministic RNG derived from training window to keep repeatability
                train_seed = _make_deterministic_seed(train['Log_Arrivals'])
                rng_bt = np.random.default_rng(train_seed)

                sim_arr = rng_bt.normal(train['Log_Arrivals'].mean(), train['Log_Arrivals'].std(), num_simulations)
                sim_mu = model.params[0] + model.params[1] * sim_arr
                sim_ret = rng_bt.normal(sim_mu, model.resid.std(), size=num_simulations)

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
            bt_df['Breach_CVaR'] = bt_df['Breach_CVaR'].astype(int)

            right_col.markdown("### 🔎 Monte Carlo VaR Backtest")
            right_col.line_chart(bt_df[['Actual_Return', 'MC_VaR']])
            right_col.markdown(f"**Observed Breach Rate (VaR):** {bt_df['Breach_VaR'].mean() * 100:.2f}%")

            # Kupiec POF test for VaR
            F = bt_df['Breach_VaR'].sum()
            T = len(bt_df)
            p = 1 - confidence_level
            observed_p = bt_df['Breach_VaR'].mean() if T > 0 else 0
            if 0 < observed_p < 1:
                stat = -2 * np.log(((1 - p) ** (T - F) * p ** F) / ((1 - observed_p) ** (T - F) * observed_p ** F))
                p_val = 1 - chi2.cdf(stat, df=1)
            else:
                stat, p_val = np.nan, np.nan

            right_col.write(f"**Kupiec POF test statistic:** {stat if not np.isnan(stat) else 'N/A'}")
            right_col.write(f"**P-value:** {p_val if not np.isnan(p_val) else 'N/A'}")
            if not np.isnan(p_val):
                if p_val < 0.05:
                    right_col.error("❌ VaR model does not fit well (reject null hypothesis).")
                else:
                    right_col.success("✅ VaR model fits well (fail to reject null hypothesis).")

            # -------------------- Optimized CVaR Backtesting (deterministic) --------------------
            right_col.markdown("### 📊 Optimized CVaR Backtesting (Interactive)")

            # Rolling CVaR predictions & actual returns (in percent, for human-friendly view)
            actual_returns = []
            predicted_cvar_pct = []
            rng_forecast = np.random.default_rng(seed + 12345)  # deterministic but different stream for forecasting

            for i in range(rolling_window, len(ds)):
                train = ds.iloc[i - rolling_window:i]
                test = ds.iloc[i]

                model = sm.OLS(train["Log_Returns"], sm.add_constant(train["Log_Arrivals"])).fit()
                resid_sigma = model.resid.std()

                # Use train-specific seed for deterministic sim
                train_seed = _make_deterministic_seed(train['Log_Arrivals'])
                rng_train = np.random.default_rng(train_seed)

                sim_arrivals = rng_train.normal(train["Log_Arrivals"].mean(), train["Log_Arrivals"].std(), num_simulations)
                sim_mu = model.params[0] + model.params[1] * sim_arrivals
                sim_returns = rng_train.normal(sim_mu, resid_sigma, num_simulations)

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
            }).reset_index(drop=True)

            # Add Breach indicator
            results_df["Breached_CVaR"] = results_df["Actual_Return (%)"] < results_df["Predicted_CVaR (%)"]

            breach_rate = results_df["Breached_CVaR"].mean() * 100 if len(results_df) > 0 else np.nan

            right_col.markdown(f"**Observed Breach Rate (CVaR):** {breach_rate:.2f}%")
            right_col.markdown("**Expected (Nominal) Breach Rate:** {:.2f}%".format((1 - confidence_level) * 100))
            if breach_rate > (1 - confidence_level) * 100:
                right_col.error("🔴 CVaR underestimates tail risk.")
            else:
                right_col.success("🟢 CVaR appears conservative or well-calibrated.")

            # Interactive CVaR plot with Plotly
            if not results_df.empty:
                fig_cvar = px.line(results_df, x="Date", y=["Actual_Return (%)", "Predicted_CVaR (%)"],
                                   labels={"value": "Return (%)", "variable": "Series"},
                                   title=f"Interactive CVaR Backtest ({col})")
                fig_cvar.update_traces(mode="lines+markers")
                # Add shading for predicted CVaR area
                fig_cvar.add_scatter(x=results_df["Date"], y=results_df["Predicted_CVaR (%)"],
                                     fill='tozeroy', name='Predicted CVaR Area', mode='none', opacity=0.1)
                right_col.plotly_chart(fig_cvar, use_container_width=True)

            # -------------------- Forecasting: When is the worst CVaR likely? --------------------
            # Forecast next `forecast_horizon` weeks deterministically using train last window
            last_train = ds.iloc[-rolling_window:] if len(ds) >= rolling_window else ds.copy()
            model_full = sm.OLS(last_train["Log_Returns"], sm.add_constant(last_train["Log_Arrivals"])).fit()
            resid_sigma_full = model_full.resid.std()
            arrivals_mean = last_train["Log_Arrivals"].mean()
            arrivals_std = last_train["Log_Arrivals"].std()

            # use separate deterministic RNG for forecast steps
            rng_fore = np.random.default_rng(seed + 99999)

            future_week_index = pd.date_range(start=ds.index[-1] + pd.Timedelta(days=7),
                                              periods=forecast_horizon, freq='7D')

            forecast_cvar_pct = []
            for h in range(forecast_horizon):
                # simulate arrivals for horizon step h
                sim_arrivals_future = rng_fore.normal(arrivals_mean, arrivals_std, num_simulations)
                sim_mu_future = model_full.params[0] + model_full.params[1] * sim_arrivals_future
                sim_returns_future = rng_fore.normal(sim_mu_future, resid_sigma_full, num_simulations)

                var_f = np.percentile(sim_returns_future, (1 - confidence_level) * 100)
                cvar_f = sim_returns_future[sim_returns_future <= var_f].mean()
                cvar_f_pct = (np.exp(cvar_f) - 1) * 100
                forecast_cvar_pct.append(cvar_f_pct)

            forecast_df = pd.DataFrame({
                "Date": future_week_index,
                "Forecasted_CVaR (%)": forecast_cvar_pct
            })

            # Which future week has the worst (lowest) CVaR?
            worst_idx = forecast_df["Forecasted_CVaR (%)"].idxmin()
            worst_row = forecast_df.loc[worst_idx]
            worst_date = pd.to_datetime(worst_row["Date"]).date()
            worst_value = worst_row["Forecasted_CVaR (%)"]

            right_col.markdown("#### 📆 Near-future CVaR forecast (deterministic)")
            right_col.write(f"Worst forecasted weekly CVaR in next {forecast_horizon} weeks is on **{worst_date}** "
                            f"with expected CVaR ≈ **{worst_value:.2f}%** (a fall of this percent).")

            # show interactive forecast chart
            fig_fore = px.line(forecast_df, x="Date", y="Forecasted_CVaR (%)", title="Forecasted CVaR (next weeks)")
            fig_fore.add_scatter(x=[worst_row["Date"]], y=[worst_value], mode='markers+text',
                                 text=[f"Worst: {worst_value:.2f}%"], textposition="bottom right", name="Worst")
            right_col.plotly_chart(fig_fore, use_container_width=True)

    # -------------------- Final outputs --------------------
    if results:
        st.subheader("📋 Model Comparison Table")
        st.dataframe(pd.DataFrame(results).set_index("Market"))

    if briefs and display_interpretations:
        st.subheader("📝 Automated Policy Briefs (Friendly language)")
        for market, text in briefs:
            with st.expander(f"Policy Brief for {market}", expanded=False):
                # Presentation: simplify language for farmers below
                st.markdown(text)
                # Plain farmer-friendly actionable bullets
                st.markdown("""
**Actionable suggestions (simple):**
- If predicted CVaR shows large falls, consider storing produce or selling smaller quantities each week.  
- If arrival forecasts show high risk weeks, contact local cooperatives for aggregation or forward sale.  
- Use crop insurance or local price-stabilization schemes if available.
                """)

else:
    st.info("Upload your weekly crop market data to begin analysis.")

# -------------------- Footer --------------------
st.markdown("""
<hr style="border:1px solid #eee" />

<div style="text-align: center; font-size: 14px; color: gray;">
    🚀 Built by <b>Suman L</b> — simple UI for farmers. <br>
    📬 Support: <a href="mailto:sumanecon.uas@outlook.com">sumanecon.uas@outlook.com</a>
</div>
""", unsafe_allow_html=True)
