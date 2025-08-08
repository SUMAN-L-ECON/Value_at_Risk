import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, chi2
from jinja2 import Template
import plotly.graph_objects as go
import hashlib
import warnings

warnings.filterwarnings("ignore")

# -------------------- Page config --------------------
st.set_page_config(page_title="Value_at_Risk Analysis App [Suman_econ_UAS(B)]",
                   layout="wide")
st.title("📉 Value at Risk Analysis App — Suman_econ_UAS(B)")

st.markdown("""
### 📌 Instructions Before Upload:
- Date column will be automatically converted to datetime and set as index (column named `Date`).
- File format: `.csv`, `.xls`, or `.xlsx`.
- Upload should contain at least one numeric price series (e.g., Modal).
- Missing values will be handled (imputed or dropped if minimal).
""")

# -------------------- Sidebar controls --------------------
st.sidebar.header("Analysis Controls")
confidence_level = st.sidebar.slider("Confidence Level (VaR / CVaR)", min_value=0.90, max_value=0.999, value=0.95, step=0.01, format="%.3f")
num_sim = st.sidebar.number_input("Monte Carlo simulations", min_value=1000, max_value=200000, value=10000, step=1000)
forecast_horizon = st.sidebar.number_input("Forecast horizon (weeks)", min_value=1, max_value=52, value=12, step=1)
rolling_window = st.sidebar.number_input("Rolling window for backtest (weeks)", min_value=20, max_value=260, value=52, step=1)
run_button = st.sidebar.button("Run Analysis")

z_score = norm.ppf(1 - confidence_level)

# -------------------- File upload --------------------
uploaded_file = st.file_uploader("Upload your crop market data (CSV, XLS, XLSX)", type=["csv", "xls", "xlsx"])

def deterministic_seed_from_series(series: pd.Series) -> int:
    """
    Create a deterministic 32-bit seed from a series values so Monte Carlo is reproducible
    for the same dataset.
    """
    # Use last 200 values if long (keeps seed stable)
    arr = series.dropna().values
    if len(arr) == 0:
        base = b"empty"
    else:
        cut = arr[-200:] if len(arr) > 200 else arr
        base = cut.tobytes()
    h = hashlib.sha256(base).digest()
    # take first 4 bytes as little-endian int
    seed = int.from_bytes(h[:4], "little")
    return seed

def make_policy_interpretation(col, breach_rate, cvar_pct, mc_cvar_pct):
    """
    Short policy interpretation template — actionable and clear.
    """
    t = Template("""
    **Market:** {{market}}
    - Observed breach rate (CVaR backtest): **{{breach_rate:.2f}}%** (nominal: {{nominal:.2f}}%)
    - Estimated CVaR (recent window): **{{cvar:.2f}}%**
    - Monte Carlo CVaR: **{{mc_cvar:.2f}}%**
    
    **Interpretation / Policy notes**
    1. If observed breach > nominal, tail risk is underestimated — consider buffer stocks or emergency procurement triggers.
    2. If CVaR is large (absolute), consider price stabilization (eg: forward contracting, targeted procurement).
    3. Increase monitoring frequency for supply-side indicators (arrivals, weather alerts). 
    4. Suggest stress-scenarios for planning: repeat worst 5% shocks for budget & logistics planning.
    """)
    return t.render(market=col, breach_rate=breach_rate, nominal=(1 - confidence_level) * 100, cvar=cvar_pct, mc_cvar=mc_cvar_pct)

# -------------------- Main run --------------------
if uploaded_file and run_button:
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

    st.markdown(f"**Date Range:** {df.index.min().date()} — {df.index.max().date()}")
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns detected. Please upload a dataset with a price/arrival series.")
        st.stop()

    selected_col = st.selectbox("Select Market Price Column (for VaR/CVaR)", options=['All'] + numeric_cols)
    date_range = st.date_input("Select date range", [df.index.min(), df.index.max()])
    df = df.loc[(df.index >= pd.to_datetime(date_range[0])) & (df.index <= pd.to_datetime(date_range[1]))]

    analysis_cols = numeric_cols if selected_col == 'All' else [selected_col]

    results, briefs = [], []

    for col in analysis_cols:
        # Prepare series
        series = df[col].copy()
        series.replace(0, np.nan, inplace=True)
        series.dropna(inplace=True)

        if len(series) < 30:
            st.warning(f"Not enough data points in {col} to compute risk models.")
            continue

        # Log returns
        log_returns = np.log(series / series.shift(1)).dropna()
        mu, sigma = log_returns.mean(), log_returns.std()

        # Historical VaR
        hist_var = np.percentile(log_returns, (1 - confidence_level) * 100)

        # Parametric VaR
        param_var = mu + z_score * sigma

        # Deterministic RNG for reproducibility per series
        seed = deterministic_seed_from_series(series)
        rng = np.random.default_rng(seed)

        # Monte Carlo returns (single-step) using mu,sigma
        sim_returns = rng.normal(mu, sigma, size=num_sim)
        mc_var = np.percentile(sim_returns, (1 - confidence_level) * 100)
        mc_cvar = sim_returns[sim_returns <= mc_var].mean()

        # Save results (percent)
        results.append({
            "Market": col,
            "Historical VaR (%)": round((np.exp(hist_var) - 1) * 100, 4),
            "Parametric VaR (%)": round((np.exp(param_var) - 1) * 100, 4),
            "Monte Carlo VaR (%)": round((np.exp(mc_var) - 1) * 100, 4),
            "Monte Carlo CVaR (%)": round((np.exp(mc_cvar) - 1) * 100, 4),
        })

        # automated brief with interpretation
        brief_text = make_policy_interpretation(col,
                                               breach_rate=0.0,  # placeholder; updated after backtest if available
                                               cvar_pct=(np.exp(hist_var) - 1) * 100,
                                               mc_cvar_pct=(np.exp(mc_cvar) - 1) * 100)
        briefs.append((col, brief_text))

        # Plot: Monte Carlo histogram (static matplotlib kept as before)
        fig, ax = plt.subplots()
        sns.histplot(sim_returns, bins=50, kde=True, ax=ax)
        ax.axvline(mc_var, color='red', linestyle='--', label=f"VaR {int(confidence_level*100)}%")
        ax.axvline(mc_cvar, color='purple', linestyle='--', label=f"CVaR {int(confidence_level*100)}%")
        ax.set_title(f"Monte Carlo Simulated Returns: {col}")
        ax.set_xlabel("Log Returns")
        ax.legend()
        st.pyplot(fig)

    # Show results table
    if results:
        st.subheader("📋 Model Comparison Table")
        st.dataframe(pd.DataFrame(results).set_index("Market"))

    # Show briefs (these will be updated later if backtest runs)
    if briefs:
        st.subheader("📝 Automated Policy Briefs")
        for market, text in briefs:
            with st.expander(f"Policy Brief (initial) for {market}"):
                st.markdown(text)

    # -------------------- Backtesting if required columns exist --------------------
    if 'Modal' in df.columns and 'Arrivals' in df.columns:
        st.subheader("🔎 Monte Carlo VaR Backtesting (Log Returns ~ Log Arrivals)")
        ds = df[['Modal', 'Arrivals']].copy()
        ds['Log_Returns'] = np.log(ds['Modal'] / ds['Modal'].shift(1))
        ds['Log_Arrivals'] = np.log(ds['Arrivals'].replace(0, np.nan)).fillna(method='bfill')
        ds.dropna(inplace=True)

        # Basic backtest for VaR and CVaR using rolling window
        backtest_results = []
        for i in range(int(rolling_window), len(ds)):
            train = ds.iloc[i - int(rolling_window):i]
            test = ds.iloc[i]

            X = sm.add_constant(train['Log_Arrivals'])
            y = train['Log_Returns']
            model = sm.OLS(y, X).fit()

            # deterministic RNG using training modal series for reproducibility
            seed_bt = deterministic_seed_from_series(train['Modal'])
            rng_bt = np.random.default_rng(seed_bt)

            sim_arr = rng_bt.normal(train['Log_Arrivals'].mean(), train['Log_Arrivals'].std(), num_sim)
            sim_mu = model.params[0] + model.params[1] * sim_arr
            sim_ret = rng_bt.normal(sim_mu, model.resid.std(), size=num_sim)

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

        st.markdown(f"**Observed Breach Rate (VaR):** {bt_df['Breach_VaR'].mean() * 100:.2f}%")
        st.markdown(f"**Observed Breach Rate (CVaR):** {bt_df['Breach_CVaR'].mean() * 100:.2f}%")

        # Kupiec test for VaR (as earlier)
        F = bt_df['Breach_VaR'].sum()
        T = len(bt_df)
        p = 1 - confidence_level
        observed_p = bt_df['Breach_VaR'].mean()
        # avoid log(0)
        observed_p = max(min(observed_p, 1 - 1e-10), 1e-10)
        stat = -2 * np.log(((1 - p) ** (T - F) * p ** F) / ((1 - observed_p) ** (T - F) * observed_p ** F))
        p_val = 1 - chi2.cdf(stat, df=1)
        st.markdown(f"**Kupiec's POF Test Statistic:** {stat:.4f}")
        st.markdown(f"**P-value:** {p_val:.4f}")
        if p_val < 0.05:
            st.error("❌ VaR model does not fit well (reject null hypothesis).")
        else:
            st.success("✅ VaR model fits well (fail to reject null hypothesis).")

        # Update briefs with actual breach info and policy notes
        breach_rate_cvar_pct = bt_df['Breach_CVaR'].mean() * 100
        # Update the first brief if available (safe updating)
        if briefs:
            updated_briefs = []
            for market, _text in briefs:
                # compute mc_cvar_pct for that market if exists in results
                matching = [r for r in results if r['Market'] == market]
                mc_cvar_pct = matching[0]['Monte Carlo CVaR (%)'] if matching else np.nan
                new_text = make_policy_interpretation(market, breach_rate_cvar_pct, cvar_pct=(np.exp(hist_var) - 1) * 100 if 'hist_var' in locals() else np.nan, mc_cvar_pct=mc_cvar_pct)
                updated_briefs.append((market, new_text))
            # display updated briefs
            st.subheader("📝 Updated Policy Briefs (Post-Backtest)")
            for market, text in updated_briefs:
                with st.expander(f"Policy Brief (backtest) for {market}"):
                    st.markdown(text)

        # Interactive CVaR Backtest Plot using Plotly
        st.subheader("📊 Interactive CVaR Backtest Plot")
        # convert to percentage for plotting
        plot_df = bt_df.copy()
        plot_df['Actual_Return_pct'] = (np.exp(plot_df['Actual_Return']) - 1) * 100
        plot_df['MC_CVaR_pct'] = (np.exp(plot_df['MC_CVaR']) - 1) * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Actual_Return_pct'],
                                 mode='lines+markers', name='Actual Return (%)'))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MC_CVaR_pct'],
                                 mode='lines', name=f'Predicted CVaR {int(confidence_level*100)}% (%)'))
        fig.update_layout(title="Interactive Backtest: Actual Return vs Predicted CVaR",
                          xaxis_title="Date", yaxis_title="Return (%)",
                          hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # -------------------- Optimized CVaR Backtesting (reproducible) --------------------
        st.subheader("🔁 Optimized CVaR Backtesting (Deterministic RNG)")

        actual_returns = []
        predicted_cvar_pct = []

        # use ds series seed for reproducibility
        seed_ds = deterministic_seed_from_series(ds['Modal'])
        rng_ds = np.random.default_rng(seed_ds)

        for i in range(int(rolling_window), len(ds)):
            train = ds.iloc[i - int(rolling_window):i]
            test = ds.iloc[i]

            model = sm.OLS(train["Log_Returns"], sm.add_constant(train["Log_Arrivals"])).fit()
            resid_sigma = model.resid.std()

            # seed per rolling window to keep deterministic reproducibility
            seed_window = deterministic_seed_from_series(train['Modal'])
            rng_window = np.random.default_rng(seed_window)

            sim_arrivals = rng_window.normal(train["Log_Arrivals"].mean(), train["Log_Arrivals"].std(), num_sim)
            sim_mu = model.params[0] + model.params[1] * sim_arrivals
            sim_returns = rng_window.normal(sim_mu, resid_sigma, num_sim)

            var = np.percentile(sim_returns, (1 - confidence_level) * 100)
            cvar = sim_returns[sim_returns <= var].mean()
            cvar_pct = (np.exp(cvar) - 1) * 100

            actual_log_ret = test["Log_Returns"]
            actual_ret_pct = (np.exp(actual_log_ret) - 1) * 100
            actual_returns.append(actual_ret_pct)
            predicted_cvar_pct.append(cvar_pct)

        results_df = pd.DataFrame({
            "Date": ds.index[int(rolling_window):],
            "Actual_Return (%)": actual_returns,
            "Predicted_CVaR (%)": predicted_cvar_pct
        })
        results_df["Breached_CVaR"] = results_df["Actual_Return (%)"] < results_df["Predicted_CVaR (%)"]

        breach_rate = results_df["Breached_CVaR"].mean() * 100
        st.markdown(f"**Observed Breach Rate (Optimized CVaR):** {breach_rate:.2f}%")
        st.markdown("**Expected (Nominal) Breach Rate:** {:.2f}%".format((1 - confidence_level) * 100))
        if breach_rate > (1 - confidence_level) * 100:
            st.error("🔴 CVaR underestimates tail risk.")
        else:
            st.success("🟢 CVaR appears conservative or well-calibrated.")

        # Plot static CVaR backtest (matplotlib) kept for users who like it
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(results_df["Date"], results_df["Actual_Return (%)"], label="Actual Return")
        ax2.plot(results_df["Date"], results_df["Predicted_CVaR (%)"], label="Predicted CVaR")
        ax2.fill_between(results_df["Date"], results_df["Predicted_CVaR (%)"], -50, alpha=0.1)
        ax2.axhline(0, linestyle='--', color='black', linewidth=0.8)
        ax2.set_title("Optimized Backtest: CVaR for Weekly Prices")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Weekly Return (%)")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

        # -------------------- Forecast: when the worst CVaR fall will occur --------------------
        st.subheader("🔮 Forecast: When might the worst CVaR fall occur?")

        # Use deterministic RNG using last modal series
        seed_forecast = deterministic_seed_from_series(ds['Modal'])
        rng_fore = np.random.default_rng(seed_forecast)

        # For forecasting use regression if available or simple mu/sigma
        last_train = ds.iloc[-int(rolling_window):] if len(ds) >= int(rolling_window) else ds
        try:
            reg = sm.OLS(last_train["Log_Returns"], sm.add_constant(last_train["Log_Arrivals"])).fit()
            use_regression = True
        except Exception:
            use_regression = False

        # collect index offsets where the minimum return occurs in each simulation
        min_positions = []
        sims = num_sim  # use same num_sim
        horizon = int(forecast_horizon)

        # Pre-compute last date and weekly frequency assumption
        last_date = ds.index[-1]
        # infer weekly periods via pandas frequency if available; else assume 7 days
        freq = '7D'
        if hasattr(ds.index, 'inferred_freq') and ds.index.inferred_freq is not None:
            freq = ds.index.inferred_freq

        for s in range(sims):
            if use_regression:
                # simulate arrivals path: assume random normal increments around last log arrivals mean/std
                sim_arr_path = rng_fore.normal(last_train["Log_Arrivals"].mean(), last_train["Log_Arrivals"].std(), horizon)
                sim_mu_path = reg.params[0] + reg.params[1] * sim_arr_path
                sim_ret_path = rng_fore.normal(sim_mu_path, reg.resid.std())
            else:
                mu_f = last_train["Log_Returns"].mean()
                sigma_f = last_train["Log_Returns"].std()
                sim_ret_path = rng_fore.normal(mu_f, sigma_f, horizon)

            # find index of minimum return across horizon (worst fall)
            min_idx = int(np.argmin(sim_ret_path))
            min_positions.append(min_idx)

        # compute expected week index and distribution
        min_positions = np.array(min_positions)
        # most frequent week index where worst falls happen
        mode_index = int(pd.Series(min_positions).mode()[0])
        # expected (mean) index
        expected_index = int(np.round(min_positions.mean()))
        # convert index to date
        predicted_mode_date = pd.to_datetime(last_date) + pd.to_timedelta((mode_index + 1) * 7, unit='D')
        predicted_expected_date = pd.to_datetime(last_date) + pd.to_timedelta((expected_index + 1) * 7, unit='D')

        st.markdown(f"- Based on {sims:,} deterministic simulations over a {horizon}-week horizon:")
        st.markdown(f"  - **Most likely week for worst fall (mode):** Week {mode_index+1} → around **{predicted_mode_date.date()}**")
        st.markdown(f"  - **Expected week index (mean):** Week {expected_index+1} → around **{predicted_expected_date.date()}**")
        st.markdown("  - *Note:* This uses a short-horizon simulation and deterministic seeding — results are reproducible for the same dataset and settings.")

        # Small distribution summary plot using plotly
        dist_fig = go.Figure()
        counts = pd.Series(min_positions).value_counts().sort_index()
        dist_fig.add_trace(go.Bar(x=[f"Week {i+1}" for i in counts.index], y=counts.values))
        dist_fig.update_layout(title="Distribution of simulated worst-fall week (across simulations)",
                               xaxis_title="Week index (1..horizon)", yaxis_title="Number of simulations")
        st.plotly_chart(dist_fig, use_container_width=True)

else:
    if uploaded_file:
        st.info("Change controls in the left sidebar and press **Run Analysis** when ready.")
    else:
        st.info("Upload your weekly crop market data to begin analysis.")

# -------------------- Footer --------------------
st.markdown("""
<hr style="border:1px solid #ccc" />
<div style="text-align: center; font-size: 14px; color: gray;">
    🚀 This app was built by <b>Suman L</b> <br>
    📬 For support or collaboration, contact: <a href="mailto:sumanecon.uas@outlook.com">sumanecon.uas@outlook.com</a>
</div>
""", unsafe_allow_html=True)
