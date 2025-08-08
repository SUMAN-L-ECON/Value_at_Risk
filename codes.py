# Value_at_Risk Analysis App (Merged + Optimized CVaR Backtesting + Dynamic Breach Stats)
# Author: Suman L (adapted & merged per request)
# Notes: Keeps all original models (Historical, Parametric, Monte Carlo, Monte Carlo CVaR),
#       adds Optimized CVaR Backtesting, dynamic breach statistics, and automatic "Best Performing Model".
#       No PDF/email functionality. No yfinance or external fetches.
# Run with: streamlit run this_file.py

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from scipy.stats import norm, chi2
from jinja2 import Template
import warnings
warnings.filterwarnings("ignore")

# ---------------- Streamlit config ----------------
st.set_page_config(page_title="Value_at_risk Analysis App [Suman_econ_UAS(B)]", layout="wide")
st.title("📉 Value at Risk Analysis App Developed by Suman_econ_UAS(B)")

st.markdown("""
### 📌 Instructions Before Upload:
- Date column will be automatically converted to datetime and set as index.
- File format: `.csv`, `.xls`, or `.xlsx`.
- Data should contain at least one price series.
- Missing values will be automatically handled by imputation or dropped if minor.
""")

# ---------------- File upload ----------------
uploaded_file = st.file_uploader("Upload your crop market data", type=["csv", "xls", "xlsx"])

if uploaded_file:
    # read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # process date column
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df.set_index('Date', inplace=True)
        df = df.sort_index()

    # show date range
    st.markdown(f"**Date Range:** {df.index.min().date()} to {df.index.max().date()}")

    # numeric column selection
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found in the uploaded file. Please upload data with at least one price series.")
        st.stop()

    selected_col = st.selectbox("Select Market Price Column", options=['All'] + numeric_cols)

    # date selection
    date_range = st.date_input("Select date range to analyze", [df.index.min(), df.index.max()])
    df = df.loc[(df.index >= pd.to_datetime(date_range[0])) & (df.index <= pd.to_datetime(date_range[1]))]

    analysis_cols = numeric_cols if selected_col == 'All' else [selected_col]

    # model parameters (exposed as small UI controls for flexibility)
    st.sidebar.header("Model & Backtest Parameters")
    confidence_level = st.sidebar.slider("Confidence level (VaR/CVaR)", min_value=0.90, max_value=0.99, value=0.95, step=0.01)
    num_sim = st.sidebar.number_input("Monte Carlo simulations (per step)", min_value=1000, max_value=200000, value=10000, step=1000)
    rolling_window = st.sidebar.number_input("Rolling window for backtesting (weeks)", min_value=26, max_value=260, value=52, step=1)
    use_regression_for_mc_backtest = st.sidebar.checkbox("Use Regression (Log_Arrivals) for MC VaR backtesting (if Modal & Arrivals exist)", value=True)

    z_score = norm.ppf(1 - confidence_level)
    # For percent conversions later
    pct_label = lambda x: (np.exp(x) - 1) * 100

    # containers to accumulate model-level backtest metrics
    master_results = []  # for VaR estimates summary per market
    backtest_summary_records = []  # for comparing models (Historical, Parametric, MonteCarlo)
    cvar_backtest_df = None  # will hold optimized CVaR backtest results if executed

    # ---------------- VaR Analysis per market ----------------
    st.header("🧮 VaR & CVaR Analysis (per selected market price series)")
    for col in analysis_cols:
        st.subheader(f"Market: {col}")
        series = df[col].copy()
        series.replace(0, np.nan, inplace=True)
        series.dropna(inplace=True)

        if len(series) < 30:
            st.warning(f"Not enough data points in {col} to compute risk models.")
            continue

        # log returns
        log_returns = np.log(series / series.shift(1)).dropna()
        mu, sigma = log_returns.mean(), log_returns.std()

        # Historical VaR (simple unconditional quantile)
        hist_var = np.percentile(log_returns, (1 - confidence_level) * 100)

        # Parametric VaR (assuming normal returns)
        param_var = mu + z_score * sigma

        # Monte Carlo VaR & CVaR (unconditional simulation from fitted normal)
        sim_returns = np.random.normal(mu, sigma, size=int(num_sim))
        mc_var = np.percentile(sim_returns, (1 - confidence_level) * 100)
        mc_cvar = sim_returns[sim_returns <= mc_var].mean()

        # store results in master table
        master_results.append({
            "Market": col,
            "Historical VaR (%)": round(pct_label(hist_var), 4),
            "Parametric VaR (%)": round(pct_label(param_var), 4),
            "Monte Carlo VaR (%)": round(pct_label(mc_var), 4),
            "Monte Carlo CVaR (%)": round(pct_label(mc_cvar), 4),
            "Num Observations": len(log_returns)
        })

        # brief policy note per market
        brief_template = Template("""
        In the recent analysis for {{ market }}, we found the following risk indicators:
        - Historical VaR at {{ cl*100|int }}%: {{ hist }}%%
        - Parametric VaR at {{ cl*100|int }}%: {{ param }}%%
        - Monte Carlo VaR at {{ cl*100|int }}%: {{ mc }}%%
        - Monte Carlo CVaR (Expected Shortfall) at {{ cl*100|int }}%: {{ cvar }}%%
        """)
        brief = brief_template.render(market=col,
                                      cl=confidence_level,
                                      hist=round(pct_label(hist_var), 4),
                                      param=round(pct_label(param_var), 4),
                                      mc=round(pct_label(mc_var), 4),
                                      cvar=round(pct_label(mc_cvar), 4))

        # display quick table and plot
        st.markdown(f"**Data points (returns):** {len(log_returns)}")
        st.write(pd.DataFrame([master_results[-1]]).set_index("Market"))

        # Monte Carlo histogram of simulated returns (log returns)
        fig, ax = plt.subplots()
        sns.histplot(sim_returns, bins=60, kde=True, ax=ax)
        ax.axvline(mc_var, color='red', linestyle='--', label=f"VaR {int(confidence_level*100)}%")
        ax.axvline(mc_cvar, color='purple', linestyle='--', label=f"CVaR {int(confidence_level*100)}%")
        ax.set_title(f"Monte Carlo Simulated Log-Returns: {col}")
        ax.set_xlabel("Log Returns")
        ax.legend()
        st.pyplot(fig)

        st.markdown("#### 📌 Interpretations")
        st.markdown(f"- Historical VaR: **{pct_label(hist_var):.4f}%**")
        st.markdown(f"- Parametric VaR: **{pct_label(param_var):.4f}%**")
        st.markdown(f"- Monte Carlo VaR: **{pct_label(mc_var):.4f}%**")
        st.markdown(f"- Monte Carlo CVaR: **{pct_label(mc_cvar):.4f}%**")
        st.markdown("—" * 20)

    # show model comparison table across markets
    if master_results:
        st.subheader("📋 Model Comparison Table (All selected markets)")
        st.dataframe(pd.DataFrame(master_results).set_index("Market"))

    # ---------------- Monte Carlo VaR Backtesting (original regression-based approach) ----------------
    # This preserves original logic: uses Modal & Arrivals if present and if the user opts in (checkbox)
    if 'Modal' in df.columns and 'Arrivals' in df.columns:
        st.header("🔎 Monte Carlo VaR Backtesting (Original: Log_Returns ~ Log_Arrivals)")
        ds = df[['Modal', 'Arrivals']].copy()
        ds['Log_Returns'] = np.log(ds['Modal'] / ds['Modal'].shift(1))
        # Guard against zeros in Arrivals, use bfill as before but more robust
        ds['Log_Arrivals'] = np.log(ds['Arrivals'].replace(0, np.nan)).fillna(method='bfill')
        ds.dropna(inplace=True)

        # Lists for backtesting results (original)
        backtest_results = []
        for i in range(int(rolling_window), len(ds)):
            train = ds.iloc[i - int(rolling_window):i]
            test = ds.iloc[i]
            X = sm.add_constant(train['Log_Arrivals'])
            y = train['Log_Returns']
            model = sm.OLS(y, X).fit()

            # simulate arrivals and returns conditional on arrivals
            sim_arr = np.random.normal(train['Log_Arrivals'].mean(), train['Log_Arrivals'].std(), int(num_sim))
            sim_mu = model.params[0] + model.params[1] * sim_arr
            sim_ret = np.random.normal(sim_mu, model.resid.std(), size=int(num_sim))

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

        bt_df = pd.DataFrame(backtest_results).set_index('Date').sort_index()
        bt_df['Breach_VaR'] = bt_df['Breach_VaR'].astype(int)
        bt_df['Breach_CVaR'] = bt_df['Breach_CVaR'].astype(int)

        # show simple line chart for log returns vs MC VaR (log scale)
        st.line_chart(bt_df[['Actual_Return', 'MC_VaR']])

        observed_breach_rate_vaR = bt_df['Breach_VaR'].mean() * 100
        st.markdown(f"**Observed Breach Rate (MC-VaR):** {observed_breach_rate_vaR:.2f}%")

        # Kupiec's POF test (VaR)
        F = bt_df['Breach_VaR'].sum()
        T = len(bt_df)
        p = 1 - confidence_level
        observed_p = bt_df['Breach_VaR'].mean()
        # avoid zero/one issues in POF formula by bounding observed_p
        observed_p_bounded = max(min(observed_p, 1 - 1e-10), 1e-10)
        stat = -2 * np.log(((1 - p)**(T - F) * p**F) / ((1 - observed_p_bounded)**(T - F) * observed_p_bounded**F))
        p_val = 1 - chi2.cdf(stat, df=1)

        st.markdown(f"**Kupiec's POF Test Statistic (MC-VaR):** {stat:.4f}")
        st.markdown(f"**P-value:** {p_val:.4f}")
        if p_val < 0.05:
            st.error("❌ VaR model (MC-Reg) does not fit well (reject null hypothesis).")
        else:
            st.success("✅ VaR model (MC-Reg) fits well (fail to reject null hypothesis).")

        # dynamic breach stats for MC (regression-based)
        # Convert log returns to percent for interpretation
        bt_df['Actual_Return_pct'] = (np.exp(bt_df['Actual_Return']) - 1) * 100
        bt_df['MC_VaR_pct'] = (np.exp(bt_df['MC_VaR']) - 1) * 100
        bt_df['Loss_when_breach_pct'] = np.where(bt_df['Actual_Return'] < bt_df['MC_VaR'],
                                                 bt_df['Actual_Return_pct'] - bt_df['MC_VaR_pct'],
                                                 0)
        mc_breach_pct = bt_df['Breach_VaR'].mean() * 100
        mc_avg_loss_when_breach = bt_df.loc[bt_df['Breach_VaR'] == 1, 'Loss_when_breach_pct'].abs().mean()
        st.markdown(f"**MC-VaR Backtest — % days breached:** {mc_breach_pct:.2f}%")
        st.markdown(f"**MC-VaR Backtest — Average breach loss (% points):** {mc_avg_loss_when_breach:.4f}%")

        # store summary for comparison later
        backtest_summary_records.append({
            'Model': 'MC-Reg-VaR',
            'Observed_Breach_Rate_pct': mc_breach_pct,
            'Avg_Breach_Loss_pct': mc_avg_loss_when_breach if not np.isnan(mc_avg_loss_when_breach) else 0.0,
            'Num_Observations': len(bt_df)
        })

    else:
        st.info("Columns 'Modal' and 'Arrivals' are required for Monte Carlo regression-based VaR backtesting. Skipping MC-Reg backtest.")

    # ---------------- Rolling-window VaR Backtesting: Historical & Parametric & (Optional) Unconditional MC ----------------
    # We'll compute rolling-window backtests using log returns only (unconditional).
    # This provides consistent comparison across Historical, Parametric and unconditional Monte Carlo VaR.
    st.header("🔁 Rolling-window Backtesting (Historical, Parametric, Unconditional Monte Carlo)")
    # need at least one numeric series to perform backtesting; choose first analysis_col if multiple
    bt_col_for_univariate = analysis_cols[0]
    series_bt = df[bt_col_for_univariate].copy().replace(0, np.nan).dropna()
    log_returns_bt = np.log(series_bt / series_bt.shift(1)).dropna()

    if len(log_returns_bt) < int(rolling_window) + 1:
        st.warning("Not enough observations to perform rolling-window backtesting for Historical/Parametric/Unconditional MC models.")
    else:
        # storage
        hist_breaches = []
        param_breaches = []
        uni_mc_breaches = []
        hist_loss_list = []
        param_loss_list = []
        uni_mc_loss_list = []
        dates_list = []

        for i in range(int(rolling_window), len(log_returns_bt)):
            train = log_returns_bt.iloc[i - int(rolling_window):i]
            test = log_returns_bt.iloc[i]
            dates_list.append(log_returns_bt.index[i])

            # Historical VaR (rolling empirical quantile)
            hist_v = np.percentile(train, (1 - confidence_level) * 100)
            breach_hist = test < hist_v
            hist_breaches.append(int(breach_hist))

            # parametric VaR (rolling mean & std)
            mu_t = train.mean()
            sigma_t = train.std()
            param_v = mu_t + z_score * sigma_t
            breach_param = test < param_v
            param_breaches.append(int(breach_param))

            # unconditional Monte Carlo (simulate from rolling mu & sigma)
            sim_ret_uncond = np.random.normal(mu_t, sigma_t, int(num_sim))
            mc_v_uncond = np.percentile(sim_ret_uncond, (1 - confidence_level) * 100)
            breach_uni_mc = test < mc_v_uncond
            uni_mc_breaches.append(int(breach_uni_mc))

            # losses (convert to percent space for interpretability)
            test_pct = (np.exp(test) - 1) * 100
            hist_v_pct = (np.exp(hist_v) - 1) * 100
            param_v_pct = (np.exp(param_v) - 1) * 100
            mc_v_uncond_pct = (np.exp(mc_v_uncond) - 1) * 100

            hist_loss_list.append((test_pct - hist_v_pct) if breach_hist else 0.0)
            param_loss_list.append((test_pct - param_v_pct) if breach_param else 0.0)
            uni_mc_loss_list.append((test_pct - mc_v_uncond_pct) if breach_uni_mc else 0.0)

        # compile results
        rolling_bt_df = pd.DataFrame({
            'Date': dates_list,
            'Hist_Breach': hist_breaches,
            'Param_Breach': param_breaches,
            'UniMC_Breach': uni_mc_breaches,
            'Hist_Loss_pct': hist_loss_list,
            'Param_Loss_pct': param_loss_list,
            'UniMC_Loss_pct': uni_mc_loss_list
        }).set_index('Date')

        # summary metrics
        hist_breach_rate = rolling_bt_df['Hist_Breach'].mean() * 100
        param_breach_rate = rolling_bt_df['Param_Breach'].mean() * 100
        unimc_breach_rate = rolling_bt_df['UniMC_Breach'].mean() * 100

        hist_avg_loss = rolling_bt_df.loc[rolling_bt_df['Hist_Breach'] == 1, 'Hist_Loss_pct'].abs().mean()
        param_avg_loss = rolling_bt_df.loc[rolling_bt_df['Param_Breach'] == 1, 'Param_Loss_pct'].abs().mean()
        unimc_avg_loss = rolling_bt_df.loc[rolling_bt_df['UniMC_Breach'] == 1, 'UniMC_Loss_pct'].abs().mean()

        # handle NaN if no breaches
        hist_avg_loss = 0.0 if np.isnan(hist_avg_loss) else hist_avg_loss
        param_avg_loss = 0.0 if np.isnan(param_avg_loss) else param_avg_loss
        unimc_avg_loss = 0.0 if np.isnan(unimc_avg_loss) else unimc_avg_loss

        st.markdown("### Rolling-window Backtest Summary")
        bt_summary_table = pd.DataFrame([
            {
                'Model': 'Historical',
                'Observed_Breach_Rate_pct': round(hist_breach_rate, 4),
                'Avg_Breach_Loss_pct': round(hist_avg_loss, 4),
                'Num_Obs': len(rolling_bt_df)
            },
            {
                'Model': 'Parametric',
                'Observed_Breach_Rate_pct': round(param_breach_rate, 4),
                'Avg_Breach_Loss_pct': round(param_avg_loss, 4),
                'Num_Obs': len(rolling_bt_df)
            },
            {
                'Model': 'Unconditional-MC',
                'Observed_Breach_Rate_pct': round(unimc_breach_rate, 4),
                'Avg_Breach_Loss_pct': round(unimc_avg_loss, 4),
                'Num_Obs': len(rolling_bt_df)
            }
        ])
        st.dataframe(bt_summary_table.set_index('Model'))

        # append to backtest_summary_records
        backtest_summary_records.extend(bt_summary_table.to_dict('records'))

        # plot breach events over time for visual check
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(rolling_bt_df.index, (rolling_bt_df['Hist_Breach'] * (rolling_bt_df['Hist_Loss_pct'] != 0)), label='Hist Breach (flag)')
        ax.plot(rolling_bt_df.index, (rolling_bt_df['Param_Breach'] * (rolling_bt_df['Param_Loss_pct'] != 0)), label='Param Breach (flag)')
        ax.plot(rolling_bt_df.index, (rolling_bt_df['UniMC_Breach'] * (rolling_bt_df['UniMC_Loss_pct'] != 0)), label='UniMC Breach (flag)')
        ax.set_title(f"Backtest Breach Flags over Time: {bt_col_for_univariate}")
        ax.set_ylabel("Breach Flag (0/1)")
        ax.set_xlabel("Date")
        ax.legend()
        st.pyplot(fig)

    # ---------------- Optimized CVaR Backtesting (new addition) ----------------
    # This section uses a regression-based approach similar to user's snippet, and outputs breach stats and plots.
    if 'Modal' in df.columns and 'Arrivals' in df.columns:
        st.header("📊 Optimized CVaR Backtesting (Rolling Regression + Simulated CVaR)")
        ds = df[['Modal', 'Arrivals']].copy()
        ds['Log_Returns'] = np.log(ds['Modal'] / ds['Modal'].shift(1))
        ds['Log_Arrivals'] = np.log(ds['Arrivals'].replace(0, np.nan)).fillna(method='bfill')
        ds.dropna(inplace=True)

        if len(ds) < int(rolling_window) + 1:
            st.warning("Not enough observations in Modal/Arrivals to run Optimized CVaR Backtesting.")
        else:
            num_simulations = int(num_sim)
            actual_returns = []
            predicted_cvar_pct = []
            cvar_breach_flags = []
            dates_for_cvar = []

            for i in range(int(rolling_window), len(ds)):
                train = ds.iloc[i - int(rolling_window):i]
                test = ds.iloc[i]

                # regression model
                model = sm.OLS(train["Log_Returns"], sm.add_constant(train["Log_Arrivals"])).fit()
                resid_sigma = model.resid.std(ddof=1)

                # simulate arrivals conditional on rolling stats
                sim_arrivals = np.random.normal(train["Log_Arrivals"].mean(), train["Log_Arrivals"].std(ddof=1), num_simulations)
                sim_mu = model.params[0] + model.params[1] * sim_arrivals
                sim_returns = np.random.normal(sim_mu, resid_sigma, num_simulations)

                var = np.percentile(sim_returns, (1 - confidence_level) * 100)
                cvar = sim_returns[sim_returns <= var].mean()
                cvar_pct = (np.exp(cvar) - 1) * 100

                actual_log_ret = test["Log_Returns"]
                actual_ret_pct = (np.exp(actual_log_ret) - 1) * 100

                actual_returns.append(actual_ret_pct)
                predicted_cvar_pct.append(cvar_pct)
                dates_for_cvar.append(ds.index[i])
                cvar_breach_flags.append(int(actual_ret_pct < cvar_pct))

            results_df = pd.DataFrame({
                "Date": dates_for_cvar,
                "Actual_Return (%)": actual_returns,
                "Predicted_CVaR (%)": predicted_cvar_pct,
                "Breached_CVaR": cvar_breach_flags
            }).set_index('Date')

            breach_rate_cvar = results_df["Breached_CVaR"].mean() * 100
            avg_loss_cvar = (results_df.loc[results_df['Breached_CVaR'] == 1, 'Actual_Return (%)'] - results_df.loc[results_df['Breached_CVaR'] == 1, 'Predicted_CVaR (%)']).abs().mean()
            avg_loss_cvar = 0.0 if np.isnan(avg_loss_cvar) else avg_loss_cvar

            st.markdown(f"**Observed Breach Rate (Optimized CVaR):** {breach_rate_cvar:.2f}%")
            st.markdown(f"**Expected (Nominal) Breach Rate:** {(1 - confidence_level) * 100:.2f}%")
            st.markdown(f"**Average loss when CVaR breached (percentage points):** {avg_loss_cvar:.4f}%")

            # show table and plot
            st.dataframe(results_df.head(10))
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(results_df.index, results_df["Actual_Return (%)"], label="Actual Return", linewidth=1)
            ax.plot(results_df.index, results_df["Predicted_CVaR (%)"], label="Predicted CVaR", linewidth=1.25)
            ax.fill_between(results_df.index, results_df["Predicted_CVaR (%)"], -100, alpha=0.12)
            ax.axhline(0, linestyle='--', color='black', linewidth=0.8)
            ax.set_title("Optimized Backtest: CVaR for Weekly Prices (Regression-based)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Weekly Return (%)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # store CVaR backtest results for final comparison
            cvar_backtest_df = results_df.copy()
            backtest_summary_records.append({
                'Model': 'Optimized-CVaR',
                'Observed_Breach_Rate_pct': round(breach_rate_cvar, 6),
                'Avg_Breach_Loss_pct': round(avg_loss_cvar, 6),
                'Num_Obs': len(results_df)
            })
    else:
        st.info("Columns 'Modal' and 'Arrivals' required for Optimized CVaR Backtesting. Skipping this section.")

    # ---------------- Final Dynamic Breach Statistics Table & Automatic Best Model ----------------
    st.header("📊 Final Backtest Comparison & Best Performing Model (Automatic)")

    if backtest_summary_records:
        summary_df = pd.DataFrame(backtest_summary_records).drop_duplicates(subset=['Model']).set_index('Model')
        # show nicely
        st.subheader("Backtest Summary (Observed breach rates & Avg breach loss)")
        st.dataframe(summary_df)

        # Nominal breach rate expected
        nominal_breach_pct = (1 - confidence_level) * 100
        st.markdown(f"**Nominal (Expected) Breach Rate:** {nominal_breach_pct:.2f}%")

        # Compute selection metric: closeness to nominal (abs diff), then tie-breaker by smaller avg loss
        summary_df = summary_df.assign(
            Abs_Diff_From_Nominal=lambda d: (d['Observed_Breach_Rate_pct'] - nominal_breach_pct).abs()
        )
        # choose best performing
        best_model_row = summary_df.sort_values(['Abs_Diff_From_Nominal', 'Avg_Breach_Loss_pct']).iloc[0]
        best_model_name = best_model_row.name if isinstance(best_model_row.name, str) else best_model_row.name
        st.markdown("### 🏆 Best Performing Model (by closest breach rate to nominal; tie-breaker: smaller avg loss)")
        st.markdown(f"**{best_model_name}**")
        st.write(best_model_row.to_frame().T)

        # Provide an automated interpretation
        interpret_lines = []
        interpret_lines.append(f"- Selected model: **{best_model_name}**")
        interpret_lines.append(f"- Observed breach rate: **{best_model_row['Observed_Breach_Rate_pct']:.4f}%** (Nominal: {nominal_breach_pct:.2f}%)")
        interpret_lines.append(f"- Average breach loss: **{best_model_row['Avg_Breach_Loss_pct']:.4f}%**")
        interpret_lines.append(f"- Rationale: This model's observed breach rate is closest to the nominal expectation; if two models are similarly close, the one with smaller average breach loss is preferred.")

        st.markdown("\n".join(interpret_lines))

    else:
        st.info("No backtest summary records were created (check data & Modal/Arrivals availability).")

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; font-size: 14px; color: gray;">
        🚀 This app was built by <b>Suman L</b> <br>
        📬 For support or collaboration, contact: <a href="mailto:sumanecon.uas@outlook.com">sumanecon.uas@outlook.com</a>
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("Upload your weekly crop market data to begin analysis.")
