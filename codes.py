# full_vaR_cvar_app.py
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from scipy.stats import norm, chi2
from jinja2 import Template
from fpdf import FPDF
import smtplib
import base64
import warnings
import os
from datetime import datetime

warnings.filterwarnings("ignore")

# -----------------------
# Page config / Header
# -----------------------
st.set_page_config(page_title="Value_at_Risk Analysis App [Suman_econ_UAS(B)]", layout="wide")
st.title("📉 Value at Risk Analysis App — Suman_econ_UAS(B)")
st.markdown("""
**Instructions**
- Upload `.csv`, `.xls`, or `.xlsx` file with a `Date` column (or index convertible to datetime).
- File should contain at least one price series column (e.g., "Modal" or other market prices).
- App will compute: Historical VaR, Parametric VaR, Monte Carlo VaR, Monte Carlo CVaR, backtesting, and optimized CVaR backtest.
""")

# -----------------------
# Utility functions
# -----------------------
def read_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    return df

def create_policy_brief(market, hist_var, param_var, mc_var, mc_cvar):
    tmpl = Template("""
    In the recent analysis for {{ market }}:
    - Historical VaR (95%): {{ hist }}%%
    - Parametric VaR (95%): {{ param }}%%
    - Monte Carlo VaR (95%): {{ mc }}%%
    - Monte Carlo CVaR (95%): {{ cvar }}%%
    """)
    return tmpl.render(market=market,
                       hist=round((np.exp(hist_var)-1)*100, 2),
                       param=round((np.exp(param_var)-1)*100, 2),
                       mc=round((np.exp(mc_var)-1)*100, 2),
                       cvar=round((np.exp(mc_cvar)-1)*100, 2))

def fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return buf

def send_email_gmail(sender, password, recipient, subject, body, attachments=None):
    """
    Send an email with attachments through Gmail SMTP.
    attachments: list of tuples (filename, bytes_data)
    """
    # Build a simple MIME message manually (base64 for attachments)
    import email, email.mime.application, email.mime.multipart, email.mime.text
    msg = email.mime.multipart.MIMEMultipart()
    msg['From'] = sender
    msg['To'] = recipient
    msg['Subject'] = subject

    # Attach body
    msg.attach(email.mime.text.MIMEText(body, 'plain'))

    # Attach files
    if attachments:
        for fname, fbytes in attachments:
            part = email.mime.application.MIMEApplication(fbytes.read(), Name=fname)
            part['Content-Disposition'] = f'attachment; filename="{fname}"'
            msg.attach(part)

    # Send via Gmail SMTP
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login(sender, password)
    server.sendmail(sender, recipient, msg.as_string())
    server.quit()

def generate_pdf_report(title, description, df_results, df_bt_summary, figs, briefs, best_model_text):
    """
    Create a PDF report with tables and images.
    Returns BytesIO of PDF.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, title, ln=True, align='C')
    pdf.ln(4)
    pdf.multi_cell(0, 6, description)
    pdf.ln(6)

    # Add results table (VaR summary)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 6, "VaR / CVaR Model Comparison", ln=True)
    pdf.set_font("Arial", size=10)

    # Convert df_results to text table
    if df_results is not None and not df_results.empty:
        # header
        colnames = df_results.columns.tolist()
        # attempt to print a compact table
        pdf.set_font("Arial", size=9)
        for col in colnames:
            pdf.cell(40, 6, str(col)[:15], border=1)
        pdf.ln()
        for i in range(min(20, len(df_results))):
            row = df_results.iloc[i]
            for col in colnames:
                pdf.cell(40, 6, str(row[col])[:15], border=1)
            pdf.ln()
    pdf.ln(6)

    # Backtest summary table
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 6, "Backtest Summary (Model-level)", ln=True)
    pdf.set_font("Arial", size=10)
    if df_bt_summary is not None and not df_bt_summary.empty:
        colnames = df_bt_summary.columns.tolist()
        for col in colnames:
            pdf.cell(45, 6, str(col)[:18], border=1)
        pdf.ln()
        for i in range(len(df_bt_summary)):
            row = df_bt_summary.iloc[i]
            for col in colnames:
                pdf.cell(45, 6, str(round(row[col], 4))[:18], border=1)
            pdf.ln()
    pdf.ln(6)

    # Add text briefs
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 6, "Automated Briefs", ln=True)
    pdf.set_font("Arial", size=9)
    for m, txt in briefs:
        pdf.multi_cell(0, 6, f"{m}:\n{txt}")
        pdf.ln(2)

    pdf.ln(4)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 6, "Best Performing Model Summary", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, best_model_text)
    pdf.ln(6)

    # Add figures
    for i, figbuf in enumerate(figs):
        pdf.add_page()
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 6, f"Figure {i+1}", ln=True)
        pdf.ln(2)
        # figbuf is BytesIO
        fname_img = f"fig_{i+1}.png"
        with open(fname_img, 'wb') as f:
            f.write(figbuf.getbuffer())
        pdf.image(fname_img, w=180)
        try:
            os.remove(fname_img)
        except:
            pass

    # Return BytesIO
    out = BytesIO()
    out.write(pdf.output(dest='S').encode('latin-1'))
    out.seek(0)
    return out

# -----------------------
# File upload & preprocessing
# -----------------------
uploaded_file = st.file_uploader("Upload your crop market data", type=["csv", "xls", "xlsx"])

if not uploaded_file:
    st.info("Upload your weekly crop market data to begin analysis.")
    st.stop()

# read
df_raw = read_file(uploaded_file)

# if Date present, set index and sort
if 'Date' in df_raw.columns:
    df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')
    df_raw.dropna(subset=['Date'], inplace=True)
    df_raw.set_index('Date', inplace=True)
    df_raw = df_raw.sort_index()
else:
    # attempt to convert index to datetime
    try:
        df_raw.index = pd.to_datetime(df_raw.index)
        df_raw = df_raw.sort_index()
    except:
        st.error("No 'Date' column and index cannot be parsed as datetime. Please include a Date column.")
        st.stop()

st.markdown(f"**Data loaded:** {uploaded_file.name} — Date range: {df_raw.index.min().date()} to {df_raw.index.max().date()}")
st.write("Preview of data (first 5 rows):")
st.dataframe(df_raw.head())

# Column selection
numeric_cols = df_raw.select_dtypes(include='number').columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found in uploaded data. Need at least one price/number column.")
    st.stop()

selected_col = st.selectbox("Select Market Price Column to analyze", options=['All'] + numeric_cols)
date_range = st.date_input("Select date range to analyze", [df_raw.index.min(), df_raw.index.max()])
df = df_raw.loc[(df_raw.index >= pd.to_datetime(date_range[0])) & (df_raw.index <= pd.to_datetime(date_range[1]))].copy()

analysis_cols = numeric_cols if selected_col == 'All' else [selected_col]

# Parameters
confidence_level = st.sidebar.slider("Confidence Level (VaR)", min_value=0.90, max_value=0.99, value=0.95, step=0.01)
num_sim = st.sidebar.number_input("Number of Monte Carlo simulations", min_value=1000, max_value=200000, value=10000, step=1000)
window = st.sidebar.number_input("Rolling window for backtests (weeks)", min_value=20, max_value=520, value=52, step=1)
z_score = norm.ppf(1 - confidence_level)

# storage for final report
all_results = []
all_briefs = []
all_figs = []  # BytesIO buffers to embed in PDF
backtest_model_summary = []

# -----------------------
# VaR / CVaR Model computations (per original logic)
# -----------------------
for col in analysis_cols:
    series = df[col].copy()
    series.replace(0, np.nan, inplace=True)
    series.dropna(inplace=True)

    if len(series) < 30:
        st.warning(f"Not enough data points in {col} to compute risk models.")
        continue

    st.header(f"🔎 Analysis for market series: **{col}**")
    # log returns
    log_returns = np.log(series / series.shift(1)).dropna()
    mu, sigma = log_returns.mean(), log_returns.std()

    # Historical VaR (empirical)
    hist_var = np.percentile(log_returns, (1 - confidence_level) * 100)

    # Parametric VaR (Gaussian)
    param_var = mu + z_score * sigma

    # Monte Carlo using return distribution
    sim_returns = np.random.normal(mu, sigma, size=num_sim)
    mc_var = np.percentile(sim_returns, (1 - confidence_level) * 100)
    mc_cvar = sim_returns[sim_returns <= mc_var].mean()

    # Present numeric results (converted to % change)
    res_row = {
        "Market": col,
        "Historical VaR (%)": round((np.exp(hist_var) - 1) * 100, 4),
        "Parametric VaR (%)": round((np.exp(param_var) - 1) * 100, 4),
        "Monte Carlo VaR (%)": round((np.exp(mc_var) - 1) * 100, 4),
        "Monte Carlo CVaR (%)": round((np.exp(mc_cvar) - 1) * 100, 4),
        "Mu (log ret)": float(mu),
        "Sigma (log ret)": float(sigma)
    }
    all_results.append(res_row)

    # Policy brief
    brief_text = Template("""
    For {{ market }}:
    - Historical VaR ({{ conf }}): {{ hist }}%%
    - Parametric VaR ({{ conf }}): {{ param }}%%
    - Monte Carlo VaR ({{ conf }}): {{ mc }}%%
    - Monte Carlo CVaR ({{ conf }}): {{ cvar }}%%
    """).render(market=col, conf=f"{int(confidence_level*100)}%", hist=res_row["Historical VaR (%)"],
                param=res_row["Parametric VaR (%)"], mc=res_row["Monte Carlo VaR (%)"],
                cvar=res_row["Monte Carlo CVaR (%)"])
    all_briefs.append((col, brief_text))

    # Plot simulated returns histogram
    fig, ax = plt.subplots()
    sns.histplot(sim_returns, bins=60, kde=True, ax=ax)
    ax.axvline(mc_var, color='red', linestyle='--', label=f"VaR {int(confidence_level*100)}%")
    ax.axvline(mc_cvar, color='purple', linestyle='--', label=f"CVaR {int(confidence_level*100)}%")
    ax.set_title(f"Monte Carlo Simulated Log-Returns — {col}")
    ax.set_xlabel("Log Returns")
    ax.legend()
    st.pyplot(fig)
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    all_figs.append(buf)

# Show overall results table
if all_results:
    results_df = pd.DataFrame(all_results).set_index("Market")
    st.subheader("📋 Model Comparison Table (VaR / CVaR estimates)")
    st.dataframe(results_df.style.format("{:.4f}"))
else:
    st.info("No valid numeric series to analyze.")
    st.stop()

# Show briefs
st.subheader("📝 Automated Policy Briefs")
for market, text in all_briefs:
    with st.expander(f"Policy Brief — {market}"):
        st.markdown(text)

# -----------------------
# Monte Carlo VaR Backtesting (Original regression-based section)
# This preserves the code you had: Log_Returns ~ Log_Arrivals
# -----------------------
if 'Modal' in df.columns and 'Arrivals' in df.columns:
    st.subheader("🔎 Monte Carlo VaR Backtesting (Regression-based: Log_Returns ~ Log_Arrivals)")
    ds = df[['Modal', 'Arrivals']].copy()
    ds['Log_Returns'] = np.log(ds['Modal'] / ds['Modal'].shift(1))
    ds['Log_Arrivals'] = np.log(ds['Arrivals'].replace(0, np.nan)).fillna(method='bfill')
    ds.dropna(inplace=True)

    backtest_results = []

    for i in range(window, len(ds)):
        train = ds.iloc[i - int(window):i]
        test = ds.iloc[i]
        X = sm.add_constant(train['Log_Arrivals'])
        y = train['Log_Returns']
        model = sm.OLS(y, X).fit()

        sim_arr = np.random.normal(train['Log_Arrivals'].mean(), train['Log_Arrivals'].std(), num_sim)
        sim_mu = model.params[0] + model.params[1] * sim_arr
        sim_ret = np.random.normal(sim_mu, model.resid.std(), size=num_sim)

        mc_var_bt = np.percentile(sim_ret, (1 - confidence_level) * 100)
        mc_cvar_bt = sim_ret[sim_ret <= mc_var_bt].mean()

        actual_ret = test['Log_Returns']
        backtest_results.append({
            'Date': ds.index[i],
            'Actual_Return': actual_ret,
            'MC_VaR': mc_var_bt,
            'MC_CVaR': mc_cvar_bt,
            'Breach_VaR': float(actual_ret < mc_var_bt),
            'Breach_CVaR': float(actual_ret < mc_cvar_bt)
        })

    bt_df = pd.DataFrame(backtest_results).set_index('Date')
    bt_df['Breach_VaR'] = bt_df['Breach_VaR'].astype(int)
    bt_df['Breach_CVaR'] = bt_df['Breach_CVaR'].astype(int)

    st.line_chart(bt_df[['Actual_Return', 'MC_VaR']])
    st.markdown(f"**Observed Breach Rate (Regression MC VaR):** {bt_df['Breach_VaR'].mean()*100:.2f}%")

    # Kupiec Test (for VaR breaches)
    F = int(bt_df['Breach_VaR'].sum())
    T = len(bt_df)
    p = 1 - confidence_level
    observed_p = bt_df['Breach_VaR'].mean()
    # avoid division by zero / log issues
    if observed_p in [0, 1] or p in [0, 1]:
        stat, p_val = np.nan, np.nan
        st.warning("Kupiec test couldn't be computed due to zero/one observed breach rate.")
    else:
        stat = -2 * np.log(((1 - p) ** (T - F) * p ** F) / ((1 - observed_p) ** (T - F) * observed_p ** F))
        p_val = 1 - chi2.cdf(stat, df=1)
        st.markdown(f"**Kupiec's POF Test Statistic:** {stat:.4f}")
        st.markdown(f"**P-value:** {p_val:.4f}")
        if p_val < 0.05:
            st.error("❌ VaR model (regression-based) does not fit well (reject null).")
        else:
            st.success("✅ VaR model (regression-based) fits (fail to reject null).")

    # Save regression backtest figure
    fig_bt, ax = plt.subplots(figsize=(10, 4))
    ax.plot(bt_df.index, (np.exp(bt_df['Actual_Return']) - 1) * 100, label="Actual Return (%)")
    ax.plot(bt_df.index, (np.exp(bt_df['MC_VaR']) - 1) * 100, label="MC VaR (%)")
    ax.set_title("Regression-based Monte Carlo VaR Backtest (Actual vs MC VaR)")
    ax.legend()
    ax.axhline(0, linestyle='--', color='black')
    st.pyplot(fig_bt)
    buf_bt = BytesIO()
    fig_bt.savefig(buf_bt, format='png', bbox_inches='tight')
    buf_bt.seek(0)
    all_figs.append(buf_bt)

else:
    st.info("Columns 'Modal' and 'Arrivals' not both present — skipping regression-based Monte Carlo VaR backtest.")

# -----------------------
# Rolling model-level backtest for Historical, Parametric, Monte Carlo (based on returns only)
# This yields dynamic breach stats and "best performing model" selection
# -----------------------
st.subheader("📈 Rolling Backtest (Model-level on log-returns) — Historical / Parametric / MC")
model_bt_results = []

for col in analysis_cols:
    series = df[col].dropna().copy()
    series.replace(0, np.nan, inplace=True)
    series.dropna(inplace=True)
    if len(series) < int(window) + 5:
        st.warning(f"Not enough length in {col} for rolling backtest (need > window).")
        continue

    log_returns = np.log(series / series.shift(1)).dropna()
    dates = log_returns.index

    recs = []
    for i in range(int(window), len(log_returns)):
        train = log_returns.iloc[i - int(window):i]
        test = log_returns.iloc[i]

        # Historical VaR (empirical from train)
        hist_v = np.percentile(train, (1 - confidence_level) * 100)

        # Parametric VaR from train
        mu_t = train.mean()
        sigma_t = train.std()
        param_v = mu_t + z_score * sigma_t

        # Monte Carlo based on train's mu/sigma
        sim_ret = np.random.normal(mu_t, sigma_t, size=num_sim)
        mc_v = np.percentile(sim_ret, (1 - confidence_level) * 100)

        # actual breach flags
        breach_hist = float(test < hist_v)
        breach_param = float(test < param_v)
        breach_mc = float(test < mc_v)

        # losses when breached (in percent)
        loss_hist = (np.exp(test) - 1) * 100 if breach_hist else np.nan
        loss_param = (np.exp(test) - 1) * 100 if breach_param else np.nan
        loss_mc = (np.exp(test) - 1) * 100 if breach_mc else np.nan

        recs.append({
            "Date": dates[i],
            "Actual_LogRet": test,
            "Hist_Var": hist_v,
            "Param_Var": param_v,
            "MC_Var": mc_v,
            "Breach_Hist": breach_hist,
            "Breach_Param": breach_param,
            "Breach_MC": breach_mc,
            "Loss_Hist_pct": loss_hist,
            "Loss_Param_pct": loss_param,
            "Loss_MC_pct": loss_mc
        })

    recs_df = pd.DataFrame(recs).set_index('Date')

    # Compute dynamic stats
    stats = {}
    for model_name, breach_col, loss_col in [
        ("Historical", "Breach_Hist", "Loss_Hist_pct"),
        ("Parametric", "Breach_Param", "Loss_Param_pct"),
        ("MonteCarlo", "Breach_MC", "Loss_MC_pct")
    ]:
        breach_rate = recs_df[breach_col].mean() * 100
        avg_loss_when_breached = np.nanmean(recs_df[loss_col])  # percent
        stats[f"{model_name}_BreachRate_pct"] = breach_rate
        stats[f"{model_name}_AvgLossWhenBreached_pct"] = avg_loss_when_breached

    # Best model selection: prioritize breach rate close to nominal (1 - confidence_level)*100 and lower avg loss
    nominal_pct = (1 - confidence_level) * 100
    # score = abs(breach_rate - nominal) + 0.01 * avg_loss_when_breached (smaller is better)
    scores = {}
    for m in ["Historical", "Parametric", "MonteCarlo"]:
        br = stats[f"{m}_BreachRate_pct"]
        al = stats[f"{m}_AvgLossWhenBreached_pct"]
        # if NaN, penalize
        al_sub = 0 if np.isnan(al) else al
        scores[m] = abs(br - nominal_pct) + 0.02 * max(0, al_sub)  # weight avg loss lightly

    best_model = min(scores, key=scores.get)
    best_text = (f"For series {col}, based on rolling backtest: Best model = {best_model}. "
                 f"Nominal breach rate = {nominal_pct:.2f}%. Breach rates: "
                 f"Hist={stats['Historical_BreachRate_pct']:.2f}%, "
                 f"Param={stats['Parametric_BreachRate_pct']:.2f}%, "
                 f"MC={stats['MonteCarlo_BreachRate_pct']:.2f}%. "
                 f"Average loss when breached (pct): Hist={stats['Historical_AvgLossWhenBreached_pct']:.2f}, "
                 f"Param={stats['Parametric_AvgLossWhenBreached_pct']:.2f}, "
                 f"MC={stats['MonteCarlo_AvgLossWhenBreached_pct']:.2f}.")

    st.markdown(f"**Rolling backtest summary for {col}:**")
    st.markdown(best_text)

    # Add model-level summary table to results
    summary_row = {
        "Market": col,
        "Nominal_BreachPct": nominal_pct,
        "Hist_BreachPct": stats["Historical_BreachRate_pct"],
        "Hist_AvgLossPct": stats["Historical_AvgLossWhenBreached_pct"],
        "Param_BreachPct": stats["Parametric_BreachRate_pct"],
        "Param_AvgLossPct": stats["Parametric_AvgLossWhenBreached_pct"],
        "MC_BreachPct": stats["MonteCarlo_BreachRate_pct"],
        "MC_AvgLossPct": stats["MonteCarlo_AvgLossWhenBreached_pct"],
        "BestModel": best_model
    }
    backtest_model_summary.append(summary_row)

    # store plots for this recs_df
    fig_rb, ax = plt.subplots(figsize=(11, 4))
    ax.plot(recs_df.index, (np.exp(recs_df['Actual_LogRet']) - 1) * 100, label='Actual Return (%)')
    ax.plot(recs_df.index, (np.exp(recs_df['Hist_Var']) - 1) * 100, label='Hist VaR (%)', alpha=0.8)
    ax.plot(recs_df.index, (np.exp(recs_df['Param_Var']) - 1) * 100, label='Param VaR (%)', alpha=0.8)
    ax.plot(recs_df.index, (np.exp(recs_df['MC_Var']) - 1) * 100, label='MC VaR (%)', alpha=0.8)
    ax.set_title(f"Rolling Backtest — {col} (Actual vs VaR models)")
    ax.legend()
    ax.axhline(0, linestyle='--', color='black', linewidth=0.6)
    st.pyplot(fig_rb)
    buf_rb = BytesIO()
    fig_rb.savefig(buf_rb, format='png', bbox_inches='tight')
    buf_rb.seek(0)
    all_figs.append(buf_rb)

# Show aggregated backtest model summary
if backtest_model_summary:
    bt_summary_df = pd.DataFrame(backtest_model_summary).set_index("Market")
    st.subheader("📊 Rolling Backtest — Model-level Summary")
    st.dataframe(bt_summary_df.style.format("{:.3f}"))

# -----------------------
# Optimized CVaR Backtesting (your requested code integrated into Streamlit)
# -----------------------
st.subheader("📊 Optimized CVaR Backtesting (Regression-based Rolling CVaR predictions)")

if 'Modal' in df.columns and 'Arrivals' in df.columns:
    ds = df[['Modal', 'Arrivals']].copy()
    ds['Log_Returns'] = np.log(ds['Modal'] / ds['Modal'].shift(1))
    ds['Log_Arrivals'] = np.log(ds['Arrivals'].replace(0, np.nan)).fillna(method='bfill')
    ds.dropna(inplace=True)

    rolling_window = int(window)
    num_simulations = int(num_sim)

    actual_returns = []
    predicted_cvar_pct = []

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
        actual_returns.append(actual_ret_pct)
        predicted_cvar_pct.append(cvar_pct)

    results_df = pd.DataFrame({
        "Date": ds.index[rolling_window:],
        "Actual_Return (%)": actual_returns,
        "Predicted_CVaR (%)": predicted_cvar_pct
    }).set_index("Date")

    results_df["Breached_CVaR"] = results_df["Actual_Return (%)"] < results_df["Predicted_CVaR (%)"]
    breach_rate = results_df["Breached_CVaR"].mean() * 100

    st.markdown(f"**Observed Breach Rate (Optimized CVaR):** {breach_rate:.2f}%")
    st.markdown("**Expected (Nominal) Breach Rate:** {:.2f}%".format((1 - confidence_level) * 100))
    if breach_rate > (1 - confidence_level) * 100:
        st.error("🔴 CVaR underestimates tail risk (breach rate higher than nominal).")
    else:
        st.success("🟢 CVaR appears conservative or well-calibrated (breach rate <= nominal).")

    # Plot CVaR Backtest
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(results_df.index, results_df["Actual_Return (%)"], label="Actual Return", linewidth=1)
    ax.plot(results_df.index, results_df["Predicted_CVaR (%)"], label="Predicted CVaR", linewidth=1)
    ax.fill_between(results_df.index, results_df["Predicted_CVaR (%)"], -50, alpha=0.12)
    ax.axhline(0, linestyle='--', color='black', linewidth=0.8)
    ax.set_title("Optimized Backtest: CVaR for Weekly Prices (Regression-based)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weekly Return (%)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    buf_cvar = BytesIO()
    fig.savefig(buf_cvar, format='png', bbox_inches='tight')
    buf_cvar.seek(0)
    all_figs.append(buf_cvar)
else:
    st.info("Optimized CVaR Backtesting requires 'Modal' and 'Arrivals' columns.")

# -----------------------
# Export / Email functionality
# -----------------------
st.subheader("✉️ Export and Email Results")

# Prepare default email sender info (from secrets if available)
default_sender = ""
default_password = ""
if st.secrets.get("gmail"):
    default_sender = st.secrets["gmail"].get("EMAIL_SENDER", "")
    default_password = st.secrets["gmail"].get("EMAIL_PASSWORD", "")

sender_email = st.text_input("Sender Gmail address (will be used to send the report)", value=default_sender, help="Preferably use an App Password for Gmail.")
sender_password = st.text_input("Sender App Password (Gmail) or SMTP password", value=default_password, type="password")
recipient_email = st.text_input("Recipient email address", value="")
report_name = f"VaR_CVaR_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

# Create aggregated DataFrames for report
df_results_report = pd.DataFrame(all_results)
df_bt_summary_report = pd.DataFrame(backtest_model_summary)

if st.button("Generate & Preview PDF Report"):
    st.info("Generating PDF report — this may take a few seconds.")
    # description
    descr = f"Value-at-Risk & CVaR analysis for {uploaded_file.name}\nDate range: {df.index.min().date()} to {df.index.max().date()}\nGenerated: {datetime.now().isoformat()}"
    best_model_text = ""
    if not df_bt_summary_report.empty:
        # Pick first best model summary to display (or aggregate)
        best_model_text = "\n".join([f"{r['Market']}: BestModel = {r['BestModel']}" for _, r in df_bt_summary_report.reset_index().iterrows()])
    else:
        best_model_text = "No rolling backtest summary available."

    pdf_bytes = generate_pdf_report(
        title="VaR & CVaR — Analysis Report",
        description=descr,
        df_results=df_results_report,
        df_bt_summary=df_bt_summary_report,
        figs=all_figs,
        briefs=all_briefs,
        best_model_text=best_model_text
    )

    # Show PDF download link
    b64 = base64.b64encode(pdf_bytes.read()).decode('utf-8')
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{report_name}">⬇️ Download PDF Report</a>'
    st.markdown(href, unsafe_allow_html=True)
    # Reset buffer pointer
    pdf_bytes.seek(0)

    # store for email sending
    st.session_state['last_report'] = pdf_bytes.getvalue()

if st.button("Send Report via Gmail"):
    if not sender_email or not sender_password or not recipient_email:
        st.error("Please provide sender email, password, and recipient email.")
    elif 'last_report' not in st.session_state:
        st.error("Generate the PDF report first using 'Generate & Preview PDF Report'.")
    else:
        try:
            pdf_bytes_raw = BytesIO(st.session_state['last_report'])
            subject = f"VaR & CVaR Report — {uploaded_file.name}"
            body = f"Attached is the VaR/CVaR analysis report for {uploaded_file.name}.\nGenerated: {datetime.now().isoformat()}"
            attachments = [(report_name, pdf_bytes_raw)]
            send_email_gmail(sender_email, sender_password, recipient_email, subject, body, attachments=attachments)
            st.success(f"Report sent to {recipient_email} via Gmail.")
        except Exception as e:
            st.error(f"Error sending email: {e}")

# Footer / contact
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:gray;">
    🚀 This app was built by <b>Suman L</b> — For help: <a href="mailto:sumanecon.uas@outlook.com">sumanecon.uas@outlook.com</a>
</div>
""", unsafe_allow_html=True)
