# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import smtplib
import ssl
import sys
import subprocess

# -------- AUTO-INSTALL MISSING PACKAGES --------
required_packages = ["fpdf"]
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

from fpdf import FPDF
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from io import BytesIO

# -------- PAGE CONFIG --------
st.set_page_config(page_title="VaR Analysis Tool", layout="wide")
st.title("📊 Value at Risk (VaR) Analysis Tool with Email & PDF Export")

# -------- FILE UPLOAD --------
uploaded_file = st.file_uploader("Upload CSV/XLSX file", type=["csv", "xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # -------- USER INPUTS --------
    price_col = st.selectbox("Select Price Column", df.columns)
    confidence_level = st.slider("Select Confidence Level (%)", 90, 99, 95)
    initial_investment = st.number_input("Portfolio Value", min_value=1000, value=100000, step=1000)

    returns = df[price_col].pct_change().dropna()

    # -------- HISTORIC VAR --------
    hist_var = np.percentile(returns, 100 - confidence_level) * initial_investment

    # -------- PARAMETRIC VAR --------
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    from scipy.stats import norm
    param_var = (mean_ret - std_ret * norm.ppf(confidence_level / 100)) * initial_investment

    # -------- MONTE CARLO VAR --------
    np.random.seed(42)
    sim_returns = np.random.normal(mean_ret, std_ret, 100000)
    mc_var = np.percentile(sim_returns, 100 - confidence_level) * initial_investment

    # -------- CVaR --------
    cvar_hist = returns[returns <= np.percentile(returns, 100 - confidence_level)].mean() * initial_investment

    # -------- DYNAMIC BREACH STATISTICS --------
    breaches_hist = (returns < (hist_var / initial_investment)).sum()
    breach_pct_hist = breaches_hist / len(returns) * 100
    avg_loss_breach_hist = returns[returns < (hist_var / initial_investment)].mean() * initial_investment

    breaches_param = (returns < (param_var / initial_investment)).sum()
    breach_pct_param = breaches_param / len(returns) * 100
    avg_loss_breach_param = returns[returns < (param_var / initial_investment)].mean() * initial_investment

    breaches_mc = (returns < (mc_var / initial_investment)).sum()
    breach_pct_mc = breaches_mc / len(returns) * 100
    avg_loss_breach_mc = returns[returns < (mc_var / initial_investment)].mean() * initial_investment

    # -------- SELECT BEST MODEL --------
    breach_df = pd.DataFrame({
        "Model": ["Historic", "Parametric", "Monte Carlo"],
        "Breach %": [breach_pct_hist, breach_pct_param, breach_pct_mc],
        "Avg Loss When Breached": [avg_loss_breach_hist, avg_loss_breach_param, avg_loss_breach_mc]
    })
    best_model = breach_df.loc[breach_df["Breach %"].idxmin(), "Model"]

    # -------- DISPLAY RESULTS --------
    st.subheader("Results")
    st.write(f"Historic VaR: {hist_var:,.2f}")
    st.write(f"Parametric VaR: {param_var:,.2f}")
    st.write(f"Monte Carlo VaR: {mc_var:,.2f}")
    st.write(f"CVaR (Historic): {cvar_hist:,.2f}")
    st.write(f"Best Performing Model (Lowest Breach %): **{best_model}**")

    st.subheader("Breach Statistics")
    st.dataframe(breach_df)

    # -------- PLOT RETURNS --------
    fig, ax = plt.subplots()
    sns.histplot(returns, bins=50, kde=True, ax=ax)
    ax.axvline(hist_var / initial_investment, color="red", linestyle="--", label="Historic VaR")
    ax.axvline(param_var / initial_investment, color="blue", linestyle="--", label="Parametric VaR")
    ax.axvline(mc_var / initial_investment, color="green", linestyle="--", label="Monte Carlo VaR")
    ax.legend()
    st.pyplot(fig)

    # -------- PDF EXPORT --------
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, "Value at Risk Analysis Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, f"Historic VaR: {hist_var:,.2f}\nParametric VaR: {param_var:,.2f}\nMonte Carlo VaR: {mc_var:,.2f}\nCVaR (Historic): {cvar_hist:,.2f}\nBest Model: {best_model}")
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)

    st.download_button("Download PDF Report", data=pdf_output, file_name="var_report.pdf", mime="application/pdf")

    # -------- EMAIL RESULTS --------
    st.subheader("Email Report")
    email_to = st.text_input("Recipient Email")
    email_user = st.text_input("Your Gmail Address")
    email_pass = st.text_input("Your Gmail App Password", type="password")

    if st.button("Send Email Report"):
        try:
            msg = MIMEMultipart()
            msg["From"] = email_user
            msg["To"] = email_to
            msg["Subject"] = "VaR Analysis Report"
            msg.attach(MIMEText("Please find attached your VaR Analysis Report.", "plain"))

            part = MIMEBase("application", "octet-stream")
            part.set_payload(pdf_output.getvalue())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename= var_report.pdf")
            msg.attach(part)

            context = ssl.create_default_context()
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                server.login(email_user, email_pass)
                server.send_message(msg)

            st.success("Email sent successfully!")
        except Exception as e:
            st.error(f"Error sending email: {e}")
