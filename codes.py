# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import yfinance as yf
import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# ---------------------------
# 1. APP CONFIG
# ---------------------------
st.set_page_config(page_title="VaR & CVaR Backtesting Suite", layout="wide")

# ---------------------------
# 2. FUNCTIONS
# ---------------------------

def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data['Returns'] = data['Adj Close'].pct_change().dropna()
    return data.dropna()

def var_historical(returns, alpha):
    return np.percentile(returns, (1-alpha)*100)

def var_parametric(returns, alpha):
    mean = returns.mean()
    std = returns.std()
    return norm.ppf(1-alpha, mean, std)

def var_monte_carlo(returns, alpha, simulations=10000):
    mean = returns.mean()
    std = returns.std()
    sim_returns = np.random.normal(mean, std, simulations)
    return np.percentile(sim_returns, (1-alpha)*100)

def cvar(returns, alpha):
    var_level = var_historical(returns, alpha)
    return returns[returns <= var_level].mean()

def backtest_var(returns, var_series):
    breaches = returns < var_series
    breach_count = breaches.sum()
    breach_percent = breach_count / len(returns) * 100
    avg_loss_breach = returns[breaches].mean()
    return breach_count, breach_percent, avg_loss_breach

def generate_pdf(report_title, summary_dict, fig):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = [Paragraph(report_title, styles['Title']), Spacer(1, 20)]
    
    for key, value in summary_dict.items():
        story.append(Paragraph(f"<b>{key}:</b> {value}", styles['Normal']))
        story.append(Spacer(1, 10))
    
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    story.append(Image(img_buffer))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def send_email_with_attachment(sender_email, app_password, recipient_email, subject, body, attachment_bytes, filename="report.pdf"):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    
    msg.attach(MIMEText(body, 'plain'))
    
    part = MIMEApplication(attachment_bytes.read(), Name=filename)
    attachment_bytes.seek(0)
    part['Content-Disposition'] = f'attachment; filename="{filename}"'
    msg.attach(part)
    
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.login(sender_email, app_password)
    server.send_message(msg)
    server.quit()

# ---------------------------
# 3. STREAMLIT UI
# ---------------------------
st.title("📊 VaR & CVaR Analysis with Backtesting and Email Reports")

ticker = st.text_input("Enter Stock Ticker", "AAPL")
start_date = st.date_input("Start Date", datetime.date(2015, 1, 1))
end_date = st.date_input("End Date", datetime.date.today())
alpha = st.slider("Confidence Level (α)", 0.90, 0.99, 0.95)

if st.button("Run Analysis"):
    data = load_data(ticker, start_date, end_date)
    returns = data['Returns']

    # VaR Calculations
    var_hist = var_historical(returns, alpha)
    var_para = var_parametric(returns, alpha)
    var_mc = var_monte_carlo(returns, alpha)
    cvar_val = cvar(returns, alpha)

    # Backtesting
    hist_series = pd.Series(var_hist, index=returns.index)
    para_series = pd.Series(var_para, index=returns.index)
    mc_series = pd.Series(var_mc, index=returns.index)

    hist_bt = backtest_var(returns, hist_series)
    para_bt = backtest_var(returns, para_series)
    mc_bt = backtest_var(returns, mc_series)

    # Model comparison by breach %
    models_bt = {
        "Historical": hist_bt[1],
        "Parametric": para_bt[1],
        "Monte Carlo": mc_bt[1]
    }
    best_model = min(models_bt, key=models_bt.get)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(returns.index, returns, label="Returns")
    ax.axhline(var_hist, color='r', linestyle='--', label=f'Hist VaR {alpha}')
    ax.axhline(var_para, color='g', linestyle='--', label=f'Param VaR {alpha}')
    ax.axhline(var_mc, color='b', linestyle='--', label=f'MC VaR {alpha}')
    ax.legend()
    st.pyplot(fig)

    # Summary
    summary = {
        "Ticker": ticker,
        "VaR Historical": round(var_hist, 5),
        "VaR Parametric": round(var_para, 5),
        "VaR Monte Carlo": round(var_mc, 5),
        "CVaR": round(cvar_val, 5),
        "Best Model (Lowest Breach %)": best_model,
        "Historical Breach %": f"{hist_bt[1]:.2f}%",
        "Parametric Breach %": f"{para_bt[1]:.2f}%",
        "Monte Carlo Breach %": f"{mc_bt[1]:.2f}%",
        "Avg Loss on Breach (Hist)": round(hist_bt[2], 5)
    }
    st.write(summary)

    # PDF generation
    pdf_buffer = generate_pdf(f"VaR & CVaR Report - {ticker}", summary, fig)
    st.download_button("Download PDF Report", pdf_buffer, file_name=f"{ticker}_VaR_Report.pdf")

    # Email section
    st.subheader("📧 Email the report")
    sender_email = st.text_input("Sender Gmail")
    app_password = st.text_input("Gmail App Password", type="password")
    recipient_email = st.text_input("Recipient Email")
    
    if st.button("Send Email Report"):
        send_email_with_attachment(
            sender_email, app_password, recipient_email,
            f"VaR & CVaR Report - {ticker}",
            "Please find attached the VaR & CVaR backtesting report.",
            pdf_buffer
        )
        st.success("Email sent successfully!")
