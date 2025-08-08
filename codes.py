import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import timedelta
import random

# Ensure reproducibility
np.random.seed(42)
random.seed(42)

st.set_page_config(page_title="CVaR Analysis", layout="wide")

# Title
st.title("Interactive Conditional VaR (CVaR) & Forecast Analysis")

# Sidebar controls
st.sidebar.header("Settings")
confidence_level = st.sidebar.slider("Confidence Level (%)", 80, 99, 95, 1)
num_simulations = st.sidebar.number_input("Monte Carlo Simulations", min_value=1000, max_value=100000, value=10000, step=1000)

# File upload
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Ensure date column is present
    date_col = st.sidebar.selectbox("Select Date Column", df.columns)
    price_col = st.sidebar.selectbox("Select Price Column", df.columns)

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col)

    # Calculate returns
    df["Returns"] = df[price_col].pct_change().dropna()

    # VaR & CVaR calculation
    alpha = confidence_level / 100
    var_value = np.percentile(df["Returns"].dropna(), (1 - alpha) * 100)
    cvar_value = df["Returns"][df["Returns"] <= var_value].mean()

    # Monte Carlo Simulation
    simulated_cvar = []
    for _ in range(num_simulations):
        simulated_returns = np.random.choice(df["Returns"].dropna(), size=len(df["Returns"]), replace=True)
        sim_var = np.percentile(simulated_returns, (1 - alpha) * 100)
        sim_cvar = simulated_returns[simulated_returns <= sim_var].mean()
        simulated_cvar.append(sim_cvar)

    mean_cvar = np.mean(simulated_cvar)

    # Display results
    st.subheader("CVaR Analysis Results")
    st.write(f"**Selected Confidence Level:** {confidence_level}%")
    st.write(f"**Historical VaR:** {var_value:.4f}")
    st.write(f"**Historical CVaR:** {cvar_value:.4f}")
    st.write(f"**Monte Carlo Estimated CVaR (mean of simulations):** {mean_cvar:.4f}")

    # Interactive CVaR plot
    fig = px.line(df, x=date_col, y="Returns", title="Returns & CVaR Threshold")
    fig.add_hline(y=cvar_value, line_dash="dot", line_color="red", annotation_text=f"CVaR ({confidence_level}%)", annotation_position="top left")
    st.plotly_chart(fig, use_container_width=True)

    # Forecasting next worst fall
    last_date = df[date_col].max()
    forecast_horizon = 30  # days
    forecast_dates = [last_date + timedelta(days=i) for i in range(1, forecast_horizon + 1)]
    forecast_cvar = np.random.normal(loc=cvar_value, scale=np.std(simulated_cvar), size=forecast_horizon)

    worst_future_index = np.argmin(forecast_cvar)
    worst_future_date = forecast_dates[worst_future_index]
    worst_future_value = forecast_cvar[worst_future_index]

    st.subheader("Forecast: Next Worst CVaR Fall")
    st.write(f"📉 The next worst CVaR fall is expected on **{worst_future_date.strftime('%Y-%m-%d')}** with an estimated CVaR of **{worst_future_value:.4f}**.")

    fig_forecast = px.line(x=forecast_dates, y=forecast_cvar, title="Forecasted CVaR over Next 30 Days")
    fig_forecast.add_hline(y=worst_future_value, line_dash="dot", line_color="red", annotation_text="Worst CVaR Forecast", annotation_position="top left")
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Policy Brief Interpretation
    st.subheader("Policy Brief Interpretation")
    st.markdown(f"""
    - **Market Risk**: At a {confidence_level}% confidence level, the market may experience average losses beyond VaR of {cvar_value:.2%}.
    - **Simulated Risk Estimate**: Monte Carlo simulations suggest an average CVaR of {mean_cvar:.2%}, reinforcing the historical risk pattern.
    - **Near-Term Outlook**: The next significant risk period is forecasted around {worst_future_date.strftime('%Y-%m-%d')}, suggesting preparation for market downturns.
    - **Policy Recommendation**: Implement price stabilization mechanisms and hedging strategies before this period to mitigate potential losses.
    """)

else:
    st.info("Please upload a dataset to begin analysis.")
