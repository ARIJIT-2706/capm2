import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import minimize

st.set_page_config(layout="wide")
st.title("📊 Portfolio Optimization App")

# Sidebar - Input
st.sidebar.header("Portfolio Settings")

option = st.sidebar.radio("Choose input method", ("Upload CSV", "Select Tickers"))

if option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV with 'Date' as index and stock symbols as columns", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, index_col="Date", parse_dates=True)
else:
    tickers = st.sidebar.text_input("Enter comma-separated stock tickers (e.g., AAPL, MSFT, TSLA)", value="AAPL,MSFT,GOOG").split(',')
    start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.today())
    df = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]

# Data Preview
if 'df' in locals():
    st.subheader("Stock Price Data")
    st.dataframe(df.tail())

    # Return Calculation
    returns = df.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    st.subheader("Annualized Returns and Risk")
    annual_returns = mean_returns * 252
    annual_risk = returns.std() * np.sqrt(252)
    metrics_df = pd.DataFrame({'Return': annual_returns, 'Risk': annual_risk})
    st.bar_chart(metrics_df)

    # Portfolio Optimization
    def portfolio_performance(weights, mean_returns, cov_matrix):
        returns = np.dot(weights, mean_returns) * 252
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        return std, returns

    def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
        p_var, p_ret = portfolio_performance(weights, mean_returns, cov_matrix)
        return -(p_ret - risk_free_rate) / p_var

    def check_sum(weights):
        return np.sum(weights) - 1

    num_assets = len(mean_returns)
    initial_weights = [1. / num_assets] * num_assets
    bounds = tuple((0, 1) for asset in range(num_assets))
    constraints = ({'type': 'eq', 'fun': check_sum})

    optimized = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(mean_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    st.subheader("Optimized Portfolio Weights")
    weights_df = pd.DataFrame({'Ticker': df.columns, 'Weight': optimized.x})
    weights_df.set_index("Ticker", inplace=True)
    st.dataframe(weights_df.style.format("{:.2%}"))

    # Plot Efficient Frontier (Optional)
    st.subheader("Portfolio Return vs Risk")

    def simulate_portfolios(num_portfolios, mean_returns, cov_matrix):
        results = np.zeros((3, num_portfolios))
        for i in range(num_portfolios):
            weights = np.random.dirichlet(np.ones(num_assets), size=1)[0]
            portfolio_std_dev, portfolio_return = portfolio_performance(weights, mean_returns, cov_matrix)
            results[0,i] = portfolio_std_dev
            results[1,i] = portfolio_return
            results[2,i] = (portfolio_return - 0.01) / portfolio_std_dev
        return results

    results = simulate_portfolios(5000, mean_returns, cov_matrix)
    max_sharpe_idx = np.argmax(results[2])

    fig, ax = plt.subplots()
    scatter = ax.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis', alpha=0.5)
    ax.scatter(results[0,max_sharpe_idx], results[1,max_sharpe_idx], c='red', marker='*', s=100)
    ax.set_xlabel('Risk (Std Dev)')
    ax.set_ylabel('Return')
    ax.set_title('Efficient Frontier')
    fig.colorbar(scatter, label='Sharpe Ratio')
    st.pyplot(fig)
