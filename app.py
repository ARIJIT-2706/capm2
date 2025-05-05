# importing libraries
import streamlit as st
import pandas_datareader.data as web
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
import io

# Set page config
st.set_page_config(
    page_title="Advanced Portfolio Analytics",
    page_icon="📊",
    layout="wide",
)

# Set color theme
primary_color = "#1E88E5"  # Professional blue
secondary_color = "#FFC107"  # Amber accent
text_color = "#212121"  # Dark text
background_color = "#FAFAFA"  # Light background

# Apply custom CSS
st.markdown("""
<style>
    .main {
        background-color: #FAFAFA;
    }
    h1, h2, h3 {
        color: #1E88E5;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
    }
    .stButton>button:hover {
        background-color: #1565C0;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #FAFAFA;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Function to plot interactive plot
def interactive_plot(df):
    fig = px.line()
    for i in df.columns[1:]:
        fig.add_scatter(x=df['Date'], y=df[i], name=i)
    fig.update_layout(
        width=450,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=text_color)
    )
    return fig

# Function to normalize the prices based on the initial price
def normalize(df):
    x = df.copy()
    for i in x.columns[1:]:
        x[i] = x[i] / x[i][0]
    return x

# Function to calculate the daily returns 
def daily_return(df):
    df_daily_return = df.copy()
    for i in df.columns[1:]:
        for j in range(1, len(df)):
            df_daily_return[i][j] = ((df[i][j] - df[i][j-1]) / df[i][j-1]) * 100
        df_daily_return[i][0] = 0
    return df_daily_return

# Function to calculate beta
def calculate_beta(stocks_daily_return, stock):
    # Fit a polynomial between the stock and the S&P500
    b, a = np.polyfit(stocks_daily_return['sp500'], stocks_daily_return[stock], 1)
    return b, a

# Function to get market data
def get_sp500_data(start_date, end_date):
    try:
        # Try using pandas-datareader first
        SP500 = web.DataReader(['sp500'], 'fred', start_date, end_date)
        SP500.reset_index(inplace=True)
        SP500.columns = ['Date', 'sp500']
        return SP500
    except Exception as e:
        st.warning("Couldn't fetch S&P500 data from FRED. Using ^GSPC from Yahoo Finance instead.")
        # Use ^GSPC from Yahoo Finance as a fallback
        sp_data = yf.download('^GSPC', start=start_date, end=end_date)
        sp_data = sp_data[['Close']].reset_index()
        sp_data.columns = ['Date', 'sp500']
        return sp_data

# Function to calculate portfolio performance
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

# Function to generate random portfolios for efficient frontier
def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_performance(weights, mean_returns, cov_matrix)
        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
        
    return results, weights_record

# Function to optimize for minimum volatility
def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    
    result = minimize(
        lambda weights: portfolio_performance(weights, mean_returns, cov_matrix)[0],
        np.array(num_assets * [1. / num_assets]),
        args=args,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result

# Function to optimize for maximum Sharpe Ratio
def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    
    def neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
        p_var, p_ret = portfolio_performance(weights, mean_returns, cov_matrix)
        return -(p_ret - risk_free_rate) / p_var if p_var > 0 else float('inf')  # Added check to avoid division by zero
    
    result = minimize(
        neg_sharpe,
        np.array(num_assets * [1. / num_assets]),
        args=args,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result

# Function to generate efficient frontier
def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficient_portfolios = []
    num_assets = len(mean_returns)
    
    for ret in returns_range:
        constraints = (
            {'type': 'eq', 'fun': lambda x: portfolio_performance(x, mean_returns, cov_matrix)[1] - ret},
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        )
        bound = (0.0, 1.0)
        bounds = tuple(bound for asset in range(num_assets))
        
        effi_result = minimize(
            lambda x: portfolio_performance(x, mean_returns, cov_matrix)[0],
            np.array(num_assets * [1. / num_assets]),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if effi_result.success:  # Only add successful optimizations
            efficient_portfolios.append(effi_result['x'])
        
    return efficient_portfolios

# Create tabs for the different calculators
tab1, tab2, tab3 = st.tabs(["Asset Pricing", "Individual Stock Analysis", "Portfolio Optimization"])

# Tab 1: Capital Asset Pricing Model for multiple stocks
with tab1:
    st.title('Asset Pricing Analytics 📈')

    # Getting input from user
    col1, col2 = st.columns([1, 1])
    with col1:
        stocks_list = st.multiselect(
            "Select Stocks",
            ('TSLA', 'AAPL', 'NFLX', 'MGM', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'JPM', 'V', 'DIS', 'INTC'),
            ['TSLA', 'AAPL', 'MSFT', 'NFLX'],
            key="stock_list",
        )
    with col2:
        year = st.number_input("Analysis Period (Years)", 1, 10, 3, key="multi_year")

    if st.button("Analyze Assets", key="calc_capm"):
        if not stocks_list:
            st.error("Please select at least one stock to analyze.")
        else:
            try:
                # Downloading data for SP500
                end = datetime.date.today()
                start = datetime.date(end.year - year, end.month, end.day)
                SP500 = get_sp500_data(start, end)

                # Downloading data for the stocks
                stocks_df = pd.DataFrame()
                for stock in stocks_list:
                    try:
                        data = yf.download(stock, period=f'{year}y')
                        if not data.empty:
                            stocks_df[f'{stock}'] = data['Close']
                        else:
                            st.warning(f"No data available for {stock}. Skipping this stock.")
                    except Exception as e:
                        st.warning(f"Error fetching data for {stock}: {e}. Skipping this stock.")
                
                if stocks_df.empty:
                    st.error("Could not retrieve any valid stock data. Please try different stocks.")
                    st.stop()
                    
                stocks_df.reset_index(inplace=True)
                
                # Ensure both dataframes have datetime format for Date column before merging
                stocks_df['Date'] = stocks_df['Date'].astype('datetime64[ns]')
                stocks_df['Date'] = stocks_df['Date'].apply(lambda x: str(x)[:10])
                stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
                SP500['Date'] = pd.to_datetime(SP500['Date'])
                
                # Merge dataframes
                stocks_df = pd.merge(stocks_df, SP500, on='Date', how='inner')

                if len(stocks_df) < 2:
                    st.error("Not enough data points after merging. Please try a larger time period or different stocks.")
                    st.stop()

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown('### Price History')
                    st.dataframe(stocks_df.head(), use_container_width=True)
                with col2:
                    st.markdown('### Recent Data')
                    st.dataframe(stocks_df.tail(), use_container_width=True)

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown('### Absolute Price Performance')
                    # Plot interactive chart
                    st.plotly_chart(interactive_plot(stocks_df))

                with col2:
                    st.markdown('### Normalized Price Performance')
                    # Plot normalized interactive chart
                    st.plotly_chart(interactive_plot(normalize(stocks_df)))

                # Calculating daily return 
                stocks_daily_return = daily_return(stocks_df)

                beta = {}
                alpha = {}

                for i in stocks_daily_return.columns:
                    # Ignoring the Date and S&P500 Columns 
                    if i != 'Date' and i != 'sp500':
                        # Calculate beta and alpha for all stocks
                        b, a = calculate_beta(stocks_daily_return, i)
                        beta[i] = b
                        alpha[i] = a

                col1, col2 = st.columns([1, 1])

                # Create DataFrame properly from dictionary
                beta_df = pd.DataFrame({
                    'Stock': list(beta.keys()),
                    'Beta Value': [round(i, 2) for i in beta.values()]
                })

                with col1:
                    st.markdown('### Market Sensitivity (β)')
                    st.dataframe(beta_df, use_container_width=True)

                # Calculate return for any security using CAPM  
                rf = 0  # Risk free rate of return
                rm = stocks_daily_return['sp500'].mean() * 252  # Market portfolio return
                
                # Create DataFrame properly
                return_df = pd.DataFrame({
                    'Stock': list(beta.keys()),
                    'Expected Return (%)': [round(rf + (value * (rm - rf)), 2) for value in beta.values()]
                })

                with col2:
                    st.markdown('### Expected Returns')
                    st.dataframe(return_df, use_container_width=True)

                # Create a chart comparing beta and expected returns
                fig = px.scatter(
                    x=beta_df['Beta Value'], 
                    y=return_df['Expected Return (%)'],
                    text=beta_df['Stock'],
                    labels={'x': 'Beta (β)', 'y': 'Expected Return (%)'},
                    title='Risk-Return Profile'
                )
                fig.update_traces(
                    marker=dict(size=12, color=primary_color, opacity=0.8),
                    textposition="top center",
                    textfont=dict(size=12)
                )
                fig.update_layout(
                    template="plotly_white",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=text_color)
                )
                
                # Add Security Market Line
                x_range = np.linspace(min(beta_df['Beta Value']) - 0.2, max(beta_df['Beta Value']) + 0.2, 100)
                y_range = rf + (x_range * (rm - rf))
                fig.add_trace(
                    go.Scatter(
                        x=x_range, 
                        y=y_range, 
                        mode='lines', 
                        name='Security Market Line',
                        line=dict(color='crimson', width=2, dash='dash')
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.write("Please select valid stocks and years")

# Tab 2: Calculate Beta for individual stock
with tab2:
    st.title('Individual Stock Analysis')

    # Getting input from user
    col1, col2 = st.columns([1, 1])
    with col1:
        stock = st.selectbox("Select Stock", ('TSLA', 'AAPL', 'NFLX', 'MGM', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'JPM', 'V', 'DIS', 'INTC'))
    with col2:
        year = st.number_input("Analysis Period (Years)", 1, 10, 3, key="single_year")

    if st.button("Analyze Stock", key="calc_beta"):
        try:
            # Downloading data for SP500
            end = datetime.date.today()
            start = datetime.date(end.year - year, end.month, end.day)
            SP500 = get_sp500_data(start, end)

            # Downloading data for the stock
            stocks_df = yf.download(stock, period=f'{year}y')
            if stocks_df.empty:
                st.error(f"No data available for {stock}. Please select a different stock.")
                st.stop()
                
            stocks_df = stocks_df[['Close']]
            stocks_df.columns = [f'{stock}']
            stocks_df.reset_index(inplace=True)
            
            stocks_df['Date'] = stocks_df['Date'].astype('datetime64[ns]')
            stocks_df['Date'] = stocks_df['Date'].apply(lambda x: str(x)[:10])
            stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
            
            # Ensure both dataframes have datetime format for Date column
            SP500['Date'] = pd.to_datetime(SP500['Date'])
            
            # Merge dataframes
            stocks_df = pd.merge(stocks_df, SP500, on='Date', how='inner')

            if len(stocks_df) < 2:
                st.error("Not enough data points after merging. Please try a larger time period.")
                st.stop()

            # Calculating daily return 
            stocks_daily_return = daily_return(stocks_df)
            
            # Calculate beta and alpha
            beta, alpha = calculate_beta(stocks_daily_return, stock)

            # Risk free rate of return
            rf = 0

            # Market portfolio return
            rm = stocks_daily_return['sp500'].mean() * 252

            # Calculate return
            return_value = round(rf + (beta * (rm - rf)), 2)

            # Showing results in a well-formatted card
            st.markdown(f"""
            <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;">
                <h3 style="color: {primary_color}; margin-bottom: 15px;">{stock} Analysis Results</h3>
                <div style="display: flex; gap: 40px;">
                    <div>
                        <p style="font-size: 16px; margin-bottom: 8px;">Beta (β)</p>
                        <p style="font-size: 28px; font-weight: bold; color: {text_color};">{round(beta, 2)}</p>
                        <p style="font-size: 14px; color: #757575;">Measure of market sensitivity</p>
                    </div>
                    <div>
                        <p style="font-size: 16px; margin-bottom: 8px;">Expected Return</p>
                        <p style="font-size: 28px; font-weight: bold; color: {primary_color};">{return_value}%</p>
                        <p style="font-size: 14px; color: #757575;">Based on CAPM model</p>
                    </div>
                    <div>
                        <p style="font-size: 16px; margin-bottom: 8px;">Alpha (α)</p>
                        <p style="font-size: 28px; font-weight: bold; color: {text_color};">{round(alpha, 4)}</p>
                        <p style="font-size: 14px; color: #757575;">Excess return measure</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Create tabs for different chart views
            chart_tab1, chart_tab2 = st.tabs(["Return Analysis", "Price History"])
            
            with chart_tab1:
                # Creating scatter plot with regression line
                fig = px.scatter(
                    stocks_daily_return, 
                    x='sp500', 
                    y=stock, 
                    title=f"{stock} Daily Returns vs Market Returns",
                    labels={'sp500': 'S&P 500 Daily Return (%)', stock: f'{stock} Daily Return (%)'}
                )
                
                fig.add_scatter(
                    x=stocks_daily_return['sp500'],
                    y=beta * stocks_daily_return['sp500'] + alpha,
                    mode='lines',
                    name='Regression Line',
                    line=dict(color="crimson", width=2)
                )
                
                fig.update_layout(
                    template="plotly_white",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=text_color),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with chart_tab2:
                # Create price history chart
                fig = px.line(
                    stocks_df, 
                    x='Date', 
                    y=[stock, 'sp500'], 
                    title=f"{stock} vs S&P 500 Price History",
                    labels={'value': 'Price', 'variable': 'Asset'}
                )
                
                fig.update_layout(
                    template="plotly_white",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=text_color),
                    height=500,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Display interpretation
            st.markdown(f"""
            ### Interpretation
            
            - **Beta ({round(beta, 2)})**: 
              {'This stock is more volatile than the market.' if beta > 1 else 'This stock is less volatile than the market.' if beta < 1 else 'This stock has the same volatility as the market.'}
              {'For every 1% move in the market, this stock tends to move {round(beta, 2)}%.' if beta != 1 else ''}
              
            - **Expected Return ({return_value}%)**:
              Based on the Capital Asset Pricing Model, this is the theoretical required return for the level of systematic risk.
              
            - **Alpha ({round(alpha, 4)})**:
              {'Positive alpha indicates the stock has outperformed what would be expected given its beta.' if alpha > 0 else 'Negative alpha indicates the stock has underperformed what would be expected given its beta.' if alpha < 0 else 'Zero alpha indicates the stock has performed exactly as expected given its beta.'}
            """)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.write("Please select a valid stock and year")

# Tab 3: Modern Portfolio Theory
with tab3:
    st.title('Portfolio Optimization')

    col1, col2 = st.columns([2, 1])
    
    with col1:
        portfolio_stocks = st.multiselect(
            "Select Portfolio Assets",
            ('TSLA', 'AAPL', 'NFLX', 'MGM', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'JPM', 'V', 'DIS', 'INTC', 'PG', 'KO', 'GS', 'WMT'),
            ['AAPL', 'MSFT', 'AMZN', 'JPM', 'KO'],
            key="portfolio_stocks",
        )
    
    with col2:
        portfolio_years = st.number_input("Analysis Period (Years)", 1, 10, value=3, key="portfolio_years")
        investment_amount = st.number_input("Investment Amount ($)", 10000, 10000000, value=100000, step=10000, key="investment_amount")
    
    if st.button("Optimize Portfolio", key="optimize_portfolio"):
        if len(portfolio_stocks) < 2:
            st.error("Please select at least 2 stocks for portfolio optimization")
        else:
            try:
                with st.spinner("Calculating optimal portfolio allocation..."):
                    # Download stock data
                    end_date = datetime.date.today()
                    start_date = datetime.date(end_date.year - portfolio_years, end_date.month, end_date.day)
                    
                    # Get stock data
                    stock_data = yf.download(portfolio_stocks, start=start_date, end=end_date)['Adj Close']
                    
                    # Check if we got valid data
                    if stock_data.empty:
                        st.error("Could not retrieve any stock data. Please try different stocks.")
                        st.stop()
                    
                    # Drop columns with all NaN values
                    stock_data = stock_data.dropna(axis=1, how='all')
                    
                    # Check if we have enough stocks after dropping NaN columns
                    if len(stock_data.columns) < 2:
                        st.error("Not enough valid stocks to perform optimization. Please select more stocks.")
                        st.stop()
                    
                    # If some stocks were dropped, update the portfolio_stocks list
                    portfolio_stocks = list(stock_data.columns)
                    
                    # Fill any remaining NaN values with forward fill then backward fill
                    stock_data = stock_data.fillna(method='ffill').fillna(method='bfill')
                    
                    # Calculate daily returns
                    returns = stock_data.pct_change().dropna()
                    
                    # Make sure we have enough data points
                    if len(returns) < 30:
                        st.error("Not enough data points for reliable optimization. Please try a longer period.")
                        st.stop()
                    
                    # Calculate mean returns and covariance matrix
                    mean_returns = returns.mean()
                    cov_matrix = returns.cov()
                    
                    # Risk free rate
                    risk_free_rate = 0.01
                    
                    # Calculate results for random portfolios
                    num_portfolios = 5000
                    results, weights = random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate)
                    
                    # Create results DataFrame
                    portfolios = pd.DataFrame({
                        'StdDev': results[0, :],
                        'Return': results[1, :],
                        'SharpeRatio': results[2, :]
                    })
                    
                    # Calculate minimum volatility portfolio
                    min_vol_result = min_variance(mean_returns, cov_matrix)
                    min_vol_std, min_vol_return = portfolio_performance(min_vol_result['x'], mean_returns, cov_matrix)
                    
                    # Calculate maximum Sharpe ratio portfolio
                    max_sharpe_result = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
                    max_sharpe_std, max_sharpe_return = portfolio_performance(max_sharpe_result['x'], mean_returns, cov_matrix)
                    
                    # Calculate efficient frontier
                    # Use more conservative targets to avoid optimization failures
                    target_returns = np.linspace(min_vol_return, max(mean_returns) * 252 * 0.8, 20)
                    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target_returns)
                    
                    efficient_std = []
                    efficient_ret = []
                    
                    for portfolio in efficient_portfolios:
                        std, ret = portfolio_performance(portfolio, mean_returns, cov_matrix)
                        efficient_std.append(std)
                        efficient_ret.append(ret)
                    
                    # Create DataFrame for allocation display
                    min_vol_allocation = pd.DataFrame({
                        'Asset': portfolio_stocks,
                        'Weight (%)': [round(i * 100, 2) for i in min_vol_result['x']],
                        'Allocation ($)': [round(i * investment_amount, 2) for i in min_vol_result['x']]
                    }).sort_values(by='Weight (%)', ascending=False)
                    
                    max_sharpe_allocation = pd.DataFrame({
                        'Asset': portfolio_stocks,
                        'Weight (%)': [round(i * 100, 2) for i in max_sharpe_result['x']],
                        'Allocation ($)': [round(i * investment_amount, 2) for i in max_sharpe_result['x']]
                    }).sort_values(by='Weight (%)', ascending=False)
                    
                    # Display annual expected returns and volatility
                    min_vol_annual_ret = round(min_vol_return * 100, 2)
                    min_vol_annual_std = round(min_vol_std * 100, 2)
                    max_sharpe_annual_ret = round(max_sharpe_return * 100, 2)
                    max_sharpe_annual_std = round(max_sharpe_std * 100, 2)
                    max_sharpe_ratio = round((max_sharpe_return - risk_free_rate) / max_sharpe_std, 2)
                    
                    # Display results
                    st.markdown("""
                    ## Portfolio Optimization Results
                    
                    Modern Portfolio Theory helps you find the optimal asset allocation based on your risk preferences.
                    Two key portfolios are shown below:
                    """)
                    
                    # Create two columns for the two
        weights_df = pd.DataFrame({'Ticker': df.columns, 'Weight': optimized.x})
        weights_df.set_index('Ticker', inplace=True)
        st.dataframe(weights_df.style.format("{:.2%}"))

        st.subheader("Portfolio Performance")

        def portfolio_performance(weights, mean_returns, cov_matrix):
            returns = np.dot(weights, mean_returns) * 252
            std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
            sharpe_ratio = (returns - 0.01) / std
            return returns, std, sharpe_ratio

        ret, risk, sharpe = portfolio_performance(optimized.x, mean_returns, cov_matrix)

        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Return", f"{ret:.2%}")
        col2.metric("Risk (Volatility)", f"{risk:.2%}")
        col3.metric("Sharpe Ratio", f"{sharpe:.2f}")

        st.subheader("Efficient Frontier")

        def simulate_random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate=0.01):
            results = {'returns': [], 'volatility': [], 'sharpe': [], 'weights': []}
            for _ in range(num_portfolios):
                weights = np.random.dirichlet(np.ones(len(mean_returns)), 1)[0]
                portfolio_return = np.dot(weights, mean_returns) * 252
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
                results['returns'].append(portfolio_return)
                results['volatility'].append(portfolio_volatility)
                results['sharpe'].append(sharpe_ratio)
                results['weights'].append(weights)
            return results

        results = simulate_random_portfolios(5000, mean_returns, cov_matrix)

        max_sharpe_idx = np.argmax(results['sharpe'])

        fig, ax = plt.subplots(figsize=(10, 6))
        sc = ax.scatter(results['volatility'], results['returns'], c=results['sharpe'], cmap='viridis', alpha=0.7)
        ax.scatter(results['volatility'][max_sharpe_idx], results['returns'][max_sharpe_idx], c='red', marker='*', s=200, label='Max Sharpe')
        ax.set_xlabel('Volatility (Std. Deviation)')
        ax.set_ylabel('Expected Return')
        ax.set_title('Efficient Frontier')
        ax.legend()
        plt.colorbar(sc, label='Sharpe Ratio')
        st.pyplot(fig)

