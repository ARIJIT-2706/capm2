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
        return -(p_ret - risk_free_rate) / p_var
    
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
        
        result = minimize(
            lambda x: portfolio_performance(x, mean_returns, cov_matrix)[0],
            np.array(num_assets * [1. / num_assets]),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        efficient_portfolios.append(result['x'])
        
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
        year = st.number_input("Analysis Period (Years)", 1, 10, key="multi_year")

    if st.button("Analyze Assets", key="calc_capm"):
        try:
            # Downloading data for SP500
            end = datetime.date.today()
            start = datetime.date(end.year - year, end.month, end.day)
            SP500 = get_sp500_data(start, end)

            # Downloading data for the stocks
            stocks_df = pd.DataFrame()
            for stock in stocks_list:
                data = yf.download(stock, period=f'{year}y')
                stocks_df[f'{stock}'] = data['Close']
            stocks_df.reset_index(inplace=True)
            
            # Ensure both dataframes have datetime format for Date column before merging
            stocks_df['Date'] = stocks_df['Date'].astype('datetime64[ns]')
            stocks_df['Date'] = stocks_df['Date'].apply(lambda x: str(x)[:10])
            stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
            SP500['Date'] = pd.to_datetime(SP500['Date'])
            
            # Merge dataframes
            stocks_df = pd.merge(stocks_df, SP500, on='Date', how='inner')

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
        year = st.number_input("Analysis Period (Years)", 1, 10, key="single_year")

    if st.button("Analyze Stock", key="calc_beta"):
        try:
            # Downloading data for SP500
            end = datetime.date.today()
            start = datetime.date(end.year - year, end.month, end.day)
            SP500 = get_sp500_data(start, end)

            # Downloading data for the stock
            stocks_df = yf.download(stock, period=f'{year}y')
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
                    
                    # Calculate daily returns
                    returns = stock_data.pct_change().dropna()
                    
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
                    target_returns = np.linspace(min_vol_return, max(mean_returns) * 252, 50)
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
                    
                    # Create two columns for the two key portfolios
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); height: 100%;">
                            <h3 style="color: {primary_color};">Minimum Volatility Portfolio</h3>
                            <p>This portfolio has the lowest possible risk.</p>
                            <div style="margin: 20px 0; display: flex; gap: 20px;">
                                <div style="text-align: center;">
                                    <p style="font-size: 16px; margin-bottom: 8px;">Expected Return</p>
                                    <p style="font-size: 24px; font-weight: bold; color: {primary_color};">{min_vol_annual_ret}%</p>
                                </div>
                                <div style="text-align: center;">
                                    <p style="font-size: 16px; margin-bottom: 8px;">Volatility</p>
                                    <p style="font-size: 24px; font-weight: bold; color: {text_color};">{min_vol_annual_std}%</p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.dataframe(min_vol_allocation, use_container_width=True)

                    with col2:
                        st.markdown(f"""
                        <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); height: 100%;">
                            <h3 style="color: {primary_color};">Maximum Sharpe Ratio Portfolio</h3>
                            <p>This portfolio has the best risk-adjusted return.</p>
                            <div style="margin: 20px 0; display: flex; gap: 20px;">
                                <div style="text-align: center;">
                                    <p style="font-size: 16px; margin-bottom: 8px;">Expected Return</p>
                                    <p style="font-size: 24px; font-weight: bold; color: {primary_color};">{max_sharpe_annual_ret}%</p>
                                </div>
                                <div style="text-align: center;">
                                    <p style="font-size: 16px; margin-bottom: 8px;">Volatility</p>
                                    <p style="font-size: 24px; font-weight: bold; color: {text_color};">{max_sharpe_annual_std}%</p>
                                </div>
                                <div style="text-align: center;">
                                    <p style="font-size: 16px; margin-bottom: 8px;">Sharpe Ratio</p>
                                    <p style="font-size: 24px; font-weight: bold; color: {secondary_color};">{max_sharpe_ratio}</p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.dataframe(max_sharpe_allocation, use_container_width=True)
                    
                    # Create efficient frontier plot
                    fig = go.Figure()
                    
                    # Add random portfolios
                    fig.add_trace(go.Scatter(
                        x=portfolios['StdDev'], 
                        y=portfolios['Return'],
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=portfolios['SharpeRatio'],
                            colorscale='Viridis',
                            colorbar=dict(title='Sharpe Ratio'),
                            opacity=0.7),
                            name='Random Portfolios'
                    ))
                    
                    # Add efficient frontier
                    fig.add_trace(go.Scatter(
                        x=efficient_std, 
                        y=efficient_ret,
                        mode='lines',
                        line=dict(color='black', width=3),
                        name='Efficient Frontier'
                    ))
                    
                    # Add minimum volatility portfolio
                    fig.add_trace(go.Scatter(
                        x=[min_vol_std], 
                        y=[min_vol_return],
                        mode='markers',
                        marker=dict(size=15, color='red', symbol='star'),
                        name='Minimum Volatility'
                    ))
                    
                    # Add maximum Sharpe ratio portfolio
                    fig.add_trace(go.Scatter(
                        x=[max_sharpe_std], 
                        y=[max_sharpe_return],
                        mode='markers',
                        marker=dict(size=15, color='green', symbol='star'),
                        name='Maximum Sharpe Ratio'
                    ))
                    
                    # Add individual assets
                    for i, asset in enumerate(portfolio_stocks):
                        ann_return = mean_returns[i] * 252
                        ann_std = returns[asset].std() * np.sqrt(252)
                        fig.add_trace(go.Scatter(
                            x=[ann_std], 
                            y=[ann_return],
                            mode='markers+text',
                            marker=dict(size=10),
                            text=asset,
                            textposition="top center",
                            name=asset
                        ))
                    
                    # Update layout
                    fig.update_layout(
                        title='Efficient Frontier',
                        xaxis=dict(title='Annualized Volatility (Standard Deviation)'),
                        yaxis=dict(title='Annualized Return'),
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
                        font=dict(color=text_color),
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display interpretation and educational content
                    st.markdown("""
                    ### Portfolio Theory Explained
                    
                    **Key Concepts:**
                    
                    1. **Efficient Frontier:** The curved line representing the optimal portfolios that offer the highest expected return for a defined level of risk.
                    
                    2. **Minimum Volatility Portfolio:** The portfolio with the lowest possible risk, located at the leftmost point of the efficient frontier.
                    
                    3. **Maximum Sharpe Ratio Portfolio:** The portfolio with the best risk-adjusted return, offering the optimal balance between risk and reward.
                    
                    4. **Diversification Benefit:** By combining assets with different risk-return profiles, you can achieve better performance than individual assets alone.
                    """)
                    
                    # Add risk preference selector
                    st.markdown("### Customize Your Portfolio Based on Risk Preference")
                    risk_preference = st.slider(
                        "Risk Tolerance (Higher = More Aggressive)",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.1
                    )
                    
                    # Calculate custom portfolio based on risk preference
                    # A weighted combination of min volatility and max Sharpe portfolios
                    custom_weights = (1 - risk_preference) * min_vol_result['x'] + risk_preference * max_sharpe_result['x']
                    custom_std, custom_return = portfolio_performance(custom_weights, mean_returns, cov_matrix)
                    
                    custom_allocation = pd.DataFrame({
                        'Asset': portfolio_stocks,
                        'Weight (%)': [round(i * 100, 2) for i in custom_weights],
                        'Allocation ($)': [round(i * investment_amount, 2) for i in custom_weights]
                    }).sort_values(by='Weight (%)', ascending=False)
                    
                    custom_annual_ret = round(custom_return * 100, 2)
                    custom_annual_std = round(custom_std * 100, 2)
                    custom_sharpe = round((custom_return - risk_free_rate) / custom_std, 2)
                    
                    st.markdown(f"""
                    <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                        <h3 style="color: {primary_color};">Your Personalized Portfolio</h3>
                        <p>Based on your risk preference of {risk_preference:.1f} (where 0 is most conservative and 1 is most aggressive)</p>
                        <div style="margin: 20px 0; display: flex; gap: 20px;">
                            <div style="text-align: center;">
                                <p style="font-size: 16px; margin-bottom: 8px;">Expected Return</p>
                                <p style="font-size: 24px; font-weight: bold; color: {primary_color};">{custom_annual_ret}%</p>
                            </div>
                            <div style="text-align: center;">
                                <p style="font-size: 16px; margin-bottom: 8px;">Volatility</p>
                                <p style="font-size: 24px; font-weight: bold; color: {text_color};">{custom_annual_std}%</p>
                            </div>
                            <div style="text-align: center;">
                                <p style="font-size: 16px; margin-bottom: 8px;">Sharpe Ratio</p>
                                <p style="font-size: 24px; font-weight: bold; color: {secondary_color};">{custom_sharpe}</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.dataframe(custom_allocation, use_container_width=True)
                    
                    # Add portfolio performance projection
                    st.markdown("### Portfolio Performance Projection")
                    projection_years = st.slider("Projection Period (Years)", 1, 20, 10)
                    
                    # Create projection
                    years = np.arange(0, projection_years + 1)
                    
                    # Calculate expected growth for different portfolios
                    min_vol_growth = [investment_amount * (1 + min_vol_return) ** year for year in years]
                    max_sharpe_growth = [investment_amount * (1 + max_sharpe_return) ** year for year in years]
                    custom_growth = [investment_amount * (1 + custom_return) ** year for year in years]
                    
                    # Calculate confidence intervals (simple approximation)
                    min_vol_upper = [investment_amount * (1 + min_vol_return + 1.96 * min_vol_std / np.sqrt(252)) ** year for year in years]
                    min_vol_lower = [investment_amount * (1 + min_vol_return - 1.96 * min_vol_std / np.sqrt(252)) ** year for year in years]
                    
                    max_sharpe_upper = [investment_amount * (1 + max_sharpe_return + 1.96 * max_sharpe_std / np.sqrt(252)) ** year for year in years]
                    max_sharpe_lower = [investment_amount * (1 + max_sharpe_return - 1.96 * max_sharpe_std / np.sqrt(252)) ** year for year in years]
                    
                    custom_upper = [investment_amount * (1 + custom_return + 1.96 * custom_std / np.sqrt(252)) ** year for year in years]
                    custom_lower = [investment_amount * (1 + custom_return - 1.96 * custom_std / np.sqrt(252)) ** year for year in years]
                    
                    # Create projection plot
                    fig = go.Figure()
                    
                    # Add min volatility projection
                    fig.add_trace(go.Scatter(
                        x=years,
                        y=min_vol_growth,
                        line=dict(color='red', width=2),
                        name='Minimum Volatility Portfolio'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=years,
                        y=min_vol_upper,
                        line=dict(color='red', width=0),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=years,
                        y=min_vol_lower,
                        line=dict(color='red', width=0),
                        fill='tonexty',
                        fillcolor='rgba(255, 0, 0, 0.1)',
                        showlegend=False
                    ))
                    
                    # Add max Sharpe projection
                    fig.add_trace(go.Scatter(
                        x=years,
                        y=max_sharpe_growth,
                        line=dict(color='green', width=2),
                        name='Maximum Sharpe Portfolio'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=years,
                        y=max_sharpe_upper,
                        line=dict(color='green', width=0),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=years,
                        y=max_sharpe_lower,
                        line=dict(color='green', width=0),
                        fill='tonexty',
                        fillcolor='rgba(0, 255, 0, 0.1)',
                        showlegend=False
                    ))
                    
                    # Add custom projection
                    fig.add_trace(go.Scatter(
                        x=years,
                        y=custom_growth,
                        line=dict(color=primary_color, width=3),
                        name='Your Custom Portfolio'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=years,
                        y=custom_upper,
                        line=dict(color=primary_color, width=0),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=years,
                        y=custom_lower,
                        line=dict(color=primary_color, width=0),
                        fill='tonexty',
                        fillcolor=f'rgba(30, 136, 229, 0.1)',
                        showlegend=False
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title='Portfolio Value Projection',
                        xaxis=dict(title='Years'),
                        yaxis=dict(title='Portfolio Value ($)'),
                        template="plotly_white",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color=text_color),
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add histogram of returns comparison
                    st.markdown("### Return Distribution Comparison")
                    
                    # Historical returns for comparison
                    historical_returns_df = returns.copy()
                    
                    # Calculate portfolio returns based on weights
                    min_vol_returns = historical_returns_df.dot(min_vol_result['x'])
                    max_sharpe_returns = historical_returns_df.dot(max_sharpe_result['x'])
                    custom_returns = historical_returns_df.dot(custom_weights)
                    
                    # Create distribution plot
                    fig = go.Figure()
                    
                    # Add histograms
                    fig.add_trace(go.Histogram(
                        x=min_vol_returns * 100,
                        opacity=0.6,
                        name='Minimum Volatility',
                        marker=dict(color='red')
                    ))
                    
                    fig.add_trace(go.Histogram(
                        x=max_sharpe_returns * 100,
                        opacity=0.6,
                        name='Maximum Sharpe',
                        marker=dict(color='green')
                    ))
                    
                    fig.add_trace(go.Histogram(
                        x=custom_returns * 100,
                        opacity=0.6,
                        name='Your Custom Portfolio',
                        marker=dict(color=primary_color)
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title='Daily Return Distribution',
                        xaxis=dict(title='Daily Return (%)'),
                        yaxis=dict(title='Frequency'),
                        barmode='overlay',
                        template="plotly_white",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color=text_color)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add drawdown analysis
                    st.markdown("### Drawdown Analysis")
                    
                    # Calculate cumulative returns
                    min_vol_cum_returns = (1 + min_vol_returns).cumprod()
                    max_sharpe_cum_returns = (1 + max_sharpe_returns).cumprod()
                    custom_cum_returns = (1 + custom_returns).cumprod()
                    
                    # Calculate drawdowns
                    min_vol_drawdown = min_vol_cum_returns / min_vol_cum_returns.cummax() - 1
                    max_sharpe_drawdown = max_sharpe_cum_returns / max_sharpe_cum_returns.cummax() - 1
                    custom_drawdown = custom_cum_returns / custom_cum_returns.cummax() - 1
                    
                    # Create drawdown plot
                    fig = go.Figure()
                    
                    # Add drawdown lines
                    fig.add_trace(go.Scatter(
                        x=min_vol_drawdown.index,
                        y=min_vol_drawdown * 100,
                        line=dict(color='red', width=2),
                        name='Minimum Volatility'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=max_sharpe_drawdown.index,
                        y=max_sharpe_drawdown * 100,
                        line=dict(color='green', width=2),
                        name='Maximum Sharpe'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=custom_drawdown.index,
                        y=custom_drawdown * 100,
                        line=dict(color=primary_color, width=3),
                        name='Your Custom Portfolio'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title='Portfolio Drawdowns',
                        xaxis=dict(title='Date'),
                        yaxis=dict(title='Drawdown (%)'),
                        template="plotly_white",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color=text_color),
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary statistics
                    min_vol_max_drawdown = round(min_vol_drawdown.min() * 100, 2)
                    max_sharpe_max_drawdown = round(max_sharpe_drawdown.min() * 100, 2)
                    custom_max_drawdown = round(custom_drawdown.min() * 100, 2)
                    
                    # Calculate VaR and CVaR (95%)
                    min_vol_var = round(np.percentile(min_vol_returns, 5) * 100, 2)
                    max_sharpe_var = round(np.percentile(max_sharpe_returns, 5) * 100, 2)
                    custom_var = round(np.percentile(custom_returns, 5) * 100, 2)
                    
                    min_vol_cvar = round(min_vol_returns[min_vol_returns <= np.percentile(min_vol_returns, 5)].mean() * 100, 2)
                    max_sharpe_cvar = round(max_sharpe_returns[max_sharpe_returns <= np.percentile(max_sharpe_returns, 5)].mean() * 100, 2)
                    custom_cvar = round(custom_returns[custom_returns <= np.percentile(custom_returns, 5)].mean() * 100, 2)
                    
                    # Create stats table
                    stats_df = pd.DataFrame({
                        'Portfolio': ['Minimum Volatility', 'Maximum Sharpe', 'Your Custom Portfolio'],
                        'Expected Return (%)': [min_vol_annual_ret, max_sharpe_annual_ret, custom_annual_ret],
                        'Volatility (%)': [min_vol_annual_std, max_sharpe_annual_std, custom_annual_std],
                        'Sharpe Ratio': [round((min_vol_return - risk_free_rate) / min_vol_std, 2), max_sharpe_ratio, custom_sharpe],
                        'Max Drawdown (%)': [min_vol_max_drawdown, max_sharpe_max_drawdown, custom_max_drawdown],
                        'VaR 95% Daily (%)': [min_vol_var, max_sharpe_var, custom_var],
                        'CVaR 95% Daily (%)': [min_vol_cvar, max_sharpe_cvar, custom_cvar]
                    })
                    
                    st.markdown("### Risk-Return Profile Summary")
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # Stress testing
                    st.markdown("### Portfolio Stress Testing")
                    
                    stress_scenarios = {
                        "Market Crash (-20%)": -0.20,
                        "Moderate Correction (-10%)": -0.10,
                        "Minor Dip (-5%)": -0.05,
                        "Strong Bull Market (+10%)": 0.10,
                        "Exceptional Rally (+20%)": 0.20
                    }
                    
                    # Create sensitivity analysis
                    stress_results = []
                    
                    for scenario, market_change in stress_scenarios.items():
                        # Simple approximation using beta relationship
                        stock_betas = {}
                        for stock in portfolio_stocks:
                            b, _ = calculate_beta(returns, stock)
                            stock_betas[stock] = b
                        
                        min_vol_impact = sum([min_vol_result['x'][i] * stock_betas.get(stock, 1) * market_change 
                                            for i, stock in enumerate(portfolio_stocks)])
                        max_sharpe_impact = sum([max_sharpe_result['x'][i] * stock_betas.get(stock, 1) * market_change 
                                                for i, stock in enumerate(portfolio_stocks)])
                        custom_impact = sum([custom_weights[i] * stock_betas.get(stock, 1) * market_change 
                                            for i, stock in enumerate(portfolio_stocks)])
                        
                        min_vol_value = investment_amount * (1 + min_vol_impact)
                        max_sharpe_value = investment_amount * (1 + max_sharpe_impact)
                        custom_value = investment_amount * (1 + custom_impact)
                        
                        stress_results.append({
                            'Scenario': scenario,
                            'Minimum Volatility ($)': round(min_vol_value, 2),
                            'Minimum Volatility (%)': round(min_vol_impact * 100, 2),
                            'Maximum Sharpe ($)': round(max_sharpe_value, 2),
                            'Maximum Sharpe (%)': round(max_sharpe_impact * 100, 2),
                            'Your Portfolio ($)': round(custom_value, 2),
                            'Your Portfolio (%)': round(custom_impact * 100, 2)
                        })
                    
                    stress_df = pd.DataFrame(stress_results)
                    st.dataframe(stress_df, use_container_width=True)
                    
                    # Download portfolio report
                    st.markdown("### Download Portfolio Report")
                    
                    # Prepare downloadable content
                    buffer = io.StringIO()
                    buffer.write("# Advanced Portfolio Analysis Report\n\n")
                    buffer.write(f"Date: {datetime.date.today().strftime('%Y-%m-%d')}\n")
                    buffer.write(f"Analysis Period: {portfolio_years} years\n")
                    buffer.write(f"Initial Investment: ${investment_amount}\n\n")
                    
                    buffer.write("## Portfolio Assets\n")
                    for stock in portfolio_stocks:
                        buffer.write(f"- {stock}\n")
                    
                    buffer.write("\n## Optimal Portfolio Allocations\n\n")
                    buffer.write("### Minimum Volatility Portfolio\n")
                    buffer.write(f"Expected Return: {min_vol_annual_ret}%\n")
                    buffer.write(f"Volatility: {min_vol_annual_std}%\n\n")
                    buffer.write(min_vol_allocation.to_markdown())
                    
                    buffer.write("\n\n### Maximum Sharpe Ratio Portfolio\n")
                    buffer.write(f"Expected Return: {max_sharpe_annual_ret}%\n")
                    buffer.write(f"Volatility: {max_sharpe_annual_std}%\n")
                    buffer.write(f"Sharpe Ratio: {max_sharpe_ratio}\n\n")
                    buffer.write(max_sharpe_allocation.to_markdown())
                    
                    buffer.write("\n\n### Custom Portfolio\n")
                    buffer.write(f"Risk Preference: {risk_preference:.1f}\n")
                    buffer.write(f"Expected Return: {custom_annual_ret}%\n")
                    buffer.write(f"Volatility: {custom_annual_std}%\n")
                    buffer.write(f"Sharpe Ratio: {custom_sharpe}\n\n")
                    buffer.write(custom_allocation.to_markdown())
                    
                    buffer.write("\n\n## Risk Metrics Summary\n\n")
                    buffer.write(stats_df.to_markdown())
                    
                    buffer.write("\n\n## Stress Test Results\n\n")
                    buffer.write(stress_df.to_markdown())
                    
                    report = buffer.getvalue()
                    
                    st.download_button(
                        label="Download Report (Markdown)",
                        data=report,
                        file_name=f"portfolio_report_{datetime.date.today().strftime('%Y%m%d')}.md",
                        mime="text/markdown"
                    )
                    
            except Exception as e:
                st.error(f"An error occurred during portfolio optimization: {e}")
                st.write("Please check your inputs and try again.")

# Add footer with information
st.markdown("""
---
### About Advanced Portfolio Analytics

This tool implements Modern Portfolio Theory concepts to help investors optimize their portfolios for maximum return and minimum risk.
The analysis includes:

- Asset pricing analysis using market models
- Individual stock risk-return profiling
- Portfolio optimization using efficient frontier analysis
- Customized allocations based on risk preferences
- Forward projections and stress testing

*All projections are based on historical data and should be used as educational tools, not financial advice.*
""")

# Import necessary libraries at top that were missing
import io
