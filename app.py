# importing libraries
import streamlit as st
import pandas_datareader.data as web
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Portfolio Returns Calculator",
    page_icon="💼",
    layout="wide",
)

# Define professional color palette inspired by financial institutions
# Using a palette of blue (trust/stability), green (growth), and gray (professionalism)
COLOR_PRIMARY = "#0F3057"    # Dark blue - primary color
COLOR_SECONDARY = "#00587A"  # Medium blue - secondary color
COLOR_ACCENT = "#008891"     # Teal - accent color
COLOR_TEXT_DARK = "#333333"  # Dark gray - for text
PLOT_BG = "#F7F7F7"          # Light gray background for plots

# Custom CSS to apply the color scheme
st.markdown("""
<style>
    .main {
        background-color: #FFFFFF;
    }
    .stButton>button {
        background-color: """ + COLOR_PRIMARY + """;
        color: white;
    }
    .stButton>button:hover {
        background-color: """ + COLOR_SECONDARY + """;
        color: white;
    }
    h1, h2, h3 {
        color: """ + COLOR_PRIMARY + """;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #FFFFFF;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: """ + COLOR_PRIMARY + """;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# App header and introduction
st.image("https://www.svgrepo.com/show/9778/pie-chart.svg", width=100)
st.title('Portfolio Returns Calculator')

with st.expander("📚 Learn about this tool"):
    st.markdown("""
    ### What is this tool?
    This calculator helps investors estimate expected returns for stocks based on their market risk. It uses the **Capital Asset Pricing Model (CAPM)**, a foundational concept in modern finance theory.
    
    ### How does it work?
    1. **Beta calculation**: The tool calculates how volatile a stock is compared to the overall market (S&P 500).
    2. **Expected return**: Using beta, it estimates what return you might expect for taking on that level of risk.
    
    ### Key terms:
    - **Beta**: Measures a stock's volatility compared to the market. A beta of 1 means the stock moves with the market.
    - **Expected Return**: The estimated return of an investment based on its risk level.
    - **S&P 500**: An index of 500 large US companies, often used as a benchmark for the overall market.
    
    ### How to use this tool:
    - Select stocks you're interested in analyzing
    - Choose a time period (in years)
    - Review the calculated beta and expected returns
    """)

# Function to plot interactive plot with custom styling
def interactive_plot(df):
    fig = px.line()
    colors = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT, "#6E7582", "#4C8BF5", "#2C4770"]
    
    for i, col in enumerate(df.columns[1:]):
        color_idx = i % len(colors)
        fig.add_scatter(x=df['Date'], y=df[col], name=col, line=dict(color=colors[color_idx]))
    
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
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PLOT_BG,
        font=dict(color=COLOR_TEXT_DARK)
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

# Create tabs for the two different calculators
tab1, tab2 = st.tabs(["📊 Portfolio Analysis", "📈 Single Stock Analysis"])

# Tab 1: Capital Asset Pricing Model for multiple stocks
with tab1:
    st.header('Portfolio Risk & Return Analysis')
    
    st.markdown("""
    This tool analyzes multiple stocks to calculate their risk (beta) and expected returns based on historical data.
    """)

    # Getting input from user
    col1, col2 = st.columns([1, 1])
    with col1:
        stocks_list = st.multiselect(
            "Select up to 4 stocks for your portfolio",
            ('TSLA', 'AAPL', 'NFLX', 'MGM', 'MSFT', 'AMZN', 'NVDA', 'GOOGL'),
            ['TSLA', 'AAPL', 'MSFT', 'NFLX'],
            key="stock_list",
        )
    with col2:
        year = st.number_input("Analysis period (years)", 1, 10, key="multi_year")

    if st.button("Analyze Portfolio", key="calc_capm"):
        if not stocks_list:
            st.error("Please select at least one stock to analyze.")
        else:
            try:
                with st.spinner("Fetching market data..."):
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
                    
                    stocks_df['Date'] = stocks_df['Date'].astype('datetime64[ns]')
                    stocks_df['Date'] = stocks_df['Date'].apply(lambda x: str(x)[:10])
                    stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
                    
                    # Ensure both dataframes have datetime format for Date column before merging
                    SP500['Date'] = pd.to_datetime(SP500['Date'])
                    
                    # Merge dataframes
                    stocks_df = pd.merge(stocks_df, SP500, on='Date', how='inner')

                st.success("Analysis complete!")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown('### Price Trends')
                    # Plot interactive chart
                    st.plotly_chart(interactive_plot(stocks_df), use_container_width=True)

                with col2:
                    st.markdown('### Normalized Performance')
                    # Plot normalized interactive chart
                    st.plotly_chart(interactive_plot(normalize(stocks_df)), use_container_width=True)

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
                    st.markdown('### Risk Analysis (Beta Values)')
                    st.markdown("""
                    **Beta interpretation:**
                    - β > 1: More volatile than market
                    - β = 1: Same volatility as market
                    - β < 1: Less volatile than market
                    """)
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
                    st.markdown('### Expected Returns (CAPM)')
                    st.markdown("""
                    Based on each stock's risk level, these are the expected annual returns according to CAPM.
                    """)
                    st.dataframe(return_df, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.write("Please select valid stocks and years")

# Tab 2: Calculate Beta for individual stock
with tab2:
    st.header('Single Stock Risk Analysis')
    
    st.markdown("""
    This tool analyzes an individual stock's relationship with the market to determine its risk profile and expected return.
    """)

    # Getting input from user
    col1, col2 = st.columns([1, 1])
    with col1:
        stock = st.selectbox("Select a stock to analyze", ('TSLA', 'AAPL', 'NFLX', 'MGM', 'MSFT', 'AMZN', 'NVDA', 'GOOGL'))
    with col2:
        year = st.number_input("Analysis period (years)", 1, 10, key="single_year")

    if st.button("Analyze Stock", key="calc_beta"):
        try:
            with st.spinner("Analyzing market data..."):
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

            st.success("Analysis complete!")
            
            # Showing results
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown('### Risk Analysis')
                
                # Custom styling for beta value
                beta_rounded = round(beta, 2)
                if beta_rounded > 1.1:
                    beta_color = "red"
                    risk_level = "High Risk"
                elif beta_rounded < 0.9:
                    beta_color = "green"
                    risk_level = "Low Risk"
                else:
                    beta_color = "orange"
                    risk_level = "Medium Risk"
                
                st.markdown(f"""
                <div style="background-color:#f8f9fa; padding:20px; border-radius:10px;">
                    <h3 style="margin:0;">Beta: <span style="color:{beta_color};">{beta_rounded}</span></h3>
                    <p>Risk level: <strong>{risk_level}</strong></p>
                    <h3 style="margin-top:20px;">Expected Return: {return_value}%</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                **What this means:**
                - **Beta > 1**: Stock is more volatile than the market
                - **Beta = 1**: Stock moves with the market
                - **Beta < 1**: Stock is less volatile than the market
                """)
            
            # Creating scatter plot with regression line
            fig = px.scatter(stocks_daily_return, x='sp500', y=stock, 
                             title=f"{stock} Daily Returns vs S&P500",
                             labels={"sp500": "S&P 500 Return (%)", stock: f"{stock} Return (%)"},
                             color_discrete_sequence=[COLOR_SECONDARY])
            
            fig.add_scatter(
                x=stocks_daily_return['sp500'],
                y=beta * stocks_daily_return['sp500'] + alpha,
                mode='lines',
                name='Regression Line',
                line=dict(color=COLOR_PRIMARY, width=2)
            )
            
            fig.update_layout(
                plot_bgcolor=PLOT_BG,
                paper_bgcolor=PLOT_BG,
                font=dict(color=COLOR_TEXT_DARK),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("How to interpret this chart"):
                st.markdown("""
                This scatter plot shows the relationship between daily returns of your selected stock and the market (S&P 500):
                
                - **Each point**: Represents one day's returns for both the stock and the market
                - **Regression line**: Shows the average relationship between stock and market movements
                - **Steeper line (higher beta)**: Indicates the stock tends to amplify market movements
                - **Flatter line (lower beta)**: Indicates the stock is less responsive to market movements
                
                The beta value is the slope of this regression line.
                """)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.write("Please select a valid stock and year")

# Add footer with disclaimer
st.markdown("""---""")
st.markdown("""
<div style="text-align: center; color: #888888; font-size: 0.8em;">
    <p>Disclaimer: This tool is for educational purposes only. Past performance is not indicative of future results.</p>
</div>
""", unsafe_allow_html=True)
