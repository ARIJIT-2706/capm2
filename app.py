import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
from datetime import date, timedelta

# Set page configuration
st.set_page_config(
    page_title="Investment Analysis Suite",
    layout="wide"
)

# Custom CSS for professional color scheme - using financial blues and neutrals
st.markdown("""
<style>
    /* Main colors: Dark blue, lighter blue accent, and neutral gray */
    :root {
        --main-color: #0A3161;      /* Deep navy blue - primary */
        --secondary-color: #1E5C97; /* Medium blue - secondary */
        --neutral-color: #F0F3F5;   /* Light gray - background */
        --text-color: #333333;      /* Dark gray - text */
    }
    
    .stApp {
        background-color: var(--neutral-color);
        color: var(--text-color);
    }
    
    .stButton button {
        background-color: var(--main-color);
        color: white;
        border-radius: 4px;
        border: none;
        padding: 8px 16px;
    }
    
    .stButton button:hover {
        background-color: var(--secondary-color);
    }
    
    h1, h2, h3 {
        color: var(--main-color);
        font-weight: 600;
    }
    
    .key-metric {
        background-color: white;
        border-left: 4px solid var(--main-color);
        padding: 15px;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    .info-box {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .st-expander {
        border-left: 2px solid var(--secondary-color);
    }

    /* Customize the sidebar */
    .css-1d391kg {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.sidebar.title("Investment Analysis Suite")
    
    app_mode = st.sidebar.selectbox("Select Analysis Tool", 
                                    ["Asset Return Predictor", "Investment Risk Calculator"])
    
    if app_mode == "Asset Return Predictor":
        expected_return_calculator()
    else:
        beta_calculator()

def expected_return_calculator():
    st.title("Asset Return Predictor")
    
    st.markdown("""
    <div class="info-box">
        <h3>What is this tool?</h3>
        <p>The Asset Return Predictor helps you estimate the expected return of an investment based on 
        market risk factors using the Capital Asset Pricing Model (CAPM). This tool is valuable for:</p>
        <ul>
            <li>Evaluating if an investment offers adequate returns for its risk</li>
            <li>Comparing different investment opportunities</li>
            <li>Setting reasonable return expectations for your portfolio</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Parameters")
        rf = st.number_input("Risk-free Rate (%)", min_value=0.0, max_value=20.0, value=3.0, step=0.1)
        market_return = st.number_input("Expected Market Return (%)", min_value=0.0, max_value=30.0, value=8.0, step=0.1)
        beta = st.number_input("Asset Beta (β)", min_value=-2.0, max_value=5.0, value=1.2, step=0.01)
        
    with col2:
        st.subheader("Results")
        if st.button("Calculate Expected Return"):
            rf_decimal = rf / 100
            market_return_decimal = market_return / 100
            
            # CAPM formula: Expected Return = Risk-free Rate + Beta * (Expected Market Return - Risk-free Rate)
            expected_return = rf_decimal + beta * (market_return_decimal - rf_decimal)
            expected_return_percentage = expected_return * 100
            
            st.markdown(f"""
            <div class="key-metric">
                <h3>Expected Return: {expected_return_percentage:.2f}%</h3>
                <p>Based on:</p>
                <ul>
                    <li>Risk-free rate: {rf:.2f}%</li>
                    <li>Market risk premium: {(market_return - rf):.2f}%</li>
                    <li>Asset beta (β): {beta:.2f}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Visualization of risk-return relationship
            st.subheader("Risk-Return Visualization")
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Create data for Security Market Line
            beta_range = np.linspace(0, 2.5, 100)
            expected_returns = rf_decimal + beta_range * (market_return_decimal - rf_decimal)
            
            # Plot the Security Market Line
            ax.plot(beta_range, expected_returns * 100, 'b-', label='Security Market Line')
            
            # Highlight the specified asset
            ax.scatter([beta], [expected_return * 100], color='red', s=100, label='Your Asset')
            
            # Mark risk-free rate
            ax.scatter([0], [rf], color='green', s=100, label='Risk-free Rate')
            
            ax.set_xlabel('Beta (Risk)')
            ax.set_ylabel('Expected Return (%)')
            ax.set_title('Risk-Return Trade-off')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            st.pyplot(fig)
    
    # Explanation section
    with st.expander("Learn More About Asset Return Prediction"):
        st.markdown("""
        ### How the Asset Return Predictor Works
        
        This tool uses the Capital Asset Pricing Model (CAPM), a fundamental concept in finance that helps investors:
        
        - **Estimate the required return** for taking on additional risk compared to a risk-free investment
        - **Evaluate if an investment is fairly priced** based on its risk characteristics
        
        #### The Formula
        Expected Return = Risk-free Rate + Beta × (Expected Market Return - Risk-free Rate)
        
        #### Key Components
        
        - **Risk-free Rate**: The return on a zero-risk investment (typically government bonds)
        - **Beta**: Measures how much an asset's price moves compared to the overall market (β=1 means it moves exactly with the market)
        - **Market Risk Premium**: The additional return investors expect for taking on market risk (Market Return - Risk-free Rate)
        
        #### Interpreting Results
        
        - Higher beta = Higher expected return (but also higher risk)
        - The Security Market Line shows the risk-return tradeoff for different beta values
        - Assets above the line may be undervalued (offering more return than their risk would suggest)
        - Assets below the line may be overvalued (offering less return than their risk would suggest)
        """)

def beta_calculator():
    st.title("Investment Risk Calculator")
    
    st.markdown("""
    <div class="info-box">
        <h3>What is this tool?</h3>
        <p>The Investment Risk Calculator helps you measure an asset's volatility relative to the market (Beta). 
        This essential metric shows how much an investment tends to move in relation to the broader market, helping you:</p>
        <ul>
            <li>Assess the systematic risk of your investments</li>
            <li>Build a portfolio with your desired level of market sensitivity</li>
            <li>Make more informed decisions about diversification</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Input section
    st.subheader("Enter Asset Information")
    col1, col2 = st.columns(2)
    
    with col1:
        ticker = st.text_input("Asset Ticker Symbol (e.g. AAPL, MSFT)", "AAPL")
        benchmark = st.text_input("Benchmark Index Ticker Symbol", "^GSPC")  # S&P 500 by default
    
    with col2:
        end_date = st.date_input("End Date", date.today())
        period = st.selectbox("Analysis Period", ["1 Year", "2 Years", "3 Years", "5 Years"])
        
        # Calculate start date based on period
        if period == "1 Year":
            start_date = end_date - timedelta(days=365)
        elif period == "2 Years":
            start_date = end_date - timedelta(days=730)
        elif period == "3 Years":
            start_date = end_date - timedelta(days=1095)
        else:
            start_date = end_date - timedelta(days=1825)
    
    # Process and display results
    if st.button("Calculate Risk Profile"):
        try:
            # Get data
            asset_data = yf.download(ticker, start=start_date, end=end_date)
            market_data = yf.download(benchmark, start=start_date, end=end_date)
            
            # Calculate returns
            asset_returns = asset_data['Adj Close'].pct_change().dropna()
            market_returns = market_data['Adj Close'].pct_change().dropna()
            
            # Create DataFrame with both returns
            df = pd.DataFrame({'Asset': asset_returns, 'Market': market_returns}).dropna()
            
            # Calculate Beta
            covariance = df.cov().iloc[0, 1]
            market_variance = df['Market'].var()
            beta = covariance / market_variance
            
            # Calculate correlation
            correlation = df.corr().iloc[0, 1]
            
            # Calculate asset volatility vs market volatility
            asset_volatility = df['Asset'].std() * (252 ** 0.5)  # Annualized
            market_volatility = df['Market'].std() * (252 ** 0.5)  # Annualized
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="key-metric">
                    <h3>Beta (β): {beta:.2f}</h3>
                    <p>Interpretation:</p>
                    <ul>
                        {"<li>Less volatile than the market</li>" if beta < 0.9 else ""}
                        {"<li>Similar volatility to the market</li>" if 0.9 <= beta <= 1.1 else ""}
                        {"<li>More volatile than the market</li>" if beta > 1.1 else ""}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="key-metric">
                    <h3>Correlation with Market: {correlation:.2f}</h3>
                    <p>How closely the asset moves with the market</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="key-metric">
                    <h3>Asset Volatility: {asset_volatility*100:.2f}%</h3>
                    <p>Annualized standard deviation of returns</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="key-metric">
                    <h3>Market Volatility: {market_volatility*100:.2f}%</h3>
                    <p>Annualized standard deviation of market returns</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Create visualization
            st.subheader("Return Comparison")
            
            # Scatter plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df['Market'], df['Asset'], alpha=0.5, color='#1E5C97')
            
            # Regression line
            coefficients = np.polyfit(df['Market'], df['Asset'], 1)
            line = np.poly1d(coefficients)
            x_range = np.linspace(df['Market'].min(), df['Market'].max(), 100)
            ax.plot(x_range, line(x_range), color='#0A3161', linewidth=2)
            
            # Reference line (beta = 1)
            ax.plot(x_range, x_range, 'k--', alpha=0.3, label='β = 1.0')
            
            ax.set_xlabel('Market Returns')
            ax.set_ylabel(f'{ticker} Returns')
            ax.set_title(f'{ticker} vs {benchmark} Returns')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
            ax.legend([f'Data Points', f'Regression Line (β = {beta:.2f})', 'Market Line (β = 1.0)'])
            
            st.pyplot(fig)
            
            # Historical price chart
            st.subheader("Historical Price Performance")
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            
            # Normalize price data
            asset_norm = asset_data['Adj Close'] / asset_data['Adj Close'].iloc[0]
            market_norm = market_data['Adj Close'] / market_data['Adj Close'].iloc[0]
            
            ax2.plot(asset_norm, label=ticker, linewidth=2, color='#0A3161')
            ax2.plot(market_norm, label=benchmark, linewidth=2, color='#1E5C97', alpha=0.7)
            
            ax2.set_title(f'Normalized Price: {ticker} vs {benchmark}')
            ax2.set_ylabel('Normalized Price (Base=1)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            st.pyplot(fig2)
            
        except Exception as e:
            st.error(f"Error: {e}")
            st.warning("Please check the ticker symbols and try again. Make sure you're using valid symbols.")
    
    # Educational information
    with st.expander("Learn More About Beta and Investment Risk"):
        st.markdown("""
        ### Understanding Beta and Investment Risk
        
        Beta (β) is a key measure of an investment's volatility compared to the market, helping you:
        
        #### Interpreting Beta Values
        
        - **β > 1**: More volatile than the market
          - Example: β = 1.5 means the asset tends to move 150% for each 100% move in the market
          - Higher potential returns but also higher risk
        
        - **β = 1**: Same volatility as the market
          - Moves in tandem with market fluctuations
        
        - **β < 1**: Less volatile than the market
          - Example: β = 0.7 means the asset tends to move 70% for each 100% move in the market
          - More stability during market downturns, but potentially lower returns in bull markets
        
        - **Negative β**: Moves opposite to the market
          - Rare for most stocks, but can be useful for portfolio diversification
        
        #### Using Beta in Your Investment Strategy
        
        - **Growth investors** may target high-beta stocks for greater upside potential
        - **Conservative investors** often prefer low-beta stocks for stability
        - **Portfolio management** uses beta to balance risk across different holdings
        
        #### Limitations to Consider
        
        - Beta is backward-looking and may not predict future performance
        - Based on historical price movements, not fundamentals
        - Most useful when measured over longer time periods
        - Different benchmarks can yield different beta values
        """)

if __name__ == "__main__":
    main()
