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
    page_title="CAPM Calculator",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)

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
        )
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

# Create tabs for the two different calculators
tab1, tab2 = st.tabs(["Multiple Stocks CAPM", "Individual Stock Beta"])

# Tab 1: Capital Asset Pricing Model for multiple stocks
with tab1:
    st.title('Capital Asset Pricing Model 📈')

    # Getting input from user
    col1, col2 = st.columns([1, 1])
    with col1:
        stocks_list = st.multiselect(
            "Choose 4 Stocks",
            ('TSLA', 'AAPL', 'NFLX', 'MGM', 'MSFT', 'AMZN', 'NVDA', 'GOOGL'),
            ['TSLA', 'AAPL', 'MSFT', 'NFLX'],
            key="stock_list",
        )
    with col2:
        year = st.number_input("Number of Years", 1, 10, key="multi_year")

    if st.button("Calculate CAPM", key="calc_capm"):
        try:
            # Downloading data for SP500
            end = datetime.date.today()
            start = datetime.date(datetime.date.today().year - year, datetime.date.today().month, datetime.date.today().day)
            SP500 = web.DataReader(['sp500'], 'fred', start, end)

            # Downloading data for the stocks
            stocks_df = pd.DataFrame()
            for stock in stocks_list:
                data = yf.download(stock, period=f'{year}y')
                stocks_df[f'{stock}'] = data['Close']
            stocks_df.reset_index(inplace=True)
            SP500.reset_index(inplace=True)
            SP500.columns = ['Date', 'sp500']
            stocks_df['Date'] = stocks_df['Date'].astype('datetime64[ns]')
            stocks_df['Date'] = stocks_df['Date'].apply(lambda x: str(x)[:10])
            stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
            stocks_df = pd.merge(stocks_df, SP500, on='Date', how='inner')

            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown('### Dataframe head')
                st.dataframe(stocks_df.head(), use_container_width=True)
            with col2:
                st.markdown('### Dataframe tail')
                st.dataframe(stocks_df.tail(), use_container_width=True)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown('### Price of all the Stocks')
                # Plot interactive chart
                st.plotly_chart(interactive_plot(stocks_df))

            with col2:
                st.markdown('### Price of all the Stocks (After Normalizing)')
                # Plot normalized interactive chart
                st.plotly_chart(interactive_plot(normalize(stocks_df)))

            # Calculating daily return 
            stocks_daily_return = daily_return(stocks_df)

            beta = {}
            alpha = {}

            for i in stocks_daily_return.columns:
                # Ignoring the date and S&P500 Columns 
                if i != 'Date' and i != 'sp500':
                    # Calculate beta and alpha for all stocks
                    b, a = calculate_beta(stocks_daily_return, i)
                    beta[i] = b
                    alpha[i] = a

            col1, col2 = st.columns([1, 1])

            beta_df = pd.DataFrame(columns=['Stock', 'Beta Value'])
            beta_df['Stock'] = beta.keys()
            beta_df['Beta Value'] = [str(round(i, 2)) for i in beta.values()]

            with col1:
                st.markdown('### Calculated Beta Value ')
                st.dataframe(beta_df, use_container_width=True)

            # Calculate return for any security using CAPM  
            rf = 0  # Risk free rate of return
            rm = stocks_daily_return['sp500'].mean() * 252  # Market portfolio return
            return_df = pd.DataFrame()
            stock_list = []
            return_value = []
            for stock, value in beta.items():
                stock_list.append(stock)
                # Calculate return
                return_value.append(str(round(rf + (value * (rm - rf)), 2)))
            return_df['Stock'] = stock_list
            return_df['Return Value'] = return_value

            with col2:
                st.markdown('### Calculated Return using CAPM ')
                st.dataframe(return_df, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.write("Please select valid stocks and years")

# Tab 2: Calculate Beta for individual stock
with tab2:
    st.title('Calculate Beta and Return for Individual Stock')

    # Getting input from user
    col1, col2 = st.columns([1, 1])
    with col1:
        stock = st.selectbox("Choose a stock", ('TSLA', 'AAPL', 'NFLX', 'MGM', 'MSFT', 'AMZN', 'NVDA', 'GOOGL'))
    with col2:
        year = st.number_input("Number of Years", 1, 10, key="single_year")

    if st.button("Calculate Beta", key="calc_beta"):
        try:
            # Downloading data for SP500
            end = datetime.date.today()
            start = datetime.date(datetime.date.today().year - year, datetime.date.today().month, datetime.date.today().day)
            SP500 = web.DataReader(['sp500'], 'fred', start, end)

            # Downloading data for the stock
            stocks_df = yf.download(stock, period=f'{year}y')
            stocks_df = stocks_df[['Close']]
            stocks_df.columns = [f'{stock}']
            stocks_df.reset_index(inplace=True)
            SP500.reset_index(inplace=True)
            SP500.columns = ['Date', 'sp500']
            stocks_df['Date'] = stocks_df['Date'].astype('datetime64[ns]')
            stocks_df['Date'] = stocks_df['Date'].apply(lambda x: str(x)[:10])
            stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
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

            # Showing results
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f'### Beta : {round(beta, 2)}')
                st.markdown(f'### Return : {return_value}%')
            
            # Creating scatter plot with regression line
            fig = px.scatter(stocks_daily_return, x='sp500', y=stock, title=f"{stock} Daily Returns vs S&P500")
            fig.add_scatter(
                x=stocks_daily_return['sp500'],
                y=beta * stocks_daily_return['sp500'] + alpha,
                mode='lines',
                name='Regression Line',
                line=dict(color="crimson")
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.write("Please select a valid stock and year")