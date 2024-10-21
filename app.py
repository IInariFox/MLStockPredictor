# Machine Learning-Driven Financial Portfolio Prediction System

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scipy.optimize import minimize
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Set up Streamlit app
st.title('Machine Learning-Driven Financial Portfolio Prediction System')
st.write('This app predicts stock prices using LSTM models and optimizes portfolio allocation using Modern Portfolio Theory.')

# Define stock tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Get user input for date range
end_date = datetime.today()
start_date = end_date - timedelta(days=365*5)  # Last 5 years

# Function to fetch stock data
@st.cache
def get_stock_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        data[ticker] = stock_data['Close']
    return pd.DataFrame(data)

# Fetch data
price_data = get_stock_data(tickers, start_date, end_date)
st.write('### Historical Stock Prices')
st.line_chart(price_data)

# Prepare data for LSTM
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Parameters
seq_length = 60  # Past 60 days
epochs = 10
batch_size = 32

models = {}
future_prices = {}
expected_returns = {}

# Loop through each stock to build and train LSTM models
for ticker in tickers:
    st.write(f'#### Processing {ticker}')
    stock_prices = price_data[ticker].values.reshape(-1,1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(stock_prices)

    X, y = create_sequences(scaled_data, seq_length)

    # Split data into training and testing
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(seq_length,1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    models[ticker] = model

    # Predict next day's price
    last_seq = scaled_data[-seq_length:]
    last_seq = last_seq.reshape(1, seq_length, 1)
    next_price_scaled = model.predict(last_seq)
    next_price = scaler.inverse_transform(next_price_scaled)
    future_prices[ticker] = next_price[0][0]

    # Calculate expected return
    last_price = stock_prices[-1][0]
    expected_return = (next_price[0][0] - last_price) / last_price
    expected_returns[ticker] = expected_return

    st.write(f"Predicted next price for {ticker}: ${next_price[0][0]:.2f}")
    st.write(f"Expected return for {ticker}: {expected_return*100:.2f}%")

# Calculate covariance matrix of historical returns
returns = price_data.pct_change().dropna()
cov_matrix = returns.cov()

# Define portfolio optimization functions
def portfolio_performance(weights, expected_returns, cov_matrix):
    portfolio_return = np.dot(weights, list(expected_returns.values()))
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_std

def neg_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate=0):
    p_return, p_std = portfolio_performance(weights, expected_returns, cov_matrix)
    return - (p_return - risk_free_rate) / p_std

# Set optimization constraints and bounds
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0,1) for _ in range(len(tickers)))
init_guess = len(tickers) * [1./len(tickers),]

# Optimize portfolio
opt_results = minimize(neg_sharpe_ratio, init_guess,
                       args=(expected_returns, cov_matrix),
                       method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = opt_results.x
portfolio_return, portfolio_std = portfolio_performance(optimal_weights, expected_returns, cov_matrix)
sharpe_ratio = (portfolio_return) / portfolio_std

# Display optimal portfolio allocation
st.write('### Optimal Portfolio Allocation')
for ticker, weight in zip(tickers, optimal_weights):
    st.write(f"{ticker}: {weight*100:.2f}%")

st.write(f"Expected Portfolio Return: {portfolio_return*100:.2f}%")
st.write(f"Expected Portfolio Volatility: {portfolio_std*100:.2f}%")
st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")
