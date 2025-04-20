import numpy as np
import pandas as pd
import datetime
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data import TimeFrame
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Request SPY data with daily bars
request = StockBarsRequest(
    symbol_or_symbols=["SPY"],
    timeframe=TimeFrame.Day,
    start=datetime.datetime(2021, 1, 1),
    end=datetime.datetime.now()
)

bars = client.get_stock_bars(request).df
dataset = bars[bars.index.get_level_values(0) == 'SPY'].copy()
dataset.reset_index(inplace=True, drop=True)

# Plot candlestick chart with SMAs
def plot_stock_data(df):
    # Calculate SMAs
    df['SMA5'] = df['close'].rolling(window=5).mean()
    df['SMA20'] = df['close'].rolling(window=20).mean()
    df['SMA50'] = df['close'].rolling(window=50).mean()

    # Create figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        subplot_titles=('Candlestick with SMAs', 'Volume'),
                        row_width=[0.7, 0.3])

    # Add candlestick
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['open'],
                                 high=df['high'],
                                 low=df['low'],
                                 close=df['close'],
                                 name="OHLC"),
                  row=1, col=1)

    # Add SMAs
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA5'],
                             line=dict(color="blue", width=1),
                             name='SMA 5'),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'],
                             line=dict(color="orange", width=1),
                             name='SMA 20'),
                  row=1, col=1)

    # Add Volume
    colors = ['green' if c > o else 'red' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'],
                         marker_color=colors,
                         width=2.0,
                         name='Volume'),
                  row=2, col=1)

    # Update layout
    fig.update_layout(
        title='SPY Stock Price and Volume Analysis',
        yaxis_title='Stock Price (USD)',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False,
        height=800
    )

    fig.show()

# Add technical indicators
def add_indicators(df):
    df['SMA5'] = df['close'].rolling(window=5).mean()
    df['SMA20'] = df['close'].rolling(window=20).mean()
    df['SMA50'] = df['close'].rolling(window=50).mean()

    df['Price_Change'] = df['close'].pct_change()
    df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=7).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=7).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df

# Prepare data for LSTM
def prepare_data(df, look_back=10):
    features = ['close', 'SMA5', 'SMA20', 'SMA50', 'Price_Change', 'RSI']
    df = df.dropna()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])

    X, y = [], []
    for i in range(look_back, len(scaled_data) - 1):
        X.append(scaled_data[i - look_back:i])
        y.append(df['Target'].iloc[i])

    return np.array(X), np.array(y)

# Create LSTM model
def create_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# Main execution
if __name__ == "__main__":
    symbol = 'SPY'
    look_back = 10

    print("Fetching stock data...")
    df = dataset.copy()

    print("Plotting stock data...")
    plot_stock_data(df)

    print("Calculating technical indicators...")
    df = add_indicators(df)

    print("Preparing data for LSTM...")
    X, y = prepare_data(df, look_back)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print("Training model...")
    model = create_model(input_shape=(look_back, X.shape[2]))
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    # Evaluate model
    train_score = model.evaluate(X_train, y_train, verbose=0)
    test_score = model.evaluate(X_test, y_test, verbose=0)

    print("\nModel Performance Explanation:")
    print(f"Train accuracy: {train_score[1]:.4f}")
    print(f"Test accuracy: {test_score[1]:.4f}")

    # Make prediction for tomorrow
    last_sequence = X[-1:]
    tomorrow_pred = model.predict(last_sequence)[0][0]

    print(f"\nPrediction for tomorrow:")
    print(f"Probability of price increase: {tomorrow_pred:.2%}")
    print(f"Predicted direction: {'UP' if tomorrow_pred > 0.5 else 'DOWN'}")