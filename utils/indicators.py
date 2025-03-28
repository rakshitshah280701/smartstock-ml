import pandas as pd
import ta

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # Force Series from DataFrame (guaranteed 1D)
    close = pd.Series(df["Close"].values.ravel(), index=df.index)
    high = pd.Series(df["High"].values.ravel(), index=df.index)
    low = pd.Series(df["Low"].values.ravel(), index=df.index)

    # Add indicators
    df["rsi"] = ta.momentum.RSIIndicator(close=close).rsi()
    df["macd"] = ta.trend.MACD(close=close).macd_diff()
    
    bb = ta.volatility.BollingerBands(close=close)
    df["bollinger_h"] = bb.bollinger_hband()
    df["bollinger_l"] = bb.bollinger_lband()
    
    df["stoch"] = ta.momentum.StochasticOscillator(high=high, low=low, close=close).stoch()
    df["adx"] = ta.trend.ADXIndicator(high=high, low=low, close=close).adx()

    df.dropna(inplace=True)
    return df
