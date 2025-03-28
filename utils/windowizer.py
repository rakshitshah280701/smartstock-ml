import numpy as np
import pandas as pd

def create_sequences(df: pd.DataFrame, window_size: int = 30) -> tuple:
    """
    Create sequences of shape (samples, window_size, features) for training.
    Input: past `window_size` days of features.
    Target: next day's close price.

    Returns:
        X: np.array of shape (n_samples, window_size, n_features)
        y: np.array of shape (n_samples,)
    """
    df = df.copy()

    # Ensure the 'Close' column exists
    if "Close" not in df.columns:
        raise ValueError("DataFrame must include 'Close' column.")

    # Extract target
    y = df["Close"].values

    # Drop target column from features
    feature_df = df.drop(columns=["Close"])
    X = feature_df.values

    X_seq, y_seq = [], []

    for i in range(window_size, len(df) - 1):
        X_seq.append(X[i - window_size:i])
        y_seq.append(y[i + 1])  # predict day after the window

    return np.array(X_seq), np.array(y_seq)
